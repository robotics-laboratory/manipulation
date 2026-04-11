"""SmolVLA actor-critic wrapper for PPO with Flow-SDE.

Design
------
SmolVLAPPOActor wraps VLAFlowMatching (policy.model) and adds:

1. ``sample_actions_with_logprob`` — runs the hybrid ODE-SDE denoising pass
   to produce an action chunk and a tractable log-probability from one
   stochastic SDE step.  Also computes a value estimate from the frozen
   VLM prefix features.

2. ``recompute_logprob_and_value`` — efficiently recomputes log π_new(a|s)
   and V_new(s) for a stored transition using current weights (needed inside
   the PPO update loop).  Only ONE ``denoise_step`` call is made per sample.

SmolVLACritic
-------------
A 4-layer MLP that estimates V(s) from mean-pooled VLM prefix embeddings
(shape: B × vlm_hidden_dim → scalar).  The VLM backbone is frozen; the critic
only learns a lightweight regression head.

Critic input: mean-pool of non-padding VLM prefix tokens, shape (B, 960).
The VLM hidden size of 960 is from SmolVLM2-500M.  A guard uses the actual
model weight shape at init-time to be robust to different checkpoints.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from .flow_sde import hybrid_ode_sde_sample, recompute_log_prob

if TYPE_CHECKING:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ---------------------------------------------------------------------------
# Critic MLP
# ---------------------------------------------------------------------------


class SmolVLACritic(nn.Module):
    """4-layer MLP value head on mean-pooled VLM prefix features.

    Args:
        vlm_hidden_dim: VLM text-transformer hidden size (960 for SmolVLM2-500M).
        hidden_dims:    Hidden layer widths.
    """

    def __init__(
        self,
        vlm_hidden_dim: int = 960,
        hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
    ) -> None:
        super().__init__()
        layers = []
        in_dim = vlm_hidden_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, prefix_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prefix_features: Mean-pooled VLM prefix output, shape (B, vlm_hidden_dim).
        Returns:
            values: Shape (B,).
        """
        return self.net(prefix_features).squeeze(-1)


# ---------------------------------------------------------------------------
# Actor-critic wrapper
# ---------------------------------------------------------------------------


class SmolVLAPPOActor:
    """Thin wrapper around SmolVLAPolicy that adds PPO-compatible interfaces.

    This is NOT an nn.Module — it owns the policy and critic as members and
    provides convenience methods for rollout collection and PPO updates.

    Args:
        policy:       Loaded SmolVLAPolicy (full model including VLM).
        noise_level:  SDE noise magnitude ``a`` (default 0.5, per piRL paper).
        num_steps:    Number of denoising steps K (overrides policy.config.num_steps
                      if provided, else uses the policy's default).
        lora_rank:    If > 0, apply LoRA adapters to the VLM backbone for
                      parameter-efficient backbone finetuning.
    """

    def __init__(
        self,
        policy: "SmolVLAPolicy",
        noise_level: float = 0.5,
        num_steps: int | None = None,
        lora_rank: int = 0,
    ) -> None:
        self.policy = policy
        self.flow_model = policy.model  # VLAFlowMatching
        self.noise_level = noise_level
        self.K = num_steps if num_steps is not None else policy.config.num_steps

        # Infer VLM hidden size from the actual projection weight.
        # state_proj maps (max_state_dim,) → (expert_hidden_size,).
        # The VLM text model hidden size is available through vlm_with_expert.
        try:
            vlm_hs = self.flow_model.vlm_with_expert.vlm.config.text_config.hidden_size
        except AttributeError:
            vlm_hs = 960  # SmolVLM2-500M default
        self.vlm_hidden_dim = vlm_hs

        self.critic = SmolVLACritic(vlm_hidden_dim=vlm_hs)

        if lora_rank > 0:
            self._apply_lora(lora_rank)

    # ------------------------------------------------------------------
    # LoRA setup
    # ------------------------------------------------------------------

    def _apply_lora(self, rank: int) -> None:
        """Add LoRA adapters to VLM backbone attention projections."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.0,
                bias="none",
            )
            self.flow_model.vlm_with_expert = get_peft_model(
                self.flow_model.vlm_with_expert, lora_cfg
            )
            print(f"[SmolVLA-PPO] LoRA (rank={rank}) applied to VLM backbone.", flush=True)
        except ImportError:
            print("[SmolVLA-PPO] WARNING: peft not available, skipping LoRA.", flush=True)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def get_trainable_param_groups(
        self,
        action_lr: float = 5e-6,
        critic_lr: float = 1e-4,
        lora_lr: float = 5e-6,
    ) -> list[dict]:
        """Return AdamW parameter groups.

        Freezes everything except:
        - Action head (action_in_proj, action_out_proj, action_time_mlp_*,
          state_proj, vlm_with_expert.lm_expert)
        - LoRA adapter weights (if present)
        - Critic MLP
        """
        _ACTION_HEAD_PREFIXES = (
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "vlm_with_expert.lm_expert",
            "state_proj",
        )

        # Freeze everything first.
        for p in self.policy.parameters():
            p.requires_grad_(False)

        action_head_params: list[torch.nn.Parameter] = []
        lora_params: list[torch.nn.Parameter] = []

        for name, param in self.flow_model.named_parameters():
            is_action_head = any(name.startswith(pfx) for pfx in _ACTION_HEAD_PREFIXES)
            is_lora = "lora_" in name
            if is_action_head:
                param.requires_grad_(True)
                action_head_params.append(param)
            elif is_lora:
                param.requires_grad_(True)
                lora_params.append(param)

        if not action_head_params:
            print(
                "[SmolVLA-PPO] WARNING: no action-head params matched — "
                "unfreezing full flow_model as fallback.",
                flush=True,
            )
            for p in self.flow_model.parameters():
                p.requires_grad_(True)
            action_head_params = [p for p in self.flow_model.parameters()]

        groups: list[dict] = [
            {"params": action_head_params, "lr": action_lr, "name": "action_head"},
            {"params": list(self.critic.parameters()), "lr": critic_lr, "name": "critic"},
        ]
        if lora_params:
            groups.append({"params": lora_params, "lr": lora_lr, "name": "lora"})

        n_action = sum(p.numel() for p in action_head_params)
        n_critic = sum(p.numel() for p in self.critic.parameters())
        n_lora = sum(p.numel() for p in lora_params)
        print(
            f"[SmolVLA-PPO] Trainable: action_head={n_action:,}, "
            f"critic={n_critic:,}, lora={n_lora:,}",
            flush=True,
        )
        return groups

    # ------------------------------------------------------------------
    # Helpers — prefix embedding (shared between rollout and update)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _embed_prefix_and_kv(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, object, torch.Tensor]:
        """Run the frozen VLM prefix forward pass.

        Returns:
            prefix_pad_masks: (B, prefix_len).
            prefix_features:  Mean-pooled VLM output (B, vlm_hidden_dim) for critic.
            past_key_values:  KV cache for subsequent denoise_step calls.
            noise:            Fresh Gaussian noise for denoising (B, chunk_size, max_action_dim).
        """
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        flow = self.flow_model

        # Prepare inputs (mirrors SmolVLAPolicy._prepare_batch / prepare_images).
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # VLM forward — captures both prefix outputs (for critic) and KV cache.
        prefix_outputs, past_kv = flow.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=flow.config.use_cache,
            fill_kv_cache=True,
        )

        # Mean-pool VLM token outputs over valid (non-padding) positions.
        # prefix_outputs[0] shape: (B, prefix_len, vlm_hidden_dim)
        vlm_out = prefix_outputs[0].float()  # (B, prefix_len, hidden)
        mask = prefix_pad_masks.unsqueeze(-1).float()  # (B, prefix_len, 1)
        prefix_features = (vlm_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, hidden)

        bsize = state.shape[0]
        noise = flow.sample_noise(
            (bsize, flow.config.chunk_size, flow.config.max_action_dim), device
        )
        return prefix_pad_masks, prefix_features, past_kv, noise

    # ------------------------------------------------------------------
    # Rollout: sample actions with log-probability
    # ------------------------------------------------------------------

    def sample_actions_with_logprob(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
    ) -> dict[str, Any]:
        """Generate an action chunk and compute SDE log-prob for PPO rollout.

        Args:
            batch:  Pre-processed observation dict from the LeRobot preprocessor.
            device: Compute device.

        Returns:
            Dictionary with keys:
              ``action_chunk``  (B, chunk_size, action_dim) — final denoised actions
              ``log_prob``      (B,) — log π_old(a|s) from the SDE step
              ``value``         (B,) — V(s) from critic
              ``k_sde``         int  — which denoising step was stochastic
              ``x_k_sde``       (B, chunk_size, max_action_dim) CPU — state before SDE step
              ``x_next_sde``    (B, chunk_size, max_action_dim) CPU — state after SDE step
              ``prefix_pad_masks_cpu``  CPU tensor — for PPO update
        """
        prefix_pad_masks, prefix_features, past_kv, noise = self._embed_prefix_and_kv(
            batch, device
        )

        # Critic value estimate (no grad needed at rollout time).
        with torch.no_grad():
            value = self.critic(prefix_features)  # (B,)

        # Hybrid ODE-SDE sampling (no grad — acting, not updating).
        action_chunk, log_prob, k_sde, x_k_sde, x_next_sde = hybrid_ode_sde_sample(
            flow_model=self.flow_model,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_kv,
            noise=noise,
            K=self.K,
            noise_level=self.noise_level,
        )

        # Slice to real action_dim.
        action_dim = self.policy.config.action_feature.shape[0]
        action_chunk = action_chunk[..., :action_dim]

        return {
            "action_chunk": action_chunk.detach(),       # (B, chunk_size, action_dim)
            "log_prob": log_prob.detach(),               # (B,)
            "value": value.detach(),                     # (B,)
            "k_sde": k_sde,                              # int
            "x_k_sde": x_k_sde,                         # CPU (B, chunk_size, max_action_dim)
            "x_next_sde": x_next_sde,                   # CPU (B, chunk_size, max_action_dim)
            "prefix_pad_masks_cpu": prefix_pad_masks.cpu(),  # CPU
        }

    # ------------------------------------------------------------------
    # PPO update: recompute log-prob and value with current weights
    # ------------------------------------------------------------------

    def recompute_logprob_and_value(
        self,
        batch: dict[str, torch.Tensor],
        x_k_sde: torch.Tensor,
        x_next_sde: torch.Tensor,
        k_sde: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute log π_new(a|s) and V_new(s) with current weights.

        The VLM prefix forward is run with no-grad (frozen backbone).
        Only the action head's single denoise_step and the critic MLP
        are differentiated.

        Args:
            batch:      Preprocessed observation dict (same as used in rollout).
            x_k_sde:    Stored state before SDE step (B, chunk_size, max_action_dim).
            x_next_sde: Stored state after SDE step (B, chunk_size, max_action_dim).
            k_sde:      SDE step index used during rollout.
            device:     Compute device.

        Returns:
            new_log_prob: (B,) with gradient through action head.
            new_value:    (B,) with gradient through critic MLP.
        """
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        flow = self.flow_model

        # VLM prefix forward — frozen, no gradient needed.
        with torch.no_grad():
            images, img_masks = self.policy.prepare_images(batch)
            state = self.policy.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

            prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_outputs, past_kv = flow.vlm_with_expert.forward(
                attention_mask=prefix_att_2d,
                position_ids=prefix_pos_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=flow.config.use_cache,
                fill_kv_cache=True,
            )

            # Critic features.
            vlm_out = prefix_outputs[0].float()
            mask = prefix_pad_masks.unsqueeze(-1).float()
            prefix_features = (vlm_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Critic forward — with gradient through critic MLP.
        new_value = self.critic(prefix_features)  # (B,)

        # Action head denoise_step — with gradient through action head.
        # past_kv from the no-grad prefix forward is fine to use here:
        # gradients flow only through the action head's computation of v_t.
        new_log_prob = recompute_log_prob(
            flow_model=flow,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_kv,
            x_k_sde=x_k_sde.to(device),
            x_next_sde=x_next_sde.to(device),
            k_sde=k_sde,
            K=self.K,
            noise_level=self.noise_level,
        )

        return new_log_prob, new_value
