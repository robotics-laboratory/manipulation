"""Value function for RECAP (RL with Experience and Corrections via Advantage-conditioned Policies).

Overview
--------
RECAP (arXiv:2511.14759) trains a value function V(s) to predict the Monte Carlo
return from a given observation, then uses it to compute per-decision-point advantages:

    A(s, a) = MC_return_from_s - V(s)

Advantages are binarized: the top ``positive_percentile``% across the whole buffer
become "positive" labels; the rest become "negative".

Architecture
------------
Identical to the ``SmolVLACritic`` in ``smolvla_ppo_actor.py``:

* Run the frozen VLM prefix forward pass (embed_prefix → vlm_with_expert.forward
  with fill_kv_cache=True) to obtain prefix token embeddings.
* Mean-pool valid (non-padding) prefix tokens → shape (B, vlm_hidden_dim).
* Pass through a 4-layer MLP → scalar value estimate.

The VLM backbone is always frozen here; only the MLP head is trained.

Value function training (per iteration)
----------------------------------------
For each decision point in the episode buffer we have:
    * obs (images + state + language)  — stored as numpy
    * MC return from that decision point to episode end

The MLP is trained via MSE(V(s), R(s)) for a few epochs per iteration.

The return from a decision point is computed as:
    R_k = sum_{i=k}^{T-1} reward_i
where each reward_i is the ``chunk_reward`` of decision point i.
(RECAP uses undiscounted Monte Carlo; discount gamma can be applied optionally.)
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# ---------------------------------------------------------------------------
# MLP value head (same architecture as SmolVLACritic)
# ---------------------------------------------------------------------------


class _ValueMLP(nn.Module):
    """4-layer MLP mapping pooled VLM features → scalar value."""

    def __init__(
        self,
        vlm_hidden_dim: int = 960,
        hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = vlm_hidden_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, vlm_hidden_dim)
        Returns:
            values: (B,)
        """
        return self.net(features).squeeze(-1)


# ---------------------------------------------------------------------------
# SmolVLAValueFunction
# ---------------------------------------------------------------------------


class SmolVLAValueFunction:
    """Standalone value function for RECAP.

    Wraps the frozen VLM prefix encoder from SmolVLAPolicy and adds a
    trainable MLP value head.  Does NOT modify the policy parameters.

    Args:
        policy:       Loaded SmolVLAPolicy (VLM used read-only).
        device:       Compute device.
        hidden_dims:  MLP hidden layer sizes.
        gamma:        Discount factor for Monte Carlo returns (1.0 = undiscounted).
    """

    def __init__(
        self,
        policy: "SmolVLAPolicy",
        device: torch.device,
        hidden_dims: tuple[int, ...] = (512, 256, 128, 64),
        gamma: float = 1.0,
    ) -> None:
        self.policy = policy
        self.flow_model = policy.model
        self.device = device
        self.gamma = gamma

        # Infer VLM hidden size.
        try:
            vlm_hs = self.flow_model.vlm_with_expert.vlm.config.text_config.hidden_size
        except AttributeError:
            vlm_hs = 960  # SmolVLM2-500M default
        self.vlm_hidden_dim = vlm_hs

        self.mlp = _ValueMLP(vlm_hidden_dim=vlm_hs, hidden_dims=hidden_dims).to(device)

    # ------------------------------------------------------------------
    # Prefix feature extraction (frozen VLM, no grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _prefix_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run frozen VLM prefix forward and return mean-pooled features.

        Args:
            batch: Preprocessed observation dict with keys
                   ``observation.language_tokens``,
                   ``observation.language_attention_mask``,
                   ``observation.state``, and image keys.

        Returns:
            features: (B, vlm_hidden_dim) float32.
        """
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        flow = self.flow_model
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        prefix_embs, prefix_pad_masks, prefix_att_masks = flow.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_outputs, _ = flow.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=flow.config.use_cache,
            fill_kv_cache=True,
        )

        # Mean-pool VLM output tokens over valid positions.
        vlm_out = prefix_outputs[0].float()         # (B, prefix_len, hidden)
        mask = prefix_pad_masks.unsqueeze(-1).float()  # (B, prefix_len, 1)
        features = (vlm_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # (B, hidden)
        return features

    # ------------------------------------------------------------------
    # Value prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_value_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict V(s) for a preprocessed batch.

        Args:
            batch: Preprocessed obs dict (can contain multiple envs stacked).
        Returns:
            values: (B,) float32 CPU.
        """
        features = self._prefix_features(batch)
        return self.mlp(features).cpu()

    # ------------------------------------------------------------------
    # Monte Carlo return computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mc_returns(
        episode_rewards: list[float],
        gamma: float = 1.0,
    ) -> list[float]:
        """Compute Monte Carlo returns for each decision point in an episode.

        R_k = r_k + gamma * r_{k+1} + gamma^2 * r_{k+2} + ...

        Args:
            episode_rewards: Per-decision-point chunk rewards (in episode order).
            gamma:           Discount factor.

        Returns:
            List of returns, same length as episode_rewards.
        """
        T = len(episode_rewards)
        returns = [0.0] * T
        running = 0.0
        for k in reversed(range(T)):
            running = episode_rewards[k] + gamma * running
            returns[k] = running
        return returns

    # ------------------------------------------------------------------
    # Training the value function
    # ------------------------------------------------------------------

    def train_on_episodes(
        self,
        episodes,  # list[EpisodeData]
        preprocess,
        language_instruction: str,
        camera_mapping: dict[str, str],
        num_epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-4,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """Train the MLP value head on Monte Carlo returns from collected episodes.

        Args:
            episodes:             Completed episode data (EpisodeData objects).
            preprocess:           LeRobot preprocessor function.
            language_instruction: Base task instruction (no advantage prefix).
            camera_mapping:       Isaac sensor name → policy image key mapping.
            num_epochs:           Training epochs over the collected data.
            batch_size:           Mini-batch size.
            lr:                   Learning rate (used only when optimizer is None).
            optimizer:            Optional external optimizer; created if None.

        Returns:
            Dict with ``mean_loss`` and ``num_updates``.
        """
        from lerobot.utils.constants import OBS_STATE
        import numpy as np

        if not episodes:
            return {"mean_loss": float("nan"), "num_updates": 0}

        # Build a flat list of (obs_batch, mc_return) pairs.
        samples: list[tuple[dict, float]] = []
        for ep in episodes:
            rewards = [dp.chunk_reward for dp in ep.decision_points]
            mc_returns = self.compute_mc_returns(rewards, self.gamma)
            for dp, ret in zip(ep.decision_points, mc_returns):
                frame: dict[str, Any] = {
                    "language_instruction": language_instruction,
                    "task": language_instruction,
                }
                for policy_key, img_np in dp.images_np.items():
                    frame[policy_key] = np.expand_dims(img_np.astype(np.float32), axis=0)
                state_arr = dp.state_np[:6] if dp.state_np.shape[0] >= 6 else dp.state_np
                frame[str(OBS_STATE)] = np.expand_dims(state_arr.astype(np.float32), axis=0)
                samples.append((frame, ret))

        if optimizer is None:
            optimizer = torch.optim.AdamW(self.mlp.parameters(), lr=lr)

        self.mlp.train()
        total_loss = 0.0
        num_updates = 0
        indices = list(range(len(samples)))

        for _epoch in range(num_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                mb_idx = indices[start: start + batch_size]
                if not mb_idx:
                    continue

                # Build and preprocess each frame individually, then stack.
                processed: list[dict] = []
                targets: list[float] = []
                for i in mb_idx:
                    frame, ret = samples[i]
                    processed.append(preprocess(frame))
                    targets.append(ret)

                # Stack tensors.
                stacked: dict[str, torch.Tensor] = {}
                for key in processed[0]:
                    vals = []
                    for pf in processed:
                        v = pf[key]
                        if isinstance(v, np.ndarray):
                            vals.append(torch.as_tensor(v, device=self.device))
                        elif isinstance(v, torch.Tensor):
                            vals.append(v.to(self.device))
                        else:
                            vals.append(v)
                    if isinstance(vals[0], torch.Tensor):
                        stacked[key] = torch.cat(vals, dim=0)
                    else:
                        stacked[key] = vals[0]

                target_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

                # Forward (features no-grad, MLP with grad).
                features = self._prefix_features(stacked)  # no-grad on VLM
                preds = self.mlp(features)                  # grad through MLP

                loss = nn.functional.mse_loss(preds, target_tensor)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        self.mlp.eval()
        return {
            "mean_loss": total_loss / max(num_updates, 1),
            "num_updates": num_updates,
        }

    # ------------------------------------------------------------------
    # Advantage computation and binarization
    # ------------------------------------------------------------------

    def compute_advantages(
        self,
        episodes,  # list[EpisodeData]
        preprocess,
        language_instruction: str,
        camera_mapping: dict[str, str],
        positive_percentile: float = 30.0,
    ) -> list[tuple[object, str, float]]:
        """Compute per-decision-point advantages and binarize as positive/negative.

        For each decision point dp in each episode, computes:
            A(dp) = MC_return_from_dp - V(obs_dp)

        Then assigns label "positive" to the top ``positive_percentile``% of
        advantages across all decision points in the buffer, and "negative" to
        the rest.

        Args:
            episodes:            Completed episode data.
            preprocess:          LeRobot preprocessor.
            language_instruction: Base task instruction (no advantage prefix).
            camera_mapping:      Isaac sensor name → policy image key mapping.
            positive_percentile: Fraction (%) of actions labeled positive.

        Returns:
            List of (DecisionPointData, label, mc_return) tuples, where
            label is "positive" or "negative".
        """
        from lerobot.utils.constants import OBS_STATE
        import numpy as np

        if not episodes:
            return []

        # Step 1: Compute MC returns per decision point.
        dp_ret_pairs: list[tuple[object, float]] = []  # (DecisionPointData, mc_return)
        for ep in episodes:
            rewards = [dp.chunk_reward for dp in ep.decision_points]
            mc_returns = self.compute_mc_returns(rewards, self.gamma)
            for dp, ret in zip(ep.decision_points, mc_returns):
                dp_ret_pairs.append((dp, ret))

        # Step 2: Predict V(s) for each decision point in mini-batches.
        INFER_BATCH = 32
        pred_values: list[float] = []
        self.mlp.eval()
        for start in range(0, len(dp_ret_pairs), INFER_BATCH):
            chunk = dp_ret_pairs[start: start + INFER_BATCH]
            processed: list[dict] = []
            for dp, _ in chunk:
                frame: dict[str, Any] = {
                    "language_instruction": language_instruction,
                    "task": language_instruction,
                }
                for policy_key, img_np in dp.images_np.items():
                    frame[policy_key] = np.expand_dims(img_np.astype(np.float32), axis=0)
                state_arr = dp.state_np[:6] if dp.state_np.shape[0] >= 6 else dp.state_np
                frame[str(OBS_STATE)] = np.expand_dims(state_arr.astype(np.float32), axis=0)
                processed.append(preprocess(frame))

            stacked: dict[str, torch.Tensor] = {}
            for key in processed[0]:
                vals = []
                for pf in processed:
                    v = pf[key]
                    if isinstance(v, np.ndarray):
                        vals.append(torch.as_tensor(v, device=self.device))
                    elif isinstance(v, torch.Tensor):
                        vals.append(v.to(self.device))
                    else:
                        vals.append(v)
                if isinstance(vals[0], torch.Tensor):
                    stacked[key] = torch.cat(vals, dim=0)
                else:
                    stacked[key] = vals[0]

            values = self.predict_value_batch(stacked)  # (batch,) CPU
            pred_values.extend(values.tolist())

        # Step 3: Compute advantages and binarize.
        advantages = np.array(
            [ret - v for (_, ret), v in zip(dp_ret_pairs, pred_values)],
            dtype=np.float32,
        )
        threshold = np.percentile(advantages, 100.0 - positive_percentile)

        result: list[tuple[object, str, float]] = []
        for i, (dp, ret) in enumerate(dp_ret_pairs):
            label = "positive" if advantages[i] >= threshold else "negative"
            result.append((dp, label, ret))

        n_pos = sum(1 for _, lbl, _ in result if lbl == "positive")
        n_neg = len(result) - n_pos
        print(
            f"[RECAP] Advantages computed: {len(result)} decision points, "
            f"positive={n_pos} ({100*n_pos/max(len(result),1):.1f}%), "
            f"negative={n_neg}",
            flush=True,
        )
        return result
