"""SmolVLA-backed ActorCritic for RSL-RL PPO.

Uses SmolVLA's frozen VLM backbone as a visual-language feature extractor
with standard Gaussian MLP heads for actor and critic.  The backbone is
loaded once and features are computed from camera images + language instruction
inside an observation term (see ``smolvla_visual_features`` in ``mdp/observations.py``).

This class is a thin wrapper around ``rsl_rl.modules.ActorCritic`` that:
* Accepts SmolVLA-specific kwargs (``smolvla_model_path``, etc.) without errors.
* Provides a clear ``load_state_dict`` override so that checkpoints containing
  only the MLP heads / log-std can be loaded without the backbone weights.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP


class SmolVLAActorCritic(nn.Module):
    """RSL-RL-compatible ActorCritic that consumes SmolVLA visual features.

    The visual backbone lives in an observation term (frozen, runs every env
    step).  This module only holds the lightweight MLP heads and the Gaussian
    noise parameters, making PPO updates fast and memory-efficient.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        # --- standard ActorCritic knobs (kept for config compatibility) ---
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        # --- SmolVLA-specific (ignored by base, consumed here) ---
        smolvla_model_path: str = "lerobot/smolvla_base",
        language_instruction: str = "Pick the cube.",
        freeze_backbone: bool = True,
        use_lora: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            print(f"SmolVLAActorCritic: ignoring unexpected kwargs: {list(kwargs.keys())}")

        self.obs_groups = obs_groups
        self.smolvla_model_path = smolvla_model_path
        self.language_instruction = language_instruction

        # Compute flat observation size from all policy / critic groups.
        # The SmolVLA features arrive as a flat vector through an ObsTerm,
        # so standard dimension inference works.
        num_actor_obs = 0
        for grp in obs_groups["policy"]:
            t = obs[grp]
            if isinstance(t, torch.Tensor) and t.ndim == 2:
                num_actor_obs += t.shape[-1]
            else:
                print(f"SmolVLAActorCritic: skipping non-flat obs group '{grp}' for dim calc")
        num_critic_obs = 0
        for grp in obs_groups["critic"]:
            t = obs[grp]
            if isinstance(t, torch.Tensor) and t.ndim == 2:
                num_critic_obs += t.shape[-1]
            else:
                print(f"SmolVLAActorCritic: skipping non-flat obs group '{grp}' for dim calc")

        # Actor MLP
        self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        print(f"SmolVLA Actor MLP: {self.actor}")

        # Critic MLP
        self.critic = MLP(num_critic_obs, 1, list(critic_hidden_dims), activation)
        print(f"SmolVLA Critic MLP: {self.critic}")

        # Observation normalizers (identity by default)
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        if actor_obs_normalization:
            from rsl_rl.networks import EmpiricalNormalization
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
        if critic_obs_normalization:
            from rsl_rl.networks import EmpiricalNormalization
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # Gaussian action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        print(
            f"SmolVLAActorCritic: actor_obs={num_actor_obs}, critic_obs={num_critic_obs}, "
            f"actions={num_actions}, backbone='{smolvla_model_path}'"
        )

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        parts = [obs[g] for g in self.obs_groups["policy"] if isinstance(obs[g], torch.Tensor) and obs[g].ndim == 2]
        return torch.cat(parts, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        parts = [obs[g] for g in self.obs_groups["critic"] if isinstance(obs[g], torch.Tensor) and obs[g].ndim == 2]
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Distribution
    # ------------------------------------------------------------------

    def _update_distribution(self, obs_flat: torch.Tensor) -> None:
        mean = self.actor(obs_flat)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    # ------------------------------------------------------------------
    # RSL-RL interface
    # ------------------------------------------------------------------

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self):
        raise NotImplementedError

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
