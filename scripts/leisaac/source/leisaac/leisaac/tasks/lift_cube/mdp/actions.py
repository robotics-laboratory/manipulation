from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class RateLimitedJointPositionAction(JointPositionAction):
    """Joint-position action that limits target jumps between control steps."""

    cfg: "RateLimitedJointPositionActionCfg"

    def __init__(self, cfg: "RateLimitedJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._previous_processed_actions = torch.zeros_like(self.raw_actions)
        self._has_previous_action = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def IO_descriptor(self):
        descriptor = super().IO_descriptor
        descriptor.max_delta = self.cfg.max_delta
        return descriptor

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        missing_previous = ~self._has_previous_action
        if torch.any(missing_previous):
            self._previous_processed_actions[missing_previous] = self._asset.data.joint_pos[
                missing_previous
            ][:, self._joint_ids]
            self._has_previous_action[missing_previous] = True

        max_delta = float(self.cfg.max_delta)
        lower = self._previous_processed_actions - max_delta
        upper = self._previous_processed_actions + max_delta
        self._processed_actions = torch.clamp(self._processed_actions, min=lower, max=upper)
        self._previous_processed_actions[:] = self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._has_previous_action[:] = False
        else:
            self._has_previous_action[env_ids] = False


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for a joint-position action with target rate limiting."""

    class_type: type[ActionTerm] = RateLimitedJointPositionAction

    max_delta: float = 0.025
    """Maximum joint-position target change per environment step, in radians."""
