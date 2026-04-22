# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fall-restore helpers for LeRobot dataset collection.

When the cube drops low enough, respawn it on the table instead of ending the episode
(``object_dropping`` termination must be removed — see ``collect_lerobot_dataset.py``).

The **ratio** of fall restores to episode length can be filtered in the collector via
:class:`FallRestoreRatioBand` / :class:`FallRestoreRatioCap` (aliases for the same
settings object).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.events import reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

STATE_ATTR = "_so101_fall_restore_state"


class FallRestoreRatioBand:
    """Documentation-only: min/max fraction ``fall_restores / episode_length`` enforced in the collector."""

    __slots__ = ()


FallRestoreRatioCap = FallRestoreRatioBand  # backward-compatible alias


def _ensure_state(env: ManagerBasedRLEnv) -> dict:
    if not hasattr(env, STATE_ATTR):
        n = env.num_envs
        dev = env.device
        setattr(
            env,
            STATE_ATTR,
            {
                "fall_restore_count": torch.zeros(n, device=dev, dtype=torch.long),
                "last_episode_fall_count": torch.zeros(n, device=dev, dtype=torch.long),
                "force_commit": torch.zeros(n, device=dev, dtype=torch.bool),
                "last_episode_force_commit": torch.zeros(n, device=dev, dtype=torch.bool),
            },
        )
    return getattr(env, STATE_ATTR)


def get_fall_restore_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env count of fall restores in the current episode."""
    return _ensure_state(env)["fall_restore_count"]


def get_force_commit_episode(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-env flag: episode should be committed even when ratio filters would discard (set by stall logic)."""
    return _ensure_state(env)["force_commit"]


def get_last_episode_fall_count(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fall restores in the last finished episode (updated on reset)."""
    return _ensure_state(env)["last_episode_fall_count"]


def get_last_episode_force_commit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether :func:`force_commit_episode` was active at episode end (snapshot on reset)."""
    return _ensure_state(env)["last_episode_force_commit"]


def force_commit_episode(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """Mark environments so the dataset collector treats the ending episode as must-save (stall breaker)."""
    if env_ids is None or env_ids.numel() == 0:
        return
    st = _ensure_state(env)
    eids = env_ids.to(device=env.device, dtype=torch.long)
    st["force_commit"][eids] = True


def fall_restore_reset_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> None:
    """``mode="reset"``: snapshot fall counts for the finished episode, then clear counters."""
    if env_ids is None:
        return
    if isinstance(env_ids, slice):
        eids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        eids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if eids.numel() == 0:
        return
    st = _ensure_state(env)
    st["last_episode_fall_count"][eids] = st["fall_restore_count"][eids]
    st["last_episode_force_commit"][eids] = st["force_commit"][eids]
    st["fall_restore_count"][eids] = 0
    st["force_commit"][eids] = False


def fall_restore_recover_object(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    trigger_height: float = -0.02,
    pose_range: dict | None = None,
    velocity_range: dict | None = None,
    asset_cfg: SceneEntityCfg | None = None,
    max_restores_per_episode: int = 64,
    force_timeout_on_excess: bool = True,
) -> None:
    """``mode="interval"``: if the object is below ``trigger_height``, respawn it and increment counters.

    Args:
        env: RL environment.
        env_ids: Sub-environments whose interval timer fired (see EventManager).
        trigger_height: World-frame ``z`` below which the cube is considered fallen.
        pose_range: Passed to :func:`reset_root_state_uniform` for the respawn pose.
        velocity_range: Root velocity randomization (default: zeros).
        asset_cfg: Rigid object to reset.
        max_restores_per_episode: After this many restores, optionally force episode timeout and
            :func:`force_commit_episode`.
        force_timeout_on_excess: If True, set ``episode_length_buf`` to max length to end via time-out.
    """
    if env_ids is None or env_ids.numel() == 0:
        return

    if pose_range is None:
        pose_range = {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)}
    if velocity_range is None:
        velocity_range = {}
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("object", body_names="Object")

    eids = env_ids.to(device=env.device, dtype=torch.long)
    obj = env.scene[asset_cfg.name]
    z = obj.data.root_pos_w[eids, 2]
    fallen = z < trigger_height
    if not torch.any(fallen):
        return

    st = _ensure_state(env)
    apply_eids = eids[fallen]
    counts = st["fall_restore_count"][apply_eids]
    over_cap = counts >= max_restores_per_episode

    if torch.any(over_cap):
        oc_eids = apply_eids[over_cap]
        force_commit_episode(env, oc_eids)
        if force_timeout_on_excess:
            env.episode_length_buf[oc_eids] = env.max_episode_length

    under_cap = ~over_cap
    restore_eids = apply_eids[under_cap]
    if restore_eids.numel() == 0:
        return

    reset_root_state_uniform(
        env,
        restore_eids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )
    st["fall_restore_count"][restore_eids] += 1
