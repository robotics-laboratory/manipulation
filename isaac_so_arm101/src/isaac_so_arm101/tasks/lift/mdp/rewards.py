# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

from isaac_so_arm101.tasks.lift.trajectory_store import TrajectoryStore

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def flush_discriminator_metrics_to_log(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
) -> None:
    """Write pending discriminator diagnostics into ``extras['log']`` (TensorBoard).

    Run as a **global interval** event **after** rewards and resets so keys are not wiped by
    ``_reset_idx`` in the same environment step.

    Args:
        env: Vectorized RL environment.
        env_ids: Unused (global interval callback passes ``None``).
    """
    del env_ids  # global interval
    pending = getattr(env, "_discriminator_metrics_pending", None)
    if pending is None:
        return
    log = env.extras.setdefault("log", {})
    log.update(pending)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it."""
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    return reach_reward * lift_reward


def _trajectory_goal_pos_w(env: ManagerBasedRLEnv, command_name: str, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    return des_pos_w


def _get_or_create_trajectory_state(env: ManagerBasedRLEnv, trajectory_file: str):
    state = getattr(env, "_trajectory_guidance_state", None)
    if state is not None and state.get("trajectory_file") == trajectory_file:
        return state

    store = TrajectoryStore(path=trajectory_file, device=str(env.device))
    state = {
        "trajectory_file": trajectory_file,
        "store": store,
        "traj_indices": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        "last_episode_length": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
    }
    setattr(env, "_trajectory_guidance_state", state)
    return state


def trajectory_guidance_reward(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    std: float,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward alignment between student EE and matched teacher trajectory EE."""
    distance = _trajectory_guidance_distance(
        env=env,
        trajectory_file=trajectory_file,
        std=std,
        command_name=command_name,
        match_mode=match_mode,
        exact_match_tol=exact_match_tol,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    return 1 - torch.tanh(distance / std)


def _trajectory_guidance_distance(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    std: float,
    command_name: str,
    match_mode: str,
    exact_match_tol: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute/cached teacher-student EE distance for the current sim step."""
    state = _get_or_create_trajectory_state(env, trajectory_file)

    # Cache to avoid recomputing distance if multiple reward terms are evaluated.
    # `_sim_step_counter` is incremented inside the environment step.
    step_id = getattr(env, "_sim_step_counter", None)
    if step_id is not None and state.get("last_compute_step_id") == step_id and "last_distance" in state:
        return state["last_distance"]

    store: TrajectoryStore = state["store"]
    traj_indices = state["traj_indices"]
    last_episode_length = state["last_episode_length"]

    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = object_asset.data.root_pos_w[:, :3]
    goal_pos_w = _trajectory_goal_pos_w(env, command_name=command_name, robot_cfg=robot_cfg)
    current_episode_length = env.episode_length_buf.to(dtype=torch.long)

    new_episode_mask = (traj_indices < 0) | (current_episode_length <= last_episode_length)
    if torch.any(new_episode_mask):
        reset_ids = new_episode_mask.nonzero(as_tuple=False).squeeze(-1)
        matched = store.match(
            object_pos_w[reset_ids],
            goal_pos_w[reset_ids],
            mode=match_mode,
            exact_tol=exact_match_tol,
        )
        traj_indices[reset_ids] = matched

    teacher_ee = store.get_ee_pos(traj_indices, timesteps=torch.clamp(current_episode_length - 1, min=0))
    student_ee = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(student_ee - teacher_ee, dim=1)

    last_episode_length[:] = current_episode_length

    # Cache for debug terms.
    state["last_compute_step_id"] = step_id
    state["last_distance"] = distance.detach()
    state["last_new_episode_mask"] = new_episode_mask.detach()
    return distance


def trajectory_guidance_debug_distance_over_std(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    std: float,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Debug signal: (||student_ee - teacher_ee|| / std).

    This should ideally decrease as the policy learns to follow the teacher trajectory.
    """
    distance = _trajectory_guidance_distance(
        env=env,
        trajectory_file=trajectory_file,
        std=std,
        command_name=command_name,
        match_mode=match_mode,
        exact_match_tol=exact_match_tol,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    return distance / std


def _get_or_create_discriminator_state(env: ManagerBasedRLEnv, discriminator_file: str):
    state = getattr(env, "_discriminator_guidance_state", None)
    if state is not None and state.get("discriminator_file") == discriminator_file:
        # Migration: older state dict had no episode-initial layout buffers.
        if "initial_object_pos_w" not in state:
            state["initial_object_pos_w"] = torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            )
            state["initial_goal_pos_w"] = torch.zeros(
                (env.num_envs, 3), dtype=torch.float32, device=env.device
            )
        return state

    from isaac_so_arm101.tasks.lift.trajectory_discriminator import TrajectoryDiscriminator

    disc = TrajectoryDiscriminator.load(discriminator_file, device=str(env.device))
    state = {
        "discriminator_file": discriminator_file,
        "discriminator": disc,
        "prev_ee_pos": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device),
        "last_episode_length": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        # Match offline D training: g uses reset layout (initial object & goal), not moving object.
        "initial_object_pos_w": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device),
        "initial_goal_pos_w": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device),
    }
    setattr(env, "_discriminator_guidance_state", state)
    return state


def discriminator_guidance_reward(
    env: ManagerBasedRLEnv,
    discriminator_file: str,
    command_name: str = "object_pose",
    g_mode: str = "object_only",
    eps: float = 1e-6,
    logit_temperature: float = 1.5,
    center_per_env_batch: bool = True,
    center_min_std: float = 1.0e-4,
    # After batch-centering ``log_d_raw`` (recommended when D is saturated). Symmetric band is
    # intentional: "better than batch mean" can be slightly positive.
    log_d_min: float = -1.5,
    log_d_max: float = 1.5,
    log_metrics: bool = True,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Discriminator-guided exploration reward (log D) for EE trajectories.

    Adapts the paper's augmented reward:
        r_hat = r + lambda * log D(p_t, delta_p_t, g)

    to IsaacLab by using:
    - p_t: student EE position
    - delta_p_t: student EE displacement since previous sim step
    - g: task conditioning matching ``train_trajectory_discriminator.py``: **episode-initial**
      object world position (and goal), not the live moving object pose — same as
      ``TrajectoryStore.initial_object_pos`` / ``goal_pos`` seen during training.

    ``logit_temperature`` divides logits before ``softplus``. Keep **moderate** (≈1–2); huge
    values collapse all envs to ~\\log 0.5 (see earlier notes).

    **Saturated discriminator:** raw outputs are often ~\\ ``log D ≪ -3`` with only modest
    cross-env std. The **inner** ``log(eps)`` floor in :meth:`~TrajectoryDiscriminator.log_d_scores`
    maps all those values to the **same** constant → **zero variance** after any clamp to
    ``[-3, 0]``. The reward path therefore uses **``log_d_raw``** (no inner floor) and,
    by default, **batch-centering** so shaping reflects *better vs worse than peers*; then
    a **symmetric** clamp ``[log_d_min, log_d_max]``.

    If ``std(log_d_raw) <= center_min_std``, the term returns **zeros** (no bogus gradient).
    """
    state = _get_or_create_discriminator_state(env, discriminator_file)
    disc = state["discriminator"]
    prev_ee_pos = state["prev_ee_pos"]
    last_episode_length = state["last_episode_length"]
    initial_object_pos_w = state["initial_object_pos_w"]
    initial_goal_pos_w = state["initial_goal_pos_w"]

    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object_asset.data.root_pos_w[:, :3]
    goal_pos_w = _trajectory_goal_pos_w(env, command_name=command_name, robot_cfg=robot_cfg)
    student_ee = ee_frame.data.target_pos_w[..., 0, :]

    current_episode_length = env.episode_length_buf.to(dtype=torch.long)
    new_episode_mask = (last_episode_length < 0) | (current_episode_length <= last_episode_length)
    if torch.any(new_episode_mask):
        reset_ids = new_episode_mask.nonzero(as_tuple=False).squeeze(-1)
        prev_ee_pos[reset_ids] = student_ee[reset_ids]
        # Snapshot layout at episode start (after reset), aligned with teacher dataset conditioning.
        initial_object_pos_w[reset_ids] = object_pos_w[reset_ids].to(dtype=torch.float32)
        initial_goal_pos_w[reset_ids] = goal_pos_w[reset_ids].to(dtype=torch.float32)

    delta_p = student_ee - prev_ee_pos
    prev_ee_pos[:] = student_ee
    last_episode_length[:] = current_episode_length

    if g_mode == "object_goal":
        g = torch.cat([initial_object_pos_w, initial_goal_pos_w], dim=1)
    elif g_mode == "object_only":
        g = initial_object_pos_w
    else:
        raise ValueError(f"Unknown g_mode: {g_mode}")

    log_d_raw, _log_d_inner, logits = disc.log_d_scores(
        student_ee,
        delta_p,
        g,
        logit_temperature=float(logit_temperature),
        eps=float(eps),
    )
    # Use raw log D for learning: inner eps floor wipes cross-env variance when D is very confident.
    log_d = log_d_raw
    centered = log_d_raw
    if center_per_env_batch and env.num_envs > 1:
        batch_std = log_d_raw.std(unbiased=False)
        if batch_std > float(center_min_std):
            centered = log_d_raw - log_d_raw.mean()
            log_d = centered
        else:
            log_d = torch.zeros_like(log_d_raw)
    shaped = torch.clamp(log_d, min=float(log_d_min), max=float(log_d_max))

    if log_metrics:
        # Stash for flush_discriminator_metrics_to_log (interval event after reset).
        if env.num_envs > 1:
            c_std = (log_d_raw - log_d_raw.mean()).std(unbiased=False)
        else:
            c_std = log_d_raw.new_zeros(())
        env._discriminator_metrics_pending = {
            "Metrics/discriminator/log_d_raw_mean": log_d_raw.mean(),
            "Metrics/discriminator/log_d_raw_std": log_d_raw.std(unbiased=False),
            "Metrics/discriminator/logit_mean": logits.mean(),
            "Metrics/discriminator/logit_std": logits.std(unbiased=False),
            "Metrics/discriminator/log_d_centered_std": c_std,
            "Metrics/discriminator/log_d_shaped_mean": shaped.mean(),
            "Metrics/discriminator/log_d_shaped_std": shaped.std(unbiased=False),
        }

    return torch.nan_to_num(
        shaped,
        nan=0.0,
        posinf=float(log_d_max),
        neginf=float(log_d_min),
    )
