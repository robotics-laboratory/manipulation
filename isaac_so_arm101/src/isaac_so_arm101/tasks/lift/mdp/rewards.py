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
        if "last_progress" not in state:
            state["last_progress"] = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
        if "path_milestone_max" not in state:
            state["path_milestone_max"] = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        if "origin_delta" not in state:
            state["origin_delta"] = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
        if "current_segment_idx" not in state:
            state["current_segment_idx"] = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        return state

    store = TrajectoryStore(path=trajectory_file, device=str(env.device))
    state = {
        "trajectory_file": trajectory_file,
        "store": store,
        "traj_indices": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        "last_episode_length": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        "last_progress": torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device),
        "path_milestone_max": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        # Per-env offset: training_env_origin − recording_env_origin.
        # Add to recording-frame points to get training-world-frame points (or subtract from
        # student world-frame coords to project into the recording frame).
        "origin_delta": torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device),
        # Windowed projection: current segment index per env (prevents spatial shortcuts).
        "current_segment_idx": torch.zeros((env.num_envs,), dtype=torch.long, device=env.device),
    }
    setattr(env, "_trajectory_guidance_state", state)
    return state


def _resolve_gripper_joint_idx(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> int:
    cache_key = "_so101_gripper_joint_idx"
    if hasattr(env, cache_key):
        return int(getattr(env, cache_key))
    robot = env.scene[robot_cfg.name]
    ids, names = robot.find_joints("gripper")
    if len(ids) != 1:
        raise RuntimeError(f"Expected one gripper joint, got {names!r}")
    setattr(env, cache_key, int(ids[0]))
    return int(ids[0])


def _student_gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    idx = _resolve_gripper_joint_idx(env, robot_cfg)
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_pos[:, idx]


def _env_cfg_fixed_traj_index(env: ManagerBasedRLEnv) -> int | None:
    """Optional ``trajectory_guidance_fixed_traj_index`` on env config (fixed-layout guided training)."""
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return None
    v = getattr(cfg, "trajectory_guidance_fixed_traj_index", None)
    if v is None:
        return None
    return int(v)


def _trajectory_guidance_ensure_matched(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    command_name: str,
    match_mode: str,
    exact_match_tol: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> dict:
    """Run trajectory matching and student EE once per sim step (shared by all guidance terms)."""
    state = _get_or_create_trajectory_state(env, trajectory_file)
    step_id = getattr(env, "_sim_step_counter", None)
    if step_id is not None and state.get("trajectory_match_step_id") == step_id:
        return state

    store: TrajectoryStore = state["store"]
    traj_indices = state["traj_indices"]
    last_episode_length = state["last_episode_length"]
    last_progress = state["last_progress"]
    path_milestone_max = state["path_milestone_max"]

    object_asset: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = object_asset.data.root_pos_w[:, :3]
    goal_pos_w = _trajectory_goal_pos_w(env, command_name=command_name, robot_cfg=robot_cfg)
    student_ee = ee_frame.data.target_pos_w[..., 0, :]
    current_episode_length = env.episode_length_buf.to(dtype=torch.long)

    fixed_traj_idx = _env_cfg_fixed_traj_index(env)

    origin_delta = state["origin_delta"]

    new_episode_mask = (traj_indices < 0) | (current_episode_length <= last_episode_length)
    if torch.any(new_episode_mask):
        reset_ids = new_episode_mask.nonzero(as_tuple=False).squeeze(-1)
        if fixed_traj_idx is not None:
            n_traj = store.initial_object_pos.shape[0]
            idx = max(0, min(fixed_traj_idx, n_traj - 1))
            traj_indices[reset_ids] = int(idx)
        else:
            origins = env.scene.env_origins[reset_ids, :3]
            matched = store.match(
                object_pos_w[reset_ids],
                goal_pos_w[reset_ids],
                mode=match_mode,
                exact_tol=exact_match_tol,
                env_origins=origins,
            )
            traj_indices[reset_ids] = matched
        last_progress[reset_ids] = 0.0
        path_milestone_max[reset_ids] = -1
        state["current_segment_idx"][reset_ids] = 0

        # Compute per-env origin offset so we can convert between recording and training frames.
        # origin_delta = training_origin − recording_origin
        reset_matched = traj_indices[reset_ids]
        training_origins = env.scene.env_origins[reset_ids, :3]
        if store.use_ee_local:
            recording_origins = (
                store.ee_trajectories[reset_matched, 0, :] - store.initial_ee_pos_local[reset_matched]
            )
        elif store.use_local_layout:
            recording_origins = (
                store.initial_object_pos[reset_matched] - store.initial_object_pos_local[reset_matched]
            )
        else:
            recording_origins = torch.zeros_like(training_origins)
        origin_delta[reset_ids] = training_origins - recording_origins

    last_episode_length[:] = current_episode_length

    state["trajectory_match_step_id"] = step_id
    state["student_ee_buf"] = student_ee
    state["last_new_episode_mask"] = new_episode_mask.detach()
    return state


_TRAJECTORY_PROJ_WINDOW: int = 15
"""Forward-looking search window (segments) for windowed polyline projection."""

_TRAJECTORY_MAX_SEG_ADVANCE: int = 5
"""Max segment advance per RL step.  Prevents "speed-running" through the trajectory."""

_TRAJECTORY_ADVANCE_LATERAL_GATE: float = 0.10
"""Lateral distance (metres) within which the agent is considered "on the path".

The segment pointer only advances when the agent is closer than this threshold
to the current projection, forcing it to physically follow each section of the
trajectory instead of cutting corners.  10 cm tolerates small deviations from
grasping (cube shifts the EE) while still preventing large spatial shortcuts.
"""


def _trajectory_guidance_get_cached_path_geometry(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    command_name: str,
    match_mode: str,
    exact_match_tol: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> dict:
    """Strict sequential polyline projection + path-aligned teacher gripper (cached once per step).

    Enforces that the agent follows the trajectory **in order**:

    1. Projection searches only a forward-biased window around ``current_segment_idx``.
    2. The segment pointer **never moves backward**.
    3. Per-step advance is capped at ``_TRAJECTORY_MAX_SEG_ADVANCE``.
    4. Advance is only allowed when lateral distance < ``_TRAJECTORY_ADVANCE_LATERAL_GATE``
       (the agent must physically be close to the path to "unlock" the next section).

    On episode reset ``current_segment_idx`` is set to 0, forcing the agent to
    traverse the entire trajectory from the start every episode.
    """
    state = _trajectory_guidance_ensure_matched(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    step_id = getattr(env, "_sim_step_counter", None)
    ck = "path_proj_step_id"
    if (
        step_id is not None
        and state.get(ck) == step_id
        and "proj_progress" in state
        and "proj_lateral" in state
        and "proj_arc_len" in state
        and "proj_total_len" in state
    ):
        store: TrajectoryStore = state["store"]
        return {
            "state": state,
            "store": store,
            "progress": state["proj_progress"],
            "lateral": state["proj_lateral"],
            "arc_len": state["proj_arc_len"],
            "total_len": state["proj_total_len"],
            "teacher_gripper": state.get("proj_teacher_gripper"),
            "has_gripper": store.has_gripper,
        }

    store = state["store"]
    traj_indices = state["traj_indices"]
    student_ee = state["student_ee_buf"]
    origin_delta = state["origin_delta"]
    student_ee_in_rec = student_ee - origin_delta
    current_seg = state["current_segment_idx"]

    progress, lateral, arc_len, total_len, best_seg = store.project_ee_to_progress_windowed(
        student_ee_in_rec, traj_indices, current_seg, window=_TRAJECTORY_PROJ_WINDOW,
    )

    # --- strict sequential advancement ---
    # 1. Never go backward.
    new_seg = torch.maximum(best_seg, current_seg)
    # 2. Cap per-step forward advance.
    new_seg = torch.minimum(new_seg, current_seg + _TRAJECTORY_MAX_SEG_ADVANCE)
    # 3. Only advance when the agent is actually close to the path (lateral gate).
    close_enough = lateral <= _TRAJECTORY_ADVANCE_LATERAL_GATE
    new_seg = torch.where(close_enough, new_seg, current_seg)

    state["current_segment_idx"] = new_seg

    teacher_g: torch.Tensor | None = None
    if store.has_gripper:
        lengths = store.trajectory_lengths[traj_indices].clamp(min=1)
        t_align = torch.round(progress * (lengths - 1).to(dtype=torch.float32)).long().clamp(min=0)
        teacher_g = store.get_gripper(traj_indices, t_align)

    if step_id is not None:
        state[ck] = step_id
        state["proj_progress"] = progress
        state["proj_lateral"] = lateral
        state["proj_arc_len"] = arc_len
        state["proj_total_len"] = total_len
        state["proj_teacher_gripper"] = teacher_g

    return {
        "state": state,
        "store": store,
        "progress": progress,
        "lateral": lateral,
        "arc_len": arc_len,
        "total_len": total_len,
        "teacher_gripper": teacher_g,
        "has_gripper": store.has_gripper,
    }


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
    """Time-synchronized teacher EE vs student EE distance (cached per step)."""
    del std  # unused; kept for signature compatibility with reward term params
    state = _trajectory_guidance_ensure_matched(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    step_id = getattr(env, "_sim_step_counter", None)
    if (
        step_id is not None
        and state.get("last_time_sync_distance_step_id") == step_id
        and "last_time_sync_distance" in state
    ):
        return state["last_time_sync_distance"]

    store: TrajectoryStore = state["store"]
    traj_indices = state["traj_indices"]
    student_ee = state["student_ee_buf"]
    origin_delta = state["origin_delta"]
    current_episode_length = env.episode_length_buf.to(dtype=torch.long)

    teacher_ee = store.get_ee_pos(traj_indices, timesteps=torch.clamp(current_episode_length - 1, min=0))
    # teacher_ee is in recording world frame; shift to training world frame.
    teacher_ee_w = teacher_ee + origin_delta
    distance = torch.norm(student_ee - teacher_ee_w, dim=1)

    state["last_time_sync_distance_step_id"] = step_id
    state["last_time_sync_distance"] = distance.detach()
    state["last_compute_step_id"] = step_id
    return distance


def trajectory_guidance_reward(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    std: float,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    guidance_mode: str = "path_progress",
    path_progress_delta_std: float = 0.05,
    path_progress_scale: float = 1.0,
    lateral_penalty_weight: float = 0.0,
    lateral_std: float = 0.1,
    only_forward_progress: bool = True,
    backward_progress_penalty_weight: float = 0.0,
    progress_lateral_gate: float | None = None,
    num_path_milestones: int = 8,
    max_milestone_jump: int = 1,
    milestone_reward_scale: float = 1.0,
    milestone_lateral_gate: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward alignment with a matched teacher EE trajectory.

    - ``time_sync``: same sim step index as the teacher recording — ``1 - tanh(||Δ||/std)``.
    - ``path_progress``: reward **forward motion along the teacher polyline** (arc-length progress).
    - ``path_progress_milestones``: ordered **milestones** along arc length; limits spatial shortcuts when
      combined with ``milestone_lateral_gate`` and ``max_milestone_jump``.
    """
    if guidance_mode == "time_sync":
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
        return 1.0 - torch.tanh(distance / std)

    if guidance_mode not in ("path_progress", "path_progress_milestones"):
        raise ValueError(f"Unknown guidance_mode: {guidance_mode}")

    state = _trajectory_guidance_ensure_matched(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    step_id = getattr(env, "_sim_step_counter", None)
    if (
        step_id is not None
        and state.get("last_path_reward_step_id") == step_id
        and "path_reward" in state
    ):
        return state["path_reward"]

    geom = _trajectory_guidance_get_cached_path_geometry(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    state = geom["state"]
    last_progress = state["last_progress"]
    path_milestone_max = state["path_milestone_max"]

    progress = geom["progress"]
    lateral = geom["lateral"]
    arc_len = geom["arc_len"]
    total_len = geom["total_len"]

    if guidance_mode == "path_progress_milestones":
        K = max(2, int(num_path_milestones))
        bin_w = total_len / float(K)
        eligible_bin = torch.floor(arc_len / (bin_w + 1.0e-8)).long().clamp(0, K - 1)
        lateral_ok = lateral <= float(milestone_lateral_gate)
        effective_eligible = torch.where(lateral_ok, eligible_bin, path_milestone_max)

        cap = path_milestone_max + int(max_milestone_jump)
        next_max = torch.minimum(effective_eligible, cap)
        next_max = torch.maximum(next_max, path_milestone_max)

        r = float(milestone_reward_scale) * (next_max - path_milestone_max).to(dtype=torch.float32)
        path_milestone_max[:] = next_max
        last_progress[:] = progress

        state["last_path_reward_step_id"] = step_id
        state["last_compute_step_id"] = step_id
        state["path_reward"] = r.detach()
        state["last_path_progress"] = progress.detach()
        state["last_path_lateral"] = lateral.detach()
        state["last_path_delta_raw"] = torch.zeros_like(progress)
        return r

    delta_raw = progress - last_progress
    if progress_lateral_gate is not None:
        gated = lateral <= float(progress_lateral_gate)
        delta_raw = torch.where(gated, delta_raw, torch.zeros_like(delta_raw))
    delta = torch.relu(delta_raw) if only_forward_progress else delta_raw

    r = path_progress_scale * torch.tanh(delta / float(path_progress_delta_std))
    if only_forward_progress and float(backward_progress_penalty_weight) > 0.0:
        back = torch.relu(-delta_raw)
        r = r - float(backward_progress_penalty_weight) * torch.tanh(back / float(path_progress_delta_std))
    if lateral_penalty_weight > 0.0:
        r = r - float(lateral_penalty_weight) * torch.tanh(lateral / float(lateral_std))

    last_progress[:] = progress

    state["last_path_reward_step_id"] = step_id
    state["last_compute_step_id"] = step_id
    state["path_reward"] = r.detach()
    state["last_path_progress"] = progress.detach()
    state["last_path_lateral"] = lateral.detach()
    state["last_path_delta_raw"] = delta_raw.detach()
    return r


def teacher_gripper_alignment_reward(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    std: float,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    close_threshold: float = 0.15,
    close_boost: float = 2.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Match student gripper joint to teacher gripper at path-aligned time index.

    Teacher index ``t = round(progress * (T-1))`` where ``progress`` comes from EE projection on the
    polyline. If the dataset has no ``gripper_trajectories``, returns zeros.

    When ``close_boost > 1`` the reward is multiplied by ``close_boost`` on trajectory
    segments where the teacher gripper is below ``close_threshold`` (grasping phase),
    giving extra incentive to actually close the gripper around the object.
    """
    geom = _trajectory_guidance_get_cached_path_geometry(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    state = geom["state"]
    step_id = getattr(env, "_sim_step_counter", None)
    if (
        step_id is not None
        and state.get("last_gripper_reward_step_id") == step_id
        and "gripper_reward" in state
    ):
        return state["gripper_reward"]

    if not geom["has_gripper"] or geom["teacher_gripper"] is None:
        z = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
        state["gripper_reward"] = z
        state["last_gripper_reward_step_id"] = step_id
        return z

    teacher_g = geom["teacher_gripper"]
    student_g = _student_gripper_pos(env, robot_cfg)
    err = torch.abs(student_g - teacher_g)
    r = 1.0 - torch.tanh(err / float(std))

    if float(close_boost) > 1.0:
        teacher_closed = (teacher_g < float(close_threshold)).to(dtype=r.dtype)
        r = r * (1.0 + teacher_closed * (float(close_boost) - 1.0))

    state["last_gripper_reward_step_id"] = step_id
    state["last_compute_step_id"] = step_id
    state["gripper_reward"] = r.detach()
    return r


def trajectory_guidance_debug_distance_over_std(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    std: float,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    guidance_mode: str = "path_progress",
    lateral_std: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Debug signal for trajectory guidance.

    - ``time_sync``: ``norm(student_ee - teacher_ee(t)) / std`` (should shrink when tracking improves).
    - ``path_progress``: **lateral distance to the teacher polyline** / ``lateral_std`` (off-path error;
      uses cached tensors from the main reward when evaluated after it on the same step).
    """
    if guidance_mode == "time_sync":
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

    if guidance_mode not in ("path_progress", "path_progress_milestones"):
        raise ValueError(f"Unknown guidance_mode for debug: {guidance_mode}")

    geom = _trajectory_guidance_get_cached_path_geometry(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    lateral = geom["lateral"]
    return lateral / float(lateral_std)


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
