# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for trajectory-guided lift: align EE to the start of a **chosen** teacher polyline.

Also provides :func:`visualize_teacher_trajectory` — a global-interval event that draws matched
teacher EE polylines as small sphere markers in the viewport (useful for visual debugging during
training or play).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _write_object_root_from_trajectory_row(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    store,
    chosen: torch.Tensor,
    object_cfg: SceneEntityCfg,
) -> None:
    """Place cube at the dataset initial pose for each chosen row (world frame)."""
    obj: RigidObject = env.scene[object_cfg.name]
    eids = env_ids.to(device=env.device, dtype=torch.long)
    origins = env.scene.env_origins[eids, :3]
    default_root = obj.data.default_root_state[eids].clone()
    if getattr(store, "use_local_layout", False):
        pos_w = store.initial_object_pos_local[chosen] + origins[:, :3]
    else:
        pos_w = store.initial_object_pos[chosen]
    quat = default_root[:, 3:7]
    obj.write_root_pose_to_sim(torch.cat([pos_w, quat], dim=-1), env_ids=eids)
    zv = torch.zeros((eids.shape[0], 6), device=env.device, dtype=torch.float32)
    obj.write_root_velocity_to_sim(zv, env_ids=eids)


def reset_object_pose_from_trajectory_dataset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    trajectory_file: str,
    command_name: str = "object_pose",
    match_mode: str = "object_only",
    exact_match_tol: float = 5.0e-3,
    sample_traj_index: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Match a dataset row (same logic as EE align) and snap the cube to that row's initial pose.

    Intended for ``mode="post_command_reset"`` **before** :func:`align_ee_to_teacher_trajectory_start`
    in the same config: declare this term **above** the align term so the cube matches the teacher
    layout, then IK runs on consistent (object, goal) features.

    If you only need one call, use :func:`align_ee_to_teacher_trajectory_start` with
    ``reset_object_from_dataset=True`` instead.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    from isaac_so_arm101.tasks.lift.mdp import rewards as lift_rewards

    state = lift_rewards._get_or_create_trajectory_state(env, trajectory_file)
    store = state["store"]
    traj_indices = state["traj_indices"]
    last_progress = state["last_progress"]
    last_episode_length = state["last_episode_length"]
    path_milestone_max = state["path_milestone_max"]

    n_traj = store.initial_object_pos.shape[0]
    if n_traj <= 0:
        return

    dev = env.device
    eids = env_ids.to(device=dev, dtype=torch.long)
    n = int(eids.shape[0])

    cfg = getattr(env, "cfg", None)
    fixed = getattr(cfg, "trajectory_guidance_fixed_traj_index", None) if cfg is not None else None

    if fixed is not None:
        idx = max(0, min(int(fixed), n_traj - 1))
        chosen = torch.full((n,), idx, device=dev, dtype=torch.long)
    elif sample_traj_index:
        chosen = torch.randint(low=0, high=n_traj, size=(n,), device=dev, dtype=torch.long)
    else:
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos_w = object_asset.data.root_pos_w[eids, :3]
        goal_pos_w = lift_rewards._trajectory_goal_pos_w(env, command_name=command_name, robot_cfg=robot_cfg)[
            eids, :3
        ]
        origins = env.scene.env_origins[eids, :3]
        chosen = store.match(
            object_pos_w,
            goal_pos_w,
            mode=match_mode,
            exact_tol=exact_match_tol,
            env_origins=origins,
        )

    traj_indices[eids] = chosen
    last_progress[eids] = 0.0
    path_milestone_max[eids] = -1
    last_episode_length[eids] = 0

    origin_delta = state.get("origin_delta")
    if origin_delta is not None:
        training_origins = env.scene.env_origins[eids, :3]
        if store.use_ee_local:
            recording_origins = store.ee_trajectories[chosen, 0, :] - store.initial_ee_pos_local[chosen]
        elif store.use_local_layout:
            recording_origins = store.initial_object_pos[chosen] - store.initial_object_pos_local[chosen]
        else:
            recording_origins = torch.zeros_like(training_origins)
        origin_delta[eids] = training_origins - recording_origins

    _write_object_root_from_trajectory_row(env, env_ids, store, chosen, object_cfg)


def align_ee_to_teacher_trajectory_start(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    trajectory_file: str,
    command_name: str = "object_pose",
    match_mode: str = "object_only",
    exact_match_tol: float = 5.0e-3,
    sample_traj_index: bool = False,
    reset_object_from_dataset: bool = False,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ik_body_name: str = "gripper_link",
    arm_joint_name_pattern: str = "shoulder_.*|elbow_flex|wrist_.*",
    ee_offset_pos: tuple[float, float, float] = (0.01, 0.0, -0.09),
    ik_iters: int = 40,
) -> None:
    """Choose a teacher row **without** using student EE, then move the arm so EE matches its first waypoint.

    **Selection** (same spirit as :func:`mdp.rewards._trajectory_guidance_ensure_matched`):

    * ``env.cfg.trajectory_guidance_fixed_traj_index`` — fixed row (clamped).
    * Else if ``sample_traj_index`` — uniform random row per env.
    * Else — :meth:`TrajectoryStore.match` on cube (+ goal per ``match_mode``).

    When ``reset_object_from_dataset`` is True, the cube is moved to the matched row's
    ``initial_object_pos`` (local+origin when the dataset has locals) **before** IK.

    **IK**: differential IK (position) on arm joints to reach the dataset's first EE point (same
    convention as ``collect_trajectories`` / ``ee_frame``). Gripper joint is not changed.

    Writes ``traj_indices`` and sets ``last_episode_length`` for reset envs so the first reward step
    does not re-run matching and overwrite the choice.

    Args:
        sample_traj_index: If True, pick a random trajectory index (still **not** based on EE pose).
        reset_object_from_dataset: If True, snap the cube to the matched trajectory row before IK.
        ik_iters: Jacobian IK iterations without extra physics steps (raise if the arm is far).
    """
    if env_ids is None or len(env_ids) == 0:
        return

    from isaac_so_arm101.tasks.lift.mdp import rewards as lift_rewards

    state = lift_rewards._get_or_create_trajectory_state(env, trajectory_file)
    store = state["store"]
    traj_indices = state["traj_indices"]
    last_progress = state["last_progress"]
    last_episode_length = state["last_episode_length"]
    path_milestone_max = state["path_milestone_max"]

    n_traj = store.initial_object_pos.shape[0]
    if n_traj <= 0:
        return

    dev = env.device
    eids = env_ids.to(device=dev, dtype=torch.long)
    n = int(eids.shape[0])

    cfg = getattr(env, "cfg", None)
    fixed = getattr(cfg, "trajectory_guidance_fixed_traj_index", None) if cfg is not None else None

    if fixed is not None:
        idx = max(0, min(int(fixed), n_traj - 1))
        chosen = torch.full((n,), idx, device=dev, dtype=torch.long)
    elif sample_traj_index:
        chosen = torch.randint(low=0, high=n_traj, size=(n,), device=dev, dtype=torch.long)
    else:
        object_asset: RigidObject = env.scene[object_cfg.name]
        object_pos_w = object_asset.data.root_pos_w[eids, :3]
        goal_pos_w = lift_rewards._trajectory_goal_pos_w(env, command_name=command_name, robot_cfg=robot_cfg)[
            eids, :3
        ]
        origins = env.scene.env_origins[eids, :3]
        chosen = store.match(
            object_pos_w,
            goal_pos_w,
            mode=match_mode,
            exact_tol=exact_match_tol,
            env_origins=origins,
        )

    traj_indices[eids] = chosen
    last_progress[eids] = 0.0
    path_milestone_max[eids] = -1
    # So `_trajectory_guidance_ensure_matched` does not treat the next step as a new episode rematch.
    last_episode_length[eids] = 0

    # Compute origin_delta (training_origin − recording_origin) so reward projections
    # use the correct coordinate frame.
    origin_delta = state.get("origin_delta")
    if origin_delta is not None:
        training_origins = env.scene.env_origins[eids, :3]
        if store.use_ee_local:
            recording_origins = store.ee_trajectories[chosen, 0, :] - store.initial_ee_pos_local[chosen]
        elif store.use_local_layout:
            recording_origins = store.initial_object_pos[chosen] - store.initial_object_pos_local[chosen]
        else:
            recording_origins = torch.zeros_like(training_origins)
        origin_delta[eids] = training_origins - recording_origins

    if reset_object_from_dataset:
        _write_object_root_from_trajectory_row(env, env_ids, store, chosen, object_cfg)

    robot = env.scene[robot_cfg.name]
    origins = env.scene.env_origins[eids, :3]
    tip_w = store.get_first_teacher_ee_world(chosen, origins)

    body_ids, _ = robot.find_bodies(ik_body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one body {ik_body_name!r}, got {body_ids!r}")
    body_idx = int(body_ids[0])
    ee_jacobi_idx = body_idx - 1 if robot.is_fixed_base else body_idx

    arm_joint_ids, _ = robot.find_joints(arm_joint_name_pattern)
    if len(arm_joint_ids) == 0:
        raise RuntimeError(f"No arm joints matched {arm_joint_name_pattern!r}")

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=n, device=str(dev))

    offset_pos = torch.tensor(ee_offset_pos, device=dev, dtype=torch.float32).unsqueeze(0).expand(n, 3)
    offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev, dtype=torch.float32).unsqueeze(0).expand(n, 4)

    sim_dt = env.sim.get_physics_dt()

    for _ in range(int(ik_iters)):
        robot.update(sim_dt)

        root_pos = robot.data.root_pos_w[eids, :3]
        root_quat = robot.data.root_quat_w[eids, :7]

        ee_pos_w = robot.data.body_pos_w[eids, body_idx]
        ee_quat_w = robot.data.body_quat_w[eids, body_idx]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos, root_quat, ee_pos_w, ee_quat_w
        )
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(ee_pos_b, ee_quat_b, offset_pos, offset_rot)

        tip_des_b, _ = math_utils.subtract_frame_transforms(root_pos, root_quat, tip_w)
        ik.set_command(tip_des_b, ee_pos=None, ee_quat=ee_quat_b)

        jac = robot.root_physx_view.get_jacobians()[eids, ee_jacobi_idx, :, :][:, :, arm_joint_ids]
        base_rot = robot.data.root_quat_w[eids, :7]
        br = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jac[:, :3, :] = torch.bmm(br, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(br, jac[:, 3:, :])
        jac[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_pos), jac[:, 3:, :])
        jac[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(offset_rot), jac[:, 3:, :])

        jp_arm = robot.data.joint_pos[eids][:, arm_joint_ids]
        jp_des_arm = ik.compute(ee_pos_b, ee_quat_b, jac[:, :3, :], jp_arm)

        jp_block = robot.data.joint_pos[eids].clone()
        jp_block[:, arm_joint_ids] = jp_des_arm
        jp_full = robot.data.joint_pos.clone()
        jp_full[eids] = jp_block
        zv = torch.zeros_like(jp_full)
        robot.write_joint_state_to_sim(jp_full, zv)
        robot.set_joint_position_target(jp_full)


def visualize_teacher_trajectory(
    env: ManagerBasedRLEnv,
    env_ids,
    trajectory_file: str,
    max_envs_to_draw: int = 1,
    marker_radius: float = 0.004,
    subsample_step: int = 2,
) -> None:
    """Draw color-coded teacher EE trajectories with a live projection marker.

    Marker prototypes (indexed 0–3):

    * **0 — past** (gray): waypoints behind ``current_segment_idx - window``.
    * **1 — window** (green): waypoints inside the active search window.
    * **2 — future** (dim blue): waypoints ahead of the window.
    * **3 — projection** (red, larger): polyline vertex closest to the student EE.

    Waypoint *positions* are cached per trajectory index (only recomputed on
    episode reset).  Marker *colors* and the projection point are recomputed
    every call so the visualization tracks the robot's live progress.

    Args:
        env: Vectorized RL environment.
        env_ids: Unused (global interval callback).
        trajectory_file: Path to the teacher trajectory ``.pt`` dataset.
        max_envs_to_draw: Number of environments (starting from env 0) to visualize.
        marker_radius: Radius of the waypoint sphere markers (metres).
        subsample_step: Take every N-th waypoint to reduce marker count.
    """
    del env_ids

    from isaac_so_arm101.tasks.lift.mdp.rewards import _TRAJECTORY_PROJ_WINDOW

    state = getattr(env, "_trajectory_guidance_state", None)
    if state is None or state.get("trajectory_file") != trajectory_file:
        return

    store = state["store"]
    traj_indices = state["traj_indices"]
    origin_delta = state["origin_delta"]
    current_seg = state.get("current_segment_idx")

    n_draw = min(int(max_envs_to_draw), env.num_envs)
    if n_draw <= 0 or torch.all(traj_indices[:n_draw] < 0):
        return

    # -- create markers on first call (4 prototypes) --
    vis_state = getattr(env, "_teacher_traj_vis_state", None)
    if vis_state is None:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        r = float(marker_radius)
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/TeacherTrajectory",
            markers={
                "past": sim_utils.SphereCfg(
                    radius=r,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
                ),
                "window": sim_utils.SphereCfg(
                    radius=r * 1.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.3)),
                ),
                "future": sim_utils.SphereCfg(
                    radius=r * 0.8,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.6)),
                ),
                "projection": sim_utils.SphereCfg(
                    radius=r * 2.5,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.3)),
                ),
            },
        )
        markers = VisualizationMarkers(marker_cfg)
        vis_state = {"markers": markers, "last_traj_key": None, "cached_pts": None, "env_pt_counts": None}
        env._teacher_traj_vis_state = vis_state

    markers = vis_state["markers"]
    step = max(1, int(subsample_step))
    window = int(_TRAJECTORY_PROJ_WINDOW)

    # -- recompute waypoint positions only when matched trajectory changes --
    traj_key = tuple(traj_indices[:n_draw].tolist())
    if vis_state.get("last_traj_key") != traj_key:
        all_waypoints: list[torch.Tensor] = []
        env_pt_counts: list[int] = []
        for i in range(n_draw):
            idx = int(traj_indices[i].item())
            if idx < 0:
                env_pt_counts.append(0)
                continue
            tlen = int(store.trajectory_lengths[idx].item())
            pts = store.ee_trajectories[idx, :tlen:step, :]
            pts_world = pts + origin_delta[i].unsqueeze(0)
            all_waypoints.append(pts_world)
            env_pt_counts.append(pts_world.shape[0])

        if len(all_waypoints) == 0:
            return

        vis_state["cached_pts"] = torch.cat(all_waypoints, dim=0)
        vis_state["env_pt_counts"] = env_pt_counts
        vis_state["last_traj_key"] = traj_key

    cached_pts = vis_state.get("cached_pts")
    env_pt_counts = vis_state.get("env_pt_counts")
    if cached_pts is None or env_pt_counts is None:
        return

    # -- compute per-waypoint color indices + projection point every call --
    all_indices: list[int] = []
    proj_pts: list[torch.Tensor] = []
    offset = 0
    for i in range(n_draw):
        n_pts = env_pt_counts[i]
        if n_pts == 0:
            continue

        seg = int(current_seg[i].item()) if current_seg is not None else 0
        idx = int(traj_indices[i].item())
        tlen = int(store.trajectory_lengths[idx].item())

        for k in range(n_pts):
            orig_seg = k * step
            if orig_seg < seg - window:
                all_indices.append(0)  # past
            elif orig_seg > seg + window:
                all_indices.append(2)  # future
            else:
                all_indices.append(1)  # window

        # projection point: polyline vertex at current_segment_idx
        proj_seg = min(seg, tlen - 1)
        proj_pt = store.ee_trajectories[idx, proj_seg, :] + origin_delta[i]
        proj_pts.append(proj_pt.unsqueeze(0))

        offset += n_pts

    if len(proj_pts) == 0:
        return

    proj_tensor = torch.cat(proj_pts, dim=0)
    all_pts = torch.cat([cached_pts, proj_tensor], dim=0)
    all_indices.extend([3] * proj_tensor.shape[0])

    marker_indices = torch.tensor(all_indices, dtype=torch.int32, device=cached_pts.device)
    markers.visualize(translations=all_pts, marker_indices=marker_indices)
