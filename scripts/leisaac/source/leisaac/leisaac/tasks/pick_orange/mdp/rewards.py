from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from leisaac.utils.robot_utils import is_so101_at_rest_pose

from .terminations import task_done

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _orange_start_height_world(env: ManagerBasedRLEnv, orange: RigidObject) -> torch.Tensor:
    return orange.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]


def _orange_on_plate_mask(
    env: ManagerBasedRLEnv,
    orange_cfg: SceneEntityCfg,
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_range: tuple[float, float],
) -> torch.Tensor:
    orange: RigidObject = env.scene[orange_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    orange_pos = orange.data.root_pos_w - env.scene.env_origins
    plate_pos = plate.data.root_pos_w - env.scene.env_origins

    in_x = torch.logical_and(orange_pos[:, 0] > plate_pos[:, 0] + x_range[0], orange_pos[:, 0] < plate_pos[:, 0] + x_range[1])
    in_y = torch.logical_and(orange_pos[:, 1] > plate_pos[:, 1] + y_range[0], orange_pos[:, 1] < plate_pos[:, 1] + y_range[1])
    in_z = torch.logical_and(
        orange_pos[:, 2] > plate_pos[:, 2] + height_range[0], orange_pos[:, 2] < plate_pos[:, 2] + height_range[1]
    )
    return torch.logical_and(torch.logical_and(in_x, in_y), in_z)


def _placed_mask_per_orange(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    masks = [
        _orange_on_plate_mask(env, orange_cfg, plate_cfg, x_range=x_range, y_range=y_range, height_range=height_range)
        for orange_cfg in oranges_cfg
    ]
    return torch.stack(masks, dim=1)


def _reset_episode_progress_if_needed(env: ManagerBasedRLEnv, num_oranges: int) -> None:
    """Clear per-episode progress buffers right after an environment reset."""
    if not hasattr(env, "_pick_orange_lifted_history"):
        env._pick_orange_lifted_history = torch.zeros((env.num_envs, num_oranges), dtype=torch.bool, device=env.device)
    if not hasattr(env, "_pick_orange_progress_mask"):
        env._pick_orange_progress_mask = torch.zeros((env.num_envs, num_oranges), dtype=torch.bool, device=env.device)
    if not hasattr(env, "_pick_orange_active_idx"):
        env._pick_orange_active_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    reset_env_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=True)[0]
    if reset_env_ids.numel() > 0:
        env._pick_orange_lifted_history[reset_env_ids] = False
        env._pick_orange_progress_mask[reset_env_ids] = False
        env._pick_orange_active_idx[reset_env_ids] = 0


def _ensure_active_orange_state(env: ManagerBasedRLEnv, num_oranges: int) -> torch.Tensor:
    """Ensure and return the explicit per-env active orange index."""
    if not hasattr(env, "_pick_orange_active_idx"):
        env._pick_orange_active_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if env._pick_orange_active_idx.shape != (env.num_envs,):
        env._pick_orange_active_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._pick_orange_active_idx


def _advance_active_orange_if_needed(env: ManagerBasedRLEnv, progress_mask: torch.Tensor) -> torch.Tensor:
    """Advance each env's active orange to the first incomplete orange once current target is done."""
    active_idx = _ensure_active_orange_state(env, progress_mask.shape[1])
    all_placed = torch.all(progress_mask, dim=1)
    env_ids = torch.arange(env.num_envs, device=progress_mask.device)
    current_done = progress_mask[env_ids, active_idx]
    should_advance = torch.logical_and(current_done, torch.logical_not(all_placed))
    if torch.any(should_advance):
        first_incomplete = torch.argmin(progress_mask.int(), dim=1)
        active_idx[should_advance] = first_incomplete[should_advance]
    return active_idx


def _gripper_near_closed_mask(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
    grasp_distance: float = 0.06,
    close_joint_threshold: float = 0.7,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :3]
    gripper_closed = robot.data.joint_pos[:, -1] < close_joint_threshold

    masks = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        distance = torch.linalg.vector_norm(orange.data.root_pos_w[:, :3] - ee_pos, dim=1)
        masks.append(torch.logical_and(distance < grasp_distance, gripper_closed))
    return torch.stack(masks, dim=1)


def _currently_held_per_orange(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    lifted_height_delta: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
    grasp_distance: float = 0.06,
    close_joint_threshold: float = 0.7,
) -> torch.Tensor:
    """Per-step mask: orange is grasped now AND currently lifted above threshold."""
    grasped_now = _gripper_near_closed_mask(
        env,
        oranges_cfg=oranges_cfg,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        ee_frame_index=ee_frame_index,
        grasp_distance=grasp_distance,
        close_joint_threshold=close_joint_threshold,
    )
    lifted_now = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        start_height = _orange_start_height_world(env, orange)
        lifted_now.append(orange.data.root_pos_w[:, 2] > (start_height + lifted_height_delta))
    lifted_now = torch.stack(lifted_now, dim=1)
    return torch.logical_and(grasped_now, lifted_now)


def _lifted_history_per_orange(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    lifted_height_delta: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
    grasp_distance: float = 0.06,
    close_joint_threshold: float = 0.7,
) -> torch.Tensor:
    _reset_episode_progress_if_needed(env, len(oranges_cfg))
    held_now = _currently_held_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        lifted_height_delta=lifted_height_delta,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        ee_frame_index=ee_frame_index,
        grasp_distance=grasp_distance,
        close_joint_threshold=close_joint_threshold,
    )
    env._pick_orange_lifted_history |= held_now
    return env._pick_orange_lifted_history


def _progress_mask_per_orange(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    lifted_height_delta: float = 0.04,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
    grasp_distance: float = 0.06,
    close_joint_threshold: float = 0.7,
) -> torch.Tensor:
    """Task progress: an orange only counts once it was lifted before being on the plate."""
    placed_mask = _placed_mask_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
        x_range=x_range,
        y_range=y_range,
        height_range=height_range,
    )
    lifted_history = _lifted_history_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        lifted_height_delta=lifted_height_delta,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        ee_frame_index=ee_frame_index,
        grasp_distance=grasp_distance,
        close_joint_threshold=close_joint_threshold,
    )
    env._pick_orange_progress_mask = torch.logical_and(placed_mask, lifted_history)
    _advance_active_orange_if_needed(env, env._pick_orange_progress_mask)
    return env._pick_orange_progress_mask


def _active_target_mask(env: ManagerBasedRLEnv, placed_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-hot mask for the explicit active orange and all-placed flag."""
    active_idx = _ensure_active_orange_state(env, placed_mask.shape[1])
    all_placed = torch.all(placed_mask, dim=1)
    active_mask = torch.nn.functional.one_hot(active_idx, num_classes=placed_mask.shape[1]).bool()
    active_mask = torch.logical_and(active_mask, torch.logical_not(all_placed).unsqueeze(1))
    return active_mask, all_placed


def reach_unplaced_oranges_dense(
    env: ManagerBasedRLEnv,
    std: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
) -> torch.Tensor:
    """Reward proximity between end-effector and the first unplaced orange."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :3]

    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    active_mask, all_placed = _active_target_mask(env, placed_mask)
    distance_list = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        distance_list.append(torch.linalg.vector_norm(orange.data.root_pos_w[:, :3] - ee_pos, dim=1))
    distances = torch.stack(distance_list, dim=1)

    active_distance = torch.sum(torch.where(active_mask, distances, torch.zeros_like(distances)), dim=1)
    return torch.where(all_placed, torch.zeros_like(active_distance), 1.0 - torch.tanh(active_distance / std))


def grasp_unplaced_oranges_dense(
    env: ManagerBasedRLEnv,
    std: float,
    close_joint_threshold: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
    lift_progress_height: float = 0.05,
    lift_progress_floor: float = 0.25,
) -> torch.Tensor:
    """Reward closing gripper while near the first unplaced orange.

    Multiplied by a lift-progress factor with a small floor: grasping pays only a
    fraction (``lift_progress_floor``) when the orange is on the table and ramps to
    the full value once the orange is lifted to ``lift_progress_height`` above its
    starting height. This prevents the policy from camping on a closed-gripper pose
    over the orange without actually picking it up.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :3]
    gripper_closed = (robot.data.joint_pos[:, -1] < close_joint_threshold).float()

    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    active_mask, all_placed = _active_target_mask(env, placed_mask)
    near_scores = []
    lift_factors = []
    effective_height = max(lift_progress_height, 1.0e-6)
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        distance = torch.linalg.vector_norm(orange.data.root_pos_w[:, :3] - ee_pos, dim=1)
        near_scores.append(1.0 - torch.tanh(distance / std))

        start_height = _orange_start_height_world(env, orange)
        lift_progress = torch.clamp(
            (orange.data.root_pos_w[:, 2] - start_height) / effective_height, min=0.0, max=1.0
        )
        lift_factors.append(lift_progress_floor + (1.0 - lift_progress_floor) * lift_progress)
    near_scores = torch.stack(near_scores, dim=1)
    lift_factors = torch.stack(lift_factors, dim=1)
    near_scores = near_scores * lift_factors
    active_near = torch.sum(torch.where(active_mask, near_scores, torch.zeros_like(near_scores)), dim=1)
    return torch.where(all_placed, torch.zeros_like(active_near), active_near * gripper_closed)


def lift_unplaced_oranges_dense(
    env: ManagerBasedRLEnv,
    target_height_delta: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
    grasp_distance: float = 0.06,
    close_joint_threshold: float = 0.7,
    start_height_tolerance: float = 0.005,
) -> torch.Tensor:
    """Reward vertical lift progress only when the gripper is plausibly holding the active orange."""
    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    active_mask, all_placed = _active_target_mask(env, placed_mask)
    grasped_mask = _gripper_near_closed_mask(
        env,
        oranges_cfg=oranges_cfg,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        ee_frame_index=ee_frame_index,
        grasp_distance=grasp_distance,
        close_joint_threshold=close_joint_threshold,
    )
    progress_scores = []
    effective_target = max(target_height_delta - start_height_tolerance, 1.0e-6)
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        start_height = _orange_start_height_world(env, orange)
        effective_lift = torch.clamp(orange.data.root_pos_w[:, 2] - start_height - start_height_tolerance, min=0.0)
        progress_scores.append(torch.clamp(effective_lift / effective_target, min=0.0, max=1.0))
    progress_scores = torch.stack(progress_scores, dim=1)
    progress_scores = progress_scores * grasped_mask.float()
    active_progress = torch.sum(torch.where(active_mask, progress_scores, torch.zeros_like(progress_scores)), dim=1)
    return torch.where(all_placed, torch.zeros_like(active_progress), active_progress)


def active_orange_over_lift_penalty(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    max_height_delta: float = 0.12,
) -> torch.Tensor:
    """Penalty for lifting the active orange far above the useful carry height."""
    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    active_mask, all_placed = _active_target_mask(env, placed_mask)

    penalties = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        start_height = _orange_start_height_world(env, orange)
        height_delta = orange.data.root_pos_w[:, 2] - start_height
        penalties.append(torch.clamp(height_delta - max_height_delta, min=0.0))
    penalties = torch.stack(penalties, dim=1)
    active_penalty = torch.sum(torch.where(active_mask, penalties, torch.zeros_like(penalties)), dim=1)
    return torch.where(all_placed, torch.zeros_like(active_penalty), active_penalty)


def move_unplaced_oranges_to_plate_dense(
    env: ManagerBasedRLEnv,
    std: float,
    lifted_height_delta: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    preplace_height_above_plate: float | None = None,
) -> torch.Tensor:
    """Reward moving the currently-held active orange toward the plate or pre-place pose.

    Gated by currently-held (grasped AND lifted) rather than sticky lift-history so that
    dropping the orange to sweep it across the table immediately stops earning credit.
    """
    plate: RigidObject = env.scene[plate_cfg.name]
    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg, lifted_height_delta=lifted_height_delta)
    active_mask, all_placed = _active_target_mask(env, placed_mask)

    scores = []
    held_now = _currently_held_per_orange(env, oranges_cfg=oranges_cfg, lifted_height_delta=lifted_height_delta)
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        if preplace_height_above_plate is None:
            distance = torch.linalg.vector_norm(orange.data.root_pos_w[:, :2] - plate.data.root_pos_w[:, :2], dim=1)
        else:
            preplace_target = plate.data.root_pos_w[:, :3].clone()
            preplace_target[:, 2] += preplace_height_above_plate
            distance = torch.linalg.vector_norm(orange.data.root_pos_w[:, :3] - preplace_target, dim=1)
        scores.append(1.0 - torch.tanh(distance / std))
    scores = torch.stack(scores, dim=1)
    scores = scores * held_now.float()
    active_score = torch.sum(torch.where(active_mask, scores, torch.zeros_like(scores)), dim=1)
    return torch.where(all_placed, torch.zeros_like(active_score), active_score)


def lower_unplaced_oranges_to_plate_dense(
    env: ManagerBasedRLEnv,
    std: float,
    lifted_height_delta: float,
    target_height_above_plate: float,
    near_plate_xy: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
) -> torch.Tensor:
    """Reward lowering the held active orange toward plate height once it is near the plate."""
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_pos = plate.data.root_pos_w[:, :3]
    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg, lifted_height_delta=lifted_height_delta)
    active_mask, all_placed = _active_target_mask(env, placed_mask)
    held_now = _currently_held_per_orange(env, oranges_cfg=oranges_cfg, lifted_height_delta=lifted_height_delta)

    scores = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        orange_pos = orange.data.root_pos_w[:, :3]
        distance_xy = torch.linalg.vector_norm(orange_pos[:, :2] - plate_pos[:, :2], dim=1)
        near_plate = 1.0 - torch.tanh(distance_xy / near_plate_xy)
        target_z = plate_pos[:, 2] + target_height_above_plate
        height_score = 1.0 - torch.tanh(torch.abs(orange_pos[:, 2] - target_z) / std)
        scores.append(near_plate * height_score)
    scores = torch.stack(scores, dim=1)
    scores = scores * held_now.float()
    active_score = torch.sum(torch.where(active_mask, scores, torch.zeros_like(scores)), dim=1)
    return torch.where(all_placed, torch.zeros_like(active_score), active_score)


def oranges_on_plate_fraction(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
    lifted_height_delta: float = 0.08,
) -> torch.Tensor:
    """Reward proportional to fraction of oranges successfully placed on plate.

    Gated by ``lifted_history``: only oranges that were grasped AND lifted at least
    ``lifted_height_delta`` above their starting height at some point during the
    episode count as placed. The threshold is intentionally above what stochastic
    arm motion can produce by luck so that "sweep onto plate" never triggers credit.
    """
    placed_mask = _progress_mask_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
        lifted_height_delta=lifted_height_delta,
        x_range=x_range,
        y_range=y_range,
        height_range=height_range,
    )
    return torch.mean(placed_mask.float(), dim=1)


def rest_pose_after_all_placed(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Reward returning robot to rest pose after all oranges are placed."""
    placed_mask = _progress_mask_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
        lifted_height_delta=0.04,
        x_range=x_range,
        y_range=y_range,
        height_range=height_range,
    )
    all_placed = torch.all(placed_mask, dim=1)
    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names
    at_rest = is_so101_at_rest_pose(joint_pos, joint_names)
    return torch.logical_and(all_placed, at_rest).float()


def pick_orange_success_bonus(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Sparse bonus matching task completion after lift-gated placement progress."""
    progress_mask = _progress_mask_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
        lifted_height_delta=0.04,
        x_range=x_range,
        y_range=y_range,
        height_range=height_range,
    )
    all_placed = torch.all(progress_mask, dim=1)
    task_complete = task_done(
        env=env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
        x_range=x_range,
        y_range=y_range,
        height_range=height_range,
    )
    return torch.logical_and(all_placed, task_complete).float()


def non_active_orange_displacement_penalty(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    displacement_tolerance: float = 0.025,
) -> torch.Tensor:
    """Penalty for sweeping future oranges away before they become the active target."""
    progress_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    active_mask, _ = _active_target_mask(env, progress_mask)
    future_mask = torch.logical_and(torch.logical_not(progress_mask), torch.logical_not(active_mask))

    displacements = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        default_pos_w = orange.data.default_root_state[:, :3] + env.scene.env_origins
        xy_displacement = torch.linalg.vector_norm(orange.data.root_pos_w[:, :2] - default_pos_w[:, :2], dim=1)
        displacements.append(torch.clamp(xy_displacement - displacement_tolerance, min=0.0))
    displacement = torch.stack(displacements, dim=1)
    return torch.sum(torch.where(future_mask, displacement, torch.zeros_like(displacement)), dim=1)


def active_orange_pre_lift_displacement_penalty(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    displacement_tolerance: float = 0.015,
    lifted_height_delta: float = 0.04,
) -> torch.Tensor:
    """Penalty for pushing the active orange horizontally while it is on (or near) the table.

    Triggered per-step whenever the active orange's height stays at or below the lift
    threshold, regardless of whether it was briefly lifted earlier in the episode. This
    closes the lift-then-drop-and-sweep loophole left open by the prior history-based gate.
    """
    progress_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg, lifted_height_delta=lifted_height_delta)
    active_mask, _ = _active_target_mask(env, progress_mask)

    below_lift_list = []
    displacements = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        start_height = _orange_start_height_world(env, orange)
        below_lift_list.append(orange.data.root_pos_w[:, 2] <= (start_height + lifted_height_delta))

        default_pos_w = orange.data.default_root_state[:, :3] + env.scene.env_origins
        xy_displacement = torch.linalg.vector_norm(orange.data.root_pos_w[:, :2] - default_pos_w[:, :2], dim=1)
        displacements.append(torch.clamp(xy_displacement - displacement_tolerance, min=0.0))
    below_lift = torch.stack(below_lift_list, dim=1)
    displacement = torch.stack(displacements, dim=1)

    penalized_mask = torch.logical_and(active_mask, below_lift)
    return torch.sum(torch.where(penalized_mask, displacement, torch.zeros_like(displacement)), dim=1)
