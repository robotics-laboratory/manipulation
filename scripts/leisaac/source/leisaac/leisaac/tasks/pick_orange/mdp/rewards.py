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

    reset_env_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=True)[0]
    if reset_env_ids.numel() > 0:
        env._pick_orange_lifted_history[reset_env_ids] = False
        env._pick_orange_progress_mask[reset_env_ids] = False


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
    lifted_now = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        start_height = _orange_start_height_world(env, orange)
        lifted_now.append(orange.data.root_pos_w[:, 2] > (start_height + lifted_height_delta))
    lifted_now = torch.stack(lifted_now, dim=1)
    grasped_now = _gripper_near_closed_mask(
        env,
        oranges_cfg=oranges_cfg,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        ee_frame_index=ee_frame_index,
        grasp_distance=grasp_distance,
        close_joint_threshold=close_joint_threshold,
    )
    env._pick_orange_lifted_history |= torch.logical_and(lifted_now, grasped_now)
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
    return env._pick_orange_progress_mask


def _active_target_mask(placed_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-hot mask for first unplaced orange and all-placed flag."""
    active_idx = torch.argmin(placed_mask.int(), dim=1)
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
    active_mask, all_placed = _active_target_mask(placed_mask)
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
) -> torch.Tensor:
    """Reward closing gripper while near the first unplaced orange."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :3]
    gripper_closed = (robot.data.joint_pos[:, -1] < close_joint_threshold).float()

    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    active_mask, all_placed = _active_target_mask(placed_mask)
    near_scores = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        distance = torch.linalg.vector_norm(orange.data.root_pos_w[:, :3] - ee_pos, dim=1)
        near_scores.append(1.0 - torch.tanh(distance / std))
    near_scores = torch.stack(near_scores, dim=1)
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
    active_mask, all_placed = _active_target_mask(placed_mask)
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


def move_unplaced_oranges_to_plate_dense(
    env: ManagerBasedRLEnv,
    std: float,
    lifted_height_delta: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
) -> torch.Tensor:
    """Reward reducing XY distance from lifted first unplaced orange to plate center."""
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_xy = plate.data.root_pos_w[:, :2]
    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg, lifted_height_delta=lifted_height_delta)
    active_mask, all_placed = _active_target_mask(placed_mask)

    scores = []
    lifted_history = _lifted_history_per_orange(env, oranges_cfg=oranges_cfg, lifted_height_delta=lifted_height_delta)
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        distance_xy = torch.linalg.vector_norm(orange.data.root_pos_w[:, :2] - plate_xy, dim=1)
        scores.append(1.0 - torch.tanh(distance_xy / std))
    scores = torch.stack(scores, dim=1)
    scores = scores * lifted_history.float()
    active_score = torch.sum(torch.where(active_mask, scores, torch.zeros_like(scores)), dim=1)
    return torch.where(all_placed, torch.zeros_like(active_score), active_score)


def oranges_on_plate_fraction(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Reward proportional to fraction of oranges successfully placed on plate."""
    placed_mask = _progress_mask_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
        lifted_height_delta=0.04,
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
    active_mask, _ = _active_target_mask(progress_mask)
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
    """Penalty for pushing the active orange horizontally before a valid gripper-held lift."""
    progress_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg, lifted_height_delta=lifted_height_delta)
    active_mask, _ = _active_target_mask(progress_mask)
    lifted_history = _lifted_history_per_orange(env, oranges_cfg=oranges_cfg, lifted_height_delta=lifted_height_delta)
    penalized_mask = torch.logical_and(active_mask, torch.logical_not(lifted_history))

    displacements = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        default_pos_w = orange.data.default_root_state[:, :3] + env.scene.env_origins
        xy_displacement = torch.linalg.vector_norm(orange.data.root_pos_w[:, :2] - default_pos_w[:, :2], dim=1)
        displacements.append(torch.clamp(xy_displacement - displacement_tolerance, min=0.0))
    displacement = torch.stack(displacements, dim=1)
    return torch.sum(torch.where(penalized_mask, displacement, torch.zeros_like(displacement)), dim=1)
