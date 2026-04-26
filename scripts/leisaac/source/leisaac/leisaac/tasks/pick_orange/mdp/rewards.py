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


def reach_unplaced_oranges_dense(
    env: ManagerBasedRLEnv,
    std: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
) -> torch.Tensor:
    """Reward proximity between end-effector and nearest unplaced orange."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :3]

    placed_mask = _placed_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    distance_list = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        distance_list.append(torch.linalg.vector_norm(orange.data.root_pos_w[:, :3] - ee_pos, dim=1))
    distances = torch.stack(distance_list, dim=1)

    masked_distances = torch.where(placed_mask, torch.full_like(distances, 1.0e6), distances)
    nearest = torch.min(masked_distances, dim=1).values
    has_unplaced = torch.logical_not(torch.all(placed_mask, dim=1))
    return torch.where(has_unplaced, 1.0 - torch.tanh(nearest / std), torch.ones_like(nearest))


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
    """Reward closing gripper while near any unplaced orange."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, ee_frame_index, :3]
    gripper_closed = (robot.data.joint_pos[:, -1] < close_joint_threshold).float()

    placed_mask = _placed_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    near_scores = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        distance = torch.linalg.vector_norm(orange.data.root_pos_w[:, :3] - ee_pos, dim=1)
        near_scores.append(1.0 - torch.tanh(distance / std))
    near_scores = torch.stack(near_scores, dim=1)
    near_scores = torch.where(placed_mask, torch.zeros_like(near_scores), near_scores)
    best_near = torch.max(near_scores, dim=1).values
    return best_near * gripper_closed


def lift_unplaced_oranges_dense(
    env: ManagerBasedRLEnv,
    target_height_delta: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    start_height_tolerance: float = 0.005,
) -> torch.Tensor:
    """Reward vertical lift progress of unplaced oranges from their reset heights."""
    placed_mask = _placed_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    progress_scores = []
    effective_target = max(target_height_delta - start_height_tolerance, 1.0e-6)
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        start_height = _orange_start_height_world(env, orange)
        effective_lift = torch.clamp(orange.data.root_pos_w[:, 2] - start_height - start_height_tolerance, min=0.0)
        progress_scores.append(torch.clamp(effective_lift / effective_target, min=0.0, max=1.0))
    progress_scores = torch.stack(progress_scores, dim=1)
    progress_scores = torch.where(placed_mask, torch.zeros_like(progress_scores), progress_scores)
    return torch.max(progress_scores, dim=1).values


def move_unplaced_oranges_to_plate_dense(
    env: ManagerBasedRLEnv,
    std: float,
    lifted_height_delta: float,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
) -> torch.Tensor:
    """Reward reducing XY distance from lifted, unplaced oranges to plate center."""
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_xy = plate.data.root_pos_w[:, :2]
    placed_mask = _placed_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)

    scores = []
    for orange_cfg in oranges_cfg:
        orange: RigidObject = env.scene[orange_cfg.name]
        start_height = _orange_start_height_world(env, orange)
        lifted = orange.data.root_pos_w[:, 2] > (start_height + lifted_height_delta)
        distance_xy = torch.linalg.vector_norm(orange.data.root_pos_w[:, :2] - plate_xy, dim=1)
        scores.append((1.0 - torch.tanh(distance_xy / std)) * lifted.float())
    scores = torch.stack(scores, dim=1)
    scores = torch.where(placed_mask, torch.zeros_like(scores), scores)
    return torch.max(scores, dim=1).values


def oranges_on_plate_fraction(
    env: ManagerBasedRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.07, 0.07),
) -> torch.Tensor:
    """Reward proportional to fraction of oranges successfully placed on plate."""
    placed_mask = _placed_mask_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
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
    placed_mask = _placed_mask_per_orange(
        env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
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
    """Sparse bonus matching task_done termination."""
    return task_done(
        env=env,
        oranges_cfg=oranges_cfg,
        plate_cfg=plate_cfg,
        x_range=x_range,
        y_range=y_range,
        height_range=height_range,
    ).float()
