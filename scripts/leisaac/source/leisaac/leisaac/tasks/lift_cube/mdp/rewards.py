from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _start_height_world(
    env: ManagerBasedRLEnv,
    obj: RigidObject,
) -> torch.Tensor:
    """Default object reset height in world frame."""
    return obj.data.default_root_state[:, 2] + env.scene.env_origins[:, 2]


def _lift_height_above_robot_base(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    robot_base_name: str,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(robot_base_name)
    return obj.data.root_pos_w[:, 2] - robot.data.body_pos_w[:, base_index, 2]


def _update_episode_max_lift_height_above_base(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    robot_base_name: str,
) -> None:
    lift_height = _lift_height_above_robot_base(
        env=env, object_cfg=object_cfg, robot_cfg=robot_cfg, robot_base_name=robot_base_name
    )
    if not hasattr(env, "_lift_cube_episode_max_height_above_base"):
        env._lift_cube_episode_max_height_above_base = lift_height.clone()
    else:
        env._lift_cube_episode_max_height_above_base = torch.maximum(env._lift_cube_episode_max_height_above_base, lift_height)


def reach_object_dense(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
) -> torch.Tensor:
    """Dense reaching reward based on end-effector to cube distance."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, ee_frame_index, :3]
    distance = torch.linalg.vector_norm(obj_pos_w - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def grasp_closure_dense(
    env: ManagerBasedRLEnv,
    std: float,
    close_joint_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ee_frame_index: int = 1,
) -> torch.Tensor:
    """Reward closing gripper when the end-effector is near the cube."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[:, ee_frame_index, :3]
    distance = torch.linalg.vector_norm(obj_pos_w - ee_pos_w, dim=1)
    near_cube = 1.0 - torch.tanh(distance / std)
    gripper_closed = (robot.data.joint_pos[:, -1] < close_joint_threshold).float()
    return near_cube * gripper_closed


def lift_progress_dense(
    env: ManagerBasedRLEnv,
    target_height_delta: float,
    start_height_tolerance: float = 0.005,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Reward proportional vertical progress from reset height to lift target."""
    _update_episode_max_lift_height_above_base(
        env=env, object_cfg=object_cfg, robot_cfg=robot_cfg, robot_base_name=robot_base_name
    )

    obj: RigidObject = env.scene[object_cfg.name]
    start_height = _start_height_world(env, obj)

    z = obj.data.root_pos_w[:, 2]
    # Ignore tiny positive deltas caused by contact settling right after reset.
    effective_lift = torch.clamp(z - start_height - start_height_tolerance, min=0.0)
    effective_target = max(target_height_delta - start_height_tolerance, 1.0e-6)
    progress = effective_lift / effective_target
    return torch.clamp(progress, min=0.0, max=1.0)


def episode_max_lift_height_above_base(
    env: ManagerBasedRLEnv,
    env_ids,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
) -> dict[str, torch.Tensor]:
    """Log average maximum cube lift height above robot base for episodes being reset."""
    _update_episode_max_lift_height_above_base(
        env=env, object_cfg=object_cfg, robot_cfg=robot_cfg, robot_base_name=robot_base_name
    )
    episode_max = env._lift_cube_episode_max_height_above_base
    metric = torch.mean(episode_max[env_ids])
    episode_max[env_ids] = -1.0e9
    return {"avg": metric}


def object_is_lifted_delta(
    env: ManagerBasedRLEnv,
    minimal_height_delta: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Binary reward for lifting object above start height plus delta."""
    obj: RigidObject = env.scene[object_cfg.name]
    start_height = _start_height_world(env, obj)
    return (obj.data.root_pos_w[:, 2] > (start_height + minimal_height_delta)).float()


def cube_height_above_base_bonus(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    ramp_width: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    robot_base_name: str = "base",
) -> torch.Tensor:
    """Dense success bonus that ramps up to the same height check as the success termination."""
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    base_index = robot.data.body_names.index(robot_base_name)
    lift_height = cube.data.root_pos_w[:, 2] - robot.data.body_pos_w[:, base_index, 2]
    return torch.clamp((lift_height - (height_threshold - ramp_width)) / ramp_width, min=0.0, max=1.0)


def lifted_stillness_dense(
    env: ManagerBasedRLEnv,
    lifted_height_delta: float,
    velocity_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Reward keeping the object stable once lifted."""
    obj: RigidObject = env.scene[object_cfg.name]
    start_height = _start_height_world(env, obj)
    lifted_height = start_height + lifted_height_delta
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w[:, :3], dim=1)
    stable = 1.0 - torch.tanh(speed / velocity_std)
    lifted = (obj.data.root_pos_w[:, 2] > lifted_height).float()
    return lifted * stable


def _commanded_goal_position_w(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute commanded object target in world coordinates from base-frame command."""
    robot: Articulation = env.scene[robot_cfg.name]
    desired_pos_b = env.command_manager.get_command(command_name)[:, :3]
    desired_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], desired_pos_b
    )
    return desired_pos_w


def goal_tracking_dense(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    lifted_height_delta: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Reward tracking commanded object target after lifting starts."""
    obj: RigidObject = env.scene[object_cfg.name]
    start_height = _start_height_world(env, obj)
    lifted_height = start_height + lifted_height_delta
    desired_pos_w = _commanded_goal_position_w(env, command_name=command_name, robot_cfg=robot_cfg)
    distance = torch.linalg.vector_norm(desired_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    lifted = (obj.data.root_pos_w[:, 2] > lifted_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))


def goal_success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_tolerance: float,
    lifted_height_delta: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Sparse bonus for successfully lifted and goal-aligned cube poses."""
    obj: RigidObject = env.scene[object_cfg.name]
    start_height = _start_height_world(env, obj)
    lifted_height = start_height + lifted_height_delta
    desired_pos_w = _commanded_goal_position_w(env, command_name=command_name, robot_cfg=robot_cfg)
    distance = torch.linalg.vector_norm(desired_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    at_goal = distance < position_tolerance
    lifted = obj.data.root_pos_w[:, 2] > lifted_height
    return (at_goal & lifted).float()
