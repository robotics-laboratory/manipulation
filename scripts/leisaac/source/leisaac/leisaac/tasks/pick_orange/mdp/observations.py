import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms


def _orange_on_plate_mask(
    env: ManagerBasedRLEnv | DirectRLEnv,
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
    env: ManagerBasedRLEnv | DirectRLEnv,
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


def _progress_mask_per_orange(
    env: ManagerBasedRLEnv | DirectRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Use lift-gated progress from rewards when available; otherwise fall back to geometric placement."""
    progress_mask = getattr(env, "_pick_orange_progress_mask", None)
    if progress_mask is not None and progress_mask.shape == (env.num_envs, len(oranges_cfg)):
        reset_env_ids = (env.episode_length_buf == 0).nonzero(as_tuple=True)[0]
        if reset_env_ids.numel() > 0:
            progress_mask[reset_env_ids] = False
            lifted_history = getattr(env, "_pick_orange_lifted_history", None)
            if lifted_history is not None and lifted_history.shape == progress_mask.shape:
                lifted_history[reset_env_ids] = False
        return progress_mask
    return _placed_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)


def orange_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check if an object(orange) is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)

    return grasped


def put_orange_to_plate(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60,
) -> torch.Tensor:
    """Check if an object(orange) is placed on the specified plate."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    orange: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    plate_x, plate_y = plate.data.root_pos_w[:, 0], plate.data.root_pos_w[:, 1]
    orange_x, orange_y = orange.data.root_pos_w[:, 0], orange.data.root_pos_w[:, 1]
    orange_in_plate_x = torch.logical_and(orange_x < plate_x + x_range[1], orange_x > plate_x + x_range[0])
    orange_in_plate_y = torch.logical_and(orange_y < plate_y + y_range[1], orange_y > plate_y + y_range[0])
    orange_in_plate = torch.logical_and(orange_in_plate_x, orange_in_plate_y)

    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    orange_pos = orange.data.root_pos_w
    pos_diff = torch.linalg.vector_norm(orange_pos - end_effector_pos, dim=1)
    ee_near_to_orange = pos_diff < diff_threshold

    gripper_open = robot.data.joint_pos[:, -1] > grasp_threshold

    placed = torch.logical_and(orange_in_plate, ee_near_to_orange)
    placed = torch.logical_and(placed, gripper_open)

    return placed


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("Orange001"),
) -> torch.Tensor:
    """Object position represented in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj_pos_w)
    return obj_pos_b


def objects_positions_in_robot_root_frame(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    objects_cfg: list[SceneEntityCfg] | None = None,
) -> torch.Tensor:
    """Object positions represented in the robot base frame, concatenated in task order."""
    if objects_cfg is None:
        objects_cfg = [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")]

    robot: Articulation = env.scene[robot_cfg.name]
    positions = []
    for object_cfg in objects_cfg:
        obj: RigidObject = env.scene[object_cfg.name]
        obj_pos_b, _ = subtract_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj.data.root_pos_w[:, :3])
        positions.append(obj_pos_b)
    return torch.cat(positions, dim=1)


def active_unplaced_orange_position_in_robot_root_frame(
    env: ManagerBasedRLEnv | DirectRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the first not-yet-placed orange in robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    active_idx = torch.argmin(placed_mask.int(), dim=1)
    all_placed = torch.all(placed_mask, dim=1)

    orange_positions = torch.stack([env.scene[orange_cfg.name].data.root_pos_w[:, :3] for orange_cfg in oranges_cfg], dim=1)
    env_ids = torch.arange(env.num_envs, device=robot.data.root_pos_w.device)
    active_pos_w = orange_positions[env_ids, active_idx]
    active_pos_b, _ = subtract_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], active_pos_w)
    return torch.where(all_placed.unsqueeze(1), torch.zeros_like(active_pos_b), active_pos_b)


def placed_oranges_flags(
    env: ManagerBasedRLEnv | DirectRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
) -> torch.Tensor:
    """Binary placement flags for each orange in fixed task order."""
    return _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg).float()


def all_oranges_placed(
    env: ManagerBasedRLEnv | DirectRLEnv,
    oranges_cfg: list[SceneEntityCfg],
    plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
) -> torch.Tensor:
    """Whether all oranges have been placed on the plate."""
    placed_mask = _progress_mask_per_orange(env, oranges_cfg=oranges_cfg, plate_cfg=plate_cfg)
    return torch.all(placed_mask, dim=1).float().unsqueeze(1)
