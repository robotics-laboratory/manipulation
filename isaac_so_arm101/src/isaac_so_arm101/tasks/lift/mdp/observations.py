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

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from . import rewards as rewards_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def teacher_gripper_cmd_and_close_hint(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    close_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Teacher gripper cmd + close hint (shape ``(N, 2)``); one cached path projection per step."""

    geom = rewards_mdp._trajectory_guidance_get_cached_path_geometry(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    if not geom["has_gripper"] or geom["teacher_gripper"] is None:
        return torch.zeros((env.num_envs, 2), dtype=torch.float32, device=env.device)
    g = geom["teacher_gripper"]
    hint = (g < float(close_threshold)).to(dtype=torch.float32)
    return torch.stack((g, hint), dim=-1)


def teacher_gripper_path_aligned(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Teacher gripper joint at path-aligned time index (shape ``(N, 1)``)."""

    geom = rewards_mdp._trajectory_guidance_get_cached_path_geometry(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    if not geom["has_gripper"] or geom["teacher_gripper"] is None:
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
    return geom["teacher_gripper"].unsqueeze(-1)


def teacher_gripper_close_hint(
    env: ManagerBasedRLEnv,
    trajectory_file: str,
    command_name: str = "object_pose",
    match_mode: str = "object_goal",
    exact_match_tol: float = 1.0e-3,
    close_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary hint: 1 if teacher gripper below ``close_threshold``."""

    cmd = teacher_gripper_path_aligned(
        env,
        trajectory_file,
        command_name,
        match_mode,
        exact_match_tol,
        object_cfg,
        robot_cfg,
        ee_frame_cfg,
    )
    return (cmd.squeeze(-1) < float(close_threshold)).to(dtype=torch.float32).unsqueeze(-1)
