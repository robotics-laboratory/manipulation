"""Utilities for saving and restoring LiftCube recovery-start states."""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any

import torch
from isaaclab.utils.math import subtract_frame_transforms


RECOVERY_STATE_VERSION = 1


def _env0_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Return env-0 tensor data detached on CPU."""
    return tensor[0].detach().cpu().clone()


def _relative_root_pose(asset, env_origin: torch.Tensor) -> torch.Tensor:
    root_pose = _env0_cpu(asset.data.root_state_w[:, :7])
    root_pose[:3] -= env_origin.detach().cpu()
    return root_pose


def _mlp_policy_obs(env) -> dict[str, torch.Tensor]:
    """Compute the dense MLP teacher observation terms for inspection."""
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    default_joint_pos = robot.data.default_joint_pos
    cube_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        cube.data.root_pos_w[:, :3],
    )

    return {
        "joint_pos": _env0_cpu(joint_pos - default_joint_pos),
        "joint_vel": _env0_cpu(joint_vel),
        "cube_position": _env0_cpu(cube_pos_b),
        "actions": _env0_cpu(env.action_manager.action),
    }


def capture_lift_cube_state(
    env,
    *,
    task: str,
    instruction: str | None,
    episode_index: int,
    episode_step: int,
    last_action: torch.Tensor | None = None,
    include_mlp_obs: bool = True,
) -> dict[str, Any]:
    """Capture a minimal resettable LiftCube state for env 0."""
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    env_origin = _env0_cpu(env.scene.env_origins)

    state: dict[str, Any] = {
        "version": RECOVERY_STATE_VERSION,
        "created_at": time(),
        "task": task,
        "instruction": instruction,
        "episode_index": episode_index,
        "episode_step": episode_step,
        "is_relative": True,
        "env_origin": env_origin,
        "robot_joint_names": list(robot.data.joint_names),
        "robot_joint_pos": _env0_cpu(robot.data.joint_pos),
        "robot_joint_vel": _env0_cpu(robot.data.joint_vel),
        "robot_root_pose": _relative_root_pose(robot, env_origin),
        "robot_root_vel": _env0_cpu(robot.data.root_state_w[:, 7:]),
        "cube_root_pose": _relative_root_pose(cube, env_origin),
        "cube_root_vel": _env0_cpu(cube.data.root_state_w[:, 7:]),
    }
    if last_action is not None:
        state["last_action"] = last_action.detach().cpu().clone()
    if include_mlp_obs:
        state["mlp_policy_obs"] = _mlp_policy_obs(env)
    return state


def save_lift_cube_state(state: dict[str, Any], output_dir: str | Path, prefix: str, index: int) -> Path:
    """Save a captured recovery state as a numbered .pt file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_{index:06d}.pt"
    torch.save(state, path)
    return path


def load_lift_cube_state(path: str | Path, device: str | torch.device) -> dict[str, Any]:
    """Load a recovery state and move tensors to the requested device."""
    state = torch.load(path, map_location=device)
    if state.get("version") != RECOVERY_STATE_VERSION:
        raise ValueError(f"Unsupported recovery state version in {path}: {state.get('version')}")
    return state


def _batched(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device).unsqueeze(0)


def restore_lift_cube_state(env, state: dict[str, Any], env_id: int = 0) -> None:
    """Restore robot and cube state into a single-env LiftCube environment."""
    if env_id != 0:
        raise ValueError("Recovery state files currently store env 0 only.")

    robot = env.scene["robot"]
    cube = env.scene["cube"]
    device = env.device
    env_ids = torch.tensor([env_id], dtype=torch.long, device=device)
    env_origin = env.scene.env_origins[env_ids][0]

    robot_joint_pos = _batched(state["robot_joint_pos"], device)
    robot_joint_vel = _batched(state["robot_joint_vel"], device)
    robot_root_pose = _batched(state["robot_root_pose"], device)
    cube_root_pose = _batched(state["cube_root_pose"], device)
    if state.get("is_relative", False):
        robot_root_pose[:, :3] += env_origin
        cube_root_pose[:, :3] += env_origin

    robot_root_vel = _batched(state["robot_root_vel"], device)
    cube_root_vel = _batched(state["cube_root_vel"], device)

    robot.write_root_pose_to_sim(robot_root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_root_vel, env_ids=env_ids)
    robot.set_joint_position_target(robot_joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(robot_joint_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)

    cube.write_root_pose_to_sim(cube_root_pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(cube_root_vel, env_ids=env_ids)

