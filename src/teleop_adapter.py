from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS


LEADER_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class CompatibilityResult:
    ok: bool
    message: str
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    action_dim: int


def _find_robot_articulation(env, robot_name: str = "robot"):
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    articulations = getattr(scene, "articulations", None)
    if articulations is None:
        return None
    if hasattr(articulations, "get"):
        robot = articulations.get(robot_name)
        if robot is not None:
            return robot
    if hasattr(articulations, "__len__") and len(articulations) > 0:
        return articulations[0]
    return None


def _get_term_joint_names(env, term_name: str) -> tuple[str, ...]:
    try:
        term = env.action_manager.get_term(term_name)
    except Exception:
        return ()
    return tuple(getattr(term, "_joint_names", ()) or ())


def check_so101_jointpos_compatibility(env, robot_name: str = "robot") -> CompatibilityResult:
    robot = _find_robot_articulation(env, robot_name=robot_name)
    if robot is None:
        return CompatibilityResult(False, "Robot articulation not found in scene.", (), (), 0)

    arm_joint_names = _get_term_joint_names(env, "arm_action")
    gripper_joint_names = _get_term_joint_names(env, "gripper_action")
    if not arm_joint_names or not gripper_joint_names:
        return CompatibilityResult(
            False,
            "Task must expose action terms 'arm_action' and 'gripper_action'.",
            arm_joint_names,
            gripper_joint_names,
            int(env.action_manager.total_action_dim),
        )

    required_arm = {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"}
    if set(arm_joint_names) != required_arm:
        return CompatibilityResult(
            False,
            (
                "Unsupported arm joint layout for SO101 leader mapping. "
                f"Expected {sorted(required_arm)}, got {list(arm_joint_names)}."
            ),
            arm_joint_names,
            gripper_joint_names,
            int(env.action_manager.total_action_dim),
        )
    if len(gripper_joint_names) != 1 or gripper_joint_names[0] != "gripper":
        return CompatibilityResult(
            False,
            f"Unsupported gripper joint layout. Expected ['gripper'], got {list(gripper_joint_names)}.",
            arm_joint_names,
            gripper_joint_names,
            int(env.action_manager.total_action_dim),
        )

    total = int(env.action_manager.total_action_dim)
    if total != 6:
        return CompatibilityResult(
            False,
            f"Unsupported action dimension {total}. Expected 6 for SO101 single-arm mapping.",
            arm_joint_names,
            gripper_joint_names,
            total,
        )

    return CompatibilityResult(True, "Compatible SO101 single-arm joint-position action layout.", arm_joint_names, gripper_joint_names, total)


def make_preprocess_device_action(env, robot_name: str = "robot") -> Callable:
    robot = _find_robot_articulation(env, robot_name=robot_name)
    if robot is None:
        raise RuntimeError("Robot articulation is missing; cannot create teleop action mapping.")

    joint_names = list(getattr(robot, "joint_names", ()))
    index_by_name = {name: i for i, name in enumerate(joint_names)}
    for name in LEADER_JOINT_ORDER:
        if name not in index_by_name:
            raise RuntimeError(f"Robot articulation does not contain required joint '{name}'.")

    arm_joint_names = _get_term_joint_names(env, "arm_action")
    gripper_joint_names = _get_term_joint_names(env, "gripper_action")

    def _preprocess_device_action(_cfg_self, action: dict, _teleop_device) -> torch.Tensor:
        if action.get("so101_leader") is None:
            raise NotImplementedError("This wrapper currently supports only --teleop_device=so101leader.")
        joint_state = action.get("joint_state", {})
        motor_limits = action.get("motor_limits", {})
        out_by_term: dict[str, torch.Tensor] = {}

        arm_values = []
        for joint_name in arm_joint_names:
            if joint_name not in joint_state or joint_name not in motor_limits:
                raise RuntimeError(f"Leader state is missing joint '{joint_name}'.")
            m_min, m_max = motor_limits[joint_name]
            m_range = float(m_max - m_min)
            if abs(m_range) < 1e-8:
                raise RuntimeError(f"Invalid motor range for '{joint_name}': [{m_min}, {m_max}]")
            joint_limit_range_deg = SO101_FOLLOWER_USD_JOINT_LIMLITS[joint_name]
            joint_range = float(joint_limit_range_deg[1] - joint_limit_range_deg[0])
            motor_degree = float(joint_state[joint_name]) - float(m_min)
            processed_degree = motor_degree / m_range * joint_range + float(joint_limit_range_deg[0])
            processed_radius = processed_degree / 180.0 * np.pi
            arm_values.append(processed_radius)
        out_by_term["arm_action"] = torch.tensor(
            [arm_values], device=env.device, dtype=torch.float32
        ).repeat(env.num_envs, 1)

        grip_name = gripper_joint_names[0]
        if grip_name not in joint_state or grip_name not in motor_limits:
            raise RuntimeError(f"Leader state is missing gripper joint '{grip_name}'.")
        m_min, m_max = motor_limits[grip_name]
        m_range = float(m_max - m_min)
        if abs(m_range) < 1e-8:
            raise RuntimeError(f"Invalid motor range for '{grip_name}': [{m_min}, {m_max}]")
        joint_limit_range_deg = SO101_FOLLOWER_USD_JOINT_LIMLITS[grip_name]
        joint_range = float(joint_limit_range_deg[1] - joint_limit_range_deg[0])
        motor_degree = float(joint_state[grip_name]) - float(m_min)
        processed_degree = motor_degree / m_range * joint_range + float(joint_limit_range_deg[0])
        grip_value = processed_degree / 180.0 * np.pi
        out_by_term["gripper_action"] = torch.tensor(
            [[grip_value]], device=env.device, dtype=torch.float32
        ).repeat(env.num_envs, 1)

        # Concatenate in active action term order.
        chunks = []
        for term_name in env.action_manager.active_terms:
            if term_name not in out_by_term:
                raise RuntimeError(
                    f"Unsupported active action term '{term_name}'. "
                    "Expected only 'arm_action' and 'gripper_action'."
                )
            chunks.append(out_by_term[term_name])
        return torch.cat(chunks, dim=-1)

    return _preprocess_device_action


def collect_camera_frames(env) -> dict[str, Image.Image]:
    frames: dict[str, Image.Image] = {}
    scene = getattr(env, "scene", None)
    if scene is None:
        return frames
    sensors = getattr(scene, "sensors", None)
    if sensors is None:
        return frames

    for sensor_name, sensor in sensors.items():
        data = getattr(sensor, "data", None)
        output = getattr(data, "output", None)
        if not isinstance(output, dict) or "rgb" not in output:
            continue
        rgb = output["rgb"]
        if rgb is None or rgb.shape[0] == 0:
            continue
        rgb_np = rgb[0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
        lower = sensor_name.lower()
        if "top" in lower:
            key = "observation.images.top"
        elif "wrist" in lower:
            key = "observation.images.wrist"
        elif "side" in lower:
            key = "observation.images.side"
        else:
            key = f"observation.images.{sensor_name}"
        frames[key] = Image.fromarray(rgb_np)
    return frames


def collect_joint_state_and_action(env, env_action: torch.Tensor, robot_name: str = "robot") -> tuple[np.ndarray, np.ndarray]:
    robot = _find_robot_articulation(env, robot_name=robot_name)
    if robot is None:
        raise RuntimeError("Robot articulation not found while collecting state.")

    joint_names = list(getattr(robot, "joint_names", ()))
    index_by_name = {name: i for i, name in enumerate(joint_names)}
    required = LEADER_JOINT_ORDER
    if any(name not in index_by_name for name in required):
        raise RuntimeError(
            f"Robot joint names missing required SO101 joints. Available: {joint_names}"
        )
    joint_pos = robot.data.joint_pos[0]
    state_rad = torch.stack([joint_pos[index_by_name[name]] for name in required], dim=0)
    state_deg = torch.rad2deg(state_rad).detach().cpu().numpy().astype(np.float32)

    action_np = env_action[0, : len(required)].detach().cpu().numpy().astype(np.float32)
    action_deg = np.rad2deg(action_np).astype(np.float32)
    return state_deg, action_deg
