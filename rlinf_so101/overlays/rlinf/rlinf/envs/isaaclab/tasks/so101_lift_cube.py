# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from ..isaaclab_env import IsaaclabBaseEnv


SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class IsaaclabSO101LiftCubeEnv(IsaaclabBaseEnv):
    """RLinf wrapper for the LeIsaac SO-101 LiftCube task.

    Observations are exposed in the same compact format used by RLinf's GR00T
    rollout workers: front image, wrist image, 6D LeRobot-style joint state,
    and a repeated task instruction.
    """

    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
    ):
        self._trajectory_reward_cfg = cfg.get("trajectory_reward", {})
        self._trajectory_reference = None
        super().__init__(
            cfg,
            num_envs,
            seed_offset,
            total_num_processes,
            worker_info,
        )
        self._trajectory_reference = self._load_trajectory_reference()

    def _make_env_function(self):
        """Build the IsaacLab environment inside the subprocess worker."""

        def make_env_isaaclab():
            os.environ.pop("DISPLAY", None)
            self._ensure_leisaac_importable()

            from isaaclab.app import AppLauncher

            sim_app = AppLauncher(headless=True, enable_cameras=True).app

            import leisaac.tasks.lift_cube  # noqa: F401
            from isaaclab_tasks.utils import load_cfg_from_registry

            isaac_env_cfg = load_cfg_from_registry(
                self.isaaclab_env_id, "env_cfg_entry_point"
            )
            isaac_env_cfg.seed = self.seed
            isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs
            isaac_env_cfg.recorders = None
            if hasattr(isaac_env_cfg, "episode_length_s"):
                isaac_env_cfg.episode_length_s = self.cfg.init_params.get(
                    "episode_length_s", isaac_env_cfg.episode_length_s
                )

            task_type = self.cfg.init_params.get("task_type", "so101leader")
            if hasattr(isaac_env_cfg, "use_teleop_device"):
                isaac_env_cfg.use_teleop_device(task_type)
                self._configure_absolute_joint_actions(isaac_env_cfg)

            self._resize_camera(isaac_env_cfg, "front", "front_cam")
            self._resize_camera(isaac_env_cfg, "wrist", "wrist_cam")

            env = gym.make(
                self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array"
            ).unwrapped
            return env, sim_app

        return make_env_isaaclab

    def _ensure_leisaac_importable(self) -> None:
        candidate_paths = [
            self.cfg.init_params.get("leisaac_python_path", None),
            os.environ.get("LEISAAC_PYTHONPATH"),
            "/workspace/leisaac/source/leisaac",
            "/workspace/IsaacLab/manipulation/scripts/leisaac/source/leisaac",
            "/workspace/RLinf/manipulation/scripts/leisaac/source/leisaac",
            "/home/robotics/IsaacLab/manipulation/scripts/leisaac/source/leisaac",
            str(
                Path(__file__).resolve().parents[4]
                / "manipulation/scripts/leisaac/source/leisaac"
            ),
            str(
                Path(__file__).resolve().parents[5]
                / "manipulation/scripts/leisaac/source/leisaac"
            ),
        ]
        for path in candidate_paths:
            if path and Path(path).exists() and path not in sys.path:
                sys.path.insert(0, path)

    @staticmethod
    def _configure_absolute_joint_actions(isaac_env_cfg) -> None:
        for action_name in ("arm_action", "gripper_action"):
            action_cfg = getattr(isaac_env_cfg.actions, action_name, None)
            if action_cfg is None:
                continue
            if hasattr(action_cfg, "use_default_offset"):
                action_cfg.use_default_offset = False
            if hasattr(action_cfg, "scale"):
                action_cfg.scale = 1.0

    def _resize_camera(self, isaac_env_cfg, scene_name: str, cfg_name: str) -> None:
        camera_cfg = self.cfg.init_params.get(cfg_name, None)
        scene_camera = getattr(isaac_env_cfg.scene, scene_name, None)
        if camera_cfg is None or scene_camera is None:
            return
        scene_camera.height = camera_cfg.height
        scene_camera.width = camera_cfg.width

    def _wrap_obs(self, obs):
        policy_obs = obs["policy"]
        record_obs = obs.get("record", None)
        instruction = [self.task_description] * self.num_envs

        # Prefer record observations when available (collection/dense variants) since
        # they contain raw camera frames and absolute joint positions.
        if isinstance(record_obs, dict):
            joint_pos = record_obs.get("joint_pos_abs", record_obs.get("joint_pos", None))
            front = record_obs.get("front", None)
            wrist = record_obs.get("wrist", None)
        elif isinstance(policy_obs, dict):
            joint_pos = policy_obs.get("joint_pos", None)
            front = policy_obs.get("front", None)
            wrist = policy_obs.get("wrist", None)
        else:
            joint_pos, front, wrist = None, None, None

        if joint_pos is None:
            if not torch.is_tensor(policy_obs):
                raise TypeError(
                    "Expected `obs['policy']` to be either dict or tensor, "
                    f"got {type(policy_obs)}."
                )
            # Dense train envs may return concatenated state vectors without camera keys.
            # Use the first six joint dimensions as SO101 state fallback.
            joint_pos = policy_obs[..., : len(SO101_JOINT_NAMES)]

        states = self._convert_leisaac_state_to_lerobot(joint_pos)

        if front is None or wrist is None:
            # Some dense reward configs disable raw camera observations. Provide
            # placeholder frames so SmolVLA input contract stays valid.
            h = int(self.cfg.init_params.front_cam.height)
            w = int(self.cfg.init_params.front_cam.width)
            front = torch.zeros((self.num_envs, h, w, 3), dtype=torch.uint8, device=states.device)
            wrist = torch.zeros((self.num_envs, h, w, 3), dtype=torch.uint8, device=states.device)

        env_obs = {
            "main_images": front,
            "task_descriptions": instruction,
            "states": states,
            "wrist_images": wrist,
        }
        if isinstance(policy_obs, dict) and "ee_frame_state" in policy_obs:
            env_obs["ee_frame_states"] = policy_obs["ee_frame_state"]
        return env_obs

    def _convert_leisaac_state_to_lerobot(self, joint_pos: torch.Tensor) -> torch.Tensor:
        try:
            from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

            state_np = convert_leisaac_action_to_lerobot(joint_pos)
            return torch.as_tensor(state_np, device=joint_pos.device, dtype=torch.float32)
        except Exception:
            joint_lows = torch.tensor(
                [-110.0, -100.0, -100.0, -95.0, -160.0, -10.0],
                device=joint_pos.device,
            )
            joint_highs = torch.tensor(
                [110.0, 100.0, 90.0, 95.0, 160.0, 100.0],
                device=joint_pos.device,
            )
            motor_lows = torch.tensor(
                [-100.0, -100.0, -100.0, -100.0, -100.0, 0.0],
                device=joint_pos.device,
            )
            motor_highs = torch.tensor(
                [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                device=joint_pos.device,
            )
            joint_degrees = joint_pos.to(dtype=torch.float32) / torch.pi * 180.0
            joint_fraction = (joint_degrees - joint_lows) / (joint_highs - joint_lows)
            return joint_fraction * (motor_highs - motor_lows) + motor_lows

    def _load_trajectory_reference(self) -> torch.Tensor | None:
        ref_path = self._trajectory_reward_cfg.get("reference_path", None)
        if not ref_path:
            return None

        path = Path(ref_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Trajectory reference does not exist: {path}")

        if path.suffix == ".pt":
            reference = torch.load(path, map_location="cpu")
        elif path.suffix == ".npy":
            import numpy as np

            reference = torch.as_tensor(np.load(path), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported trajectory reference format: {path.suffix}")

        if isinstance(reference, dict):
            reference = reference.get("ee_frame_states", reference.get("states"))
        reference = torch.as_tensor(reference, dtype=torch.float32, device=self.device)
        if reference.ndim != 2:
            raise ValueError(
                f"Expected trajectory reference [T, D], got {tuple(reference.shape)}"
            )
        return reference

    def step(self, actions=None, auto_reset=True):
        obs, step_reward, terminations, truncations, infos = super().step(
            actions, auto_reset=auto_reset
        )
        trajectory_reward = self._calc_trajectory_reward(obs)
        if trajectory_reward is not None:
            infos["env_reward"] = step_reward
            self.returns += trajectory_reward - step_reward
            step_reward = trajectory_reward
            if "episode" in infos:
                infos["episode"]["return"] = self.returns.clone()
                infos["episode"]["reward"] = step_reward
                infos["episode"]["trajectory_distance"] = self._last_trajectory_distance
        return obs, step_reward, terminations, truncations, infos

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.clone()
        episode_info["success_at_end"] = terminations.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _calc_trajectory_reward(self, obs: dict[str, Any]) -> torch.Tensor | None:
        if self._trajectory_reference is None:
            return None
        if "ee_frame_states" not in obs:
            return None

        current = obs["ee_frame_states"][..., : self._trajectory_reference.shape[-1]]
        ref_idx = torch.clamp(
            self.elapsed_steps.long() - 1,
            min=0,
            max=self._trajectory_reference.shape[0] - 1,
        )
        target = self._trajectory_reference[ref_idx].to(current.device)
        distance = torch.linalg.vector_norm(current - target, dim=-1)
        self._last_trajectory_distance = distance.detach()
        sigma = float(self._trajectory_reward_cfg.get("sigma", 0.05))
        scale = float(self._trajectory_reward_cfg.get("scale", 1.0))
        return scale * torch.exp(-distance / sigma)
