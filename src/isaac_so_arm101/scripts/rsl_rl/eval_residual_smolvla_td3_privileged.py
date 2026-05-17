#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained residual checkpoint on LiftCube Collect task.

This script is focused on the privileged/state residual recipe and reports:
- base policy success/return
- residual policy success/return
- max cube height above robot base (debug)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.app import AppLauncher
from tqdm.auto import tqdm

# isort: off
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args
# isort: on


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-RewardDense-Collect-v0")
parser.add_argument("--residual_checkpoint", type=str, required=True, help="Path to residual TD3 checkpoint (.pt).")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help=(
        "Evaluation episodes. If omitted and --target_ci_half_width is set, script computes strict minimum n "
        "so Wilson CI half-width is <= target at --confidence_level. If both are omitted, defaults to 20."
    ),
)
parser.add_argument(
    "--target_ci_half_width",
    type=float,
    default=None,
    help="Optional target half-width for strict Wilson CI auto-sizing of --num_episodes.",
)
parser.add_argument(
    "--confidence_level",
    type=float,
    default=0.95,
    help="Confidence level used with --target_ci_half_width (default: 0.95).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--policy_checkpoint_path", type=str, default=None)
parser.add_argument("--policy_backend", type=str, choices=["local", "service"], default=None)
parser.add_argument("--policy_action_horizon", type=int, default=1)
parser.add_argument(
    "--policy_must_go",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use must-go policy mode by default. Pass --no-policy_must_go to disable.",
)
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=8080)
parser.add_argument("--policy_timeout_ms", type=int, default=5000)
parser.add_argument("--policy_type", type=str, default="smolvla")
parser.add_argument(
    "--skip_teleop_device_setup",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override teleop setup toggle. If omitted, use checkpoint args when available.",
)
parser.add_argument(
    "--state_source",
    type=str,
    choices=["record_joint6", "policy"],
    default=None,
    help="Override residual state source. If omitted, use checkpoint args.",
)
parser.add_argument(
    "--auto_state_source_fallback",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "If actor load fails due to shape mismatch, try alternate state_source automatically. "
        "Disabled by default to avoid mixing incompatible setup checkpoints."
    ),
)
parser.add_argument(
    "--residual_scale",
    type=float,
    default=None,
    help="Override residual scale. If omitted, use checkpoint args.",
)
parser.add_argument(
    "--residual_scale_min",
    type=float,
    default=None,
    help="Lower bound for applied residual action scale. If omitted, use checkpoint args or 0.01.",
)
parser.add_argument(
    "--residual_scale_max",
    type=float,
    default=None,
    help="Upper bound for applied residual action scale. If omitted, use checkpoint args or 0.2.",
)
parser.add_argument(
    "--print_episode_debug",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Print per-episode return/success/max-height debug (default: False).",
)
parser.add_argument(
    "--episode_progress_bar",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show tqdm progress bar across evaluation episodes.",
)
parser.add_argument(
    "--eval_base",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also evaluate base policy in this run. Disabled by default for faster residual-only checks.",
)
parser.add_argument(
    "--base_only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Evaluate only base policy (skip residual evaluation).",
)
parser.add_argument(
    "--record_video",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Record per-episode eval videos from record.front camera.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="logs/eval_videos",
    help="Directory where eval videos are saved when --record_video is enabled.",
)
parser.add_argument("--video_fps", type=int, default=25, help="FPS for saved evaluation videos.")
parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="Optional path to dump structured eval metrics (for post-hoc checkpoint selection).",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.policy import LeRobotServicePolicyClient
from leisaac.tasks.lift_cube import mdp as lift_cube_mdp
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac

import leisaac  # noqa: F401


def _z_from_confidence_level(confidence_level: float) -> float:
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"--confidence_level must be in (0, 1), got: {confidence_level}")
    alpha = 1.0 - confidence_level
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _wilson_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = float(successes) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (phat + z2 / (2.0 * float(n))) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * float(n))) / float(n)) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return float(low), float(high)


def _max_wilson_half_width_for_n(n: int, z: float) -> float:
    if n <= 0:
        return 1.0
    max_half = 0.0
    for successes in range(n + 1):
        low, high = _wilson_interval(successes, n, z)
        half = 0.5 * (high - low)
        if half > max_half:
            max_half = half
    return float(max_half)


def _required_episodes_strict_wilson(target_half_width: float, confidence_level: float) -> int:
    if target_half_width <= 0.0 or target_half_width >= 1.0:
        raise ValueError(f"--target_ci_half_width must be in (0, 1), got: {target_half_width}")
    z = _z_from_confidence_level(confidence_level)
    approx = int(math.ceil((z * z * 0.25) / (target_half_width * target_half_width)))
    n = max(1, approx - 50)
    max_n = 2_000_000
    while n <= max_n:
        if _max_wilson_half_width_for_n(n, z) <= target_half_width:
            return int(n)
        n += 1
    raise RuntimeError(
        f"Failed to find required episodes up to {max_n} for target_half_width={target_half_width} "
        f"confidence_level={confidence_level}"
    )


class LocalSmolVLAPolicy:
    _STATE_NAMES = (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )

    def __init__(self, pretrained_name_or_path: str, device: str, actions_per_chunk: int):
        try:
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.utils import build_inference_frame
        except ImportError as exc:
            raise RuntimeError(
                "Local SmolVLA backend requires lerobot smolvla package. Install lerobot[smolvla] or use --policy_backend=service."
            ) from exc

        self._build_inference_frame = build_inference_frame
        self.device = torch.device(device)
        self.actions_per_chunk = int(actions_per_chunk)
        self.policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path).to(self.device).eval()
        if hasattr(self.policy, "config") and hasattr(self.policy.config, "n_action_steps"):
            self.policy.config.n_action_steps = self.actions_per_chunk
        preprocess, postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_name_or_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.ds_features = {
            "observation.state": {
                "type": "STATE",
                "dtype": "float32",
                "shape": [6],
                "names": list(self._STATE_NAMES),
            },
            "observation.images.front": {"type": "VISUAL", "dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist": {"type": "VISUAL", "dtype": "video", "shape": [480, 640, 3]},
        }

    def reset(self) -> None:
        if hasattr(self.policy, "reset") and callable(self.policy.reset):
            self.policy.reset()
        if hasattr(self.preprocess, "reset") and callable(self.preprocess.reset):
            self.preprocess.reset()
        if hasattr(self.postprocess, "reset") and callable(self.postprocess.reset):
            self.postprocess.reset()

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        front = observation_dict["front"][0].detach().cpu().numpy().astype(np.uint8)
        wrist = observation_dict["wrist"][0].detach().cpu().numpy().astype(np.uint8)
        joint_pos = observation_dict["joint_pos"][0].detach().cpu().numpy()
        joint_pos = convert_leisaac_action_to_lerobot(joint_pos[None, :])[0]
        policy_obs = {"front": front, "wrist": wrist}
        for i, name in enumerate(self._STATE_NAMES):
            policy_obs[name] = float(joint_pos[i])
        obs_frame = self._build_inference_frame(
            observation=policy_obs,
            ds_features=self.ds_features,
            device=self.device,
            task=str(observation_dict["task_description"]),
            robot_type="",
        )
        model_input = self.preprocess(obs_frame)
        with torch.no_grad():
            action = self.policy.select_action(model_input)
        action = self.postprocess(action)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim >= 2 and action.shape[0] > self.actions_per_chunk:
            action = action[: self.actions_per_chunk]
        action = convert_lerobot_action_to_leisaac(action)
        return torch.from_numpy(action[:, None, :])


class ResidualActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_layer_norm: bool,
        use_state_gate: bool,
        gate_hidden_dim: int,
        gate_init_bias: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = obs_dim + action_dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.use_state_gate = bool(use_state_gate)
        if self.use_state_gate:
            gate_dim = max(1, int(gate_hidden_dim))
            self.gate_net = nn.Sequential(
                nn.Linear(obs_dim, gate_dim),
                nn.ReLU(),
                nn.Linear(gate_dim, 1),
            )
            nn.init.constant_(self.gate_net[-1].bias, float(gate_init_bias))
        else:
            self.gate_net = None

    def forward(self, obs: torch.Tensor, base_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_unit = torch.tanh(self.net(torch.cat([obs, base_action], dim=-1)))
        if self.use_state_gate and self.gate_net is not None:
            gate = torch.sigmoid(self.gate_net(obs))
        else:
            gate = torch.ones((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)
        return residual_unit, gate


class EvalVisualObsEncoder(nn.Module):
    """Inference-only visual encoder matching train_residual_smolvla_td3_privileged setup."""

    class _PatchEmbed2(nn.Module):
        def __init__(self, embed_dim: int, patch_size: int):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=4),
                nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.embed(x)
            return y.flatten(2).transpose(1, 2)

    class _TransformerLayer(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int):
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
            x = x + y
            x = x + self.ff(self.norm2(x))
            return x

    class _MinViT(nn.Module):
        def __init__(self, *, image_size: int, patch_size: int, embed_dim: int, num_heads: int, depth: int):
            super().__init__()
            self.patch_embed = EvalVisualObsEncoder._PatchEmbed2(embed_dim=embed_dim, patch_size=patch_size)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                num_patches = int(self.patch_embed(dummy).shape[1])
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.blocks = nn.Sequential(
                *[EvalVisualObsEncoder._TransformerLayer(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(embed_dim)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.num_patches = num_patches

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.blocks(x)
            return self.norm(x)

    def __init__(
        self,
        *,
        device: torch.device,
        state_dim: int,
        vit_image_size: int,
        vit_depth: int,
        vit_embed_dim: int,
        vit_num_heads: int,
        vit_patch_size: int,
        vit_proj_dim: int,
        project_tokens: bool,
    ):
        super().__init__()
        self.device = device
        self.state_dim = int(state_dim)
        self.vit_image_size = int(vit_image_size)
        self.vit_depth = int(vit_depth)
        self.vit_embed_dim = int(vit_embed_dim)
        self.vit_num_heads = int(vit_num_heads)
        self.vit_patch_size = int(vit_patch_size)
        self.vit_proj_dim = int(vit_proj_dim)
        self.project_tokens = bool(project_tokens)
        self.vit_front = EvalVisualObsEncoder._MinViT(
            image_size=self.vit_image_size,
            patch_size=self.vit_patch_size,
            embed_dim=self.vit_embed_dim,
            num_heads=self.vit_num_heads,
            depth=self.vit_depth,
        ).to(self.device)
        self.vit_wrist = EvalVisualObsEncoder._MinViT(
            image_size=self.vit_image_size,
            patch_size=self.vit_patch_size,
            embed_dim=self.vit_embed_dim,
            num_heads=self.vit_num_heads,
            depth=self.vit_depth,
        ).to(self.device)
        self.front_proj: nn.Linear | None = None
        self.wrist_proj: nn.Linear | None = None
        if self.project_tokens:
            self.front_proj = nn.Linear(self.vit_embed_dim, self.vit_proj_dim).to(self.device)
            self.wrist_proj = nn.Linear(self.vit_embed_dim, self.vit_proj_dim).to(self.device)
            self.output_dim = int(self.state_dim + 2 * self.vit_proj_dim)
        else:
            token_dim = self.vit_embed_dim * self.vit_front.num_patches
            self.output_dim = int(self.state_dim + 2 * token_dim)

    def _to_hwc_uint8(self, image) -> np.ndarray:
        arr = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0) * 255.0 if arr.max() <= 1.0 else np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    def _prep_image(self, image) -> torch.Tensor:
        arr = self._to_hwc_uint8(image)
        x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        if x.shape[-2] != self.vit_image_size or x.shape[-1] != self.vit_image_size:
            x = F.interpolate(x, size=(self.vit_image_size, self.vit_image_size), mode="bilinear", align_corners=False)
        x = x / 255.0 if x.max() > 1.0 else x
        return x - 0.5

    def _fit_state_dim(self, state) -> torch.Tensor:
        arr = state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        out = np.zeros((self.state_dim,), dtype=np.float32)
        n = min(self.state_dim, arr.shape[0])
        if n > 0:
            out[:n] = arr[:n]
        return torch.from_numpy(out).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_single_no_grad(self, *, front, wrist, joint_state) -> torch.Tensor:
        joint_t = self._fit_state_dim(joint_state)
        front_x = self._prep_image(front)
        wrist_x = self._prep_image(wrist)
        front_tokens = self.vit_front(front_x)
        wrist_tokens = self.vit_wrist(wrist_x)
        if self.project_tokens:
            assert self.front_proj is not None and self.wrist_proj is not None
            front_feat = self.front_proj(front_tokens.mean(dim=1))
            wrist_feat = self.wrist_proj(wrist_tokens.mean(dim=1))
        else:
            front_feat = front_tokens.flatten(1, 2)
            wrist_feat = wrist_tokens.flatten(1, 2)
        return torch.cat([joint_t, front_feat, wrist_feat], dim=-1).squeeze(0)


@dataclass
class EvalSummary:
    sr: float
    ret: float
    mean_max_height: float
    max_max_height: float
    successes: int
    episodes: int


def _configure_lerobot_absolute_joint_actions(env_cfg, task_type: str) -> None:
    if task_type != "so101leader":
        return
    for action_name in ("arm_action", "gripper_action"):
        action_cfg = getattr(env_cfg.actions, action_name, None)
        if action_cfg is not None and hasattr(action_cfg, "use_default_offset"):
            action_cfg.use_default_offset = False
            action_cfg.scale = 1.0


def _restore_dense_reward_setup(env_cfg, task: str) -> None:
    if "LiftCube" not in task:
        return
    dense_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-RewardDense-v0", device=args_cli.device, num_envs=1)
    env_cfg.rewards = dense_cfg.rewards
    env_cfg.curriculum = dense_cfg.curriculum


def _ensure_collect_env_terminations(env_cfg, task: str) -> None:
    if "Collect" not in task:
        return
    env_cfg.terminations.success = TerminationTermCfg(
        func=lift_cube_mdp.cube_height_above_base,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
            "height_threshold": 0.20,
        },
    )
    env_cfg.terminations.time_out = TerminationTermCfg(func=isaac_mdp.time_out, time_out=True)


def _create_env(task: str, device: str, skip_teleop: bool, seed: int) -> tuple[gym.Env, ManagerBasedRLEnv, str]:
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    task_type = get_task_type(task)
    if not skip_teleop:
        env_cfg.use_teleop_device(task_type)
    _configure_lerobot_absolute_joint_actions(env_cfg, task_type)
    env_cfg.seed = seed
    env_cfg.recorders = None
    _restore_dense_reward_setup(env_cfg, task)
    _ensure_collect_env_terminations(env_cfg, task)
    sim_env = gym.make(task, cfg=env_cfg, render_mode=None)
    return sim_env, sim_env.unwrapped, task_type


def _build_policy_client(env: ManagerBasedRLEnv, task_type: str, policy_backend: str, policy_ckpt: str, action_horizon: int):
    camera_infos = {k: sensor.image_shape for k, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)}
    if not camera_infos:
        raise RuntimeError("No camera sensors found for SmolVLA base policy.")
    if policy_backend == "local":
        return LocalSmolVLAPolicy(
            pretrained_name_or_path=policy_ckpt,
            device=args_cli.device,
            actions_per_chunk=action_horizon,
        )
    service_kwargs = dict(
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        timeout_ms=args_cli.policy_timeout_ms,
        camera_infos=camera_infos,
        task_type=task_type,
        policy_type=args_cli.policy_type,
        pretrained_name_or_path=policy_ckpt,
        actions_per_chunk=action_horizon,
        device=args_cli.device,
    )
    try:
        return LeRobotServicePolicyClient(force_must_go=args_cli.policy_must_go, **service_kwargs)
    except TypeError:
        # Backward compatibility with client versions that don't support force_must_go.
        return LeRobotServicePolicyClient(**service_kwargs)


def _extract_record_modalities(obs_dict: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = obs_dict["record"]
    return rec["front"][0], rec["wrist"][0], rec["joint_pos_abs"][0]


def _extract_state(obs_dict: dict, state_source: str) -> np.ndarray:
    if state_source == "policy":
        pol = obs_dict["policy"]
        if torch.is_tensor(pol):
            if pol.ndim == 2:
                pol = pol[0]
            return pol.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.asarray(pol, dtype=np.float32).reshape(-1)
    _, _, joint = _extract_record_modalities(obs_dict)
    arr = joint.detach().cpu().numpy() if torch.is_tensor(joint) else np.asarray(joint)
    return arr.astype(np.float32).reshape(-1)[:6]


def _build_base_obs(obs_dict: dict, task_description: str) -> dict:
    rec = obs_dict["record"]
    return {
        "front": rec["front"],
        "wrist": rec["wrist"],
        "joint_pos": rec["joint_pos_abs"],
        "task_description": task_description,
    }


def _extract_front_frame_np(obs_dict: dict) -> np.ndarray:
    rec = obs_dict["record"]
    front = rec["front"][0]
    frame = front.detach().cpu().numpy() if torch.is_tensor(front) else np.asarray(front)
    if frame.ndim != 3:
        raise RuntimeError(f"Unexpected front frame shape: {tuple(frame.shape)}")
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0) * 255.0 if frame.max() <= 1.0 else np.clip(frame, 0.0, 255.0)
        frame = frame.astype(np.uint8)
    return frame


def _save_episode_video(frames: list[np.ndarray], video_path: Path, fps: int) -> None:
    if len(frames) == 0:
        return
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("Video recording requires imageio. Install it or run without --record_video.") from exc
    imageio.mimwrite(str(video_path), frames, fps=max(1, int(fps)))


def _get_action_bounds(env: ManagerBasedRLEnv, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    action_space = env.single_action_space
    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
    low = torch.where(torch.isfinite(low), low, torch.full_like(low, -3.2))
    high = torch.where(torch.isfinite(high), high, torch.full_like(high, 3.2))
    return low, high


def _compose_action(
    base_action: torch.Tensor,
    residual_unit: torch.Tensor,
    residual_scale: float,
    low: torch.Tensor,
    high: torch.Tensor,
    residual_scale_min: float,
    residual_scale_max: float,
    residual_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    scale_min = float(min(residual_scale_min, residual_scale_max))
    scale_max = float(max(residual_scale_min, residual_scale_max))
    scale_cap = float(np.clip(float(residual_scale), scale_min, scale_max))
    if residual_gate is None:
        delta = scale_cap * residual_unit
    else:
        gated_scale = scale_min + (scale_cap - scale_min) * torch.clamp(residual_gate, 0.0, 1.0)
        delta = gated_scale * residual_unit
    return torch.clamp(base_action + delta, low, high)


def _cube_height_above_base_m(env: ManagerBasedRLEnv) -> float:
    cube: RigidObject = env.scene["cube"]
    robot: Articulation = env.scene["robot"]
    base_index = robot.data.body_names.index("base")
    cube_height = float(cube.data.root_pos_w[0, 2].detach().cpu().item())
    base_height = float(robot.data.body_pos_w[0, base_index, 2].detach().cpu().item())
    return cube_height - base_height


def _run_eval(
    *,
    sim_env: gym.Env,
    env: ManagerBasedRLEnv,
    task_type: str,
    policy,
    actor: ResidualActor,
    device: torch.device,
    low: torch.Tensor,
    high: torch.Tensor,
    residual_scale: float,
    residual_scale_min: float,
    residual_scale_max: float,
    state_source: str,
    state_extractor: Callable[[dict], np.ndarray],
    prompt: str,
    use_residual: bool,
    success_height_threshold: float,
    record_video: bool,
    video_dir: str,
    video_fps: int,
) -> EvalSummary:
    success_count = 0
    returns: list[float] = []
    max_heights: list[float] = []
    label = "residual" if use_residual else "base"
    actor.eval()
    ep_iter = tqdm(
        range(args_cli.num_episodes),
        desc=f"eval:{label}",
        unit="ep",
        leave=False,
        disable=not args_cli.episode_progress_bar,
    )
    for ep in ep_iter:
        obs_dict, _ = sim_env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        video_frames: list[np.ndarray] = []
        if record_video:
            video_frames.append(_extract_front_frame_np(obs_dict))
        ep_return = 0.0
        ep_max_height = -1e9
        success = False
        final_height = float("nan")
        success_by_height = False
        while simulation_app.is_running():
            ep_max_height = max(ep_max_height, _cube_height_above_base_m(env))
            state_vec = state_extractor(obs_dict)
            state_t = torch.from_numpy(state_vec.astype(np.float32)).to(device).unsqueeze(0)
            base_obs = _build_base_obs(obs_dict, prompt)
            base_action = policy.get_action(base_obs).to(device)[0, 0, :].float()
            if use_residual:
                with torch.no_grad():
                    residual_unit, residual_gate = actor(state_t, base_action.unsqueeze(0))
                    residual_unit = residual_unit.squeeze(0)
                    residual_gate = residual_gate.squeeze(0)
                action = _compose_action(
                    base_action,
                    residual_unit,
                    residual_scale,
                    low,
                    high,
                    residual_scale_min,
                    residual_scale_max,
                    residual_gate,
                )
            else:
                action = base_action
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, task_type)
            obs_dict, reward, terminated, truncated, _ = sim_env.step(action.unsqueeze(0))
            if record_video:
                video_frames.append(_extract_front_frame_np(obs_dict))
            ep_max_height = max(ep_max_height, _cube_height_above_base_m(env))
            ep_return += float(reward[0].detach().cpu().item())
            done = bool((terminated[0] | truncated[0]).detach().cpu().item())
            if done:
                final_height = _cube_height_above_base_m(env)
                success_by_height = final_height > float(success_height_threshold)
                success = bool(env.reset_terminated[0].detach().cpu().item())
                if success:
                    success_count += 1
                break
        returns.append(ep_return)
        max_heights.append(ep_max_height)
        ep_iter.set_postfix(sr=f"{success_count / max(ep + 1, 1):.3f}", ret=f"{np.mean(returns):.3f}")
        if args_cli.print_episode_debug:
            print(
                f"[EVAL:{label}] ep={ep + 1}/{args_cli.num_episodes} "
                f"ret={ep_return:.3f} success={int(success)} "
                f"max_height_above_base_m={ep_max_height:.4f} final_height_above_base_m={final_height:.4f} "
                f"success_by_height={int(success_by_height)}"
            )
        if record_video:
            video_path = Path(video_dir) / f"{label}_ep{ep + 1:03d}_success{int(success)}.mp4"
            _save_episode_video(video_frames, video_path=video_path, fps=video_fps)
            if args_cli.print_episode_debug:
                print(f"[EVAL:{label}] saved_video={video_path}")
    ep_iter.close()
    return EvalSummary(
        sr=float(success_count / max(args_cli.num_episodes, 1)),
        ret=float(np.mean(returns) if returns else 0.0),
        mean_max_height=float(np.mean(max_heights) if max_heights else 0.0),
        max_max_height=float(np.max(max_heights) if max_heights else 0.0),
        successes=int(success_count),
        episodes=int(args_cli.num_episodes),
    )


def main() -> None:
    if args_cli.num_episodes is None:
        if args_cli.target_ci_half_width is not None:
            args_cli.num_episodes = _required_episodes_strict_wilson(
                target_half_width=float(args_cli.target_ci_half_width),
                confidence_level=float(args_cli.confidence_level),
            )
            print(
                "[INFO] Auto-computed strict num_episodes for Wilson CI: "
                f"n={args_cli.num_episodes} (target_half_width={args_cli.target_ci_half_width:.4f}, "
                f"confidence={args_cli.confidence_level:.3f})"
            )
        else:
            args_cli.num_episodes = 20
    elif args_cli.num_episodes <= 0:
        raise ValueError(f"--num_episodes must be > 0 when provided, got: {args_cli.num_episodes}")

    ckpt = torch.load(args_cli.residual_checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    policy_backend = args_cli.policy_backend or str(ckpt_args.get("policy_backend", "local"))
    policy_ckpt = args_cli.policy_checkpoint_path or ckpt_args.get("policy_checkpoint_path")
    if not policy_ckpt:
        raise ValueError("--policy_checkpoint_path is required (or available in checkpoint args).")
    action_horizon = int(
        args_cli.policy_action_horizon
        if args_cli.policy_action_horizon is not None
        else ckpt_args.get("policy_action_horizon", 1)
    )
    state_source = str(args_cli.state_source or ckpt_args.get("state_source", "record_joint6"))
    obs_encoder_mode = str(ckpt_args.get("obs_encoder", "state"))
    if obs_encoder_mode not in {"state", "vit"}:
        obs_encoder_mode = "state"
    residual_scale = float(args_cli.residual_scale if args_cli.residual_scale is not None else ckpt_args.get("residual_scale", 0.1))
    residual_scale_min = float(
        args_cli.residual_scale_min if args_cli.residual_scale_min is not None else ckpt_args.get("residual_scale_min", 0.01)
    )
    residual_scale_max = float(
        args_cli.residual_scale_max if args_cli.residual_scale_max is not None else ckpt_args.get("residual_scale_max", 0.2)
    )
    skip_teleop = bool(
        args_cli.skip_teleop_device_setup
        if args_cli.skip_teleop_device_setup is not None
        else ckpt_args.get("skip_teleop_device_setup", False)
    )
    success_height_threshold = 0.20
    if args_cli.base_only and not args_cli.eval_base:
        print("[INFO] --base_only enabled: forcing --eval_base.")
        args_cli.eval_base = True

    device = torch.device(args_cli.device)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    sim_env, env, task_type = _create_env(args_cli.task, args_cli.device, skip_teleop=skip_teleop, seed=args_cli.seed)
    prompt = str(getattr(env.cfg, "task_description", "Lift the red cube up."))
    print(f"[INFO] Policy checkpoint: {policy_ckpt}")
    policy = _build_policy_client(env, task_type, policy_backend, str(policy_ckpt), action_horizon)
    low, high = _get_action_bounds(env, device)

    obs0, _ = sim_env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    act_dim = int(env.single_action_space.shape[0])

    def _build_eval_setup(source: str) -> tuple[ResidualActor, Callable[[dict], np.ndarray], int]:
        state_dim_local = int(_extract_state(obs0, source).shape[0])
        obs_dim_local = state_dim_local
        vit_encoder: EvalVisualObsEncoder | None = None
        if obs_encoder_mode == "vit":
            obs_encoder_state = ckpt.get("obs_encoder")
            if not isinstance(obs_encoder_state, dict):
                raise SystemExit(
                    "[EVAL-LOAD-MISMATCH] Checkpoint expects obs_encoder=vit but no obs_encoder weights were found."
                )
            has_token_projection = any(k.startswith("front_proj.") or k.startswith("wrist_proj.") for k in obs_encoder_state.keys())
            project_tokens = bool(ckpt_args.get("obs_vit_project_tokens", has_token_projection))
            if has_token_projection:
                project_tokens = True
            vit_encoder = EvalVisualObsEncoder(
                device=device,
                state_dim=state_dim_local,
                vit_image_size=int(ckpt_args.get("obs_vit_image_size", 84)),
                vit_depth=int(ckpt_args.get("obs_vit_depth", 1)),
                vit_embed_dim=int(ckpt_args.get("obs_vit_embed_dim", 128)),
                vit_num_heads=int(ckpt_args.get("obs_vit_num_heads", 4)),
                vit_patch_size=int(ckpt_args.get("obs_vit_patch_size", 8)),
                vit_proj_dim=int(ckpt_args.get("obs_vit_proj_dim", 128)),
                project_tokens=project_tokens,
            ).to(device)
            vit_encoder.load_state_dict(obs_encoder_state)
            vit_encoder.eval()
            obs_dim_local = int(vit_encoder.output_dim)

            def _extractor(obs_dict: dict) -> np.ndarray:
                front, wrist, _ = _extract_record_modalities(obs_dict)
                encoded = vit_encoder.encode_single_no_grad(front=front, wrist=wrist, joint_state=_extract_state(obs_dict, source))
                return encoded.detach().cpu().numpy().astype(np.float32).reshape(-1)

        else:

            def _extractor(obs_dict: dict) -> np.ndarray:
                return _extract_state(obs_dict, source)

        actor_local = ResidualActor(
            obs_dim=obs_dim_local,
            action_dim=act_dim,
            hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
            num_layers=int(ckpt_args.get("actor_num_layers", 2)),
            use_layer_norm=bool(ckpt_args.get("use_layer_norm", True)),
            use_state_gate=bool(ckpt_args.get("use_state_gate", False)),
            gate_hidden_dim=int(ckpt_args.get("gate_hidden_dim", 64)),
            gate_init_bias=float(ckpt_args.get("gate_init_bias", -2.0)),
        ).to(device)
        return actor_local, _extractor, obs_dim_local

    actor, state_extractor, state_dim = _build_eval_setup(state_source)
    try:
        actor.load_state_dict(ckpt["actor"])
    except RuntimeError as exc:
        # Keep strict behavior by default so incompatible setup checkpoints fail fast
        # and can be skipped by the batch selector.
        if args_cli.auto_state_source_fallback and args_cli.state_source is None:
            alt_state_source = "policy" if state_source == "record_joint6" else "record_joint6"
            alt_actor, alt_state_extractor, alt_state_dim = _build_eval_setup(alt_state_source)
            alt_actor.load_state_dict(ckpt["actor"])
            actor = alt_actor
            state_extractor = alt_state_extractor
            state_dim = alt_state_dim
            state_source = alt_state_source
            print(
                "[INFO] Actor load mismatch resolved by auto-switching state_source to "
                f"{state_source} (obs_encoder={obs_encoder_mode}, state_dim={state_dim})."
            )
        else:
            if "size mismatch" in str(exc):
                raise SystemExit(
                    "[EVAL-LOAD-MISMATCH] Incompatible checkpoint for this eval setup "
                    f"(obs_encoder={obs_encoder_mode}, state_source={state_source}, state_dim={state_dim})."
                ) from None
            raise exc

    print(f"[INFO] Loaded checkpoint: {args_cli.residual_checkpoint}")
    print(
        f"[INFO] Eval config: task={args_cli.task}, episodes={args_cli.num_episodes}, "
        f"obs_encoder={obs_encoder_mode}, state_source={state_source}, residual_state_dim={state_dim}, "
        f"obs_vit_project_tokens={bool(ckpt_args.get('obs_vit_project_tokens', False))}, "
        f"use_state_gate={bool(ckpt_args.get('use_state_gate', False))}, "
        f"residual_scale={residual_scale:.3f}, residual_scale_range=[{residual_scale_min:.3f},{residual_scale_max:.3f}], "
        f"skip_teleop={skip_teleop}, "
        f"eval_base={args_cli.eval_base}, success_height_threshold={success_height_threshold:.3f}, "
        f"record_video={args_cli.record_video}, video_dir={args_cli.video_dir}, video_fps={args_cli.video_fps}"
    )

    base: EvalSummary | None = None
    if args_cli.eval_base:
        base = _run_eval(
            sim_env=sim_env,
            env=env,
            task_type=task_type,
            policy=policy,
            actor=actor,
            device=device,
            low=low,
            high=high,
            residual_scale=residual_scale,
            residual_scale_min=residual_scale_min,
            residual_scale_max=residual_scale_max,
            state_source=state_source,
            state_extractor=state_extractor,
            prompt=prompt,
            use_residual=False,
            success_height_threshold=success_height_threshold,
            record_video=args_cli.record_video,
            video_dir=args_cli.video_dir,
            video_fps=args_cli.video_fps,
        )
    residual: EvalSummary | None = None
    if not args_cli.base_only:
        residual = _run_eval(
            sim_env=sim_env,
            env=env,
            task_type=task_type,
            policy=policy,
            actor=actor,
            device=device,
            low=low,
            high=high,
            residual_scale=residual_scale,
        residual_scale_min=residual_scale_min,
        residual_scale_max=residual_scale_max,
            state_source=state_source,
            state_extractor=state_extractor,
            prompt=prompt,
            use_residual=True,
            success_height_threshold=success_height_threshold,
            record_video=args_cli.record_video,
            video_dir=args_cli.video_dir,
            video_fps=args_cli.video_fps,
        )

    if base is not None and residual is not None:
        print(
            f"[RESULT] base_sr={base.sr:.3f} residual_sr={residual.sr:.3f} "
            f"base_ret={base.ret:.3f} residual_ret={residual.ret:.3f}"
        )
        print(
            f"[HEIGHT] base_mean_max={base.mean_max_height:.4f}m residual_mean_max={residual.mean_max_height:.4f}m "
            f"base_global_max={base.max_max_height:.4f}m residual_global_max={residual.max_max_height:.4f}m"
        )
    elif residual is not None:
        print(f"[RESULT] residual_sr={residual.sr:.3f} residual_ret={residual.ret:.3f}")
        print(
            f"[HEIGHT] residual_mean_max={residual.mean_max_height:.4f}m residual_global_max={residual.max_max_height:.4f}m"
        )
    elif base is not None:
        print(f"[RESULT] base_sr={base.sr:.3f} base_ret={base.ret:.3f}")
        print(f"[HEIGHT] base_mean_max={base.mean_max_height:.4f}m base_global_max={base.max_max_height:.4f}m")
    else:
        raise RuntimeError("Nothing evaluated: enable --eval_base or disable --base_only.")

    if args_cli.output_json:
        payload: dict[str, object] = {
            "checkpoint": str(args_cli.residual_checkpoint),
            "task": str(args_cli.task),
            "num_episodes": int(args_cli.num_episodes),
            "seed": int(args_cli.seed),
            "mode": "base_only" if args_cli.base_only else ("base_and_residual" if args_cli.eval_base else "residual_only"),
        }
        if residual is not None:
            payload["residual"] = {
                "success_rate": float(residual.sr),
                "successes": int(residual.successes),
                "episodes": int(residual.episodes),
                "mean_return": float(residual.ret),
                "mean_max_height": float(residual.mean_max_height),
                "max_max_height": float(residual.max_max_height),
            }
        if base is not None:
            payload["base"] = {
                "success_rate": float(base.sr),
                "successes": int(base.successes),
                "episodes": int(base.episodes),
                "mean_return": float(base.ret),
                "mean_max_height": float(base.mean_max_height),
                "max_max_height": float(base.max_max_height),
            }
        output_path = Path(args_cli.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[RESULT] wrote_json={output_path}")

    sim_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
