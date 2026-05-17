# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual off-policy TD3 finetuning on top of a frozen SmolVLA base policy.

This variant uses trainable in-agent visual encoders (ViT) on raw front/wrist images,
instead of relying on fixed env-side image feature extractors.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.app import AppLauncher

# isort: off
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args
# isort: on


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-RewardDense-Vision-Collect-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_steps", type=int, default=250_000)
parser.add_argument("--warmup_steps", type=int, default=5_000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--replay_size", type=int, default=20_000)
parser.add_argument(
    "--replay_image_size",
    type=int,
    default=96,
    help="Replay image storage resolution (images are resized before writing to buffer).",
)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--policy_delay", type=int, default=2)
parser.add_argument("--encoder_lr", type=float, default=1e-4)
parser.add_argument("--actor_lr", type=float, default=1e-4)
parser.add_argument("--critic_lr", type=float, default=1e-4)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--exploration_std", type=float, default=0.1)
parser.add_argument("--target_noise_std", type=float, default=0.1)
parser.add_argument("--target_noise_clip", type=float, default=0.3)
parser.add_argument("--residual_scale", type=float, default=0.2)
parser.add_argument("--residual_reg_weight", type=float, default=1e-3)
parser.add_argument("--eval_interval", type=int, default=10_000)
parser.add_argument("--eval_episodes", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=25_000)
parser.add_argument("--log_interval", type=int, default=1_000)
parser.add_argument("--step_hz", type=float, default=60.0)
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=8080)
parser.add_argument("--policy_timeout_ms", type=int, default=5000)
parser.add_argument("--policy_action_horizon", type=int, default=1)
parser.add_argument("--policy_language_instruction", type=str, default="Lift the red cube up.")
parser.add_argument("--policy_checkpoint_path", type=str, required=True)
parser.add_argument(
    "--policy_must_go",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use must-go policy mode by default. Pass --no-policy_must_go to disable.",
)
parser.add_argument("--policy_type", type=str, default="smolvla")
parser.add_argument(
    "--policy_backend",
    type=str,
    default="local",
    choices=["local", "service"],
    help="Backend for base policy inference. 'local' avoids policy-server RPC overhead.",
)
parser.add_argument("--log_dir", type=str, default="logs/residual_td3/lift_cube")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--vision_encoder",
    type=str,
    default="vit_b16",
    choices=["vit_b16"],
    help="Trainable visual encoder architecture used by residual actor/critic.",
)
parser.add_argument(
    "--encoder_pretrained",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Initialize visual encoder from ImageNet weights when available.",
)
parser.add_argument("--image_size", type=int, default=224, help="Encoder input resolution.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs != 1:
    raise ValueError("Residual TD3 currently supports --num_envs 1 only.")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv, mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.policy import LeRobotServicePolicyClient
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
from leisaac.tasks.lift_cube import mdp as lift_cube_mdp
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type

import leisaac  # noqa: F401


class RateLimiter:
    """Enforce wall-clock stepping and keep RTX sensors rendering."""

    def __init__(self, hz: float):
        if hz <= 0:
            raise ValueError("--step_hz must be positive.")
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        while self.last_time < time.time():
            self.last_time += self.sleep_duration


class ImageReplayBuffer:
    """Replay buffer storing raw camera observations and proprio state."""

    def __init__(self, capacity: int, image_shape: tuple[int, int, int], state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.front = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.wrist = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.base = np.zeros((capacity, action_dim), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_front = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.next_wrist = np.zeros((capacity, *image_shape), dtype=np.uint8)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_base = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        front: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        base: np.ndarray,
        act: np.ndarray,
        next_front: np.ndarray,
        next_wrist: np.ndarray,
        next_state: np.ndarray,
        next_base: np.ndarray,
        rew: float,
        done: bool,
    ) -> None:
        i = self.ptr
        self.front[i] = front
        self.wrist[i] = wrist
        self.state[i] = state
        self.base[i] = base
        self.act[i] = act
        self.next_front[i] = next_front
        self.next_wrist[i] = next_wrist
        self.next_state[i] = next_state
        self.next_base[i] = next_base
        self.rew[i, 0] = rew
        self.done[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "front": torch.from_numpy(self.front[idx]).to(device),
            "wrist": torch.from_numpy(self.wrist[idx]).to(device),
            "state": torch.from_numpy(self.state[idx]).to(device),
            "base": torch.from_numpy(self.base[idx]).to(device),
            "act": torch.from_numpy(self.act[idx]).to(device),
            "next_front": torch.from_numpy(self.next_front[idx]).to(device),
            "next_wrist": torch.from_numpy(self.next_wrist[idx]).to(device),
            "next_state": torch.from_numpy(self.next_state[idx]).to(device),
            "next_base": torch.from_numpy(self.next_base[idx]).to(device),
            "rew": torch.from_numpy(self.rew[idx]).to(device),
            "done": torch.from_numpy(self.done[idx]).to(device),
        }


class LocalSmolVLAPolicy:
    """In-process SmolVLA policy wrapper to avoid policy-server overhead."""

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
                "Local SmolVLA backend requires lerobot smolvla package. "
                "Install lerobot[smolvla] or use --policy_backend=service."
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
            "observation.images.front": {
                "type": "VISUAL",
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist": {
                "type": "VISUAL",
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
        }

    def reset(self) -> None:
        # Match service-policy episodic reset semantics as closely as possible.
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
        policy_obs = {
            "front": front,
            "wrist": wrist,
        }
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


def _to_replay_uint8(image_hwc: torch.Tensor, replay_size: int) -> np.ndarray:
    """Resize HWC image tensor to replay resolution and convert to uint8 numpy."""
    if image_hwc.dim() != 3:
        raise RuntimeError(f"Expected HWC image tensor, got shape={tuple(image_hwc.shape)}")
    img = image_hwc.unsqueeze(0).permute(0, 3, 1, 2).float()
    if img.shape[-2] != replay_size or img.shape[-1] != replay_size:
        img = F.interpolate(img, size=(replay_size, replay_size), mode="bilinear", align_corners=False)
    img = img.clamp(0.0, 255.0).to(dtype=torch.uint8)
    return img.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()


def _estimate_replay_image_bytes(capacity: int, image_shape: tuple[int, int, int]) -> int:
    # Stored image tensors per transition: front, wrist, next_front, next_wrist
    per_transition = int(np.prod(image_shape)) * 4
    return capacity * per_transition


class MultiCameraVitEncoder(nn.Module):
    """ViT encoder with one trainable backbone per camera."""

    def __init__(self, image_size: int, pretrained: bool):
        super().__init__()
        try:
            from torchvision.models import ViT_B_16_Weights, vit_b_16
        except ImportError as exc:
            raise RuntimeError("torchvision is required for ViT encoder in residual trainer.") from exc

        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.front_encoder = vit_b_16(weights=weights)
        self.wrist_encoder = vit_b_16(weights=weights)
        self.front_encoder.heads = nn.Identity()
        self.wrist_encoder.heads = nn.Identity()
        self.image_size = image_size
        self.out_dim = 768 * 2

    @staticmethod
    def _hwc_uint8_to_nchw_float(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        return x.permute(0, 3, 1, 2).contiguous()

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        img = self._hwc_uint8_to_nchw_float(img)
        if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
            img = F.interpolate(img, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return img

    def forward(self, front: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        front_in = self._prep(front)
        wrist_in = self._prep(wrist)
        front_feat = self.front_encoder(front_in)
        wrist_feat = self.wrist_encoder(wrist_in)
        return torch.cat([front_feat, wrist_feat], dim=-1)


class ResidualActor(nn.Module):
    """Actor over encoded visual+state feature."""

    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(feat))


class TwinQCritic(nn.Module):
    """Twin Q critics over encoded feature + action."""

    def __init__(self, feat_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = feat_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([feat, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([feat, action], dim=-1))


@dataclass
class TD3Stats:
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    residual_l2: float = 0.0
    delta_l2: float = 0.0


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _configure_lerobot_absolute_joint_actions(env_cfg, task_type: str) -> None:
    if task_type != "so101leader":
        return
    for action_name in ("arm_action", "gripper_action"):
        action_cfg = getattr(env_cfg.actions, action_name, None)
        if action_cfg is not None and hasattr(action_cfg, "use_default_offset"):
            action_cfg.use_default_offset = False
            action_cfg.scale = 1.0


def _ensure_collect_env_terminations(env_cfg) -> None:
    """Collect config disables success/time_out; restore them for RL episodes."""
    if "Collect" not in args_cli.task:
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


def _build_policy_client(env: ManagerBasedRLEnv, task_type: str):
    camera_infos = {k: sensor.image_shape for k, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)}
    if not camera_infos:
        raise RuntimeError("No camera sensors found for SmolVLA base policy.")
    if args_cli.policy_backend == "local":
        print("[INFO] Base policy backend: local (in-process SmolVLA, no policy-server RPC).")
        return LocalSmolVLAPolicy(
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            device=args_cli.device,
            actions_per_chunk=args_cli.policy_action_horizon,
        )
    print(f"[INFO] Base policy backend: service ({args_cli.policy_host}:{args_cli.policy_port}).")
    return LeRobotServicePolicyClient(
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        timeout_ms=args_cli.policy_timeout_ms,
        camera_infos=camera_infos,
        task_type=task_type,
        policy_type=args_cli.policy_type,
        pretrained_name_or_path=args_cli.policy_checkpoint_path,
        actions_per_chunk=args_cli.policy_action_horizon,
        force_must_go=args_cli.policy_must_go,
        device=args_cli.device,
    )


def _create_env(task: str, device: str) -> tuple[gym.Env, ManagerBasedRLEnv, str]:
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    task_type = get_task_type(task)
    env_cfg.use_teleop_device(task_type)
    _configure_lerobot_absolute_joint_actions(env_cfg, task_type)
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    _ensure_collect_env_terminations(env_cfg)
    sim_env = gym.make(task, cfg=env_cfg, render_mode=None)
    return sim_env, sim_env.unwrapped, task_type


def _build_base_obs(obs_dict: dict, task_description: str) -> dict:
    if "record" not in obs_dict:
        raise RuntimeError(
            "Residual SmolVLA training expects a 'record' observation group with front/wrist images and joint_pos_abs. "
            "Use a *-Collect task config."
        )
    rec = obs_dict["record"]
    required = ("front", "wrist", "joint_pos_abs")
    for key in required:
        if key not in rec:
            raise RuntimeError(f"Missing '{key}' in record observations for SmolVLA base policy.")
    return {
        "front": rec["front"],
        "wrist": rec["wrist"],
        "joint_pos": rec["joint_pos_abs"],
        "task_description": task_description,
    }


def _extract_residual_obs(obs_dict: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rec = obs_dict["record"]
    front = rec["front"][0]
    wrist = rec["wrist"][0]
    state = rec["joint_pos_abs"][0].float()
    return front, wrist, state


def _build_feature(encoder: MultiCameraVitEncoder, front: torch.Tensor, wrist: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    vis = encoder(front.unsqueeze(0), wrist.unsqueeze(0)).squeeze(0)
    return torch.cat([vis, state], dim=-1)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = residual_scale * residual_unit
    final_action = torch.clamp(base_action + delta, low, high)
    return final_action, delta


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t_p, s_p in zip(target.parameters(), source.parameters()):
        t_p.data.mul_(1.0 - tau).add_(tau * s_p.data)


def _evaluate(
    encoder: MultiCameraVitEncoder,
    actor: ResidualActor,
    device: torch.device,
    residual_scale: float,
    low: torch.Tensor,
    high: torch.Tensor,
    base_only: bool,
    episodes: int,
) -> tuple[float, float]:
    sim_env, env, task_type = _create_env(args_cli.task, args_cli.device)
    policy = _build_policy_client(env, task_type)
    prompt = str(getattr(env.cfg, "task_description", args_cli.policy_language_instruction))
    success_count = 0
    returns = []
    encoder.eval()
    actor.eval()
    for _ in range(episodes):
        obs_dict, _ = sim_env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0
        while simulation_app.is_running():
            front, wrist, state = _extract_residual_obs(obs_dict)
            front = front.to(device)
            wrist = wrist.to(device)
            state = state.to(device)
            base_obs = _build_base_obs(obs_dict, prompt)
            base_chunk = policy.get_action(base_obs).to(device)
            base_action = base_chunk[0, 0, :].float()
            if base_only:
                action = base_action
            else:
                with torch.no_grad():
                    feat = _build_feature(encoder, front, wrist, state)
                    residual_unit = actor(feat.unsqueeze(0)).squeeze(0)
                action, _ = _compose_action(base_action, residual_unit, residual_scale, low, high)
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, task_type)
            obs_dict, reward, terminated, truncated, _ = sim_env.step(action.unsqueeze(0))
            ep_return += float(reward[0].detach().cpu().item())
            done = bool((terminated[0] | truncated[0]).detach().cpu().item())
            if done:
                success = bool(env.reset_terminated[0].detach().cpu().item())
                if success:
                    success_count += 1
                break
        returns.append(ep_return)
    sim_env.close()
    encoder.train()
    actor.train()
    return success_count / max(episodes, 1), float(np.mean(returns) if returns else 0.0)


def main() -> None:
    os.makedirs(args_cli.log_dir, exist_ok=True)
    print(f"[INFO] Requested base policy backend: {args_cli.policy_backend}")
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    wall_start_time = time.time()

    device = torch.device(args_cli.device)
    sim_env, env, task_type = _create_env(args_cli.task, args_cli.device)
    prompt = str(getattr(env.cfg, "task_description", args_cli.policy_language_instruction))
    base_policy = _build_policy_client(env, task_type)
    low, high = _get_action_bounds(env, device)

    obs_dict, _ = sim_env.reset()
    if hasattr(base_policy, "reset"):
        base_policy.reset()
    front0, wrist0, state0 = _extract_residual_obs(obs_dict)
    front0 = front0.to(device)
    wrist0 = wrist0.to(device)
    state0 = state0.to(device)
    base_obs0 = _build_base_obs(obs_dict, prompt)
    base_action0 = base_policy.get_action(base_obs0).to(device)[0, 0, :].float()

    encoder = MultiCameraVitEncoder(image_size=args_cli.image_size, pretrained=args_cli.encoder_pretrained).to(device)
    encoder_target = MultiCameraVitEncoder(image_size=args_cli.image_size, pretrained=args_cli.encoder_pretrained).to(device)
    encoder_target.load_state_dict(encoder.state_dict())

    with torch.no_grad():
        feat0 = _build_feature(encoder, front0, wrist0, state0)
    feat_dim = int(feat0.numel())
    state_dim = int(state0.numel())
    image_shape = (args_cli.replay_image_size, args_cli.replay_image_size, int(front0.shape[-1]))
    act_dim = int(base_action0.numel())
    replay_img_bytes = _estimate_replay_image_bytes(args_cli.replay_size, image_shape)
    replay_img_gib = replay_img_bytes / (1024**3)
    if replay_img_gib > 24.0:
        raise RuntimeError(
            f"Replay image storage estimate is too large ({replay_img_gib:.1f} GiB). "
            "Reduce --replay_size or --replay_image_size."
        )
    print(
        f"[INFO] Replay image storage estimate: {replay_img_gib:.2f} GiB "
        f"(size={args_cli.replay_size}, image={image_shape[0]}x{image_shape[1]})."
    )
    replay = ImageReplayBuffer(args_cli.replay_size, image_shape, state_dim, act_dim)

    actor = ResidualActor(feat_dim, act_dim, args_cli.hidden_dim).to(device)
    actor_target = ResidualActor(feat_dim, act_dim, args_cli.hidden_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = TwinQCritic(feat_dim, act_dim, args_cli.hidden_dim).to(device)
    critic_target = TwinQCritic(feat_dim, act_dim, args_cli.hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=args_cli.encoder_lr)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args_cli.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args_cli.critic_lr)
    rate_limiter = RateLimiter(args_cli.step_hz)

    front = front0
    wrist = wrist0
    state = state0
    feat = feat0
    base_action = base_action0
    episode_return = 0.0
    episode_count = 0
    recent_returns = deque(maxlen=20)
    recent_success = deque(maxlen=20)
    update_count = 0
    last_stats = TD3Stats()

    if hasattr(env, "reward_manager") and len(getattr(env.reward_manager, "active_terms", [])) == 0:
        raise RuntimeError(
            f"Task {args_cli.task} has zero active reward terms. "
            "Use a dense-reward task such as LeIsaac-SO101-LiftCube-RewardDense-Vision-Collect-v0."
        )

    for step in range(1, args_cli.total_steps + 1):
        if step <= args_cli.warmup_steps:
            residual_unit = torch.empty(act_dim, device=device).uniform_(-1.0, 1.0)
        else:
            with torch.no_grad():
                residual_unit = actor(feat.unsqueeze(0)).squeeze(0)
                residual_unit = torch.clamp(
                    residual_unit + torch.randn_like(residual_unit) * args_cli.exploration_std, -1.0, 1.0
                )
        action, delta = _compose_action(base_action, residual_unit, args_cli.residual_scale, low, high)

        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, task_type)
        next_obs_dict, reward, terminated, truncated, _ = sim_env.step(action.unsqueeze(0))
        rew = float(reward[0].detach().cpu().item())
        done = bool((terminated[0] | truncated[0]).detach().cpu().item())
        success = bool(env.reset_terminated[0].detach().cpu().item()) if done else False

        next_front, next_wrist, next_state = _extract_residual_obs(next_obs_dict)
        next_front = next_front.to(device)
        next_wrist = next_wrist.to(device)
        next_state = next_state.to(device)
        with torch.no_grad():
            next_feat = _build_feature(encoder, next_front, next_wrist, next_state)

        if done:
            next_base_action = torch.zeros_like(base_action)
        else:
            next_base_obs = _build_base_obs(next_obs_dict, prompt)
            next_base_action = base_policy.get_action(next_base_obs).to(device)[0, 0, :].float()

        replay.add(
            front=_to_replay_uint8(front, args_cli.replay_image_size),
            wrist=_to_replay_uint8(wrist, args_cli.replay_image_size),
            state=state.detach().cpu().numpy(),
            base=base_action.detach().cpu().numpy(),
            act=action.detach().cpu().numpy(),
            next_front=_to_replay_uint8(next_front, args_cli.replay_image_size),
            next_wrist=_to_replay_uint8(next_wrist, args_cli.replay_image_size),
            next_state=next_state.detach().cpu().numpy(),
            next_base=next_base_action.detach().cpu().numpy(),
            rew=rew,
            done=done,
        )

        episode_return += rew
        if done:
            episode_count += 1
            recent_returns.append(episode_return)
            recent_success.append(float(success))
            episode_return = 0.0
            obs_dict = next_obs_dict
            if hasattr(base_policy, "reset"):
                base_policy.reset()
            front, wrist, state = _extract_residual_obs(obs_dict)
            front = front.to(device)
            wrist = wrist.to(device)
            state = state.to(device)
            with torch.no_grad():
                feat = _build_feature(encoder, front, wrist, state)
            base_obs = _build_base_obs(obs_dict, prompt)
            base_action = base_policy.get_action(base_obs).to(device)[0, 0, :].float()
        else:
            front, wrist, state = next_front, next_wrist, next_state
            feat = next_feat
            base_action = next_base_action

        if replay.size >= args_cli.batch_size:
            batch = replay.sample(args_cli.batch_size, device)

            feat_batch = torch.cat(
                [encoder(batch["front"], batch["wrist"]), batch["state"]],
                dim=-1,
            )
            with torch.no_grad():
                next_feat_batch = torch.cat(
                    [encoder_target(batch["next_front"], batch["next_wrist"]), batch["next_state"]],
                    dim=-1,
                )
                target_residual = actor_target(next_feat_batch)
                target_noise = torch.clamp(
                    torch.randn_like(target_residual) * args_cli.target_noise_std,
                    -args_cli.target_noise_clip,
                    args_cli.target_noise_clip,
                )
                target_residual = torch.clamp(target_residual + target_noise, -1.0, 1.0)
                target_action, _ = _compose_action(
                    batch["next_base"], target_residual, args_cli.residual_scale, low, high
                )
                q1_t, q2_t = critic_target(next_feat_batch, target_action)
                target_q = batch["rew"] + (1.0 - batch["done"]) * args_cli.gamma * torch.min(q1_t, q2_t)

            q1, q2 = critic(feat_batch, batch["act"])
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            encoder_opt.zero_grad(set_to_none=True)
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            encoder_opt.step()
            critic_opt.step()

            actor_loss = torch.tensor(0.0, device=device)
            if update_count % args_cli.policy_delay == 0:
                with torch.no_grad():
                    feat_actor = torch.cat([encoder(batch["front"], batch["wrist"]), batch["state"]], dim=-1)
                actor_residual = actor(feat_actor)
                actor_action, actor_delta = _compose_action(
                    batch["base"], actor_residual, args_cli.residual_scale, low, high
                )
                actor_loss = -critic.q1_only(feat_actor, actor_action).mean()
                actor_loss = actor_loss + args_cli.residual_reg_weight * (actor_delta.pow(2).mean())
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                _soft_update(actor_target, actor, args_cli.tau)
                _soft_update(critic_target, critic, args_cli.tau)
                _soft_update(encoder_target, encoder, args_cli.tau)

            update_count += 1
            last_stats = TD3Stats(
                critic_loss=float(critic_loss.detach().cpu().item()),
                actor_loss=float(actor_loss.detach().cpu().item()),
                residual_l2=float(residual_unit.norm(p=2).detach().cpu().item()),
                delta_l2=float(delta.norm(p=2).detach().cpu().item()),
            )

        if step % args_cli.log_interval == 0:
            mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            mean_success = float(np.mean(recent_success)) if recent_success else 0.0
            elapsed_s = time.time() - wall_start_time
            steps_per_s = step / max(elapsed_s, 1e-6)
            remaining_steps = max(args_cli.total_steps - step, 0)
            eta_s = remaining_steps / max(steps_per_s, 1e-6)
            print(
                f"[TRAIN] step={step} episodes={episode_count} replay={replay.size} "
                f"elapsed={_format_seconds(elapsed_s)} eta={_format_seconds(eta_s)} steps_per_s={steps_per_s:.2f} "
                f"return20={mean_return:.3f} success20={mean_success:.3f} "
                f"critic_loss={last_stats.critic_loss:.4f} actor_loss={last_stats.actor_loss:.4f} "
                f"residual_l2={last_stats.residual_l2:.4f} delta_l2={last_stats.delta_l2:.4f}"
            )

        if step % args_cli.eval_interval == 0:
            base_sr, base_ret = _evaluate(
                encoder=encoder,
                actor=actor,
                device=device,
                residual_scale=args_cli.residual_scale,
                low=low,
                high=high,
                base_only=True,
                episodes=args_cli.eval_episodes,
            )
            residual_sr, residual_ret = _evaluate(
                encoder=encoder,
                actor=actor,
                device=device,
                residual_scale=args_cli.residual_scale,
                low=low,
                high=high,
                base_only=False,
                episodes=args_cli.eval_episodes,
            )
            print(
                f"[EVAL] step={step} base_sr={base_sr:.3f} residual_sr={residual_sr:.3f} "
                f"base_ret={base_ret:.3f} residual_ret={residual_ret:.3f}"
            )

        if step % args_cli.save_interval == 0:
            ckpt = {
                "step": step,
                "encoder": encoder.state_dict(),
                "encoder_target": encoder_target.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_target": actor_target.state_dict(),
                "critic_target": critic_target.state_dict(),
                "encoder_opt": encoder_opt.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "args": vars(args_cli),
            }
            ckpt_path = os.path.join(args_cli.log_dir, f"residual_td3_step_{step:08d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"[CHECKPOINT] Saved {ckpt_path}")

        rate_limiter.sleep(env)

    final_path = os.path.join(args_cli.log_dir, "residual_td3_final.pt")
    torch.save(
        {
            "step": args_cli.total_steps,
            "encoder": encoder.state_dict(),
            "encoder_target": encoder_target.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_target": actor_target.state_dict(),
            "critic_target": critic_target.state_dict(),
            "encoder_opt": encoder_opt.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "args": vars(args_cli),
        },
        final_path,
    )
    print(f"[DONE] Saved final checkpoint: {final_path}")

    sim_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
