# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Residual off-policy TD3 with privileged MLP-style observations.

This ablation keeps the frozen SmolVLA base policy but trains the residual actor/critic
on the privileged flat policy vector (includes cube-relative state like MLP setup).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from isaaclab.app import AppLauncher

# isort: off
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args
# isort: on


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--recipe",
    type=str,
    default="resfit_vit",
    choices=["resfit_vit", "privileged_cube_pose", "guidance"],
    help=(
        "Training recipe preset. "
        "privileged_cube_pose uses full policy-state (incl. cube pose) without vision encoder. "
        "guidance enforces sparse + trajectory-guidance reward."
    ),
)
parser.add_argument(
    "--reward_mode",
    type=str,
    default="dense",
    choices=["dense", "sparse"],
    help="Reward used for residual updates: dense env reward or sparse terminal success reward.",
)
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-RewardDense-Collect-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_steps", type=int, default=250_000)
parser.add_argument("--warmup_steps", type=int, default=10_000)
parser.add_argument("--learning_starts", type=int, default=10_000)
parser.add_argument(
    "--critic_warmup_steps",
    type=int,
    default=10_000,
    help="Number of critic-only updates before enabling actor updates (ResFiT-style).",
)
parser.add_argument(
    "--warmup_noise_scale",
    type=float,
    default=0.05,
    help="Uniform action noise scale during warmup; applied as base_action + U[-s, s].",
)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--replay_size", type=int, default=300_000)
parser.add_argument(
    "--num_updates_per_iteration",
    type=int,
    default=4,
    help="Number of gradient updates per environment step once replay is ready (ResFiT/RLPD-style).",
)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--policy_delay", type=int, default=2)
parser.add_argument("--actor_lr", type=float, default=1e-6)
parser.add_argument("--critic_lr", type=float, default=1e-4)
parser.add_argument(
    "--actor_lr_warmup_updates",
    type=int,
    default=0,
    help="Linearly warm up actor LR over this many actor-update attempts (0 disables).",
)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--exploration_std", type=float, default=0.05)
parser.add_argument(
    "--exploration_std_min",
    type=float,
    default=None,
    help="Final exploration std after linear decay (None keeps fixed --exploration_std).",
)
parser.add_argument(
    "--exploration_std_decay_steps",
    type=int,
    default=0,
    help="Linearly decay exploration std from --exploration_std to --exploration_std_min over this many env steps.",
)
parser.add_argument("--target_noise_std", type=float, default=0.05)
parser.add_argument("--target_noise_clip", type=float, default=0.05)
parser.add_argument(
    "--target_action_noise",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Add TD3 target policy smoothing noise on residual target actions.",
)
parser.add_argument(
    "--use_base_policy_for_warmup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Warmup exploration mode: base_action + noise (True) or pure random action (False).",
)
parser.add_argument("--residual_scale", type=float, default=0.05)
parser.add_argument(
    "--residual_scale_min",
    type=float,
    default=0.01,
    help="Lower bound for applied residual action scale.",
)
parser.add_argument(
    "--residual_scale_max",
    type=float,
    default=0.2,
    help="Upper bound for applied residual action scale.",
)
parser.add_argument(
    "--progressive_clipping_steps",
    type=int,
    default=0,
    help="Linearly ramp residual scale from 0 to --residual_scale over this many env steps (0 disables).",
)
parser.add_argument(
    "--no_residual",
    action="store_true",
    default=False,
    help="Disable residual action contribution (runs pure base policy for sanity checks).",
)
parser.add_argument("--residual_reg_weight", type=float, default=1e-3)
parser.add_argument(
    "--use_state_gate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable state-dependent gate g(s) in [0,1] for residual scaling.",
)
parser.add_argument(
    "--gate_hidden_dim",
    type=int,
    default=64,
    help="Hidden dimension for state-dependent residual gate MLP.",
)
parser.add_argument(
    "--gate_init_bias",
    type=float,
    default=-2.0,
    help="Initial bias for gate head logits (negative keeps early gate conservative).",
)
parser.add_argument(
    "--actor_last_layer_init_scale",
    type=float,
    default=0.0,
    help="Residual actor last-layer initialization scale (0.0 starts near base policy behavior).",
)
parser.add_argument(
    "--actor_last_layer_init_distribution",
    type=str,
    choices=["normal", "orthogonal", "xavier_uniform"],
    default="normal",
    help="Initialization distribution for residual actor last layer.",
)
parser.add_argument("--n_step", type=int, default=5, help="n-step return horizon for TD targets.")
parser.add_argument(
    "--clip_q_target_to_reward_range",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Clip TD target Q to [0, 1] when using sparse reward mode.",
)
parser.add_argument(
    "--obs_encoder",
    type=str,
    default="vit",
    choices=["state", "vit"],
    help="Observation encoder for residual RL. 'vit' uses a shallow ResFiT-style visual encoder.",
)
parser.add_argument(
    "--state_source",
    type=str,
    default="record_joint6",
    choices=["record_joint6", "policy"],
    help="Source for state vector used by residual learner. 'policy' uses full env policy observation (e.g., 21D cube-pose privileged state).",
)
parser.add_argument(
    "--obs_vit_proj_dim",
    type=int,
    default=128,
    help="Per-camera pooled feature dimension when token projection is enabled (state + 2*proj_dim total).",
)
parser.add_argument(
    "--obs_vit_project_tokens",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Project pooled ViT tokens to compact per-camera features for faster training.",
)
parser.add_argument(
    "--obs_vit_image_size",
    type=int,
    default=84,
    help="Input image size for shallow ViT encoder.",
)
parser.add_argument("--obs_vit_depth", type=int, default=1, help="Shallow ViT depth (ResFiT default: 1).")
parser.add_argument("--obs_vit_embed_dim", type=int, default=128, help="Shallow ViT embed dim (ResFiT default: 128).")
parser.add_argument("--obs_vit_num_heads", type=int, default=4, help="Shallow ViT attention heads (ResFiT default: 4).")
parser.add_argument(
    "--obs_vit_patch_size",
    type=int,
    default=8,
    help="Patch embed kernel size for shallow ViT (ResFiT default: 8).",
)
parser.add_argument(
    "--obs_train_encoder",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Train shallow visual encoder during RL updates (ResFiT-style).",
)
parser.add_argument(
    "--obs_random_shift_pad",
    type=int,
    default=4,
    help="DrQ-style random shift padding applied to images during updates.",
)
parser.add_argument(
    "--obs_use_drq_aug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable DrQ-style random shift augmentation in update batches.",
)
parser.add_argument("--actor_num_layers", type=int, default=2, help="Number of hidden layers in residual actor MLP.")
parser.add_argument("--critic_num_layers", type=int, default=2, help="Number of hidden layers per Q head.")
parser.add_argument(
    "--use_layer_norm",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use LayerNorm in actor/critic MLP heads (ResFiT/RLPD-style).",
)
parser.add_argument("--num_q", type=int, default=10, help="Number of Q heads in critic ensemble.")
parser.add_argument("--min_q_heads", type=int, default=2, help="How many random Q heads to min over for TD target.")
parser.add_argument(
    "--policy_gradient_type",
    type=str,
    default="ensemble_mean",
    choices=["ensemble_mean", "min_random_pair", "q1"],
    help="How actor objective aggregates ensemble Q values.",
)
parser.add_argument(
    "--offline_source",
    type=str,
    default="none",
    choices=["none", "hf", "local_npz"],
    help="Offline replay source. 'hf' expects a Hugging Face dataset in LeRobot-like format.",
)
parser.add_argument("--offline_hf_dataset", type=str, default=None)
parser.add_argument("--offline_hf_split", type=str, default="train")
parser.add_argument("--offline_hf_config", type=str, default=None)
parser.add_argument(
    "--offline_use_base_policy_for_base_actions",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Offline replay base-action mode: True uses base-policy-inferred base actions; "
        "False reuses dataset GT actions as base actions."
    ),
)
parser.add_argument(
    "--offline_state_normalize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Normalize residual state vectors using offline dataset stats when available.",
)
parser.add_argument(
    "--offline_state_min_std",
    type=float,
    default=1e-3,
    help="Lower bound on state std when building offline-derived normalizer.",
)
parser.add_argument(
    "--offline_hf_cache_dir",
    type=str,
    default="logs/residual_td3/hf_cache",
    help="Directory for cached preprocessed HF transitions (includes base/next_base actions).",
)
parser.add_argument(
    "--offline_hf_rebuild_cache",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Force rebuilding HF cache even if a matching cache file exists.",
)
parser.add_argument("--offline_local_npz_dir", type=str, default=None)
parser.add_argument("--offline_max_samples", type=int, default=300_000)
parser.add_argument(
    "--offline_mix_ratio",
    type=float,
    default=0.5,
    help="Fraction of each training batch drawn from offline replay.",
)
parser.add_argument(
    "--sampling_strategy",
    type=str,
    choices=["uniform", "prioritized_replay"],
    default="uniform",
    help="Replay sampling strategy for online/offline buffers.",
)
parser.add_argument(
    "--priority_alpha",
    type=float,
    default=0.6,
    help="PER alpha exponent (used when --sampling_strategy=prioritized_replay).",
)
parser.add_argument(
    "--priority_beta_start",
    type=float,
    default=0.4,
    help="Initial PER beta (importance sampling correction).",
)
parser.add_argument(
    "--priority_beta_end",
    type=float,
    default=1.0,
    help="Final PER beta at end of training.",
)
parser.add_argument(
    "--priority_eps",
    type=float,
    default=1e-6,
    help="Small epsilon added to priorities for numerical stability.",
)
parser.add_argument("--eval_interval", type=int, default=10_000)
parser.add_argument("--eval_episodes", type=int, default=10)
parser.add_argument(
    "--eval_first",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run one evaluation pass before training starts (only when in-loop eval is enabled).",
)
parser.add_argument(
    "--eval_base_during_training",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run base-policy evaluation at eval intervals (default: False to reduce overhead).",
)
parser.add_argument("--save_interval", type=int, default=25_000)
parser.add_argument("--log_interval", type=int, default=1_000)
parser.add_argument(
    "--heartbeat_interval_s",
    type=float,
    default=0,
    help="Time-based progress heartbeat interval in seconds (prints regardless of log_interval).",
)
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
parser.add_argument(
    "--skip_teleop_device_setup",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Whether to skip env_cfg.use_teleop_device(...) in this trainer. "
        "Default is False (teleop setup enabled). Pass --skip_teleop_device_setup to disable."
    ),
)
parser.add_argument("--policy_type", type=str, default="smolvla")
parser.add_argument(
    "--policy_backend",
    type=str,
    default="local",
    choices=["local", "service"],
    help="Backend for base policy inference. 'local' avoids policy-server RPC overhead.",
)
_DEFAULT_LOG_DIR = "logs/residual_td3/lift_cube_privileged"
parser.add_argument(
    "--log_dir",
    type=str,
    default=_DEFAULT_LOG_DIR,
    help=(
        "Checkpoint/log directory. If left at default, trainer auto-expands to a unique per-run path "
        "(includes recipe and optional timestamp) to avoid overwrites."
    ),
)
parser.add_argument(
    "--log_dir_use_timestamp",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="When auto-expanding default --log_dir, append timestamp for unique run folders.",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--traj_guidance_db",
    type=str,
    default=None,
    help="Directory containing trajectory_*.npz reference trajectories for guidance/reset.",
)
parser.add_argument(
    "--traj_guidance_sampling",
    type=str,
    choices=["round_robin", "random"],
    default="round_robin",
    help="How to select a reference trajectory for each new episode.",
)
parser.add_argument("--traj_guidance_seed", type=int, default=7)
parser.add_argument(
    "--traj_guidance_lambda",
    type=float,
    default=0.5,
    help="Overall scale multiplier for trajectory guidance reward added to task reward.",
)
parser.add_argument("--traj_guidance_progress_weight", type=float, default=1.0)
parser.add_argument("--traj_guidance_xy_weight", type=float, default=0.25)
parser.add_argument(
    "--traj_guidance_gripper_weight",
    type=float,
    default=0.25,
    help="Weight for gripper-state alignment term in trajectory guidance reward.",
)
parser.add_argument("--traj_guidance_xy_scale", type=float, default=0.08)
parser.add_argument("--traj_guidance_gripper_scale", type=float, default=15.0)
parser.add_argument("--traj_guidance_progress_power", type=float, default=1.0)
parser.add_argument(
    "--traj_guidance_reset_from_reference",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Reset cube pose to sampled reference initial pose each episode.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.recipe == "privileged_cube_pose":
    # Lightweight sanity recipe to verify residual learning with privileged cube pose.
    args_cli.obs_encoder = "state"
    args_cli.state_source = "policy"
    args_cli.offline_source = "none"
    args_cli.offline_mix_ratio = 0.0
    args_cli.obs_train_encoder = False
    args_cli.obs_use_drq_aug = False

if args_cli.recipe == "guidance":
    # Guidance recipe is explicitly sparse + trajectory guidance.
    args_cli.obs_encoder = "state"
    args_cli.state_source = "policy"
    args_cli.offline_source = "none"
    args_cli.offline_mix_ratio = 0.0
    args_cli.obs_train_encoder = False
    args_cli.obs_use_drq_aug = False
    args_cli.reward_mode = "sparse"
    if not args_cli.traj_guidance_db:
        raise ValueError("--traj_guidance_db is required when --recipe=guidance.")
    if "GuidanceSparse" not in str(args_cli.task):
        print(
            "[INFO] Guidance recipe: overriding --task to "
            "'LeIsaac-SO101-LiftCube-GuidanceSparse-Collect-v0'."
        )
        args_cli.task = "LeIsaac-SO101-LiftCube-GuidanceSparse-Collect-v0"

# Any run using trajectory guidance should use sparse task reward.
if args_cli.traj_guidance_db and args_cli.reward_mode != "sparse":
    print(
        "[INFO] Trajectory guidance enabled: overriding --reward_mode to 'sparse' "
        "(sparse+guidance reward)."
    )
    args_cli.reward_mode = "sparse"

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv, mdp as isaac_mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.policy import LeRobotServicePolicyClient
from leisaac.tasks.lift_cube import mdp as lift_cube_mdp
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac

import leisaac  # noqa: F401


class RateLimiter:
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


class ReplayBuffer:
    def __init__(
        self,
        action_dim: int,
        capacity: int,
        image_size: int,
        state_dim: int,
        store_images: bool = True,
        sampling_strategy: str = "uniform",
        priority_alpha: float = 0.6,
        priority_eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.state_dim = int(state_dim)
        self.store_images = bool(store_images)
        self.sampling_strategy = str(sampling_strategy)
        self.priority_alpha = float(priority_alpha)
        self.priority_eps = float(priority_eps)
        self.ptr = 0
        self.size = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.obs_joint = np.zeros((capacity, self.state_dim), dtype=np.float32)
        replay_image_size = int(image_size) if self.store_images else 1
        self.obs_front = np.zeros((capacity, replay_image_size, replay_image_size, 3), dtype=np.uint8)
        self.obs_wrist = np.zeros((capacity, replay_image_size, replay_image_size, 3), dtype=np.uint8)
        self.base = np.zeros((capacity, action_dim), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_joint = np.zeros((capacity, self.state_dim), dtype=np.float32)
        self.next_front = np.zeros((capacity, replay_image_size, replay_image_size, 3), dtype=np.uint8)
        self.next_wrist = np.zeros((capacity, replay_image_size, replay_image_size, 3), dtype=np.uint8)
        self.next_base = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.discount = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs_joint: np.ndarray,
        obs_front: np.ndarray,
        obs_wrist: np.ndarray,
        base: np.ndarray,
        act: np.ndarray,
        next_joint: np.ndarray,
        next_front: np.ndarray,
        next_wrist: np.ndarray,
        next_base: np.ndarray,
        rew: float,
        done: bool,
        discount: float,
    ) -> None:
        i = self.ptr
        self.obs_joint[i] = np.asarray(obs_joint, dtype=np.float32).reshape(-1)[: self.state_dim]
        if self.store_images:
            self.obs_front[i] = obs_front
            self.obs_wrist[i] = obs_wrist
        self.base[i] = base
        self.act[i] = act
        self.next_joint[i] = np.asarray(next_joint, dtype=np.float32).reshape(-1)[: self.state_dim]
        if self.store_images:
            self.next_front[i] = next_front
            self.next_wrist[i] = next_wrist
        self.next_base[i] = next_base
        self.rew[i, 0] = rew
        self.done[i, 0] = float(done)
        self.discount[i, 0] = float(discount)
        self.priorities[i] = float(self.max_priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_indices(
        self,
        idx: np.ndarray,
        device: torch.device,
        is_weight: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = {
            "obs_joint": torch.from_numpy(self.obs_joint[idx]).to(device),
            "obs_front": torch.from_numpy(self.obs_front[idx]).to(device),
            "obs_wrist": torch.from_numpy(self.obs_wrist[idx]).to(device),
            "base": torch.from_numpy(self.base[idx]).to(device),
            "act": torch.from_numpy(self.act[idx]).to(device),
            "next_joint": torch.from_numpy(self.next_joint[idx]).to(device),
            "next_front": torch.from_numpy(self.next_front[idx]).to(device),
            "next_wrist": torch.from_numpy(self.next_wrist[idx]).to(device),
            "next_base": torch.from_numpy(self.next_base[idx]).to(device),
            "rew": torch.from_numpy(self.rew[idx]).to(device),
            "done": torch.from_numpy(self.done[idx]).to(device),
            "discount": torch.from_numpy(self.discount[idx]).to(device),
        }
        if is_weight is None:
            is_weight = np.ones((len(idx),), dtype=np.float32)
        batch["is_weight"] = torch.from_numpy(np.asarray(is_weight, dtype=np.float32)[:, None]).to(device)
        return batch

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch, _ = self.sample_with_indices(batch_size, device)
        return batch

    def sample_with_indices(
        self,
        batch_size: int,
        device: torch.device,
        beta: float = 1.0,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        if self.size <= 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if self.sampling_strategy == "prioritized_replay":
            priorities = self.priorities[: self.size]
            if priorities.sum() <= 0.0:
                probs = np.full((self.size,), 1.0 / self.size, dtype=np.float32)
            else:
                scaled = np.power(priorities + self.priority_eps, self.priority_alpha)
                probs = scaled / np.maximum(scaled.sum(), 1e-12)
            idx = np.random.choice(self.size, size=batch_size, replace=True, p=probs)
            weights = np.power(self.size * probs[idx], -float(beta))
            weights = weights / np.maximum(weights.max(), 1e-12)
            batch = self.sample_indices(idx, device, is_weight=weights.astype(np.float32))
            return batch, idx.astype(np.int64)
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = self.sample_indices(idx, device)
        return batch, idx.astype(np.int64)

    def update_priorities(self, idx: np.ndarray, priorities: np.ndarray) -> None:
        if self.sampling_strategy != "prioritized_replay":
            return
        if idx is None or len(idx) == 0:
            return
        idx_np = np.asarray(idx, dtype=np.int64).reshape(-1)
        pri_np = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if pri_np.shape[0] != idx_np.shape[0]:
            return
        pri_np = np.maximum(pri_np + self.priority_eps, self.priority_eps)
        self.priorities[idx_np] = pri_np
        if pri_np.size > 0:
            self.max_priority = float(max(self.max_priority, float(pri_np.max())))


class RandomShiftsAug(nn.Module):
    """DrQ-style random shift augmentation."""

    def __init__(self, pad: int = 4):
        super().__init__()
        self.pad = int(pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad <= 0:
            return x
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW images, got shape={tuple(x.shape)}")
        n, _, h, w = x.shape
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        crop_max = 2 * self.pad + 1
        top = torch.randint(0, crop_max, (n,), device=x.device)
        left = torch.randint(0, crop_max, (n,), device=x.device)
        rows = top[:, None] + torch.arange(h, device=x.device)[None, :]
        cols = left[:, None] + torch.arange(w, device=x.device)[None, :]
        batch = torch.arange(n, device=x.device)[:, None, None]
        # Gather each sample's (h, w) crop directly from padded image.
        out = x[batch, :, rows[:, :, None], cols[:, None, :]]
        return out.permute(0, 3, 1, 2).contiguous()


class VisualObsEncoder(nn.Module):
    """Shared shallow ViT encoder (ResFiT-style) for both online and offline batches."""

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
            return y.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, N, C]

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
            self.patch_embed = VisualObsEncoder._PatchEmbed2(embed_dim=embed_dim, patch_size=patch_size)
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                num_patches = int(self.patch_embed(dummy).shape[1])
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.blocks = nn.Sequential(
                *[VisualObsEncoder._TransformerLayer(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
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
        mode: str,
        state_dim: int,
        vit_image_size: int,
        vit_depth: int,
        vit_embed_dim: int,
        vit_num_heads: int,
        vit_patch_size: int,
        vit_proj_dim: int,
        project_tokens: bool,
        train_encoder: bool,
        use_drq_aug: bool,
        random_shift_pad: int,
    ):
        super().__init__()
        self.device = device
        self.mode = mode
        self.state_dim = int(state_dim)
        self.vit_image_size = int(vit_image_size)
        self.vit_depth = int(vit_depth)
        self.vit_embed_dim = int(vit_embed_dim)
        self.vit_num_heads = int(vit_num_heads)
        self.vit_patch_size = int(vit_patch_size)
        self.vit_proj_dim = int(vit_proj_dim)
        self.project_tokens = bool(project_tokens)
        self.train_encoder = bool(train_encoder)
        self.use_drq_aug = bool(use_drq_aug)
        self.aug = RandomShiftsAug(pad=random_shift_pad)

        self.vit_front: VisualObsEncoder._MinViT | None = None
        self.vit_wrist: VisualObsEncoder._MinViT | None = None
        self.front_proj: nn.Linear | None = None
        self.wrist_proj: nn.Linear | None = None
        if self.mode == "vit":
            self.vit_front = VisualObsEncoder._MinViT(
                image_size=self.vit_image_size,
                patch_size=self.vit_patch_size,
                embed_dim=self.vit_embed_dim,
                num_heads=self.vit_num_heads,
                depth=self.vit_depth,
            ).to(self.device)
            self.vit_wrist = VisualObsEncoder._MinViT(
                image_size=self.vit_image_size,
                patch_size=self.vit_patch_size,
                embed_dim=self.vit_embed_dim,
                num_heads=self.vit_num_heads,
                depth=self.vit_depth,
            ).to(self.device)
            if not self.train_encoder:
                self.vit_front.requires_grad_(False).eval()
                self.vit_wrist.requires_grad_(False).eval()
            self.params_per_camera = int(sum(p.numel() for p in self.vit_front.parameters()))
            self.params_total = int(self.params_per_camera * 2)
            if self.project_tokens:
                self.front_proj = nn.Linear(self.vit_embed_dim, self.vit_proj_dim).to(self.device)
                self.wrist_proj = nn.Linear(self.vit_embed_dim, self.vit_proj_dim).to(self.device)
                self.output_dim = self.state_dim + 2 * self.vit_proj_dim
            else:
                token_dim = self.vit_embed_dim * self.vit_front.num_patches
                self.output_dim = self.state_dim + 2 * token_dim
        else:
            self.params_per_camera = 0
            self.params_total = 0
            self.output_dim = self.state_dim

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

    def image_for_replay(self, image) -> np.ndarray:
        arr = self._to_hwc_uint8(image)
        if arr.shape[0] != self.vit_image_size or arr.shape[1] != self.vit_image_size:
            x = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)
            x = F.interpolate(x, size=(self.vit_image_size, self.vit_image_size), mode="bilinear", align_corners=False)
            arr = x.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return arr

    def _prep_images(self, images: torch.Tensor, augment: bool) -> torch.Tensor:
        # Input: [B, H, W, C] uint8 or float. Output: [B, C, H, W] float in [-0.5, 0.5].
        if images.dtype == torch.uint8:
            x = images.float() / 255.0
        else:
            x = images.float()
            if x.max() > 1.0:
                x = x / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        if augment and self.use_drq_aug:
            x = self.aug(x)
        return x - 0.5

    def encode_batch(
        self,
        *,
        joint: torch.Tensor,
        front: torch.Tensor,
        wrist: torch.Tensor,
        augment: bool,
    ) -> torch.Tensor:
        joint = joint.float()
        if self.mode == "state":
            return joint[:, : self.state_dim]
        assert self.vit_front is not None and self.vit_wrist is not None
        front_x = self._prep_images(front, augment=augment)
        wrist_x = self._prep_images(wrist, augment=augment)
        front_tokens = self.vit_front(front_x)
        wrist_tokens = self.vit_wrist(wrist_x)
        if self.project_tokens:
            assert self.front_proj is not None and self.wrist_proj is not None
            # Mean-pool token sequence per camera, then project to compact features.
            front_feat = self.front_proj(front_tokens.mean(dim=1))
            wrist_feat = self.wrist_proj(wrist_tokens.mean(dim=1))
        else:
            front_feat = front_tokens.flatten(1, 2)
            wrist_feat = wrist_tokens.flatten(1, 2)
        return torch.cat([joint[:, : self.state_dim], front_feat, wrist_feat], dim=-1)

    def encode_single_no_grad(self, *, front, wrist, joint_state) -> torch.Tensor:
        joint_np = joint_state.detach().cpu().numpy() if torch.is_tensor(joint_state) else np.asarray(joint_state)
        front_np = self.image_for_replay(front)
        wrist_np = self.image_for_replay(wrist)
        joint_t = torch.from_numpy(joint_np.astype(np.float32).reshape(1, -1)[:, : self.state_dim]).to(self.device)
        front_t = torch.from_numpy(front_np).unsqueeze(0).to(self.device)
        wrist_t = torch.from_numpy(wrist_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encode_batch(joint=joint_t, front=front_t, wrist=wrist_t, augment=False).squeeze(0)


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
        last_layer_init_scale: float,
        last_layer_init_distribution: str,
    ):
        super().__init__()
        in_dim = obs_dim + action_dim
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_last_layer(
            scale=float(last_layer_init_scale),
            distribution=str(last_layer_init_distribution),
        )
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

    def _init_last_layer(self, scale: float, distribution: str) -> None:
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            return
        if distribution == "normal":
            if scale <= 0.0:
                nn.init.constant_(last.weight, 0.0)
            else:
                nn.init.normal_(last.weight, mean=0.0, std=scale)
        elif distribution == "orthogonal":
            nn.init.orthogonal_(last.weight, gain=scale)
        elif distribution == "xavier_uniform":
            nn.init.xavier_uniform_(last.weight, gain=scale)
        else:
            raise ValueError(f"Unknown actor last-layer init distribution: {distribution}")
        nn.init.constant_(last.bias, 0.0)

    def forward(self, obs: torch.Tensor, base_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_unit = torch.tanh(self.net(torch.cat([obs, base_action], dim=-1)))
        if self.use_state_gate and self.gate_net is not None:
            gate = torch.sigmoid(self.gate_net(obs))
        else:
            gate = torch.ones((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)
        return residual_unit, gate


class EnsembleQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int, use_layer_norm: bool, num_q: int):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.num_q = int(num_q)
        self.q_heads = nn.ModuleList()
        for _ in range(self.num_q):
            layers: list[nn.Module] = []
            dim = in_dim
            for _ in range(max(1, int(num_layers))):
                layers.append(nn.Linear(dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU())
                dim = hidden_dim
            layers.append(nn.Linear(dim, 1))
            self.q_heads.append(nn.Sequential(*layers))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        qs = [head(x) for head in self.q_heads]
        return torch.stack(qs, dim=0)  # [num_q, B, 1]

    def target_min(self, q_values: torch.Tensor, min_q_heads: int) -> torch.Tensor:
        num = q_values.shape[0]
        take = max(1, min(int(min_q_heads), num))
        if take == num:
            return q_values.min(dim=0).values
        idx = torch.randperm(num, device=q_values.device)[:take]
        return q_values[idx].min(dim=0).values

    def policy_value(self, q_values: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "ensemble_mean":
            return q_values.mean(dim=0)
        if mode == "q1":
            return q_values[0]
        if mode == "min_random_pair":
            if q_values.shape[0] <= 2:
                return q_values.min(dim=0).values
            idx = torch.randperm(q_values.shape[0], device=q_values.device)[:2]
            return q_values[idx].min(dim=0).values
        raise ValueError(f"Unknown policy_gradient_type={mode}")


@dataclass
class TD3Stats:
    critic_loss: float = 0.0
    actor_loss: float = 0.0
    residual_l2: float = 0.0
    delta_l2: float = 0.0
    gate_mean: float = 1.0
    guidance_progress_term: float = 0.0
    guidance_gripper_term: float = 0.0


@dataclass
class GuidanceRewardCoefficients:
    progress_weight: float = 1.0
    xy_weight: float = 0.25
    gripper_weight: float = 0.25
    xy_scale: float = 0.08
    gripper_scale: float = 15.0
    progress_power: float = 1.0


@dataclass
class ReferenceTrajectory:
    traj_id: str
    ee_pos_w: np.ndarray  # [T, 3]
    gripper_state: np.ndarray  # [T]
    initial_cube_pose_w: np.ndarray  # [7]
    meta: dict

    @property
    def length(self) -> int:
        return int(self.ee_pos_w.shape[0])


@dataclass
class TrajectoryGuidanceState:
    traj: ReferenceTrajectory
    prev_progress: float = 0.0


@dataclass
class GuidanceStepMetrics:
    total_reward: float = 0.0
    progress_term: float = 0.0
    xy_term: float = 0.0
    gripper_term: float = 0.0


def _parse_guidance_meta(meta_value) -> dict:
    if meta_value is None:
        return {}
    if isinstance(meta_value, np.ndarray):
        if meta_value.shape == ():
            meta_value = meta_value.item()
        elif meta_value.size == 1:
            meta_value = meta_value.reshape(()).item()
    if isinstance(meta_value, bytes):
        meta_value = meta_value.decode("utf-8")
    if isinstance(meta_value, str):
        try:
            return json.loads(meta_value)
        except json.JSONDecodeError:
            return {"raw_meta": meta_value}
    if isinstance(meta_value, dict):
        return meta_value
    return {"raw_meta": str(meta_value)}


def _load_reference_trajectories(traj_db: str | Path) -> list[ReferenceTrajectory]:
    traj_dir = Path(traj_db).expanduser()
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory DB path does not exist: {traj_dir}")
    paths = sorted(traj_dir.glob("trajectory_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No trajectory_*.npz files found in: {traj_dir}")
    trajectories: list[ReferenceTrajectory] = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        ee_pos_w = np.asarray(data["ee_pos_w"], dtype=np.float32)
        if ee_pos_w.ndim != 2 or ee_pos_w.shape[1] < 2:
            raise ValueError(f"Invalid ee_pos_w shape in {path}: {ee_pos_w.shape}")
        gripper_state = np.asarray(data["gripper_state"], dtype=np.float32).reshape(-1)
        initial_cube_pose_w = np.asarray(data["initial_cube_pose_w"], dtype=np.float32).reshape(-1)
        if initial_cube_pose_w.shape[0] != 7:
            raise ValueError(f"Expected initial_cube_pose_w shape (7,), got {initial_cube_pose_w.shape} in {path}")
        if gripper_state.shape[0] != ee_pos_w.shape[0]:
            min_len = min(gripper_state.shape[0], ee_pos_w.shape[0])
            ee_pos_w = ee_pos_w[:min_len]
            gripper_state = gripper_state[:min_len]
        meta = _parse_guidance_meta(data["meta_json"] if "meta_json" in data else None)
        trajectories.append(
            ReferenceTrajectory(
                traj_id=path.stem,
                ee_pos_w=ee_pos_w[:, :3],
                gripper_state=gripper_state,
                initial_cube_pose_w=initial_cube_pose_w,
                meta=meta,
            )
        )
    return trajectories


def _select_reference_trajectory(
    trajectories: list[ReferenceTrajectory], episode_idx: int, *, strategy: str, rng: np.random.Generator
) -> ReferenceTrajectory:
    if not trajectories:
        raise ValueError("No reference trajectories available.")
    if strategy == "round_robin":
        return trajectories[episode_idx % len(trajectories)]
    if strategy == "random":
        return trajectories[int(rng.integers(len(trajectories)))]
    raise ValueError(f"Unsupported trajectory sampling strategy: {strategy}")


def _closest_index_xy(ref_xy: np.ndarray, current_xy: np.ndarray, min_index: int) -> tuple[int, float]:
    if min_index >= len(ref_xy):
        min_index = len(ref_xy) - 1
    tail = ref_xy[min_index:]
    if tail.size == 0:
        idx = len(ref_xy) - 1
        return idx, float(np.linalg.norm(ref_xy[idx] - current_xy))
    dists = np.linalg.norm(tail - current_xy[None, :], axis=1)
    rel_idx = int(np.argmin(dists))
    idx = min_index + rel_idx
    return idx, float(dists[rel_idx])


def _compute_guidance_reward(
    state: TrajectoryGuidanceState,
    *,
    current_ee_pos_w: np.ndarray,
    current_gripper_state: float,
    coeffs: GuidanceRewardCoefficients,
) -> GuidanceStepMetrics:
    traj = state.traj
    ref_xy = traj.ee_pos_w[:, :2]
    current_xy = np.asarray(current_ee_pos_w, dtype=np.float32)[:2]
    prev_idx = int(round(state.prev_progress * max(traj.length - 1, 1)))
    closest_idx, xy_error = _closest_index_xy(ref_xy, current_xy, prev_idx)
    progress = float(closest_idx / max(traj.length - 1, 1))
    delta_progress = max(progress - state.prev_progress, 0.0)
    ref_gripper = float(traj.gripper_state[closest_idx])
    gripper_error = abs(float(current_gripper_state) - ref_gripper)
    progress_term = coeffs.progress_weight * (delta_progress**coeffs.progress_power)
    xy_term = coeffs.xy_weight * np.exp(-xy_error / max(coeffs.xy_scale, 1e-6))
    gripper_term = coeffs.gripper_weight * np.exp(-gripper_error / max(coeffs.gripper_scale, 1e-6))
    state.prev_progress = max(state.prev_progress, progress)
    return GuidanceStepMetrics(
        total_reward=float(progress_term + xy_term + gripper_term),
        progress_term=float(progress_term),
        xy_term=float(xy_term),
        gripper_term=float(gripper_term),
    )


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _crossed_interval(previous: int, current: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return (previous // interval) < (current // interval)


def _reset_cube_pose_from_reference(
    env,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    pose_attribute_name: str = "_trajectory_guidance_initial_cube_pose_w",
) -> None:
    """Override cube reset pose from sampled reference trajectory initial pose(s)."""
    if env_ids.numel() == 0:
        return
    if not hasattr(env, pose_attribute_name):
        return
    target_pose_w = getattr(env, pose_attribute_name)
    if target_pose_w is None:
        return
    cube = env.scene[cube_cfg.name]
    pose_w = torch.as_tensor(target_pose_w, dtype=torch.float32, device=env.device)
    if pose_w.ndim == 1:
        pose_w = pose_w.unsqueeze(0).repeat(len(env_ids), 1)
    else:
        pose_w = pose_w[env_ids]
    cube.write_root_pose_to_sim(pose_w, env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def _configure_lerobot_absolute_joint_actions(env_cfg, task_type: str) -> None:
    if task_type != "so101leader":
        return
    for action_name in ("arm_action", "gripper_action"):
        action_cfg = getattr(env_cfg.actions, action_name, None)
        if action_cfg is not None and hasattr(action_cfg, "use_default_offset"):
            action_cfg.use_default_offset = False
            action_cfg.scale = 1.0


def _restore_dense_reward_setup(env_cfg) -> None:
    """Collect env disables reward/curriculum; restore from registered dense env config."""
    if args_cli.reward_mode != "dense":
        return
    if "LiftCube" not in args_cli.task:
        return
    dense_cfg = parse_env_cfg("LeIsaac-SO101-LiftCube-RewardDense-v0", device=args_cli.device, num_envs=1)
    env_cfg.rewards = dense_cfg.rewards
    env_cfg.curriculum = dense_cfg.curriculum


def _ensure_collect_env_terminations(env_cfg) -> None:
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


def _select_step_reward(env_reward: float, done: bool, success: bool) -> float:
    if args_cli.reward_mode == "sparse":
        # ResFiT-style sparse terminal reward for online replay:
        # 1.0 only on successful terminal transitions.
        return 1.0 if (done and success) else 0.0
    return float(env_reward)


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


def _create_env(task: str, device: str, num_envs: int | None = None) -> tuple[gym.Env, ManagerBasedRLEnv, str]:
    env_cfg = parse_env_cfg(task, device=device, num_envs=(args_cli.num_envs if num_envs is None else int(num_envs)))
    task_type = get_task_type(task)
    if not args_cli.skip_teleop_device_setup:
        env_cfg.use_teleop_device(task_type)
    else:
        print(f"[INFO] Skipping teleop-device setup in trainer (task_type={task_type}).")
    _configure_lerobot_absolute_joint_actions(env_cfg, task_type)
    env_cfg.seed = args_cli.seed
    env_cfg.recorders = None
    _restore_dense_reward_setup(env_cfg)
    _ensure_collect_env_terminations(env_cfg)
    if args_cli.traj_guidance_db and args_cli.traj_guidance_reset_from_reference:
        env_cfg.events.trajectory_guidance_reset_cube = EventTermCfg(
            func=_reset_cube_pose_from_reference,
            mode="reset",
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "pose_attribute_name": "_trajectory_guidance_initial_cube_pose_w",
            },
        )
    sim_env = gym.make(task, cfg=env_cfg, render_mode=None)
    return sim_env, sim_env.unwrapped, task_type


def _extract_record_modalities(obs_dict: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "record" not in obs_dict:
        raise RuntimeError("This setup requires *-Collect task with record observations.")
    rec = obs_dict["record"]
    required = ("front", "wrist", "joint_pos_abs")
    for key in required:
        if key not in rec:
            raise RuntimeError(f"Missing '{key}' in record observations.")
    return rec["front"], rec["wrist"], rec["joint_pos_abs"]


def _current_ee_position_w(env: ManagerBasedRLEnv, env_id: int = 0) -> np.ndarray:
    ee_frame = env.scene["ee_frame"]
    ee_index = 1 if ee_frame.data.target_pos_w.shape[1] > 1 else 0
    return ee_frame.data.target_pos_w[env_id, ee_index, :3].detach().cpu().numpy().astype(np.float32)


def _extract_current_gripper_lerobot(env: ManagerBasedRLEnv, env_id: int = 0) -> float:
    robot = env.scene["robot"]
    joint_np = robot.data.joint_pos[env_id].detach().cpu().numpy().astype(np.float32)[None, :]
    return float(convert_leisaac_action_to_lerobot(joint_np)[0, 5])


def _extract_residual_state(obs_dict: dict, env_id: int = 0) -> np.ndarray:
    if args_cli.state_source == "policy":
        if "policy" not in obs_dict:
            raise RuntimeError("Missing 'policy' observation group required by --state_source=policy.")
        pol = obs_dict["policy"]
        if torch.is_tensor(pol):
            if pol.ndim == 2:
                pol = pol[env_id]
            return pol.detach().cpu().numpy().astype(np.float32).reshape(-1)
        return np.asarray(pol, dtype=np.float32).reshape(-1)
    _, _, joint = _extract_record_modalities(obs_dict)
    joint_row = joint[env_id]
    return (joint_row.detach().cpu().numpy() if torch.is_tensor(joint_row) else np.asarray(joint_row)).astype(np.float32).reshape(
        -1
    )[:6]


def _prepare_modalities_for_replay(
    obs_dict: dict,
    obs_encoder: VisualObsEncoder,
    env_id: int = 0,
    state_standardizer: StateStandardizer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    front, wrist, _ = _extract_record_modalities(obs_dict)
    front = front[env_id]
    wrist = wrist[env_id]
    joint_np = _extract_residual_state(obs_dict, env_id=env_id)
    if state_standardizer is not None:
        joint_np = state_standardizer.transform_np(joint_np)
    if obs_encoder.mode == "state":
        # State-only recipe: avoid expensive image resizing/copies for replay.
        front_np = np.zeros((1, 1, 3), dtype=np.uint8)
        wrist_np = np.zeros((1, 1, 3), dtype=np.uint8)
    else:
        front_np = obs_encoder.image_for_replay(front)
        wrist_np = obs_encoder.image_for_replay(wrist)
    return joint_np, front_np, wrist_np


def _build_base_obs(obs_dict: dict, task_description: str) -> dict:
    if "record" not in obs_dict:
        raise RuntimeError("This ablation expects *-Collect task with record observations.")
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


def _build_base_obs_for_env(obs_dict: dict, task_description: str, env_id: int) -> dict:
    base_obs = _build_base_obs(obs_dict, task_description)
    return {
        "front": base_obs["front"][env_id : env_id + 1],
        "wrist": base_obs["wrist"][env_id : env_id + 1],
        "joint_pos": base_obs["joint_pos"][env_id : env_id + 1],
        "task_description": task_description,
    }


def _build_base_obs_from_arrays(front: np.ndarray, wrist: np.ndarray, joint_pos: np.ndarray, task_description: str) -> dict:
    return {
        "front": torch.from_numpy(front).unsqueeze(0),
        "wrist": torch.from_numpy(wrist).unsqueeze(0),
        "joint_pos": torch.from_numpy(joint_pos).unsqueeze(0),
        "task_description": task_description,
    }


def _compute_obs_vec_batch(
    obs_dict: dict,
    obs_encoder: VisualObsEncoder,
    device: torch.device,
    num_envs: int,
    state_standardizer: StateStandardizer | None = None,
) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    front, wrist, _ = _extract_record_modalities(obs_dict)
    for env_id in range(num_envs):
        state = _extract_residual_state(obs_dict, env_id=env_id)
        if state_standardizer is not None:
            state = state_standardizer.transform_np(state)
        feat = obs_encoder.encode_single_no_grad(front=front[env_id], wrist=wrist[env_id], joint_state=state).to(device)
        feats.append(feat)
    return torch.stack(feats, dim=0)


def _compute_base_action_batch(
    policy,
    obs_dict: dict,
    task_description: str,
    device: torch.device,
    num_envs: int,
) -> torch.Tensor:
    actions: list[torch.Tensor] = []
    for env_id in range(num_envs):
        base_obs = _build_base_obs_for_env(obs_dict, task_description, env_id)
        action = policy.get_action(base_obs).to(device)[0, 0, :].float()
        actions.append(action)
    return torch.stack(actions, dim=0)


def _concat_batches(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key in lhs.keys():
        out[key] = torch.cat([lhs[key], rhs[key]], dim=0)
    return out


def _sample_mixed_batch(
    *,
    online_replay: ReplayBuffer,
    offline_replay: ReplayBuffer | None,
    batch_size: int,
    offline_ratio: float,
    device: torch.device,
    priority_beta: float,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray | int]]:
    if offline_replay is None or offline_replay.size == 0 or offline_ratio <= 0.0:
        batch, online_idx = online_replay.sample_with_indices(batch_size, device, beta=priority_beta)
        return batch, {
            "online_idx": online_idx,
            "offline_idx": np.empty((0,), dtype=np.int64),
            "online_bs": int(batch_size),
            "offline_bs": 0,
        }
    offline_bs = int(round(batch_size * offline_ratio))
    offline_bs = max(0, min(batch_size, offline_bs, offline_replay.size))
    online_bs = batch_size - offline_bs
    if online_bs <= 0:
        batch, offline_idx = offline_replay.sample_with_indices(batch_size, device, beta=priority_beta)
        return batch, {
            "online_idx": np.empty((0,), dtype=np.int64),
            "offline_idx": offline_idx,
            "online_bs": 0,
            "offline_bs": int(batch_size),
        }
    online_batch, online_idx = online_replay.sample_with_indices(online_bs, device, beta=priority_beta)
    if offline_bs == 0:
        return online_batch, {
            "online_idx": online_idx,
            "offline_idx": np.empty((0,), dtype=np.int64),
            "online_bs": int(online_bs),
            "offline_bs": 0,
        }
    offline_batch, offline_idx = offline_replay.sample_with_indices(offline_bs, device, beta=priority_beta)
    return _concat_batches(online_batch, offline_batch), {
        "online_idx": online_idx,
        "offline_idx": offline_idx,
        "online_bs": int(online_bs),
        "offline_bs": int(offline_bs),
    }


def _resolve_lerobot_dataset_class():
    for import_path in (
        "lerobot.common.datasets.lerobot_dataset",
        "lerobot.common.datasets",
        "lerobot.datasets.lerobot_dataset",
        "lerobot.datasets",
    ):
        try:
            mod = __import__(import_path, fromlist=["LeRobotDataset"])
            dataset_cls = getattr(mod, "LeRobotDataset", None)
            if dataset_cls is not None:
                return dataset_cls
        except Exception:
            continue
    raise RuntimeError("Could not import LeRobotDataset. Install lerobot dataset extras.")


def _as_np_1d(x, dtype=np.float32) -> np.ndarray:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return arr.astype(dtype).reshape(-1)


def _fit_state_dim(state: np.ndarray, state_dim: int) -> np.ndarray:
    vec = np.asarray(state, dtype=np.float32).reshape(-1)
    out = np.zeros((int(state_dim),), dtype=np.float32)
    n = min(out.shape[0], vec.shape[0])
    if n > 0:
        out[:n] = vec[:n]
    return out


class StateStandardizer:
    """Dataset-stat normalizer for residual state vectors."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, min_std: float):
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean.shape != std.shape:
            raise ValueError(f"State normalizer mean/std shape mismatch: {mean.shape} vs {std.shape}")
        self.mean = mean
        self.std = np.maximum(std, float(min_std))

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def transform_np(self, vec: np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            x = _fit_state_dim(x, self.dim)
        return (x - self.mean) / self.std


def _extract_state_stats_from_dataset_stats(stats: dict, state_dim: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(stats, dict):
        return None
    candidate_keys = ("observation.state", "state", "joint_pos_abs", "observation.joint_pos")
    state_stats = None
    for key in candidate_keys:
        if key in stats:
            state_stats = stats[key]
            break
    if not isinstance(state_stats, dict):
        return None
    mean = state_stats.get("mean", None)
    std = state_stats.get("std", None)
    if mean is None or std is None:
        return None
    mean_vec = _fit_state_dim(np.asarray(mean, dtype=np.float32).reshape(-1), state_dim)
    std_vec = _fit_state_dim(np.asarray(std, dtype=np.float32).reshape(-1), state_dim)
    return mean_vec, std_vec


def _build_state_standardizer_from_offline_hf(state_dim: int) -> StateStandardizer | None:
    if not args_cli.offline_state_normalize:
        return None
    if not args_cli.offline_hf_dataset:
        return None
    LeRobotDataset = _resolve_lerobot_dataset_class()
    try:
        ds = LeRobotDataset(repo_id=args_cli.offline_hf_dataset)
        stats = getattr(getattr(ds, "meta", None), "stats", None)
        out = _extract_state_stats_from_dataset_stats(stats, state_dim=state_dim)
        if out is None:
            print("[WARN] Offline state normalization requested, but dataset stats missing state mean/std.")
            return None
        mean_vec, std_vec = out
        standardizer = StateStandardizer(mean=mean_vec, std=std_vec, min_std=args_cli.offline_state_min_std)
        print(
            f"[INFO] State normalizer from offline dataset: repo={args_cli.offline_hf_dataset} "
            f"dim={standardizer.dim} min_std={args_cli.offline_state_min_std:g}"
        )
        return standardizer
    except Exception as exc:
        print(f"[WARN] Failed to build offline state normalizer: {exc}")
        return None


def _offline_hf_cache_path() -> Path | None:
    if not args_cli.offline_hf_cache_dir:
        return None
    cache_root = Path(args_cli.offline_hf_cache_dir).expanduser()
    dataset = str(args_cli.offline_hf_dataset or "none")
    split = str(args_cli.offline_hf_split or "train")
    config = str(args_cli.offline_hf_config or "none")
    cache_key = "|".join(
        [
            dataset,
            split,
            config,
            str(args_cli.offline_max_samples),
            str(args_cli.obs_vit_image_size),
            str(args_cli.policy_checkpoint_path),
            str(args_cli.policy_backend),
            str(args_cli.policy_action_horizon),
            str(args_cli.policy_must_go),
            str(args_cli.policy_type),
            str(args_cli.recipe),
            str(args_cli.state_source),
            str(args_cli.obs_encoder),
            str(args_cli.obs_vit_project_tokens),
            str(args_cli.obs_vit_proj_dim),
        ]
    )
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
    stem = dataset.replace("/", "__")
    return cache_root / f"hf_preprocessed_{stem}_{digest}.npz"


def _save_preprocessed_offline_cache(path: Path, replay: ReplayBuffer, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs_joint=replay.obs_joint[:count],
        obs_front=replay.obs_front[:count],
        obs_wrist=replay.obs_wrist[:count],
        base=replay.base[:count],
        act=replay.act[:count],
        next_joint=replay.next_joint[:count],
        next_front=replay.next_front[:count],
        next_wrist=replay.next_wrist[:count],
        next_base=replay.next_base[:count],
        rew=replay.rew[:count],
        done=replay.done[:count],
        discount=replay.discount[:count],
    )


def _load_preprocessed_offline_cache(path: Path, replay: ReplayBuffer) -> int:
    data = np.load(path)
    required = (
        "obs_joint",
        "obs_front",
        "obs_wrist",
        "base",
        "act",
        "next_joint",
        "next_front",
        "next_wrist",
        "next_base",
        "rew",
        "done",
        "discount",
    )
    if not all(k in data for k in required):
        raise RuntimeError(f"Cache file is missing required arrays: {path}")
    count = int(min(len(data["obs_joint"]), replay.capacity, args_cli.offline_max_samples))
    replay.obs_joint[:count] = np.asarray(data["obs_joint"][:count], dtype=np.float32)
    replay.obs_front[:count] = np.asarray(data["obs_front"][:count], dtype=np.uint8)
    replay.obs_wrist[:count] = np.asarray(data["obs_wrist"][:count], dtype=np.uint8)
    replay.base[:count] = np.asarray(data["base"][:count], dtype=np.float32)
    replay.act[:count] = np.asarray(data["act"][:count], dtype=np.float32)
    replay.next_joint[:count] = np.asarray(data["next_joint"][:count], dtype=np.float32)
    replay.next_front[:count] = np.asarray(data["next_front"][:count], dtype=np.uint8)
    replay.next_wrist[:count] = np.asarray(data["next_wrist"][:count], dtype=np.uint8)
    replay.next_base[:count] = np.asarray(data["next_base"][:count], dtype=np.float32)
    replay.rew[:count] = np.asarray(data["rew"][:count], dtype=np.float32)
    replay.done[:count] = np.asarray(data["done"][:count], dtype=np.float32)
    replay.discount[:count] = np.asarray(data["discount"][:count], dtype=np.float32)
    replay.ptr = count % replay.capacity
    replay.size = count
    return count


def _resolve_frame_key(frame: dict, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in frame:
            return key
    return None


def _load_offline_hf(
    replay: ReplayBuffer,
    *,
    base_policy,
    prompt: str,
    device: torch.device,
    obs_encoder: VisualObsEncoder,
    state_standardizer: StateStandardizer | None = None,
    use_base_policy_for_base_actions: bool = True,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> int:
    if not args_cli.offline_hf_dataset:
        raise ValueError("--offline_hf_dataset is required when --offline_source=hf.")
    cache_path = _offline_hf_cache_path()
    if cache_path is not None and cache_path.exists() and not args_cli.offline_hf_rebuild_cache:
        print(f"[INFO] Loading preprocessed HF cache: {cache_path}")
        loaded = _load_preprocessed_offline_cache(cache_path, replay)
        print(f"[INFO] Loaded {loaded} transitions from cache.")
        return loaded

    LeRobotDataset = _resolve_lerobot_dataset_class()
    ds = LeRobotDataset(repo_id=args_cli.offline_hf_dataset)
    sample0 = ds[0]
    action_key = _resolve_frame_key(sample0, ("action", "actions"))
    state_key = _resolve_frame_key(sample0, ("observation.state", "state", "observation_state"))
    reward_key = _resolve_frame_key(sample0, ("reward", "next.reward"))
    success_key = _resolve_frame_key(sample0, ("is_success", "success", "next.success"))
    front_key = _resolve_frame_key(
        sample0,
        ("observation.images.front", "observation.images_front", "observation.images.top", "observation.images_top"),
    )
    wrist_key = _resolve_frame_key(
        sample0,
        ("observation.images.wrist", "observation.images_wrist", "observation.images.up", "observation.images_up"),
    )
    if not all([action_key, state_key, front_key, wrist_key]):
        raise RuntimeError(f"LeRobot sample missing required keys. Found keys: {list(sample0.keys())}")

    hf_dataset = getattr(ds, "hf_dataset", None)
    has_episode_idx = hf_dataset is not None and "episode_index" in getattr(hf_dataset, "column_names", [])
    has_is_last = hf_dataset is not None and "is_last" in getattr(hf_dataset, "column_names", [])

    added = 0
    n = len(ds)  # type: ignore[arg-type]
    max_count = min(args_cli.offline_max_samples, replay.capacity)
    max_i = max(0, n - 1)
    pbar_total = min(max_i, max_count)
    pbar = tqdm(total=pbar_total, desc="Offline HF preprocessing", unit="transition")
    try:
        for i in range(max_i):
            if added >= max_count:
                break
            row = ds[i]  # type: ignore[index]
            nxt = ds[i + 1]  # type: ignore[index]

            same_episode = True
            if has_episode_idx:
                same_episode = int(hf_dataset["episode_index"][i]) == int(hf_dataset["episode_index"][i + 1])

            done = False
            if has_is_last:
                done = bool(hf_dataset["is_last"][i])
            if not same_episode:
                done = True

            obs_joint = _fit_state_dim(_as_np_1d(row[state_key], dtype=np.float32), replay.state_dim)
            if state_standardizer is not None:
                obs_joint = state_standardizer.transform_np(obs_joint)
            act_lerobot = _as_np_1d(row[action_key], dtype=np.float32)
            act = convert_lerobot_action_to_leisaac(act_lerobot[None, :])[0].astype(np.float32)
            if action_low is not None and action_high is not None:
                act = np.clip(act, action_low, action_high).astype(np.float32)

            obs_front = obs_encoder.image_for_replay(row[front_key])
            obs_wrist = obs_encoder.image_for_replay(row[wrist_key])
            if use_base_policy_for_base_actions:
                base_obs = _build_base_obs_from_arrays(obs_front, obs_wrist, obs_joint, prompt)
                base = base_policy.get_action(base_obs).to(device)[0, 0, :].detach().cpu().numpy().astype(np.float32)
            else:
                base = act.copy()
            if action_low is not None and action_high is not None:
                base = np.clip(base, action_low, action_high).astype(np.float32)

            if done:
                next_obs_joint = obs_joint
                next_front = obs_front
                next_wrist = obs_wrist
                next_base = np.zeros_like(base)
            else:
                next_obs_joint = _fit_state_dim(_as_np_1d(nxt[state_key], dtype=np.float32), replay.state_dim)
                if state_standardizer is not None:
                    next_obs_joint = state_standardizer.transform_np(next_obs_joint)
                next_front = obs_encoder.image_for_replay(nxt[front_key])
                next_wrist = obs_encoder.image_for_replay(nxt[wrist_key])
                if use_base_policy_for_base_actions:
                    next_base_obs = _build_base_obs_from_arrays(next_front, next_wrist, next_obs_joint, prompt)
                    next_base = (
                        base_policy.get_action(next_base_obs).to(device)[0, 0, :].detach().cpu().numpy().astype(np.float32)
                    )
                else:
                    nxt_act_lerobot = _as_np_1d(nxt[action_key], dtype=np.float32)
                    next_base = convert_lerobot_action_to_leisaac(nxt_act_lerobot[None, :])[0].astype(np.float32)
                if action_low is not None and action_high is not None:
                    next_base = np.clip(next_base, action_low, action_high).astype(np.float32)

            replay.add(
                obs_joint=obs_joint.astype(np.float32),
                obs_front=obs_front.astype(np.uint8),
                obs_wrist=obs_wrist.astype(np.uint8),
                base=base,
                act=act,
                next_joint=next_obs_joint.astype(np.float32),
                next_front=next_front.astype(np.uint8),
                next_wrist=next_wrist.astype(np.uint8),
                next_base=next_base,
                # ResFiT-style sparse offline reward:
                # 1) use explicit reward if present, else
                # 2) use explicit success flag if present, else
                # 3) fallback to terminal-only success proxy on demo data.
                rew=(
                    float(_as_np_1d(row[reward_key], dtype=np.float32)[0])
                    if reward_key is not None
                    else (
                        float(_as_np_1d(row[success_key], dtype=np.float32)[0])
                        if success_key is not None
                        else float(done)
                    )
                ),
                done=done,
                discount=args_cli.gamma,
            )
            added += 1
            pbar.update(1)
    finally:
        pbar.close()
    if cache_path is not None:
        _save_preprocessed_offline_cache(cache_path, replay, added)
        print(f"[INFO] Saved preprocessed HF cache: {cache_path}")
    return added


def _load_offline_local_npz(replay: ReplayBuffer) -> int:
    if not args_cli.offline_local_npz_dir:
        raise ValueError("--offline_local_npz_dir is required when --offline_source=local_npz.")
    npz_paths = sorted(Path(args_cli.offline_local_npz_dir).expanduser().glob("*.npz"))
    added = 0
    max_count = min(args_cli.offline_max_samples, replay.capacity)
    for path in npz_paths:
        if added >= max_count:
            break
        data = np.load(path)
        required = (
            "obs_joint",
            "obs_front",
            "obs_wrist",
            "base_action",
            "action",
            "next_obs_joint",
            "next_obs_front",
            "next_obs_wrist",
            "next_base_action",
            "reward",
            "done",
        )
        if not all(k in data for k in required):
            continue
        length = min(len(data["obs_joint"]), max_count - added)
        for i in range(length):
            replay.add(
                obs_joint=np.asarray(data["obs_joint"][i], dtype=np.float32),
                obs_front=np.asarray(data["obs_front"][i], dtype=np.uint8),
                obs_wrist=np.asarray(data["obs_wrist"][i], dtype=np.uint8),
                base=np.asarray(data["base_action"][i], dtype=np.float32),
                act=np.asarray(data["action"][i], dtype=np.float32),
                next_joint=np.asarray(data["next_obs_joint"][i], dtype=np.float32),
                next_front=np.asarray(data["next_obs_front"][i], dtype=np.uint8),
                next_wrist=np.asarray(data["next_obs_wrist"][i], dtype=np.uint8),
                next_base=np.asarray(data["next_base_action"][i], dtype=np.float32),
                rew=float(np.asarray(data["reward"][i]).item()),
                done=bool(np.asarray(data["done"][i]).item()),
                discount=args_cli.gamma,
            )
            added += 1
            if added >= max_count:
                break
    return added


def _get_action_bounds(env: ManagerBasedRLEnv, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    action_space = env.single_action_space
    low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
    high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
    low = torch.where(torch.isfinite(low), low, torch.full_like(low, -3.2))
    high = torch.where(torch.isfinite(high), high, torch.full_like(high, 3.2))
    return low, high


def _encode_obs_batch(batch: dict[str, torch.Tensor], obs_encoder: VisualObsEncoder, *, augment: bool) -> torch.Tensor:
    return obs_encoder.encode_batch(
        joint=batch["obs_joint"],
        front=batch["obs_front"],
        wrist=batch["obs_wrist"],
        augment=augment,
    )


def _encode_next_obs_batch(batch: dict[str, torch.Tensor], obs_encoder: VisualObsEncoder, *, augment: bool) -> torch.Tensor:
    return obs_encoder.encode_batch(
        joint=batch["next_joint"],
        front=batch["next_front"],
        wrist=batch["next_wrist"],
        augment=augment,
    )


def _compose_action(
    base_action: torch.Tensor,
    residual_unit: torch.Tensor,
    residual_scale: float,
    low: torch.Tensor,
    high: torch.Tensor,
    residual_scale_min: float,
    residual_scale_max: float,
    residual_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_min = float(min(residual_scale_min, residual_scale_max))
    scale_max = float(max(residual_scale_min, residual_scale_max))
    scale_cap = float(np.clip(float(residual_scale), scale_min, scale_max))
    if residual_gate is None:
        delta = scale_cap * residual_unit
    else:
        # Gate chooses when to apply stronger correction while staying inside [scale_min, scale_cap].
        gated_scale = scale_min + (scale_cap - scale_min) * torch.clamp(residual_gate, 0.0, 1.0)
        delta = gated_scale * residual_unit
    final_action = torch.clamp(base_action + delta, low, high)
    return final_action, delta


def _effective_residual_scale(step: int) -> float:
    if args_cli.progressive_clipping_steps <= 0:
        return float(args_cli.residual_scale)
    progress = min(1.0, float(step) / max(float(args_cli.progressive_clipping_steps), 1.0))
    return float(args_cli.residual_scale) * progress


def _effective_actor_lr(update_count: int) -> float:
    warmup = int(max(args_cli.actor_lr_warmup_updates, 0))
    if warmup <= 0:
        return float(args_cli.actor_lr)
    progress = min(1.0, float(update_count + 1) / float(warmup))
    return float(args_cli.actor_lr) * progress


def _effective_exploration_std(step: int) -> float:
    end = args_cli.exploration_std_min
    decay_steps = int(max(args_cli.exploration_std_decay_steps, 0))
    if end is None or decay_steps <= 0:
        return float(args_cli.exploration_std)
    start = float(args_cli.exploration_std)
    end_f = float(end)
    progress = min(1.0, max(0.0, float(step) / max(float(decay_steps), 1.0)))
    return float(start + (end_f - start) * progress)


def _effective_priority_beta(step: int) -> float:
    if args_cli.sampling_strategy != "prioritized_replay":
        return 1.0
    start = float(args_cli.priority_beta_start)
    end = float(args_cli.priority_beta_end)
    progress = min(1.0, max(0.0, float(step) / max(float(args_cli.total_steps), 1.0)))
    return float(start + (end - start) * progress)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for t_p, s_p in zip(target.parameters(), source.parameters()):
        t_p.data.mul_(1.0 - tau).add_(tau * s_p.data)


def _evaluate(
    actor: ResidualActor,
    device: torch.device,
    obs_encoder: VisualObsEncoder,
    residual_scale: float,
    low: torch.Tensor,
    high: torch.Tensor,
    base_only: bool,
    episodes: int,
    sim_env: gym.Env,
    env: ManagerBasedRLEnv,
    task_type: str,
    policy,
    prompt: str,
    state_standardizer: StateStandardizer | None = None,
) -> tuple[float, float]:
    obs_encoder.eval()
    success_count = 0
    returns = []
    for _ in range(episodes):
        obs_dict, _ = sim_env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0
        while simulation_app.is_running():
            front_b, wrist_b, _ = _extract_record_modalities(obs_dict)
            front = front_b[0]
            wrist = wrist_b[0]
            state_vec = _extract_residual_state(obs_dict, env_id=0)
            if state_standardizer is not None:
                state_vec = state_standardizer.transform_np(state_vec)
            obs_vec = obs_encoder.encode_single_no_grad(front=front, wrist=wrist, joint_state=state_vec).to(device)
            base_obs = _build_base_obs_for_env(obs_dict, prompt, env_id=0)
            base_chunk = policy.get_action(base_obs).to(device)
            base_action = base_chunk[0, 0, :].float()
            if base_only:
                action = base_action
            else:
                with torch.no_grad():
                    residual_unit, residual_gate = actor(obs_vec.unsqueeze(0), base_action.unsqueeze(0))
                    residual_unit = residual_unit.squeeze(0)
                    residual_gate = residual_gate.squeeze(0)
                action, _ = _compose_action(
                    base_action,
                    residual_unit,
                    residual_scale,
                    low,
                    high,
                    args_cli.residual_scale_min,
                    args_cli.residual_scale_max,
                    residual_gate,
                )
            if env.cfg.dynamic_reset_gripper_effort_limit:
                dynamic_reset_gripper_effort_limit_sim(env, task_type)
            obs_dict, reward, terminated, truncated, _ = sim_env.step(action.unsqueeze(0))
            done = bool((terminated[0] | truncated[0]).detach().cpu().item())
            success = bool(env.reset_terminated[0].detach().cpu().item()) if done else False
            ep_return += _select_step_reward(float(reward[0].detach().cpu().item()), done=done, success=success)
            if done:
                if success:
                    success_count += 1
                break
        returns.append(ep_return)
    return success_count / max(episodes, 1), float(np.mean(returns) if returns else 0.0)


def main() -> None:
    if args_cli.log_dir == _DEFAULT_LOG_DIR:
        base_dir = "logs/residual_td3"
        run_name = args_cli.recipe
        if args_cli.log_dir_use_timestamp:
            run_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        args_cli.log_dir = os.path.join(base_dir, run_name)
        print(f"[INFO] Auto-resolved log_dir: {args_cli.log_dir}")
    os.makedirs(args_cli.log_dir, exist_ok=True)
    print(f"[INFO] Recipe: {args_cli.recipe}")
    print(f"[INFO] Reward mode: {args_cli.reward_mode}")
    print(f"[INFO] Requested base policy backend: {args_cli.policy_backend}")
    print(
        f"[INFO] Residual TD3 hypers: residual_scale={args_cli.residual_scale:.3f}, "
        f"residual_scale_range=[{args_cli.residual_scale_min:.3f},{args_cli.residual_scale_max:.3f}], "
        f"actor_lr={args_cli.actor_lr:.1e}, critic_lr={args_cli.critic_lr:.1e}, "
        f"exploration_std={args_cli.exploration_std:.4f}, exploration_std_min="
        f"{(args_cli.exploration_std if args_cli.exploration_std_min is None else args_cli.exploration_std_min):.4f}, "
        f"exploration_std_decay_steps={args_cli.exploration_std_decay_steps}, "
        f"warmup_steps={args_cli.warmup_steps}, critic_warmup_steps={args_cli.critic_warmup_steps}, "
        f"learning_starts={args_cli.learning_starts}, n_step={args_cli.n_step}, "
        f"num_updates_per_iteration={args_cli.num_updates_per_iteration}, "
        f"num_q={args_cli.num_q}, min_q_heads={args_cli.min_q_heads}, "
        f"policy_gradient_type={args_cli.policy_gradient_type}, use_layer_norm={args_cli.use_layer_norm}, "
        f"use_state_gate={args_cli.use_state_gate}, gate_hidden_dim={args_cli.gate_hidden_dim}, "
        f"gate_init_bias={args_cli.gate_init_bias:.2f}, "
        f"actor_last_layer_init_scale={args_cli.actor_last_layer_init_scale:.2e}, "
        f"actor_last_layer_init_distribution={args_cli.actor_last_layer_init_distribution}, "
        f"use_base_policy_for_warmup={args_cli.use_base_policy_for_warmup}, "
        f"progressive_clipping_steps={args_cli.progressive_clipping_steps}, "
        f"clip_q_target_to_reward_range={args_cli.clip_q_target_to_reward_range}"
    )
    print(
        f"[INFO] Offline replay: source={args_cli.offline_source}, "
        f"mix_ratio={args_cli.offline_mix_ratio:.2f}, max_samples={args_cli.offline_max_samples}, "
        f"offline_use_base_policy_for_base_actions={args_cli.offline_use_base_policy_for_base_actions}"
    )
    print(
        f"[INFO] Replay sampling: strategy={args_cli.sampling_strategy}, "
        f"priority_alpha={args_cli.priority_alpha:.3f}, "
        f"priority_beta_start={args_cli.priority_beta_start:.3f}, "
        f"priority_beta_end={args_cli.priority_beta_end:.3f}"
    )
    if args_cli.traj_guidance_db:
        print(
            f"[INFO] Trajectory guidance: db={args_cli.traj_guidance_db}, "
            f"sampling={args_cli.traj_guidance_sampling}, lambda={args_cli.traj_guidance_lambda:.3f}, "
            f"reset_from_reference={args_cli.traj_guidance_reset_from_reference}"
        )
    if args_cli.offline_source == "hf":
        print(
            f"[INFO] HF dataset={args_cli.offline_hf_dataset}, split={args_cli.offline_hf_split}, "
            f"cache_dir={args_cli.offline_hf_cache_dir}, rebuild_cache={args_cli.offline_hf_rebuild_cache}"
        )
    if args_cli.no_residual:
        print("[INFO] --no_residual enabled: running pure base policy (delta=0).")
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    wall_start_time = time.time()

    device = torch.device(args_cli.device)
    sim_env, env, task_type = _create_env(args_cli.task, args_cli.device, num_envs=args_cli.num_envs)
    num_envs = int(getattr(env, "num_envs", args_cli.num_envs))
    eval_enabled = bool(args_cli.eval_interval > 0 and num_envs == 1)
    if args_cli.eval_interval > 0 and not eval_enabled:
        print(
            "[WARN] Disabling in-loop eval for num_envs>1: IsaacLab does not support creating a second "
            "simulation context in the same process."
        )
    prompt = str(getattr(env.cfg, "task_description", args_cli.policy_language_instruction))
    base_policy = _build_policy_client(env, task_type)
    low, high = _get_action_bounds(env, device)
    guidance_refs: list[ReferenceTrajectory] = []
    guidance_rng = np.random.default_rng(args_cli.traj_guidance_seed)
    guidance_coeffs = GuidanceRewardCoefficients(
        progress_weight=args_cli.traj_guidance_progress_weight,
        xy_weight=args_cli.traj_guidance_xy_weight,
        gripper_weight=args_cli.traj_guidance_gripper_weight,
        xy_scale=args_cli.traj_guidance_xy_scale,
        gripper_scale=args_cli.traj_guidance_gripper_scale,
        progress_power=args_cli.traj_guidance_progress_power,
    )
    if args_cli.traj_guidance_db:
        guidance_refs = _load_reference_trajectories(args_cli.traj_guidance_db)
        print(f"[INFO] Loaded reference trajectories: {len(guidance_refs)}")
    if args_cli.state_source == "policy" and args_cli.offline_source == "hf":
        raise ValueError(
            "--state_source=policy expects privileged policy observations online, but HF offline demos only provide robot state. "
            "Use --offline_source=none (recommended for privileged_cube_pose) or --state_source=record_joint6."
        )
    store_replay_images = args_cli.obs_encoder != "state"
    state_dim = 6
    if args_cli.state_source == "policy":
        obs_probe, _ = sim_env.reset()
        state_dim = int(_extract_residual_state(obs_probe, env_id=0).shape[0])
        if hasattr(base_policy, "reset"):
            base_policy.reset()
    state_standardizer: StateStandardizer | None = None
    if args_cli.offline_source == "hf":
        state_standardizer = _build_state_standardizer_from_offline_hf(state_dim=state_dim)
    print(f"[INFO] Residual state source: {args_cli.state_source} (state_dim={state_dim})")
    if state_standardizer is not None:
        print("[INFO] Residual state normalization: enabled (offline dataset stats).")
    else:
        print("[INFO] Residual state normalization: disabled.")
    print(f"[INFO] Replay image storage: {'enabled' if store_replay_images else 'disabled (state-only mode)'}")
    obs_encoder = VisualObsEncoder(
        device=device,
        mode=args_cli.obs_encoder,
        state_dim=state_dim,
        vit_image_size=args_cli.obs_vit_image_size,
        vit_depth=args_cli.obs_vit_depth,
        vit_embed_dim=args_cli.obs_vit_embed_dim,
        vit_num_heads=args_cli.obs_vit_num_heads,
        vit_patch_size=args_cli.obs_vit_patch_size,
        vit_proj_dim=args_cli.obs_vit_proj_dim,
        project_tokens=args_cli.obs_vit_project_tokens,
        train_encoder=args_cli.obs_train_encoder,
        use_drq_aug=args_cli.obs_use_drq_aug,
        random_shift_pad=args_cli.obs_random_shift_pad,
    ).to(device)
    if args_cli.obs_encoder == "vit":
        print(
            "[INFO] Visual encoder: shallow ViT "
            f"(depth={args_cli.obs_vit_depth}, embed_dim={args_cli.obs_vit_embed_dim}, "
            f"heads={args_cli.obs_vit_num_heads}, patch={args_cli.obs_vit_patch_size}, "
            f"img={args_cli.obs_vit_image_size}, train_encoder={args_cli.obs_train_encoder}, "
            f"drq_aug={args_cli.obs_use_drq_aug}, random_shift_pad={args_cli.obs_random_shift_pad}, "
            f"project_tokens={args_cli.obs_vit_project_tokens}, proj_dim={args_cli.obs_vit_proj_dim}) "
            f"params_per_camera={obs_encoder.params_per_camera:,} total={obs_encoder.params_total:,}"
        )

    if args_cli.reward_mode == "dense" and len(getattr(env.reward_manager, "active_terms", [])) == 0:
        raise RuntimeError(
            "No active reward terms after dense reward restoration. "
            "This ablation requires LiftCube dense reward terms to be active."
        )

    guidance_episode_idx = 0
    guidance_episode_counters = [0 for _ in range(num_envs)]
    guidance_states: list[TrajectoryGuidanceState | None] = [None for _ in range(num_envs)]
    initial_cube_pose_w: np.ndarray | None = None
    if guidance_refs:
        initial_cube_pose_w = np.zeros((num_envs, 7), dtype=np.float32)
        for env_id in range(num_envs):
            selected = _select_reference_trajectory(
                guidance_refs,
                guidance_episode_counters[env_id],
                strategy=args_cli.traj_guidance_sampling,
                rng=guidance_rng,
            )
            guidance_states[env_id] = TrajectoryGuidanceState(traj=selected)
            initial_cube_pose_w[env_id] = selected.initial_cube_pose_w
        if args_cli.traj_guidance_reset_from_reference:
            setattr(env, "_trajectory_guidance_initial_cube_pose_w", initial_cube_pose_w)
        print(f"[INFO] Guidance bootstrap trajectories assigned for {num_envs} envs.")

    print("[INFO] Bootstrap: resetting env for initial observation...")
    obs_dict, _ = sim_env.reset()
    if hasattr(base_policy, "reset"):
        print("[INFO] Bootstrap: resetting base policy state...")
        base_policy.reset()
    print("[INFO] Bootstrap: encoding initial observation...")
    obs_vec0 = _compute_obs_vec_batch(
        obs_dict, obs_encoder, device=device, num_envs=num_envs, state_standardizer=state_standardizer
    )
    print("[INFO] Bootstrap: querying first base action chunk...")
    base_action0 = _compute_base_action_batch(base_policy, obs_dict, prompt, device=device, num_envs=num_envs)
    print("[INFO] Bootstrap: initial base action ready.")

    obs_dim = int(obs_encoder.output_dim)
    act_dim = int(base_action0.shape[-1])
    replay = ReplayBuffer(
        act_dim,
        args_cli.replay_size,
        image_size=args_cli.obs_vit_image_size,
        state_dim=state_dim,
        store_images=store_replay_images,
        sampling_strategy=args_cli.sampling_strategy,
        priority_alpha=args_cli.priority_alpha,
        priority_eps=args_cli.priority_eps,
    )
    offline_replay: ReplayBuffer | None = None
    if args_cli.offline_source != "none":
        offline_replay = ReplayBuffer(
            act_dim,
            args_cli.offline_max_samples,
            image_size=args_cli.obs_vit_image_size,
            state_dim=state_dim,
            store_images=store_replay_images,
            sampling_strategy=args_cli.sampling_strategy,
            priority_alpha=args_cli.priority_alpha,
            priority_eps=args_cli.priority_eps,
        )
        if args_cli.offline_source == "hf":
            offline_added = _load_offline_hf(
                offline_replay,
                base_policy=base_policy,
                prompt=prompt,
                device=device,
                obs_encoder=obs_encoder,
                state_standardizer=state_standardizer,
                use_base_policy_for_base_actions=args_cli.offline_use_base_policy_for_base_actions,
                action_low=low.detach().cpu().numpy(),
                action_high=high.detach().cpu().numpy(),
            )
        else:
            offline_added = _load_offline_local_npz(offline_replay)
        print(f"[INFO] Loaded offline replay transitions: {offline_added}")

    actor = ResidualActor(
        obs_dim,
        act_dim,
        args_cli.hidden_dim,
        args_cli.actor_num_layers,
        args_cli.use_layer_norm,
        args_cli.use_state_gate,
        args_cli.gate_hidden_dim,
        args_cli.gate_init_bias,
        args_cli.actor_last_layer_init_scale,
        args_cli.actor_last_layer_init_distribution,
    ).to(device)
    actor_target = ResidualActor(
        obs_dim,
        act_dim,
        args_cli.hidden_dim,
        args_cli.actor_num_layers,
        args_cli.use_layer_norm,
        args_cli.use_state_gate,
        args_cli.gate_hidden_dim,
        args_cli.gate_init_bias,
        args_cli.actor_last_layer_init_scale,
        args_cli.actor_last_layer_init_distribution,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = EnsembleQCritic(
        obs_dim,
        act_dim,
        args_cli.hidden_dim,
        args_cli.critic_num_layers,
        args_cli.use_layer_norm,
        args_cli.num_q,
    ).to(device)
    critic_target = EnsembleQCritic(
        obs_dim,
        act_dim,
        args_cli.hidden_dim,
        args_cli.critic_num_layers,
        args_cli.use_layer_norm,
        args_cli.num_q,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args_cli.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args_cli.critic_lr)
    encoder_opt: torch.optim.Optimizer | None = None
    if args_cli.obs_encoder == "vit" and args_cli.obs_train_encoder:
        encoder_opt = torch.optim.Adam(obs_encoder.parameters(), lr=args_cli.critic_lr)
    rate_limiter = RateLimiter(args_cli.step_hz)

    obs_vec = obs_vec0.to(device)
    base_action = base_action0
    obs_joint_np = np.zeros((num_envs, state_dim), dtype=np.float32)
    if store_replay_images:
        obs_front_np = np.zeros((num_envs, args_cli.obs_vit_image_size, args_cli.obs_vit_image_size, 3), dtype=np.uint8)
        obs_wrist_np = np.zeros((num_envs, args_cli.obs_vit_image_size, args_cli.obs_vit_image_size, 3), dtype=np.uint8)
    else:
        obs_front_np = np.zeros((num_envs, 1, 1, 3), dtype=np.uint8)
        obs_wrist_np = np.zeros((num_envs, 1, 1, 3), dtype=np.uint8)
    for env_id in range(num_envs):
        j, f, w = _prepare_modalities_for_replay(
            obs_dict, obs_encoder, env_id=env_id, state_standardizer=state_standardizer
        )
        obs_joint_np[env_id] = j
        obs_front_np[env_id] = f
        obs_wrist_np[env_id] = w
    episode_return = np.zeros((num_envs,), dtype=np.float64)
    episode_count = 0
    recent_returns = deque(maxlen=20)
    recent_success = deque(maxlen=20)
    recent_guidance_progress = deque(maxlen=20)
    recent_guidance_gripper = deque(maxlen=20)
    episode_guidance_progress = np.zeros((num_envs,), dtype=np.float64)
    episode_guidance_gripper = np.zeros((num_envs,), dtype=np.float64)
    update_count = 0
    last_stats = TD3Stats()
    n_step_queues: list[deque[dict[str, np.ndarray | float | bool]]] = [deque() for _ in range(num_envs)]
    last_heartbeat_time = time.time()
    print("[INFO] Entering training loop.")
    env_step = 0
    control_step = 0
    last_exploration_std = _effective_exploration_std(0)

    if eval_enabled and args_cli.eval_first:
        should_eval_base = bool(args_cli.eval_base_during_training or args_cli.no_residual)
        if should_eval_base:
            base_sr, base_ret = _evaluate(
                actor=actor,
                device=device,
                obs_encoder=obs_encoder,
                residual_scale=args_cli.residual_scale,
                low=low,
                high=high,
                base_only=True,
                episodes=args_cli.eval_episodes,
                sim_env=sim_env,
                env=env,
                task_type=task_type,
                policy=base_policy,
                prompt=prompt,
                state_standardizer=state_standardizer,
            )
        else:
            base_sr, base_ret = float("nan"), float("nan")
        residual_sr, residual_ret = _evaluate(
            actor=actor,
            device=device,
            obs_encoder=obs_encoder,
            residual_scale=args_cli.residual_scale,
            low=low,
            high=high,
            base_only=bool(args_cli.no_residual),
            episodes=args_cli.eval_episodes,
            sim_env=sim_env,
            env=env,
            task_type=task_type,
            policy=base_policy,
            prompt=prompt,
            state_standardizer=state_standardizer,
        )
        print(
            f"[EVAL@0] step=0 residual_sr={residual_sr:.3f} residual_return={residual_ret:.3f} "
            f"base_sr={base_sr:.3f} base_return={base_ret:.3f}"
        )

    while env_step < args_cli.total_steps:
        control_step += 1
        step_for_schedule = env_step + num_envs
        exploration_std_step = _effective_exploration_std(step_for_schedule)
        last_exploration_std = exploration_std_step
        residual_scale_step = _effective_residual_scale(step_for_schedule)
        residual_gate = torch.ones((num_envs, 1), device=device, dtype=torch.float32)
        if args_cli.no_residual:
            residual_unit = torch.zeros((num_envs, act_dim), device=device)
            action = base_action
            delta = torch.zeros_like(base_action)
        elif step_for_schedule <= args_cli.warmup_steps:
            if args_cli.use_base_policy_for_warmup:
                warmup_noise = torch.empty((num_envs, act_dim), device=device).uniform_(
                    -args_cli.warmup_noise_scale, args_cli.warmup_noise_scale
                )
                action = torch.clamp(base_action + warmup_noise, low, high)
            else:
                action = torch.empty((num_envs, act_dim), device=device).uniform_(
                    -args_cli.warmup_noise_scale, args_cli.warmup_noise_scale
                )
                action = torch.clamp(action, low, high)
                warmup_noise = action - base_action
            # Normalize for logging only (keep bounded like actor output).
            residual_unit = torch.clamp(warmup_noise / max(residual_scale_step, 1e-6), -1.0, 1.0)
            delta = action - base_action
        else:
            with torch.no_grad():
                residual_unit, residual_gate = actor(obs_vec, base_action)
                residual_unit = torch.clamp(
                    residual_unit + torch.randn_like(residual_unit) * exploration_std_step, -1.0, 1.0
                )
            action, delta = _compose_action(
                base_action,
                residual_unit,
                residual_scale_step,
                low,
                high,
                args_cli.residual_scale_min,
                args_cli.residual_scale_max,
                residual_gate,
            )

        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, task_type)
        next_obs_dict, reward, terminated, truncated, step_info = sim_env.step(action)
        executed_action = action
        if isinstance(step_info, dict):
            scaled_action = step_info.get("scaled_action")
            if isinstance(scaled_action, torch.Tensor) and scaled_action.shape == action.shape:
                executed_action = scaled_action.to(device=device, dtype=action.dtype)
        done_t = (terminated | truncated).detach().to(torch.bool)
        done_np = done_t.detach().cpu().numpy().astype(bool).reshape(-1)
        reward_np = reward.detach().cpu().numpy().reshape(-1)
        success_np = env.reset_terminated.detach().cpu().numpy().astype(bool).reshape(-1)
        task_rew_np = np.asarray(
            [
                _select_step_reward(float(reward_np[i]), done=bool(done_np[i]), success=bool(success_np[i]))
                for i in range(num_envs)
            ],
            dtype=np.float32,
        )
        guidance_rew_np = np.zeros((num_envs,), dtype=np.float32)
        guidance_progress_terms = np.zeros((num_envs,), dtype=np.float32)
        guidance_gripper_terms = np.zeros((num_envs,), dtype=np.float32)
        for env_id in range(num_envs):
            if guidance_states[env_id] is not None:
                metrics = _compute_guidance_reward(
                    guidance_states[env_id],
                    current_ee_pos_w=_current_ee_position_w(env, env_id=env_id),
                    current_gripper_state=_extract_current_gripper_lerobot(env, env_id=env_id),
                    coeffs=guidance_coeffs,
                )
                guidance_rew_np[env_id] = float(metrics.total_reward)
                guidance_progress_terms[env_id] = float(metrics.progress_term)
                guidance_gripper_terms[env_id] = float(metrics.gripper_term)
        rew_np = task_rew_np + float(args_cli.traj_guidance_lambda) * guidance_rew_np
        episode_guidance_progress += guidance_progress_terms
        episode_guidance_gripper += guidance_gripper_terms

        next_obs_vec = _compute_obs_vec_batch(
            next_obs_dict,
            obs_encoder,
            device=device,
            num_envs=num_envs,
            state_standardizer=state_standardizer,
        )
        next_joint_np = np.zeros((num_envs, state_dim), dtype=np.float32)
        next_front_np = np.zeros_like(obs_front_np)
        next_wrist_np = np.zeros_like(obs_wrist_np)
        for env_id in range(num_envs):
            j, f, w = _prepare_modalities_for_replay(
                next_obs_dict, obs_encoder, env_id=env_id, state_standardizer=state_standardizer
            )
            next_joint_np[env_id] = j
            next_front_np[env_id] = f
            next_wrist_np[env_id] = w
        next_base_action = _compute_base_action_batch(base_policy, next_obs_dict, prompt, device=device, num_envs=num_envs)
        next_base_action[done_t.reshape(-1)] = 0.0

        if not args_cli.no_residual:
            base_np = base_action.detach().cpu().numpy()
            act_np = executed_action.detach().cpu().numpy()
            next_base_np = next_base_action.detach().cpu().numpy()
            for env_id in range(num_envs):
                queue = n_step_queues[env_id]
                queue.append(
                    {
                        "obs_joint": np.asarray(obs_joint_np[env_id], dtype=np.float32),
                        "obs_front": np.asarray(obs_front_np[env_id], dtype=np.uint8),
                        "obs_wrist": np.asarray(obs_wrist_np[env_id], dtype=np.uint8),
                        "base": np.asarray(base_np[env_id], dtype=np.float32),
                        "act": np.asarray(act_np[env_id], dtype=np.float32),
                        "next_joint": np.asarray(next_joint_np[env_id], dtype=np.float32),
                        "next_front": np.asarray(next_front_np[env_id], dtype=np.uint8),
                        "next_wrist": np.asarray(next_wrist_np[env_id], dtype=np.uint8),
                        "next_base": np.asarray(next_base_np[env_id], dtype=np.float32),
                        "rew": float(rew_np[env_id]),
                        "done": bool(done_np[env_id]),
                    }
                )
                while len(queue) >= args_cli.n_step or (done_np[env_id] and len(queue) > 0):
                    horizon = min(args_cli.n_step, len(queue))
                    reward_n = 0.0
                    done_n = False
                    last = queue[horizon - 1]
                    for i in range(horizon):
                        tr = queue[i]
                        reward_n += (args_cli.gamma**i) * float(tr["rew"])
                        if bool(tr["done"]):
                            done_n = True
                            horizon = i + 1
                            last = queue[i]
                            break
                    discount_n = (args_cli.gamma**horizon) if not done_n else 1.0
                    replay.add(
                        obs_joint=np.asarray(queue[0]["obs_joint"], dtype=np.float32),
                        obs_front=np.asarray(queue[0]["obs_front"], dtype=np.uint8),
                        obs_wrist=np.asarray(queue[0]["obs_wrist"], dtype=np.uint8),
                        base=np.asarray(queue[0]["base"], dtype=np.float32),
                        act=np.asarray(queue[0]["act"], dtype=np.float32),
                        next_joint=np.asarray(last["next_joint"], dtype=np.float32),
                        next_front=np.asarray(last["next_front"], dtype=np.uint8),
                        next_wrist=np.asarray(last["next_wrist"], dtype=np.uint8),
                        next_base=np.asarray(last["next_base"], dtype=np.float32),
                        rew=reward_n,
                        done=done_n,
                        discount=discount_n,
                    )
                    queue.popleft()

        episode_return += rew_np
        for env_id in range(num_envs):
            if not done_np[env_id]:
                continue
            episode_count += 1
            recent_returns.append(float(episode_return[env_id]))
            recent_success.append(float(success_np[env_id]))
            recent_guidance_progress.append(float(episode_guidance_progress[env_id]))
            recent_guidance_gripper.append(float(episode_guidance_gripper[env_id]))
            episode_return[env_id] = 0.0
            episode_guidance_progress[env_id] = 0.0
            episode_guidance_gripper[env_id] = 0.0
            if guidance_refs:
                guidance_episode_counters[env_id] += 1
                guidance_episode_idx += 1
                selected = _select_reference_trajectory(
                    guidance_refs,
                    guidance_episode_counters[env_id],
                    strategy=args_cli.traj_guidance_sampling,
                    rng=guidance_rng,
                )
                guidance_states[env_id] = TrajectoryGuidanceState(traj=selected)
                if args_cli.traj_guidance_reset_from_reference and initial_cube_pose_w is not None:
                    initial_cube_pose_w[env_id] = selected.initial_cube_pose_w

        if args_cli.traj_guidance_reset_from_reference and initial_cube_pose_w is not None:
            setattr(env, "_trajectory_guidance_initial_cube_pose_w", initial_cube_pose_w)

        if num_envs > 1 and hasattr(base_policy, "reset"):
            # Shared chunked policy state is hard to manage with asynchronous per-env resets.
            # Reset every control step to keep multi-env behavior deterministic.
            base_policy.reset()

        obs_dict = next_obs_dict
        obs_vec = next_obs_vec
        base_action = next_base_action
        obs_joint_np, obs_front_np, obs_wrist_np = next_joint_np, next_front_np, next_wrist_np

        prev_env_step = env_step
        env_step = min(env_step + num_envs, int(args_cli.total_steps))

        if (
            (not args_cli.no_residual)
            and env_step >= args_cli.learning_starts
            and replay.size >= args_cli.batch_size
        ):
            step_actor_loss = float(last_stats.actor_loss)
            for _ in range(args_cli.num_updates_per_iteration):
                priority_beta = _effective_priority_beta(env_step)
                batch, sample_meta = _sample_mixed_batch(
                    online_replay=replay,
                    offline_replay=offline_replay,
                    batch_size=args_cli.batch_size,
                    offline_ratio=args_cli.offline_mix_ratio,
                    device=device,
                    priority_beta=priority_beta,
                )
                if args_cli.obs_encoder == "vit" and args_cli.obs_train_encoder:
                    obs_encoder.train()
                else:
                    obs_encoder.eval()
                if encoder_opt is not None:
                    encoder_opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    next_obs_feat_tgt = _encode_next_obs_batch(batch, obs_encoder, augment=args_cli.obs_use_drq_aug)
                    target_residual, target_gate = actor_target(next_obs_feat_tgt, batch["next_base"])
                    if args_cli.target_action_noise:
                        target_noise = torch.clamp(
                            torch.randn_like(target_residual) * args_cli.target_noise_std,
                            -args_cli.target_noise_clip,
                            args_cli.target_noise_clip,
                        )
                        target_residual = torch.clamp(target_residual + target_noise, -1.0, 1.0)
                    target_action, _ = _compose_action(
                        batch["next_base"],
                        target_residual,
                        residual_scale_step,
                        low,
                        high,
                        args_cli.residual_scale_min,
                        args_cli.residual_scale_max,
                        target_gate,
                    )
                    q_t = critic_target(next_obs_feat_tgt, target_action)
                    q_t_min = critic_target.target_min(q_t, args_cli.min_q_heads)
                    target_q = batch["rew"] + (1.0 - batch["done"]) * batch["discount"] * q_t_min
                    if args_cli.clip_q_target_to_reward_range and args_cli.reward_mode == "sparse":
                        target_q = torch.clamp(target_q, min=0.0, max=1.0)

                obs_feat = _encode_obs_batch(batch, obs_encoder, augment=args_cli.obs_use_drq_aug)
                q_values = critic(obs_feat, batch["act"])
                td_error_sq = (q_values - target_q.unsqueeze(0).expand_as(q_values)).pow(2).mean(dim=0).squeeze(-1)
                is_weight = batch["is_weight"].squeeze(-1)
                critic_loss = (is_weight * td_error_sq).mean()
                critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                if encoder_opt is not None:
                    encoder_opt.step()
                critic_opt.step()
                if args_cli.sampling_strategy == "prioritized_replay":
                    td_priority = torch.sqrt(torch.clamp(td_error_sq.detach(), min=0.0)).cpu().numpy()
                    online_bs = int(sample_meta["online_bs"])
                    offline_bs = int(sample_meta["offline_bs"])
                    if online_bs > 0:
                        replay.update_priorities(
                            np.asarray(sample_meta["online_idx"], dtype=np.int64),
                            td_priority[:online_bs],
                        )
                    if offline_replay is not None and offline_bs > 0:
                        offline_replay.update_priorities(
                            np.asarray(sample_meta["offline_idx"], dtype=np.int64),
                            td_priority[online_bs : online_bs + offline_bs],
                        )

                actor_loss = torch.tensor(0.0, device=device)
                actor_update_enabled = update_count >= args_cli.critic_warmup_steps
                if actor_update_enabled and update_count % args_cli.policy_delay == 0:
                    obs_actor = obs_feat.detach()
                    actor_residual, actor_gate = actor(obs_actor, batch["base"])
                    actor_action, actor_delta = _compose_action(
                        batch["base"],
                        actor_residual,
                        residual_scale_step,
                        low,
                        high,
                        args_cli.residual_scale_min,
                        args_cli.residual_scale_max,
                        actor_gate,
                    )
                    actor_q = critic(obs_actor, actor_action)
                    actor_val = critic.policy_value(actor_q, args_cli.policy_gradient_type)
                    actor_loss = -actor_val.mean()
                    actor_loss = actor_loss + args_cli.residual_reg_weight * (actor_delta.pow(2).mean())
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_lr_step = _effective_actor_lr(update_count)
                    for group in actor_opt.param_groups:
                        group["lr"] = actor_lr_step
                    actor_opt.step()
                    step_actor_loss = float(actor_loss.detach().cpu().item())

                    _soft_update(actor_target, actor, args_cli.tau)
                    _soft_update(critic_target, critic, args_cli.tau)
                else:
                    # Keep critic targets fresh even during actor warmup.
                    _soft_update(critic_target, critic, args_cli.tau)

                update_count += 1
                last_stats = TD3Stats(
                    critic_loss=float(critic_loss.detach().cpu().item()),
                    actor_loss=step_actor_loss,
                    residual_l2=float(residual_unit.norm(p=2, dim=-1).mean().detach().cpu().item()),
                    delta_l2=float(delta.norm(p=2, dim=-1).mean().detach().cpu().item()),
                    gate_mean=float(residual_gate.mean().detach().cpu().item()),
                    guidance_progress_term=(
                        float(np.mean(recent_guidance_progress)) if recent_guidance_progress else 0.0
                    ),
                    guidance_gripper_term=(
                        float(np.mean(recent_guidance_gripper)) if recent_guidance_gripper else 0.0
                    ),
                )

        if _crossed_interval(prev_env_step, env_step, args_cli.log_interval) or env_step == args_cli.total_steps:
            mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            mean_success = float(np.mean(recent_success)) if recent_success else 0.0
            mean_guidance_progress = float(np.mean(recent_guidance_progress)) if recent_guidance_progress else 0.0
            mean_guidance_gripper = float(np.mean(recent_guidance_gripper)) if recent_guidance_gripper else 0.0
            elapsed_s = time.time() - wall_start_time
            steps_per_s = env_step / max(elapsed_s, 1e-6)
            remaining_steps = max(args_cli.total_steps - env_step, 0)
            eta_s = remaining_steps / max(steps_per_s, 1e-6)
            print(
                f"[TRAIN] step={env_step} episodes={episode_count} replay={replay.size} "
                f"elapsed={_format_seconds(elapsed_s)} eta={_format_seconds(eta_s)} steps_per_s={steps_per_s:.2f} "
                f"return20={mean_return:.3f} success20={mean_success:.3f} "
                f"critic_loss={last_stats.critic_loss:.3e} actor_loss={last_stats.actor_loss:.3e} "
                f"residual_l2={last_stats.residual_l2:.4f} delta_l2={last_stats.delta_l2:.4f} "
                f"gate_mean={last_stats.gate_mean:.3f} expl_std={last_exploration_std:.4f} "
                f"guidance_prog={mean_guidance_progress:.4f} "
                f"guidance_grip={mean_guidance_gripper:.4f}"
            )

        now_time = time.time()
        if args_cli.heartbeat_interval_s > 0.0 and (now_time - last_heartbeat_time) >= args_cli.heartbeat_interval_s:
            elapsed_s = now_time - wall_start_time
            steps_per_s = env_step / max(elapsed_s, 1e-6)
            remaining_steps = max(args_cli.total_steps - env_step, 0)
            eta_s = remaining_steps / max(steps_per_s, 1e-6)
            print(
                f"[HEARTBEAT] step={env_step}/{args_cli.total_steps} replay={replay.size} "
                f"updates={update_count} elapsed={_format_seconds(elapsed_s)} eta={_format_seconds(eta_s)} "
                f"sps={steps_per_s:.2f}"
            )
            last_heartbeat_time = now_time

        if eval_enabled and _crossed_interval(prev_env_step, env_step, args_cli.eval_interval):
            should_eval_base = bool(args_cli.eval_base_during_training or args_cli.no_residual)
            if should_eval_base:
                base_sr, base_ret = _evaluate(
                    actor=actor,
                    device=device,
                    obs_encoder=obs_encoder,
                    residual_scale=args_cli.residual_scale,
                    low=low,
                    high=high,
                    base_only=True,
                    episodes=args_cli.eval_episodes,
                    sim_env=sim_env,
                    env=env,
                    task_type=task_type,
                    policy=base_policy,
                    prompt=prompt,
                    state_standardizer=state_standardizer,
                )
            else:
                base_sr, base_ret = float("nan"), float("nan")
            if args_cli.no_residual:
                residual_sr, residual_ret = base_sr, base_ret
            else:
                residual_sr, residual_ret = _evaluate(
                    actor=actor,
                    device=device,
                    obs_encoder=obs_encoder,
                    residual_scale=args_cli.residual_scale,
                    low=low,
                    high=high,
                    base_only=False,
                    episodes=args_cli.eval_episodes,
                    sim_env=sim_env,
                    env=env,
                    task_type=task_type,
                    policy=base_policy,
                    prompt=prompt,
                    state_standardizer=state_standardizer,
                )
            if should_eval_base:
                print(
                    f"[EVAL] step={env_step} base_sr={base_sr:.3f} residual_sr={residual_sr:.3f} "
                    f"base_ret={base_ret:.3f} residual_ret={residual_ret:.3f}"
                )
            else:
                print(
                    f"[EVAL] step={env_step} base=skipped residual_sr={residual_sr:.3f} "
                    f"residual_ret={residual_ret:.3f}"
                )

        if _crossed_interval(prev_env_step, env_step, args_cli.save_interval):
            ckpt = {
                "step": env_step,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_target": actor_target.state_dict(),
                "critic_target": critic_target.state_dict(),
                "obs_encoder": obs_encoder.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "encoder_opt": encoder_opt.state_dict() if encoder_opt is not None else None,
                "args": vars(args_cli),
            }
            ckpt_path = os.path.join(args_cli.log_dir, f"residual_td3_step_{env_step:08d}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"[CHECKPOINT] Saved {ckpt_path}")

        rate_limiter.sleep(env)

    final_path = os.path.join(args_cli.log_dir, "residual_td3_final.pt")
    torch.save(
        {
            "step": args_cli.total_steps,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_target": actor_target.state_dict(),
            "critic_target": critic_target.state_dict(),
            "obs_encoder": obs_encoder.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic_opt": critic_opt.state_dict(),
            "encoder_opt": encoder_opt.state_dict() if encoder_opt is not None else None,
            "args": vars(args_cli),
        },
        final_path,
    )
    print(f"[DONE] Saved final checkpoint: {final_path}")

    sim_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
