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
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-RewardDense-Collect-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_steps", type=int, default=250_000)
parser.add_argument("--warmup_steps", type=int, default=5_000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--replay_size", type=int, default=300_000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--policy_delay", type=int, default=2)
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
parser.add_argument("--policy_action_horizon", type=int, default=50)
parser.add_argument("--policy_language_instruction", type=str, default="Lift the red cube up.")
parser.add_argument("--policy_checkpoint_path", type=str, required=True)
parser.add_argument("--policy_must_go", action="store_true", default=False)
parser.add_argument("--policy_type", type=str, default="smolvla")
parser.add_argument(
    "--policy_backend",
    type=str,
    default="local",
    choices=["local", "service"],
    help="Backend for base policy inference. 'local' avoids policy-server RPC overhead.",
)
parser.add_argument("--log_dir", type=str, default="logs/residual_td3/lift_cube_privileged")
parser.add_argument("--num_envs", type=int, default=1)

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
    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.base = np.zeros((capacity, action_dim), dtype=np.float32)
        self.act = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_base = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        base: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        next_base: np.ndarray,
        rew: float,
        done: bool,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.base[i] = base
        self.act[i] = act
        self.next_obs[i] = next_obs
        self.next_base[i] = next_base
        self.rew[i, 0] = rew
        self.done[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(device),
            "base": torch.from_numpy(self.base[idx]).to(device),
            "act": torch.from_numpy(self.act[idx]).to(device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(device),
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

    def __init__(self, pretrained_name_or_path: str, device: str):
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
        self.policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path).to(self.device).eval()
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
        # Keep interface compatible with service policy client.
        return

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
        action = convert_lerobot_action_to_leisaac(action)
        return torch.from_numpy(action[:, None, :])


class ResidualActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = obs_dim + action_dim
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

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))


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


def _restore_dense_reward_setup(env_cfg) -> None:
    """Collect env disables reward/curriculum; restore dense terms for this ablation."""
    if "LiftCube" not in args_cli.task:
        return
    from leisaac.tasks.lift_cube.lift_cube_env_cfg import (
        LiftCubeRewardDenseCurriculumCfg,
        LiftCubeRewardDenseRewardsCfg,
    )

    env_cfg.rewards = LiftCubeRewardDenseRewardsCfg()
    env_cfg.curriculum = LiftCubeRewardDenseCurriculumCfg()


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


def _build_policy_client(env: ManagerBasedRLEnv, task_type: str):
    camera_infos = {k: sensor.image_shape for k, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)}
    if not camera_infos:
        raise RuntimeError("No camera sensors found for SmolVLA base policy.")
    if args_cli.policy_backend == "local":
        print("[INFO] Base policy backend: local (in-process SmolVLA, no policy-server RPC).")
        return LocalSmolVLAPolicy(
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            device=args_cli.device,
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
    _restore_dense_reward_setup(env_cfg)
    _ensure_collect_env_terminations(env_cfg)
    sim_env = gym.make(task, cfg=env_cfg, render_mode=None)
    return sim_env, sim_env.unwrapped, task_type


def _policy_obs_vector(obs_dict: dict) -> torch.Tensor:
    policy_obs = obs_dict["policy"]
    if not torch.is_tensor(policy_obs):
        raise RuntimeError("Expected tensor policy observations.")
    return policy_obs[0].float()


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
    for _ in range(episodes):
        obs_dict, _ = sim_env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0
        while simulation_app.is_running():
            obs_vec = _policy_obs_vector(obs_dict).to(device)
            base_obs = _build_base_obs(obs_dict, prompt)
            base_chunk = policy.get_action(base_obs).to(device)
            base_action = base_chunk[0, 0, :].float()
            if base_only:
                action = base_action
            else:
                with torch.no_grad():
                    residual_unit = actor(obs_vec.unsqueeze(0)).squeeze(0)
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

    if len(getattr(env.reward_manager, "active_terms", [])) == 0:
        raise RuntimeError(
            "No active reward terms after dense reward restoration. "
            "This ablation requires LiftCube dense reward terms to be active."
        )

    obs_dict, _ = sim_env.reset()
    if hasattr(base_policy, "reset"):
        base_policy.reset()
    obs_vec0 = _policy_obs_vector(obs_dict)
    base_obs0 = _build_base_obs(obs_dict, prompt)
    base_action0 = base_policy.get_action(base_obs0).to(device)[0, 0, :].float()

    obs_dim = int(obs_vec0.numel())
    act_dim = int(base_action0.numel())
    replay = ReplayBuffer(obs_dim, act_dim, args_cli.replay_size)

    actor = ResidualActor(obs_dim, act_dim, args_cli.hidden_dim).to(device)
    actor_target = ResidualActor(obs_dim, act_dim, args_cli.hidden_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = TwinQCritic(obs_dim, act_dim, args_cli.hidden_dim).to(device)
    critic_target = TwinQCritic(obs_dim, act_dim, args_cli.hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args_cli.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args_cli.critic_lr)
    rate_limiter = RateLimiter(args_cli.step_hz)

    obs_vec = obs_vec0.to(device)
    base_action = base_action0
    episode_return = 0.0
    episode_count = 0
    recent_returns = deque(maxlen=20)
    recent_success = deque(maxlen=20)
    update_count = 0
    last_stats = TD3Stats()

    for step in range(1, args_cli.total_steps + 1):
        if step <= args_cli.warmup_steps:
            residual_unit = torch.empty(act_dim, device=device).uniform_(-1.0, 1.0)
        else:
            with torch.no_grad():
                residual_unit = actor(obs_vec.unsqueeze(0)).squeeze(0)
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

        next_obs_vec = _policy_obs_vector(next_obs_dict).to(device)
        if done:
            next_base_action = torch.zeros_like(base_action)
        else:
            next_base_obs = _build_base_obs(next_obs_dict, prompt)
            next_base_action = base_policy.get_action(next_base_obs).to(device)[0, 0, :].float()

        replay.add(
            obs=obs_vec.detach().cpu().numpy(),
            base=base_action.detach().cpu().numpy(),
            act=action.detach().cpu().numpy(),
            next_obs=next_obs_vec.detach().cpu().numpy(),
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
            obs_vec = _policy_obs_vector(obs_dict).to(device)
            base_obs = _build_base_obs(obs_dict, prompt)
            base_action = base_policy.get_action(base_obs).to(device)[0, 0, :].float()
        else:
            obs_vec = next_obs_vec
            base_action = next_base_action

        if replay.size >= args_cli.batch_size:
            batch = replay.sample(args_cli.batch_size, device)
            with torch.no_grad():
                target_residual = actor_target(batch["next_obs"])
                target_noise = torch.clamp(
                    torch.randn_like(target_residual) * args_cli.target_noise_std,
                    -args_cli.target_noise_clip,
                    args_cli.target_noise_clip,
                )
                target_residual = torch.clamp(target_residual + target_noise, -1.0, 1.0)
                target_action, _ = _compose_action(
                    batch["next_base"], target_residual, args_cli.residual_scale, low, high
                )
                q1_t, q2_t = critic_target(batch["next_obs"], target_action)
                target_q = batch["rew"] + (1.0 - batch["done"]) * args_cli.gamma * torch.min(q1_t, q2_t)

            q1, q2 = critic(batch["obs"], batch["act"])
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_opt.step()

            actor_loss = torch.tensor(0.0, device=device)
            if update_count % args_cli.policy_delay == 0:
                actor_residual = actor(batch["obs"])
                actor_action, actor_delta = _compose_action(
                    batch["base"], actor_residual, args_cli.residual_scale, low, high
                )
                actor_loss = -critic.q1_only(batch["obs"], actor_action).mean()
                actor_loss = actor_loss + args_cli.residual_reg_weight * (actor_delta.pow(2).mean())
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                _soft_update(actor_target, actor, args_cli.tau)
                _soft_update(critic_target, critic, args_cli.tau)

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
                actor=actor,
                device=device,
                residual_scale=args_cli.residual_scale,
                low=low,
                high=high,
                base_only=True,
                episodes=args_cli.eval_episodes,
            )
            residual_sr, residual_ret = _evaluate(
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
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_target": actor_target.state_dict(),
                "critic_target": critic_target.state_dict(),
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
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_target": actor_target.state_dict(),
            "critic_target": critic_target.state_dict(),
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
