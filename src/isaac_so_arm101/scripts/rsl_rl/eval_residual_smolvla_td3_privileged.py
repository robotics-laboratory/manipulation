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
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from isaaclab.app import AppLauncher

# isort: off
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args
# isort: on


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-RewardDense-Collect-v0")
parser.add_argument("--residual_checkpoint", type=str, required=True, help="Path to residual TD3 checkpoint (.pt).")
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--policy_checkpoint_path", type=str, default=None)
parser.add_argument("--policy_backend", type=str, choices=["local", "service"], default=None)
parser.add_argument("--policy_action_horizon", type=int, default=None)
parser.add_argument("--policy_must_go", action="store_true", default=False)
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
    "--residual_scale",
    type=float,
    default=None,
    help="Override residual scale. If omitted, use checkpoint args.",
)
parser.add_argument(
    "--print_episode_debug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print per-episode return/success/max-height debug.",
)
parser.add_argument(
    "--eval_base",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also evaluate base policy in this run. Disabled by default for faster residual-only checks.",
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
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int, use_layer_norm: bool):
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

    def forward(self, obs: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([obs, base_action], dim=-1)))


@dataclass
class EvalSummary:
    sr: float
    ret: float
    mean_max_height: float
    max_max_height: float


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
    base_action: torch.Tensor, residual_unit: torch.Tensor, residual_scale: float, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    return torch.clamp(base_action + residual_scale * residual_unit, low, high)


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
    state_source: str,
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
    for ep in range(args_cli.num_episodes):
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
            state_vec = _extract_state(obs_dict, state_source)
            state_t = torch.from_numpy(state_vec.astype(np.float32)).to(device).unsqueeze(0)
            base_obs = _build_base_obs(obs_dict, prompt)
            base_action = policy.get_action(base_obs).to(device)[0, 0, :].float()
            if use_residual:
                with torch.no_grad():
                    residual_unit = actor(state_t, base_action.unsqueeze(0)).squeeze(0)
                action = _compose_action(base_action, residual_unit, residual_scale, low, high)
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
    return EvalSummary(
        sr=float(success_count / max(args_cli.num_episodes, 1)),
        ret=float(np.mean(returns) if returns else 0.0),
        mean_max_height=float(np.mean(max_heights) if max_heights else 0.0),
        max_max_height=float(np.max(max_heights) if max_heights else 0.0),
    )


def main() -> None:
    ckpt = torch.load(args_cli.residual_checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    policy_backend = args_cli.policy_backend or str(ckpt_args.get("policy_backend", "local"))
    policy_ckpt = args_cli.policy_checkpoint_path or ckpt_args.get("policy_checkpoint_path")
    if not policy_ckpt:
        raise ValueError("--policy_checkpoint_path is required (or available in checkpoint args).")
    action_horizon = int(args_cli.policy_action_horizon or ckpt_args.get("policy_action_horizon", 1))
    state_source = str(args_cli.state_source or ckpt_args.get("state_source", "record_joint6"))
    residual_scale = float(args_cli.residual_scale if args_cli.residual_scale is not None else ckpt_args.get("residual_scale", 0.1))
    skip_teleop = bool(
        args_cli.skip_teleop_device_setup
        if args_cli.skip_teleop_device_setup is not None
        else ckpt_args.get("skip_teleop_device_setup", True)
    )
    success_height_threshold = 0.20

    device = torch.device(args_cli.device)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    sim_env, env, task_type = _create_env(args_cli.task, args_cli.device, skip_teleop=skip_teleop, seed=args_cli.seed)
    prompt = str(getattr(env.cfg, "task_description", "Lift the red cube up."))
    policy = _build_policy_client(env, task_type, policy_backend, str(policy_ckpt), action_horizon)
    low, high = _get_action_bounds(env, device)

    obs0, _ = sim_env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    state_dim = int(_extract_state(obs0, state_source).shape[0])
    act_dim = int(env.single_action_space.shape[0])

    actor = ResidualActor(
        obs_dim=state_dim,
        action_dim=act_dim,
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        num_layers=int(ckpt_args.get("actor_num_layers", 2)),
        use_layer_norm=bool(ckpt_args.get("use_layer_norm", True)),
    ).to(device)
    actor.load_state_dict(ckpt["actor"])

    print(f"[INFO] Loaded checkpoint: {args_cli.residual_checkpoint}")
    print(
        f"[INFO] Eval config: task={args_cli.task}, episodes={args_cli.num_episodes}, "
        f"state_source={state_source}, residual_scale={residual_scale:.3f}, skip_teleop={skip_teleop}, "
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
            state_source=state_source,
            prompt=prompt,
            use_residual=False,
            success_height_threshold=success_height_threshold,
            record_video=args_cli.record_video,
            video_dir=args_cli.video_dir,
            video_fps=args_cli.video_fps,
        )
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
        state_source=state_source,
        prompt=prompt,
        use_residual=True,
        success_height_threshold=success_height_threshold,
        record_video=args_cli.record_video,
        video_dir=args_cli.video_dir,
        video_fps=args_cli.video_fps,
    )

    if base is not None:
        print(
            f"[RESULT] base_sr={base.sr:.3f} residual_sr={residual.sr:.3f} "
            f"base_ret={base.ret:.3f} residual_ret={residual.ret:.3f}"
        )
        print(
            f"[HEIGHT] base_mean_max={base.mean_max_height:.4f}m residual_mean_max={residual.mean_max_height:.4f}m "
            f"base_global_max={base.max_max_height:.4f}m residual_global_max={residual.max_max_height:.4f}m"
        )
    else:
        print(f"[RESULT] residual_sr={residual.sr:.3f} residual_ret={residual.ret:.3f}")
        print(
            f"[HEIGHT] residual_mean_max={residual.mean_max_height:.4f}m residual_global_max={residual.max_max_height:.4f}m"
        )

    sim_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
