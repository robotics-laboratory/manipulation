#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect end-effector trajectories from a trained RSL-RL teacher policy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip
import isaac_so_arm101.scripts.rsl_rl.log_paths as log_paths  # isort: skip

parser = argparse.ArgumentParser(description="Collect teacher trajectories for trajectory-guided RL.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-v0", help="Task to run.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_episodes", type=int, default=500, help="Number of completed episodes to save.")
parser.add_argument(
    "--output",
    type=str,
    default=os.path.join(log_paths.rsl_rl_root(), "teacher_trajectories", "so101_lift_cube_teacher.pt"),
    help="Output .pt file for saved trajectories.",
)
parser.add_argument(
    "--success_distance_threshold",
    type=float,
    default=0.05,
    help="Goal distance threshold (m) used for success filtering in Target-Cube tasks.",
)
parser.add_argument(
    "--success_minimal_height",
    type=float,
    default=0.01,
    help="Minimum object height (m) used for success filtering.",
)
parser.add_argument("--keep_failed", action="store_true", default=False, help="If set, keep failed episodes.")
parser.add_argument(
    "--disable_task_cameras",
    action="store_true",
    default=False,
    help="Disable task camera sensors and image observation terms in the env config.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401


def _disable_task_cameras_in_env_cfg(env_cfg):
    scene = getattr(env_cfg, "scene", None)
    if scene is not None:
        if hasattr(scene, "camera_top"):
            scene.camera_top = None
        if hasattr(scene, "camera_wrist"):
            scene.camera_wrist = None

    observations = getattr(env_cfg, "observations", None)
    image_group = getattr(observations, "observation", None) if observations is not None else None
    if image_group is not None:
        for image_term in ("images_top", "images_wrist", "images_side", "images_up"):
            if hasattr(image_group, image_term):
                setattr(image_group, image_term, None)

    if hasattr(env_cfg, "image_obs_list"):
        env_cfg.image_obs_list = []


def _compute_goal_pos_w(base_env) -> torch.Tensor:
    robot = base_env.scene["robot"]
    command = base_env.command_manager.get_command("object_pose")
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    return des_pos_w


def _current_signals(base_env):
    ee_frame = base_env.scene["ee_frame"]
    object_asset = base_env.scene["object"]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    object_pos_w = object_asset.data.root_pos_w[:, :3]
    goal_pos_w = _compute_goal_pos_w(base_env)
    return ee_pos_w, object_pos_w, goal_pos_w


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.disable_task_cameras and hasattr(env_cfg, "disable_task_cameras"):
        env_cfg.disable_task_cameras = True
    if args_cli.disable_task_cameras:
        _disable_task_cameras_in_env_cfg(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    if not args_cli.checkpoint:
        raise ValueError("Please provide --checkpoint to a trained Lift-Cube teacher model.")
    checkpoint_path = Path(args_cli.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading teacher checkpoint: {checkpoint_path}")
    runner.load(str(checkpoint_path))
    policy = runner.get_inference_policy(device=base_env.device)

    num_envs = env.num_envs
    active_ee = [[] for _ in range(num_envs)]
    active_initial_object_pos = [None for _ in range(num_envs)]
    active_goal_pos = [None for _ in range(num_envs)]
    active_max_height = torch.full((num_envs,), -1.0e9, dtype=torch.float32, device=base_env.device)
    active_min_goal_dist = torch.full((num_envs,), 1.0e9, dtype=torch.float32, device=base_env.device)
    use_goal_distance_for_success = "target-cube" in args_cli.task.lower()

    obs = env.get_observations()
    _, object_pos_w, goal_pos_w = _current_signals(base_env)
    for env_id in range(num_envs):
        active_initial_object_pos[env_id] = object_pos_w[env_id].detach().cpu().to(torch.float32)
        active_goal_pos[env_id] = goal_pos_w[env_id].detach().cpu().to(torch.float32)

    collected_initial_object_pos: list[torch.Tensor] = []
    collected_goal_pos: list[torch.Tensor] = []
    collected_ee_traj: list[torch.Tensor] = []
    collected_lengths: list[int] = []
    collected_success: list[bool] = []

    while len(collected_ee_traj) < args_cli.num_episodes and simulation_app.is_running():
        ee_pos_w, object_pos_w, goal_pos_w = _current_signals(base_env)
        goal_dist = torch.norm(object_pos_w - goal_pos_w, dim=1)
        active_max_height = torch.maximum(active_max_height, object_pos_w[:, 2])
        active_min_goal_dist = torch.minimum(active_min_goal_dist, goal_dist)
        for env_id in range(num_envs):
            active_ee[env_id].append(ee_pos_w[env_id].detach().cpu().to(torch.float32))

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            for env_id_t in done_ids:
                env_id = int(env_id_t.item())
                if len(collected_ee_traj) >= args_cli.num_episodes:
                    break
                traj = active_ee[env_id]
                if len(traj) == 0:
                    continue

                is_lifted = bool((active_max_height[env_id] > args_cli.success_minimal_height).item())
                if use_goal_distance_for_success:
                    is_success = bool(is_lifted and (active_min_goal_dist[env_id] < args_cli.success_distance_threshold).item())
                else:
                    is_success = is_lifted

                if args_cli.keep_failed or is_success:
                    collected_initial_object_pos.append(active_initial_object_pos[env_id])
                    collected_goal_pos.append(active_goal_pos[env_id])
                    collected_ee_traj.append(torch.stack(traj, dim=0))
                    collected_lengths.append(len(traj))
                    collected_success.append(is_success)

                _, next_object_pos_w, next_goal_pos_w = _current_signals(base_env)
                active_initial_object_pos[env_id] = next_object_pos_w[env_id].detach().cpu().to(torch.float32)
                active_goal_pos[env_id] = next_goal_pos_w[env_id].detach().cpu().to(torch.float32)
                active_ee[env_id] = []
                active_max_height[env_id] = -1.0e9
                active_min_goal_dist[env_id] = 1.0e9

            print(
                f"[INFO] Collected {len(collected_ee_traj)}/{args_cli.num_episodes} "
                f"(successful kept: {sum(collected_success)})"
            )

    if len(collected_ee_traj) == 0:
        raise RuntimeError("No episodes were collected. Try relaxing success thresholds or using --keep_failed.")

    max_len = max(collected_lengths)
    padded = torch.zeros((len(collected_ee_traj), max_len, 3), dtype=torch.float32)
    for i, traj in enumerate(collected_ee_traj):
        padded[i, : traj.shape[0], :] = traj

    dataset = {
        "initial_object_pos": torch.stack(collected_initial_object_pos, dim=0),
        "goal_pos": torch.stack(collected_goal_pos, dim=0),
        "ee_trajectories": padded,
        "trajectory_lengths": torch.tensor(collected_lengths, dtype=torch.long),
        "success": torch.tensor(collected_success, dtype=torch.bool),
        "meta": {
            "task": args_cli.task,
            "checkpoint": str(checkpoint_path),
            "num_envs": num_envs,
            "requested_num_episodes": args_cli.num_episodes,
            "saved_num_episodes": len(collected_ee_traj),
            "success_distance_threshold": float(args_cli.success_distance_threshold),
            "success_minimal_height": float(args_cli.success_minimal_height),
            "use_goal_distance_for_success": bool(use_goal_distance_for_success),
            "keep_failed": bool(args_cli.keep_failed),
        },
    }

    output_path = Path(args_cli.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"[INFO] Saved trajectory dataset to: {output_path}")

    dump_yaml(str(output_path.with_suffix(".env.yaml")), env_cfg)
    dump_yaml(str(output_path.with_suffix(".agent.yaml")), agent_cfg)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
