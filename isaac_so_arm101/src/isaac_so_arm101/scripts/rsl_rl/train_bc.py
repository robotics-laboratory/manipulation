#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Supervised warm-start (behavioral cloning) for an RSL-RL actor prior to PPO."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip
import isaac_so_arm101.scripts.rsl_rl.log_paths as log_paths  # isort: skip

parser = argparse.ArgumentParser(
    description="Train student actor with MSE to mimic expert actions from collect_bc_dataset.py."
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to a .pt file saved by collect_bc_dataset.py.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments (for policy construction).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (student / downstream env).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--disable_task_cameras",
    action="store_true",
    default=False,
    help="Disable task camera sensors and image observation terms in the env config.",
)
parser.add_argument("--bc_epochs", type=int, default=50, help="Passes over the dataset.")
parser.add_argument("--batch_size", type=int, default=4096, help="Minibatch size on device.")
parser.add_argument("--bc_lr", type=float, default=3e-4, help="Adam LR for BC (actor-only gradients).")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output checkpoint .pt (RSL-RL format). Default: isaac_so_arm101/logs/rsl_rl/bc/<experiment>_bc_pretrained.pt",
)
parser.add_argument(
    "--include_failed_episode_transitions",
    action="store_true",
    default=False,
    help="If set, train on all transitions. Default: use only transitions from successful episodes when the "
    "dataset provides `transition_in_success_episode`.",
)
parser.add_argument(
    "--skip_obs_groups_check",
    action="store_true",
    default=False,
    help="If set, do not compare dataset meta obs_groups to the current runner.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Workaround: isaaclab is a namespace package (no __file__).
# tensordict → torch._dynamo → inspect.getfile crashes on namespace modules.
import isaaclab as _isaaclab_ns

if not getattr(_isaaclab_ns, "__file__", None):
    _isaaclab_ns.__file__ = next(iter(_isaaclab_ns.__path__), __file__)

import gymnasium as gym
import torch
import torch.nn.functional as F
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml
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


def _zero_non_actor_grads(policy) -> None:
    """Keep only actor MLP gradients; leave critic / noise std untouched."""
    for name, p in policy.named_parameters():
        if p.grad is None:
            continue
        if name.startswith("critic"):
            p.grad.zero_()
            continue
        # scalar or per-dim exploration std
        if name == "std" or name == "log_std":
            p.grad.zero_()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"BC training expects OnPolicyRunner agent config, got {agent_cfg.class_name}")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.algorithm.learning_rate = args_cli.bc_lr

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.disable_task_cameras and hasattr(env_cfg, "disable_task_cameras"):
        env_cfg.disable_task_cameras = True
    if args_cli.disable_task_cameras:
        _disable_task_cameras_in_env_cfg(env_cfg)

    dataset_path = Path(args_cli.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"BC dataset not found: {dataset_path}")

    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if "actor_obs" not in payload or "actions" not in payload:
        raise KeyError("Dataset must contain 'actor_obs' and 'actions' tensors (use collect_bc_dataset.py).")

    actor_obs_cpu: torch.Tensor = payload["actor_obs"].to(torch.float32)
    actions_cpu: torch.Tensor = payload["actions"].to(torch.float32)
    if actor_obs_cpu.ndim != 2 or actions_cpu.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got {actor_obs_cpu.shape=} {actions_cpu.shape=}")

    use_success_mask = bool(payload.get("transition_in_success_episode") is not None)
    if use_success_mask and not args_cli.include_failed_episode_transitions:
        mask = payload["transition_in_success_episode"].to(torch.bool)
        actor_obs_cpu = actor_obs_cpu[mask]
        actions_cpu = actions_cpu[mask]
        print(f"[INFO] Filtered to {actor_obs_cpu.shape[0]} transitions from successful episodes.")
    elif use_success_mask and args_cli.include_failed_episode_transitions:
        print("[INFO] Using all transitions (including failed episodes) per --include_failed_episode_transitions.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = os.path.join(log_paths.rsl_rl_root(), "bc", agent_cfg.experiment_name)
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir

    train_cfg = agent_cfg.to_dict()
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)
    policy = runner.alg.policy
    if not hasattr(policy, "actor") or not hasattr(policy, "actor_obs_normalizer"):
        raise RuntimeError("Unexpected policy layout (expected ActorCritic with actor + actor_obs_normalizer).")

    meta = payload.get("meta") or {}
    if not args_cli.skip_obs_groups_check and "obs_groups" in meta:
        dg = {k: list(v) for k, v in meta["obs_groups"].items()}
        cg = {k: list(v) for k, v in runner.cfg["obs_groups"].items()}
        if dg != cg:
            raise ValueError(
                "Dataset obs_groups differ from current runner obs_groups. "
                f"dataset={dg} current={cg}. Align task/camera settings with collection, or pass "
                "--skip_obs_groups_check if you know the shapes still match."
            )

    exp_action_dim = env.num_actions
    if actions_cpu.shape[-1] != exp_action_dim:
        raise ValueError(
            f"Action dim mismatch: dataset {actions_cpu.shape[-1]} vs env {exp_action_dim}. "
            "Use the same task/agent as the teacher collection."
        )

    # Infer actor observation size from first policy forward
    with torch.no_grad():
        probe = runner.env.get_observations()
        d_actor = policy.get_actor_obs(probe).shape[-1]
    if actor_obs_cpu.shape[-1] != d_actor:
        raise ValueError(
            f"Actor obs dim mismatch: dataset {actor_obs_cpu.shape[-1]} vs policy {d_actor}. "
            "Collect BC data on the same task/camera setup as this student config."
        )

    device = agent_cfg.device
    n = actor_obs_cpu.shape[0]
    if n == 0:
        raise RuntimeError("No transitions after filtering; relax filters or regenerate the dataset.")

    runner.train_mode()
    for epoch in range(args_cli.bc_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, n, args_cli.batch_size):
            idx = perm[start : start + args_cli.batch_size]
            obs_b = actor_obs_cpu[idx].to(device, non_blocking=True)
            tgt_b = actions_cpu[idx].to(device, non_blocking=True)

            runner.alg.optimizer.zero_grad(set_to_none=True)
            pred = policy.actor(policy.actor_obs_normalizer(obs_b))
            loss = F.mse_loss(pred, tgt_b)
            loss.backward()
            _zero_non_actor_grads(policy)
            runner.alg.optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        print(f"[INFO] BC epoch {epoch + 1}/{args_cli.bc_epochs} | mean MSE {mean_loss:.6f}")

    out_path = args_cli.output
    if not out_path:
        out_path = os.path.join(log_dir, f"{agent_cfg.experiment_name}_bc_pretrained.pt")
    out_path = os.path.abspath(out_path)
    _out_dir = os.path.dirname(out_path)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    runner.current_learning_iteration = 0
    # OnPolicyRunner sets `logger_type` / `disable_logs` inside learn() -> _prepare_logging_writer().
    # BC never calls learn(), but save() still branches on logger_type for wandb/neptune upload.
    if not hasattr(runner, "logger_type"):
        runner.logger_type = str(train_cfg.get("logger", "tensorboard")).lower()
    if not hasattr(runner, "disable_logs"):
        runner.disable_logs = False
    runner.save(out_path, infos={"bc_dataset": str(dataset_path), "bc_epochs": args_cli.bc_epochs})
    print(f"[INFO] Saved BC warm-start checkpoint to: {out_path}")

    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    dump_yaml(os.path.join(params_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(params_dir, "agent.yaml"), agent_cfg)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
