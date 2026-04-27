# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a LeRobot dataset by rolling out an RSL-RL policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Collect a LeRobot dataset with an RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Must be 1.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--num_episodes", type=int, default=50, help="Number of successful episodes to record.")
parser.add_argument(
    "--repo_id",
    type=str,
    default="igor-saprygin/so101-lift-cube",
    help="LeRobot dataset repository ID or local dataset ID.",
)
parser.add_argument("--fps", type=int, default=30, help="LeRobot dataset frames per second.")
parser.add_argument(
    "--success_only",
    action="store_true",
    default=False,
    help="Record successful episodes only. LeRobot recording currently requires this mode.",
)
parser.add_argument(
    "--no_overwrite",
    action="store_true",
    default=False,
    help="Fail if the local LeRobot dataset already exists instead of overwriting it.",
)
parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push the dataset to Hugging Face Hub.")
parser.add_argument("--max_steps", type=int, default=0, help="Optional global rollout step limit. Set 0 for unlimited.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs != 1:
    raise ValueError("LeRobotRecorderManager records env index 0 only; use --num_envs 1 for dataset collection.")
if not args_cli.success_only:
    print("[INFO] LeRobot recorder exports successful episodes only; proceeding in success-only mode.")
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import shutil
import torch
from pathlib import Path

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.assets import retrieve_file_path

try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    get_published_pretrained_checkpoint = None

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
import leisaac.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _local_lerobot_dataset_path(repo_id: str) -> Path:
    if not repo_id:
        raise ValueError("--repo_id is required for LeRobot dataset collection.")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "lerobot" / repo_id


def _prepare_lerobot_dataset_path(repo_id: str) -> None:
    dataset_path = _local_lerobot_dataset_path(repo_id)
    if not dataset_path.exists():
        return
    if args_cli.no_overwrite:
        raise FileExistsError(
            f"Local LeRobot dataset already exists: {dataset_path}. "
            "Remove --no_overwrite or choose a different --repo_id."
        )
    shutil.rmtree(dataset_path)
    print(f"[INFO] Overwriting existing local LeRobot dataset: {dataset_path}")


def _replace_lerobot_recorder(env, env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Replace Isaac Lab's recorder with the LeRobot recorder used by LeIsaac."""
    if env_cfg.recorders is None:
        raise ValueError("The collection environment must define recorder terms.")

    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    _prepare_lerobot_dataset_path(args_cli.repo_id)
    if hasattr(env.unwrapped, "recorder_manager"):
        del env.unwrapped.recorder_manager

    dataset_cfg = LeRobotDatasetCfg(repo_id=args_cli.repo_id, fps=args_cli.fps)
    env.unwrapped.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env.unwrapped)


def _push_dataset_to_hub(env) -> None:
    dataset = env.unwrapped.recorder_manager._dataset_file_handler._lerobot_dataset
    if hasattr(dataset, "push_to_hub"):
        dataset.push_to_hub()
    else:
        print("[WARN] The active LeRobot dataset object does not expose push_to_hub(); skipping upload.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect successful LeRobot episodes with an RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Collect", "").replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise RuntimeError(
                "--use_pretrained_checkpoint is not supported by this IsaacLab install. "
                "Pass an explicit --checkpoint path instead."
            )
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    _replace_lerobot_recorder(env, env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    total_steps = 0
    last_success_count = env.unwrapped.recorder_manager.exported_successful_episode_count

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            total_steps += 1
            success_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if success_count > last_success_count:
                last_success_count = success_count
                print(f"[INFO] Recorded {success_count}/{args_cli.num_episodes} successful episodes.")

            if success_count >= args_cli.num_episodes:
                print(f"[INFO] Finished recording {args_cli.num_episodes} successful episodes.")
                break
            if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
                print(f"[WARN] Reached --max_steps={args_cli.max_steps} before collecting all requested episodes.")
                break
    finally:
        if hasattr(env.unwrapped.recorder_manager, "finalize"):
            env.unwrapped.recorder_manager.finalize()
        if args_cli.push_to_hub:
            _push_dataset_to_hub(env)
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
