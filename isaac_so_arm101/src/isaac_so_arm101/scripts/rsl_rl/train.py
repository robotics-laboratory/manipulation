"""Script to train RL agent with RSL-RL.

Usage (from the manipulation/isaac_so_arm101 directory)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/train.py \\
        --task Isaac-SO-ARM101-FixedLayout-Lift-Cube-v0 \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--disable_task_cameras",
    action="store_true",
    default=False,
    help="Disable TiledCamera sensors in the env (faster MLP training, no RGB rendering).",
)
# --- SO-ARM101-specific env overrides ---
parser.add_argument(
    "--trajectory_guidance_fixed_traj_index",
    type=int,
    default=None,
    help="Pin guided-lift tasks to a fixed teacher-trajectory index (default: sample randomly).",
)
parser.add_argument(
    "--suppress_dense_teacher_rewards_after_lift",
    action="store_true",
    default=False,
    help="Zero out dense teacher-guidance rewards once the cube has been lifted.",
)
parser.add_argument(
    "--lift_suppression_min_height",
    type=float,
    default=None,
    help="Height threshold (m) above which teacher rewards are suppressed (requires --suppress_dense_teacher_rewards_after_lift).",
)
parser.add_argument(
    "--suppress_dense_teacher_after_lift_scope",
    type=str,
    default=None,
    choices=["episode", "step"],
    help="Whether suppression persists for the rest of the episode or only the current step.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks.lift  # noqa: F401
import isaac_so_arm101.tasks.reach  # noqa: F401
from isaac_so_arm101.tasks.lift.lift_env_cfg import apply_disable_task_cameras_if_set

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from log_paths import rsl_rl_root

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _apply_so_arm101_overrides(env_cfg, args_cli) -> None:
    """Apply SO-ARM101 custom env-cfg overrides from CLI args (no-op for other tasks)."""
    if args_cli.disable_task_cameras and hasattr(env_cfg, "disable_task_cameras"):
        env_cfg.disable_task_cameras = True
    if args_cli.trajectory_guidance_fixed_traj_index is not None and hasattr(
        env_cfg, "trajectory_guidance_fixed_traj_index"
    ):
        env_cfg.trajectory_guidance_fixed_traj_index = args_cli.trajectory_guidance_fixed_traj_index
    if args_cli.suppress_dense_teacher_rewards_after_lift and hasattr(
        env_cfg, "suppress_dense_teacher_rewards_after_lift"
    ):
        env_cfg.suppress_dense_teacher_rewards_after_lift = True
    if args_cli.lift_suppression_min_height is not None and hasattr(env_cfg, "lift_suppression_min_height"):
        env_cfg.lift_suppression_min_height = args_cli.lift_suppression_min_height
    if args_cli.suppress_dense_teacher_after_lift_scope is not None and hasattr(
        env_cfg, "suppress_dense_teacher_after_lift_scope"
    ):
        env_cfg.suppress_dense_teacher_after_lift_scope = args_cli.suppress_dense_teacher_after_lift_scope
    # CLI toggles disable_task_cameras after Hydra built the cfg; re-apply camera stripping.
    apply_disable_task_cameras_if_set(env_cfg)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join(rsl_rl_root(), agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = False
    _apply_so_arm101_overrides(env_cfg, args_cli)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
