"""Script to evaluate/play a trained RSL-RL checkpoint.

Usage (from the manipulation/isaac_so_arm101 directory)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/play.py \\
        --task Isaac-SO-ARM101-FixedLayout-Lift-Cube-Play-v0 \\
        --checkpoint logs/rsl_rl/lift_fixed_layout/2026-03-25_20-04-03/model_1499.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a trained RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--disable_task_cameras",
    action="store_true",
    default=False,
    help="Disable TiledCamera sensors in the env (no RGB rendering).",
)
# --- SO-ARM101-specific env overrides ---
parser.add_argument(
    "--trajectory_guidance_fixed_traj_index",
    type=int,
    default=None,
    help="Pin guided-lift tasks to a fixed teacher-trajectory index.",
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
    help="Height threshold (m) above which teacher rewards are suppressed.",
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

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks.lift  # noqa: F401
import isaac_so_arm101.tasks.reach  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from log_paths import rsl_rl_root


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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join(rsl_rl_root(), agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    _apply_so_arm101_overrides(env_cfg, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
