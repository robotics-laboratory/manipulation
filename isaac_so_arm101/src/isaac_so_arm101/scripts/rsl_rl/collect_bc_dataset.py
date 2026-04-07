"""Collect a behavioural cloning dataset using a trained RSL-RL MLP agent.

Records observations and actions into a simple numpy (.npz) dataset,
one file per episode. Suitable for lightweight BC training with train_bc.py.

Usage (from the manipulation/isaac_so_arm101 directory)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/collect_bc_dataset.py \\
        --task Isaac-SO-ARM101-FixedLayout-Lift-Cube-Play-v0 \\
        --checkpoint logs/rsl_rl/lift_fixed_layout/2026-03-25_20-04-03/model_1499.pt \\
        --num_episodes 200 \\
        --output_dir datasets/bc_lift_cube \\
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect a BC dataset from a trained RSL-RL agent.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-FixedLayout-Lift-Cube-Play-v0")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point name."
)
parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes to collect.")
parser.add_argument("--output_dir", type=str, default="datasets/bc_lift_cube", help="Output directory.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--success_only",
    action="store_true",
    default=False,
    help="Only save episodes where the cube was lifted.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (1 recommended).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks.lift  # noqa: F401
import isaac_so_arm101.tasks.reach  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from log_paths import rsl_rl_root


def _cube_is_lifted(env_unwrapped, min_height: float = 0.025) -> bool:
    obj = env_unwrapped.scene.rigid_objects["object"]
    return obj.data.root_pos_w[0, 2].item() > min_height


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    log_root = os.path.abspath(os.path.join(rsl_rl_root(), agent_cfg.experiment_name))

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Saving episodes to: {output_dir}")

    obs = env.get_observations()
    episodes_saved = 0
    episodes_attempted = 0
    ep_obs: list[np.ndarray] = []
    ep_actions: list[np.ndarray] = []
    episode_had_lift = False

    while episodes_saved < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        ep_obs.append(obs[0].cpu().numpy())
        ep_actions.append(actions[0].cpu().numpy())

        if _cube_is_lifted(env.unwrapped):
            episode_had_lift = True

        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if dones.any():
            episodes_attempted += 1
            if args_cli.success_only and not episode_had_lift:
                print(f"  Episode {episodes_attempted}: no lift – discarded")
            else:
                fname = os.path.join(output_dir, f"episode_{episodes_saved:05d}.npz")
                np.savez_compressed(
                    fname,
                    observations=np.stack(ep_obs),
                    actions=np.stack(ep_actions),
                )
                episodes_saved += 1
                tag = "LIFTED" if episode_had_lift else "no lift"
                print(f"  Episode {episodes_saved}/{args_cli.num_episodes}: {tag} → {fname}")

            ep_obs = []
            ep_actions = []
            episode_had_lift = False

    print(f"\n[DONE] {episodes_saved} episodes saved to {output_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
