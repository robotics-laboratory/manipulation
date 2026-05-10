"""Collect SO-101 reference trajectories from a dense RSL-RL teacher checkpoint.

Pass the teacher weights with ``--checkpoint`` (defined by ``rsl_rl`` CLI args, same as training).

Each saved episode contains:
- end-effector world pose trajectory (position + quaternion)
- gripper state trajectory (LeRobot motor units)
- cube height above robot base (meters per step; matches success termination geometry)
- initial cube world pose at reset (xyz + xyzw)
- episode metadata (success/timeout/return/steps/task/prompt, height stats)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-RewardDense-v0")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RL agent config entry point.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. Must be 1.")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of trajectories to save.")
parser.add_argument("--max_steps", type=int, default=256, help="Maximum steps per trajectory.")
parser.add_argument("--seed", type=int, default=None, help="Optional environment seed.")
parser.add_argument("--output_dir", type=str, default="rollouts/teacher_trajectories_so101")
parser.add_argument(
    "--auto_match_checkpoint_obs",
    action="store_true",
    default=True,
    help=(
        "If enabled, inspect actor input dim in checkpoint and auto-enable legacy 28-D "
        "dense observations when needed."
    ),
)
parser.add_argument(
    "--force_legacy_dense_obs28",
    action="store_true",
    default=False,
    help="Force legacy 28-D dense observations (adds object_pose command observations).",
)
parser.add_argument(
    "--success_only",
    action="store_true",
    default=False,
    help="Keep only successful episodes in the output set.",
)
parser.add_argument(
    "--height_success_threshold_m",
    type=float,
    default=0.20,
    help=(
        "Cube height above robot base (m) used for success in default lift task; "
        "logged in metadata for comparison only."
    ),
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.checkpoint:
    raise SystemExit(
        "Missing teacher checkpoint: pass --checkpoint /path/to/model.pt "
        "(same flag as other RSL-RL scripts; do not duplicate a second --checkpoint)."
    )

if args_cli.num_envs != 1:
    raise ValueError("Reference trajectory collection currently supports only --num_envs 1.")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
import leisaac.tasks  # noqa: F401


def _find_actor_input_dim(state: object) -> int | None:
    """Recursively find actor.0.weight input width from checkpoint dict."""
    if isinstance(state, Mapping):
        actor_weight = state.get("actor.0.weight")
        if torch.is_tensor(actor_weight) and actor_weight.ndim == 2:
            return int(actor_weight.shape[1])
        for value in state.values():
            found = _find_actor_input_dim(value)
            if found is not None:
                return found
    return None


def _infer_checkpoint_actor_input_dim(checkpoint_path: str) -> int | None:
    """Best-effort checkpoint introspection for teacher actor observation dimension."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return _find_actor_input_dim(checkpoint)


def _enable_legacy_dense_obs28(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Match older dense teacher checkpoints that used 28-D observations."""
    from leisaac.tasks.lift_cube.lift_cube_env_cfg import LiftCubeCommandsCfg

    env_cfg.commands = LiftCubeCommandsCfg()
    env_cfg.observations.policy.object_pose = ObsTerm(
        func=isaaclab_mdp.generated_commands,
        params={"command_name": "object_pose"},
    )


def _ee_frame_index(ee_frame) -> int:
    if ee_frame.data.target_pos_w.shape[1] > 1:
        return 1
    return 0


def _collect_step_state(base_env):
    cube = base_env.scene["cube"]
    robot = base_env.scene["robot"]
    ee_frame = base_env.scene["ee_frame"]
    ee_idx = _ee_frame_index(ee_frame)

    ee_pos_w = ee_frame.data.target_pos_w[0, ee_idx, :3].detach().cpu().numpy().astype(np.float32)
    ee_quat_w = ee_frame.data.target_quat_w[0, ee_idx, :4].detach().cpu().numpy().astype(np.float32)

    joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)[None, :]
    gripper_state = float(convert_leisaac_action_to_lerobot(joint_pos)[0, 5])
    cube_pos = cube.data.root_pos_w[0, :3].detach().cpu().numpy().astype(np.float32)
    cube_quat = cube.data.root_quat_w[0, :4].detach().cpu().numpy().astype(np.float32)
    # Same geometry as leisaac.tasks.lift_cube.mdp.terminations.cube_height_above_base (first env only).
    cube_height_m = float(cube.data.root_pos_w[0, 2].detach().cpu().item())
    base_index = robot.data.body_names.index("base")
    robot_base_height_m = float(robot.data.body_pos_w[0, base_index, 2].detach().cpu().item())
    cube_height_above_base_m = cube_height_m - robot_base_height_m

    return ee_pos_w, ee_quat_w, gripper_state, cube_pos, cube_quat, cube_height_above_base_m


def _save_episode(
    output_dir: Path,
    episode_idx: int,
    *,
    ee_pos_w: list[np.ndarray],
    ee_quat_w: list[np.ndarray],
    gripper_state: list[float],
    cube_height_above_base_m: list[float],
    initial_cube_pose_w: np.ndarray,
    success: bool,
    timeout: bool,
    steps: int,
    episode_return: float,
    task: str,
    prompt: str,
    height_success_threshold_m: float,
    max_cube_height_above_base_m: float,
    final_cube_height_above_base_m: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"trajectory_{episode_idx:06d}.npz"
    meta = {
        "episode": episode_idx,
        "task": task,
        "prompt": prompt,
        "success": bool(success),
        "timeout": bool(timeout),
        "steps": int(steps),
        "return": float(episode_return),
        "height_success_threshold_m": float(height_success_threshold_m),
        "max_cube_height_above_base_m": float(max_cube_height_above_base_m),
        "final_cube_height_above_base_m": float(final_cube_height_above_base_m),
    }
    np.savez_compressed(
        out_path,
        ee_pos_w=np.asarray(ee_pos_w, dtype=np.float32),
        ee_quat_w=np.asarray(ee_quat_w, dtype=np.float32),
        gripper_state=np.asarray(gripper_state, dtype=np.float32),
        cube_height_above_base_m=np.asarray(cube_height_above_base_m, dtype=np.float32),
        initial_cube_pose_w=np.asarray(initial_cube_pose_w, dtype=np.float32),
        meta_json=np.asarray(json.dumps(meta), dtype=np.str_),
    )
    return out_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    resume_path = retrieve_file_path(args_cli.checkpoint)

    use_legacy_obs28 = args_cli.force_legacy_dense_obs28
    checkpoint_obs_dim = None
    if args_cli.auto_match_checkpoint_obs and not use_legacy_obs28:
        checkpoint_obs_dim = _infer_checkpoint_actor_input_dim(resume_path)
        if checkpoint_obs_dim == 28:
            use_legacy_obs28 = True
            print("[collector] Detected teacher actor input dim=28. Enabling legacy dense 28-D observations.")

    if use_legacy_obs28:
        _enable_legacy_dense_obs28(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
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
    try:
        runner.load(resume_path)
    except RuntimeError as exc:
        if "size mismatch for actor.0.weight" in str(exc):
            raise RuntimeError(
                f"{exc}\n\nTeacher checkpoint appears incompatible with current env observations. "
                f"Detected checkpoint actor input dim: {checkpoint_obs_dim}. "
                "Try --force_legacy_dense_obs28 for older dense SO-101 checkpoints."
            ) from exc
        raise
    policy = runner.get_inference_policy(device=base_env.device)

    out_dir = Path(args_cli.output_dir).expanduser()
    summary_path = out_dir / "summary.jsonl"
    prompt = str(getattr(base_env.cfg, "task_description", "Lift the red cube up."))

    obs = env.get_observations()
    collected = 0
    attempted = 0
    waiting_for_new_episode = True
    initial_cube_pose_w = None
    ee_pos_w_buf: list[np.ndarray] = []
    ee_quat_w_buf: list[np.ndarray] = []
    gripper_buf: list[float] = []
    height_above_base_buf: list[float] = []
    episode_return = 0.0
    step_count = 0

    while simulation_app.is_running() and collected < args_cli.num_episodes:
        if waiting_for_new_episode:
            _, _, _, cube_pos, cube_quat, _ = _collect_step_state(base_env)
            initial_cube_pose_w = np.concatenate([cube_pos, cube_quat], axis=0).astype(np.float32)
            ee_pos_w_buf = []
            ee_quat_w_buf = []
            gripper_buf = []
            height_above_base_buf = []
            episode_return = 0.0
            step_count = 0
            waiting_for_new_episode = False

        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)

        ee_pos_w, ee_quat_w, gripper_state, _, _, h_ab = _collect_step_state(base_env)
        ee_pos_w_buf.append(ee_pos_w)
        ee_quat_w_buf.append(ee_quat_w)
        gripper_buf.append(gripper_state)
        height_above_base_buf.append(h_ab)

        reward_value = float(rewards[0].detach().cpu().item()) if torch.is_tensor(rewards) else float(rewards[0])
        episode_return += reward_value
        step_count += 1

        done = bool(dones[0].detach().cpu().item()) if torch.is_tensor(dones) else bool(dones[0])
        force_timeout = step_count >= args_cli.max_steps
        if force_timeout and not done:
            obs, info = env.reset()
            done = True

        if done:
            attempted += 1
            success = bool(base_env.reset_terminated[0].detach().cpu().item())
            timeout = bool(base_env.reset_time_outs[0].detach().cpu().item()) or force_timeout
            max_h_ab = max(height_above_base_buf) if height_above_base_buf else float("nan")
            final_h_ab = height_above_base_buf[-1] if height_above_base_buf else float("nan")
            keep = (not args_cli.success_only) or success
            if keep:
                out_path = _save_episode(
                    out_dir,
                    collected,
                    ee_pos_w=ee_pos_w_buf,
                    ee_quat_w=ee_quat_w_buf,
                    gripper_state=gripper_buf,
                    cube_height_above_base_m=height_above_base_buf,
                    initial_cube_pose_w=initial_cube_pose_w,
                    success=success,
                    timeout=timeout,
                    steps=step_count,
                    episode_return=episode_return,
                    task=args_cli.task,
                    prompt=prompt,
                    height_success_threshold_m=args_cli.height_success_threshold_m,
                    max_cube_height_above_base_m=max_h_ab,
                    final_cube_height_above_base_m=final_h_ab,
                )
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "episode": collected,
                                "attempted_episode": attempted - 1,
                                "success": success,
                                "timeout": timeout,
                                "steps": step_count,
                                "return": episode_return,
                                "trajectory_path": str(out_path),
                                "task": args_cli.task,
                                "prompt": prompt,
                                "height_success_threshold_m": args_cli.height_success_threshold_m,
                                "max_cube_height_above_base_m": max_h_ab,
                                "final_cube_height_above_base_m": final_h_ab,
                            }
                        )
                        + "\n"
                    )
                print(
                    f"[trajectory {collected}] success={success} timeout={timeout} steps={step_count} "
                    f"return={episode_return:.4f} max_h_ab={max_h_ab:.4f}m final_h_ab={final_h_ab:.4f}m "
                    f"(thresh={args_cli.height_success_threshold_m:.3f}m) saved={out_path}"
                )
                collected += 1
            else:
                print(
                    f"[trajectory skip] attempted={attempted - 1} success={success} timeout={timeout} "
                    f"steps={step_count} return={episode_return:.4f} "
                    f"max_h_ab={max_h_ab:.4f}m final_h_ab={final_h_ab:.4f}m"
                )
            waiting_for_new_episode = True

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
