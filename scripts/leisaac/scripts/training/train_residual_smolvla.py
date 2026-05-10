"""Residual RL training scaffold on top of LeIsaac SmolVLA policy inference.

Purpose:
    Provide project-aligned boilerplate so you can focus on core logic:
      1) base policy query
      2) observation encoding for residual learner
      3) reward composition (sparse + guidance + residual regularization)

Usage (example):
    isaaclab -p manipulation/scripts/leisaac/scripts/training/train_residual_smolvla.py \
        --task LeIsaac-SO101-LiftCube-v0 \
        --policy_type lerobot-smolvla \
        --policy_checkpoint_path <path_or_hf_repo>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher

from trajectory_guidance_reward import (
    GuidanceRewardCoefficients,
    TrajectoryGuidanceState,
    compute_guidance_reward,
    load_reference_trajectories,
    select_reference_trajectory,
)

# IMPORTANT: keep arg names aligned with policy_inference.py when possible
parser = argparse.ArgumentParser(description="Residual RL scaffold with SmolVLA base policy.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-v0")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--episode_length_s", type=float, default=5.0)
parser.add_argument("--max_env_steps", type=int, default=300_000)
parser.add_argument("--output_dir", type=str, default="manipulation/rollouts/residual_smolvla")

# Base policy options (mirrors evaluation/policy_inference.py)
parser.add_argument("--policy_type", type=str, default="lerobot-smolvla")
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=5555)
parser.add_argument("--policy_timeout_ms", type=int, default=15000)
parser.add_argument("--policy_action_horizon", type=int, default=16)
parser.add_argument("--policy_language_instruction", type=str, default="Lift the red cube up.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None)
parser.add_argument("--policy_must_go", action="store_true", default=False)

# Guidance reward options
parser.add_argument("--traj_db", type=str, default=None)
parser.add_argument("--traj_sampling_strategy", choices=("round_robin", "random"), default="round_robin")
parser.add_argument("--guidance_seed", type=int, default=7)
parser.add_argument("--guidance_progress_weight", type=float, default=1.0)
parser.add_argument("--guidance_xy_weight", type=float, default=0.25)
parser.add_argument("--guidance_gripper_weight", type=float, default=0.25)
parser.add_argument("--guidance_xy_scale", type=float, default=0.08)
parser.add_argument("--guidance_gripper_scale", type=float, default=15.0)
parser.add_argument("--guidance_progress_power", type=float, default=1.0)

# Residual learner config overrides
parser.add_argument("--obs_dim", type=int, default=6)
parser.add_argument("--act_dim", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--warmup_steps", type=int, default=5000)
parser.add_argument("--utd", type=int, default=4)
parser.add_argument("--lambda_guidance", type=float, default=0.5)
parser.add_argument("--lambda_residual_l2", type=float, default=1.0e-3)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from leisaac.policy.service_policy_clients import LeRobotServicePolicyClient
from leisaac.rl.residual import ResidualLearner, ResidualRLConfig, ReplayBuffer, TransitionBatch

import leisaac  # noqa: F401


class 

def configure_lerobot_absolute_joint_actions(env_cfg, task_type: str) -> None:
    """Apply LeRobot actions as absolute joint targets after motor-to-joint conversion."""
    if task_type != "so101leader":
        return
    for action_name in ("arm_action", "gripper_action"):
        action_cfg = getattr(env_cfg.actions, action_name, None)
        if action_cfg is not None and hasattr(action_cfg, "use_default_offset"):
            action_cfg.use_default_offset = False
            if hasattr(action_cfg, "scale"):
                action_cfg.scale = 1.0


def make_env() -> tuple[Any, str]:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    configure_lerobot_absolute_joint_actions(env_cfg, task_type)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.recorders = None
    sim_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    return sim_env, task_type


def make_base_policy(env, task_type: str) -> LeRobotServicePolicyClient:
    """Construct base policy client for SmolVLA via LeRobot server."""
    from isaaclab.sensors import Camera

    camera_infos = {
        key: sensor.image_shape for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)
    }
    return LeRobotServicePolicyClient(
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        timeout_ms=args_cli.policy_timeout_ms,
        camera_infos=camera_infos,
        task_type=task_type,
        policy_type="smolvla",
        pretrained_name_or_path=args_cli.policy_checkpoint_path,
        actions_per_chunk=args_cli.policy_action_horizon,
        force_must_go=args_cli.policy_must_go,
        device=args_cli.device,
    )


def preprocess_policy_obs(policy_obs: dict, language_instruction: str) -> dict:
    """Adapt env policy observations for LeRobot service client."""
    out = {k: policy_obs[k] for k in policy_obs.keys()}
    out["task_description"] = language_instruction
    return out


def obs_to_residual_vector(policy_obs: dict) -> np.ndarray:
    """Convert env observation dict to residual learner state vector.

    TODO:
        Replace this baseline with the state encoding you actually want.
        Recommended first pass: converted SO-101 joint positions (6D) in same
        convention your base policy uses.
    """
    joint_pos = policy_obs["joint_pos"][0].detach().cpu().numpy().astype(np.float32)
    return joint_pos[: args_cli.obs_dim]


def base_chunk_to_numpy(action_chunk: torch.Tensor) -> np.ndarray:
    """Convert base policy chunk [T, 1, A] -> [T, A]."""
    action_chunk = action_chunk.detach().cpu().numpy()
    if action_chunk.ndim == 3 and action_chunk.shape[1] == 1:
        action_chunk = action_chunk[:, 0, :]
    return np.asarray(action_chunk, dtype=np.float32)


def current_ee_position_w(env) -> np.ndarray:
    ee_frame = env.scene["ee_frame"]
    ee_index = 1 if ee_frame.data.target_pos_w.shape[1] > 1 else 0
    return ee_frame.data.target_pos_w[0, ee_index, :3].detach().cpu().numpy().astype(np.float32)


def extract_gripper_scalar(policy_obs: dict) -> float:
    """Extract scalar gripper state from policy obs.

    TODO:
        Ensure this matches the same convention as trajectory references.
    """
    return float(policy_obs["joint_pos"][0, -1].detach().cpu().item())


def compose_reward(
    *,
    env_reward: float,
    guidance_reward: float,
    residual_action: np.ndarray,
) -> float:
    """Compose total learner reward.

    TODO:
        Tune weights and optionally add additional task-specific terms.
    """
    residual_l2 = float(np.sum(np.square(residual_action)))
    return env_reward + args_cli.lambda_guidance * guidance_reward - args_cli.lambda_residual_l2 * residual_l2


def save_checkpoint(output_dir: Path, env_step: int, learner: ResidualLearner, replay: ReplayBuffer) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"residual_step_{env_step:08d}.pt"
    torch.save(
        {
            "env_step": env_step,
            "learner": learner.state_dict(),
            # Keep replay metadata only; full replay dump can be very large.
            "replay_size": len(replay),
            "args": vars(args_cli),
        },
        ckpt_path,
    )
    return ckpt_path


def main() -> None:
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    sim_env, task_type = make_env()
    env = sim_env.unwrapped
    base_policy = make_base_policy(env, task_type)

    cfg = ResidualRLConfig(
        task=args_cli.task,
        seed=args_cli.seed,
        device=args_cli.device,
        max_env_steps=args_cli.max_env_steps,
        warmup_steps=args_cli.warmup_steps,
        action_horizon=args_cli.policy_action_horizon,
        obs_dim=args_cli.obs_dim,
        act_dim=args_cli.act_dim,
        batch_size=args_cli.batch_size,
        utd=args_cli.utd,
        lambda_guidance=args_cli.lambda_guidance,
        lambda_residual_l2=args_cli.lambda_residual_l2,
        output_dir=args_cli.output_dir,
    )
    learner = ResidualLearner(cfg)
    replay = ReplayBuffer(capacity=cfg.replay_capacity, obs_dim=cfg.obs_dim, act_dim=cfg.act_dim)

    # Optional guidance setup
    guidance_state: TrajectoryGuidanceState | None = None
    guidance_coeffs = GuidanceRewardCoefficients(
        progress_weight=args_cli.guidance_progress_weight,
        xy_weight=args_cli.guidance_xy_weight,
        gripper_weight=args_cli.guidance_gripper_weight,
        xy_scale=args_cli.guidance_xy_scale,
        gripper_scale=args_cli.guidance_gripper_scale,
        progress_power=args_cli.guidance_progress_power,
    )
    guidance_rng = np.random.default_rng(args_cli.guidance_seed)
    guidance_refs = []
    if args_cli.traj_db:
        guidance_refs = load_reference_trajectories(args_cli.traj_db)

    output_dir = Path(args_cli.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"

    env_step = 0
    episode_idx = 0
    obs_dict, _ = sim_env.reset(seed=args_cli.seed)
    base_policy.reset()

    # Initialize guidance trajectory for episode
    if guidance_refs:
        selected = select_reference_trajectory(
            guidance_refs,
            episode_idx,
            strategy=args_cli.traj_sampling_strategy,
            rng=guidance_rng,
        )
        guidance_state = TrajectoryGuidanceState(traj=selected)

    try:
        while simulation_app.is_running() and env_step < cfg.max_env_steps:
            policy_obs = preprocess_policy_obs(obs_dict["policy"], args_cli.policy_language_instruction)

            # Base chunk [T, 1, A]
            base_chunk = base_policy.get_action(policy_obs).to(env.device)
            base_chunk_np = base_chunk_to_numpy(base_chunk)

            # Execute chunk step-by-step
            for chunk_i in range(min(cfg.action_horizon, base_chunk_np.shape[0])):
                if env_step >= cfg.max_env_steps:
                    break

                curr_policy_obs = obs_dict["policy"]
                obs_vec = obs_to_residual_vector(curr_policy_obs)
                base_action = base_chunk_np[chunk_i]

                obs_t = torch.as_tensor(obs_vec[None, :], device=learner.device, dtype=torch.float32)
                base_t = torch.as_tensor(base_action[None, :], device=learner.device, dtype=torch.float32)
                with torch.no_grad():
                    residual_t = learner.infer_residual(obs_t, base_t)
                residual = residual_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

                alpha = cfg.residual_alpha(env_step)
                exec_action = learner.compose_action(
                    base_action=base_t,
                    residual_action=residual_t,
                    alpha=alpha,
                ).squeeze(0)

                # Keep existing leisaac runtime behavior
                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, task_type)

                next_obs_dict, reward, terminated, truncated, _ = sim_env.step(exec_action[None, :])

                env_reward = float(reward[0].detach().cpu().item())
                done = bool(terminated[0].detach().cpu().item() or truncated[0].detach().cpu().item())

                # Guidance reward (optional)
                guidance_reward = 0.0
                if guidance_state is not None:
                    next_policy_obs = next_obs_dict["policy"]
                    guidance_step = compute_guidance_reward(
                        guidance_state,
                        current_ee_pos_w=current_ee_position_w(env),
                        current_gripper_state=extract_gripper_scalar(next_policy_obs),
                        coeffs=guidance_coeffs,
                    )
                    guidance_reward = float(guidance_step.reward)

                total_reward = compose_reward(
                    env_reward=env_reward,
                    guidance_reward=guidance_reward,
                    residual_action=residual,
                )

                # Need next base action for residual Bellman target.
                next_policy_obs_for_base = preprocess_policy_obs(
                    next_obs_dict["policy"],
                    args_cli.policy_language_instruction,
                )
                next_base_chunk = base_policy.get_action(next_policy_obs_for_base)
                next_base_chunk_np = base_chunk_to_numpy(next_base_chunk)
                next_base_action = next_base_chunk_np[0]

                transition = TransitionBatch(
                    obs=obs_vec.astype(np.float32),
                    base_action=base_action.astype(np.float32),
                    exec_action=exec_action.detach().cpu().numpy().astype(np.float32),
                    reward=float(total_reward),
                    next_obs=obs_to_residual_vector(next_obs_dict["policy"]).astype(np.float32),
                    next_base_action=next_base_action.astype(np.float32),
                    done=done,
                )
                replay.add(transition)

                # Learner updates
                stats = None
                if env_step >= cfg.warmup_steps and len(replay) >= cfg.batch_size:
                    for _ in range(cfg.utd):
                        batch = replay.sample(batch_size=cfg.batch_size, device=learner.device)
                        stats = learner.update(batch)

                if env_step % cfg.log_interval == 0:
                    payload = {
                        "env_step": env_step,
                        "episode_idx": episode_idx,
                        "alpha": alpha,
                        "replay_size": len(replay),
                        "env_reward": env_reward,
                        "guidance_reward": guidance_reward,
                        "total_reward": total_reward,
                        "done": done,
                    }
                    if stats is not None:
                        payload.update(
                            {
                                "critic_loss": stats.critic_loss,
                                "actor_loss": stats.actor_loss,
                                "q1_mean": stats.q1_mean,
                                "q2_mean": stats.q2_mean,
                                "target_q_mean": stats.target_q_mean,
                            }
                        )
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")

                if env_step > 0 and env_step % cfg.checkpoint_interval == 0:
                    ckpt = save_checkpoint(output_dir, env_step, learner, replay)
                    print(f"[checkpoint] saved {ckpt}")

                env_step += 1
                obs_dict = next_obs_dict

                if done:
                    episode_idx += 1
                    obs_dict, _ = sim_env.reset()
                    base_policy.reset()

                    # Start new guidance episode
                    if guidance_refs:
                        selected = select_reference_trajectory(
                            guidance_refs,
                            episode_idx,
                            strategy=args_cli.traj_sampling_strategy,
                            rng=guidance_rng,
                        )
                        guidance_state = TrajectoryGuidanceState(traj=selected)
                    else:
                        guidance_state = None
                    break

        final_ckpt = save_checkpoint(output_dir, env_step, learner, replay)
        print(f"[done] env_step={env_step} final_checkpoint={final_ckpt}")
    finally:
        sim_env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
