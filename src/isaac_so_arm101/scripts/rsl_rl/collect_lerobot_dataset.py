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
parser.add_argument("--task", type=str, default=None, help="Name of the collection task.")
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
    help="Backward-compatible alias for --target_repo_id.",
)
parser.add_argument(
    "--source_repo_id",
    type=str,
    default=None,
    help=(
        "Optional source LeRobot dataset repo to copy before collection. "
        "Use this to create base-plus-extension ablation datasets."
    ),
)
parser.add_argument(
    "--target_repo_id",
    type=str,
    default=None,
    help="LeRobot dataset repo to write. Defaults to --source_repo_id when provided, otherwise --repo_id.",
)
parser.add_argument("--fps", type=int, default=30, help="LeRobot dataset frames per second.")
parser.add_argument(
    "--step_hz",
    type=float,
    default=30.0,
    help=(
        "Wall-clock environment stepping rate during collection. This gives RTX cameras time to produce fresh frames. "
        "Keep this aligned with --fps for honest LeRobot timing."
    ),
)
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
parser.add_argument(
    "--resume_dataset",
    action="store_true",
    default=False,
    help="Append successful episodes to an existing local LeRobot dataset instead of recreating it.",
)
parser.add_argument("--push_to_hub", action="store_true", default=False, help="Push the dataset to Hugging Face Hub.")
parser.add_argument("--max_steps", type=int, default=0, help="Optional global rollout step limit. Set 0 for unlimited.")
parser.add_argument(
    "--post_success_steps",
    type=int,
    default=25,
    help="Number of additional control steps to record after first detecting task success.",
)
parser.add_argument(
    "--max_action_delta",
    type=float,
    default=0.0,
    help=(
        "Optional per-step delta limit for policy actions before env.step(). "
        "Set >0 to slow/smooth teacher arm motion; 0 disables limiting."
    ),
)
parser.add_argument(
    "--limit_gripper_delta",
    action="store_true",
    default=False,
    help="Also apply --max_action_delta to the final gripper action dimension. By default only arm dims are limited.",
)
parser.add_argument(
    "--manual_decision",
    action="store_true",
    default=False,
    help="Disable automatic success/reset. Press N to mark success+reset, R to reset/skip the current episode.",
)
parser.add_argument(
    "--post_reset_warmup_steps",
    type=int,
    default=20,
    help=(
        "In manual decision mode, render this many post-reset frames before stepping the teacher again. "
        "This avoids recording stale RTX/reset frames in very short episodes."
    ),
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.target_repo_id is None:
    args_cli.target_repo_id = args_cli.source_repo_id or args_cli.repo_id
if args_cli.source_repo_id is None:
    args_cli.source_repo_id = args_cli.target_repo_id
args_cli.repo_id = args_cli.target_repo_id

if args_cli.num_envs != 1:
    raise ValueError("LeRobotRecorderManager records env index 0 only; use --num_envs 1 for dataset collection.")
if not args_cli.success_only:
    print("[INFO] LeRobot recorder exports successful episodes only; proceeding in success-only mode.")
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import shutil
import time
from pathlib import Path

import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import DatasetExportMode, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from leisaac.enhance.datasets.lerobot_dataset_handler import LeRobotDatasetCfg
from leisaac.enhance.managers import EnhanceDatasetExportMode
from leisaac.enhance.managers.lerobot_recorder_manager import LeRobotRecorderManager
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    try:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    except ModuleNotFoundError:
        get_published_pretrained_checkpoint = None

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
import leisaac.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from leisaac.tasks.lift_cube import mdp as lift_cube_mdp


class RateLimiter:
    """Match teleop collection pacing and keep RTX sensors rendering between control steps."""

    def __init__(self, hz: float):
        if hz <= 0.0:
            raise ValueError("--step_hz must be positive.")
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def _lerobot_cache_path(repo_id: str) -> Path:
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return root / "lerobot" / repo_id


def _prepare_lerobot_dataset_path(repo_id: str) -> None:
    dataset_path = _lerobot_cache_path(repo_id)
    if not dataset_path.exists():
        return
    if args_cli.no_overwrite:
        raise FileExistsError(f"Local LeRobot dataset already exists: {dataset_path}")
    shutil.rmtree(dataset_path)
    print(f"[INFO] Overwriting existing local LeRobot dataset: {dataset_path}")


def _copy_source_dataset_to_target_if_needed() -> bool:
    """Materialize a target LeRobot dataset by copying a source dataset cache."""
    if args_cli.source_repo_id == args_cli.target_repo_id:
        return False

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as err:
        raise RuntimeError("LeRobot must be installed to clone source datasets.") from err

    print(f"[INFO] Preparing target dataset from source: {args_cli.source_repo_id} -> {args_cli.target_repo_id}")
    LeRobotDataset(repo_id=args_cli.source_repo_id)
    source_path = _lerobot_cache_path(args_cli.source_repo_id)
    target_path = _lerobot_cache_path(args_cli.target_repo_id)
    if not source_path.exists():
        raise FileNotFoundError(f"Source LeRobot dataset cache does not exist after download: {source_path}")
    if target_path.exists():
        if args_cli.no_overwrite:
            raise FileExistsError(f"Target local LeRobot dataset already exists: {target_path}")
        shutil.rmtree(target_path)
        print(f"[INFO] Removed existing target local LeRobot dataset: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path)
    print(f"[INFO] Copied source dataset cache to target: {target_path}")
    return True


def _replace_lerobot_recorder(env, env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Replace Isaac Lab's recorder with the LeRobot recorder used by LeIsaac."""
    if env_cfg.recorders is None:
        raise ValueError("The collection environment must define recorder terms.")

    copied_source_dataset = _copy_source_dataset_to_target_if_needed()
    if args_cli.resume_dataset or copied_source_dataset:
        dataset_path = _lerobot_cache_path(args_cli.repo_id)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Cannot resume missing local LeRobot dataset: {dataset_path}")
        env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
        mode = "source-copy" if copied_source_dataset else "resume"
        print(f"[INFO] Appending to local LeRobot dataset ({mode}): {dataset_path}")
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        _prepare_lerobot_dataset_path(args_cli.repo_id)
    if hasattr(env.unwrapped, "recorder_manager"):
        del env.unwrapped.recorder_manager

    dataset_cfg = LeRobotDatasetCfg(repo_id=args_cli.repo_id, fps=args_cli.fps)
    env.unwrapped.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env.unwrapped)


def _get_lerobot_episode_count(env) -> int:
    dataset = env.unwrapped.recorder_manager._dataset_file_handler._lerobot_dataset
    meta = getattr(dataset, "meta", None)
    if meta is not None and hasattr(meta, "total_episodes"):
        return int(meta.total_episodes)
    if hasattr(dataset, "num_episodes"):
        return int(dataset.num_episodes)
    return 0


def _push_dataset_to_hub(env) -> None:
    dataset = env.unwrapped.recorder_manager._dataset_file_handler._lerobot_dataset
    if hasattr(dataset, "push_to_hub"):
        dataset.push_to_hub()
    else:
        print("[WARN] The active LeRobot dataset object does not expose push_to_hub(); skipping upload.")


def _detect_lift_cube_success(env) -> torch.Tensor:
    return lift_cube_mdp.cube_height_above_base(
        env=env.unwrapped,
        cube_cfg=SceneEntityCfg("cube"),
        robot_cfg=SceneEntityCfg("robot"),
        robot_base_name="base",
        height_threshold=0.20,
    )


def _set_manual_success(env, success: bool) -> None:
    env = env.unwrapped
    if not hasattr(env, "termination_manager"):
        raise RuntimeError("Manual success marking currently requires a manager-based environment.")
    env.termination_manager.set_term_cfg(
        "success",
        TerminationTermCfg(
            func=lambda env: torch.full((env.num_envs,), success, dtype=torch.bool, device=env.device),
        ),
    )
    env.termination_manager.compute()


def _limit_action_delta(
    actions: torch.Tensor,
    previous_actions: torch.Tensor | None,
    max_delta: float,
    limit_gripper: bool,
) -> torch.Tensor:
    """Clamp per-step policy action changes to make teacher trajectories easier to imitate."""
    if max_delta <= 0.0 or previous_actions is None:
        return actions

    limited_actions = actions.clone()
    delta = torch.clamp(actions - previous_actions, min=-max_delta, max=max_delta)
    if limit_gripper:
        limited_actions = previous_actions + delta
    else:
        # SO-101 action layout is 5 arm joints + gripper. Keep gripper responsive by default.
        limited_actions[..., :-1] = previous_actions[..., :-1] + delta[..., :-1]
    return limited_actions


def _warmup_after_reset(env, num_steps: int) -> None:
    """Let sensors/rendering settle after reset without advancing policy-controlled episode frames."""
    for _ in range(max(num_steps, 0)):
        env.unwrapped.sim.render()


class ManualDecisionController:
    """Keyboard callbacks for manual dataset decisions during automated teacher rollout."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self.start_recording = False
        self.mark_success = False
        self.reset_failed = False

    def __del__(self):
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset_flags(self):
        self.start_recording = False
        self.mark_success = False
        self.reset_failed = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "B":
                self.start_recording = True
            elif event.input.name == "N":
                self.mark_success = True
            elif event.input.name == "R":
                self.reset_failed = True
        return True


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
    initial_episode_count = _get_lerobot_episode_count(env) if args_cli.resume_dataset else 0
    target_episode_count = initial_episode_count + args_cli.num_episodes
    last_success_count = env.unwrapped.recorder_manager.exported_successful_episode_count
    success_detected = False
    post_success_steps = 0
    collection_complete = False
    previous_actions = None
    rate_limiter = RateLimiter(args_cli.step_hz)
    manual_controller = ManualDecisionController() if args_cli.manual_decision else None
    pending_manual_warmup_steps = args_cli.post_reset_warmup_steps if manual_controller is not None else 0
    manual_recording_active = manual_controller is None

    if abs(args_cli.step_hz - args_cli.fps) > 1e-6:
        print(
            f"[WARN] --step_hz ({args_cli.step_hz:g}) differs from --fps ({args_cli.fps}). "
            "Videos and dataset timestamps will not reflect wall-clock collection speed."
        )
    print(f"[INFO] Collection pacing: {args_cli.step_hz:g} Hz; LeRobot metadata FPS: {args_cli.fps}.")
    if args_cli.resume_dataset:
        print(
            f"[INFO] Resume mode: dataset starts with {initial_episode_count} episodes; "
            f"recording {args_cli.num_episodes} more to reach {target_episode_count}."
        )
    if args_cli.max_action_delta > 0.0:
        target_dims = "all action dims" if args_cli.limit_gripper_delta else "arm action dims only"
        print(f"[INFO] Limiting teacher action delta to {args_cli.max_action_delta:g} per step ({target_dims}).")
    if manual_controller is not None:
        print("[INFO] Manual decision mode enabled. Press B to start, N to save success+reset, R to reset/skip.")
        print(f"[INFO] Post-reset render warmup: {args_cli.post_reset_warmup_steps} frames.")

    try:
        while simulation_app.is_running():
            if manual_controller is not None:
                if manual_controller.mark_success:
                    print("Task Success!!!")
                    print("[INFO] Manual success marked; exporting episode and resetting.")
                    expected_success_count = last_success_count + 1
                    _set_manual_success(env, True)
                    obs, _ = env.reset()
                    _set_manual_success(env, False)
                    previous_actions = None
                    success_detected = False
                    post_success_steps = 0
                    if manual_recording_active:
                        print("Stop Recording!!!")
                    manual_recording_active = False
                    manual_controller.reset_flags()

                    success_count = env.unwrapped.recorder_manager.exported_successful_episode_count
                    if success_count > last_success_count:
                        last_success_count = success_count
                        total_success_count = initial_episode_count + success_count
                        print(
                            f"[INFO] Recorded successful episode {total_success_count}/{target_episode_count} "
                            f"(session {success_count}/{args_cli.num_episodes})."
                        )
                    if success_count < expected_success_count:
                        print(
                            "[WARN] Manual success reset completed, but exported success count did not increase yet "
                            f"({success_count}/{expected_success_count}). Continuing after warmup."
                        )
                    if initial_episode_count + success_count >= target_episode_count:
                        collection_complete = True
                        break
                    pending_manual_warmup_steps = args_cli.post_reset_warmup_steps
                    continue

                if manual_controller.reset_failed:
                    print("[INFO] Manual reset requested; skipping current episode.")
                    _set_manual_success(env, False)
                    obs, _ = env.reset()
                    previous_actions = None
                    success_detected = False
                    post_success_steps = 0
                    if manual_recording_active:
                        print("Stop Recording!!!")
                    manual_recording_active = False
                    manual_controller.reset_flags()
                    pending_manual_warmup_steps = args_cli.post_reset_warmup_steps
                    continue

                if pending_manual_warmup_steps > 0:
                    env.unwrapped.sim.render()
                    pending_manual_warmup_steps -= 1
                    continue

                if manual_controller.start_recording:
                    print("Start Recording!!!")
                    manual_recording_active = True
                    manual_controller.start_recording = False

                if not manual_recording_active:
                    env.unwrapped.sim.render()
                    continue

            with torch.no_grad():
                actions = policy(obs)
            actions = _limit_action_delta(
                actions=actions,
                previous_actions=previous_actions,
                max_delta=args_cli.max_action_delta,
                limit_gripper=args_cli.limit_gripper_delta,
            )
            obs, _, _, _ = env.step(actions)
            rate_limiter.sleep(env.unwrapped)
            previous_actions = actions.detach().clone()

            total_steps += 1

            if manual_controller is None:
                is_success = bool(_detect_lift_cube_success(env)[0].item())
                if is_success and not success_detected:
                    success_detected = True
                    post_success_steps = 0
                    print(
                        "[INFO] Success detected; recording "
                        f"{args_cli.post_success_steps} additional post-success steps."
                    )
                if success_detected:
                    post_success_steps += 1

            if manual_controller is None and success_detected and post_success_steps >= args_cli.post_success_steps:
                _set_manual_success(env, True)
                obs, _ = env.reset()
                _set_manual_success(env, False)
                previous_actions = None
                success_detected = False
                post_success_steps = 0

                success_count = env.unwrapped.recorder_manager.exported_successful_episode_count
                if success_count > last_success_count:
                    last_success_count = success_count
                    total_success_count = initial_episode_count + success_count
                    print(
                        f"[INFO] Recorded successful episode {total_success_count}/{target_episode_count} "
                        f"(session {success_count}/{args_cli.num_episodes})."
                    )
                if initial_episode_count + success_count >= target_episode_count:
                    collection_complete = True
                    break

            if args_cli.max_steps > 0 and total_steps >= args_cli.max_steps:
                print(f"[WARN] Reached --max_steps={args_cli.max_steps} before collection completed.")
                break
    finally:
        if hasattr(env.unwrapped.recorder_manager, "finalize"):
            env.unwrapped.recorder_manager.finalize()
        if args_cli.push_to_hub and collection_complete:
            _push_dataset_to_hub(env)
        elif args_cli.push_to_hub:
            print("[WARN] Collection did not reach requested episode count; skipping push_to_hub.")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
