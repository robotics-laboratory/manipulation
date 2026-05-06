# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect LeRobot recovery episodes by rolling out an RSL-RL policy from saved stuck states."""

import argparse
import sys

from isaaclab.app import AppLauncher

import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect recovery LeRobot episodes from saved LiftCube stuck states.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Must be 1.")
parser.add_argument("--task", type=str, default=None, help="Name of the collection task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent config.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint.")
parser.add_argument("--recovery_state_dir", type=str, required=True, help="Directory containing state_*.pt files.")
parser.add_argument("--recovery_state_glob", type=str, default="state_*.pt", help="Glob for recovery state files.")
parser.add_argument("--num_episodes", type=int, default=0, help="Max successful episodes to record. 0 means all states.")
parser.add_argument("--max_recovery_steps", type=int, default=240, help="Max rollout steps per recovery state.")
parser.add_argument("--post_success_steps", type=int, default=25, help="Extra steps to record after success.")
parser.add_argument("--post_restore_warmup_steps", type=int, default=5, help="Render frames after restoring a state.")
parser.add_argument("--repo_id", type=str, default="igor-saprygin/so101-lift-cube-rl-recovery", help="Alias for --target_repo_id.")
parser.add_argument(
    "--source_repo_id",
    type=str,
    default=None,
    help="Optional source LeRobot dataset repo to copy before recovery collection.",
)
parser.add_argument(
    "--target_repo_id",
    type=str,
    default=None,
    help="LeRobot dataset repo to write. Defaults to --source_repo_id when provided, otherwise --repo_id.",
)
parser.add_argument("--fps", type=int, default=30, help="LeRobot dataset frames per second.")
parser.add_argument("--step_hz", type=float, default=60.0, help="Wall-clock environment stepping rate.")
parser.add_argument("--no_overwrite", action="store_true", default=False)
parser.add_argument("--resume_dataset", action="store_true", default=False)
parser.add_argument("--push_to_hub", action="store_true", default=False)
parser.add_argument(
    "--manual_decision",
    action="store_true",
    default=False,
    help="Disable automatic success export. Press B to start, N to save current recovery rollout, R to discard it.",
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
    raise ValueError("Recovery collection currently supports --num_envs 1 only.")
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import shutil
import time
from pathlib import Path

import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
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

from isaac_so_arm101.scripts.rsl_rl.recovery_state_utils import load_lift_cube_state, restore_lift_cube_state


class RateLimiter:
    """Match teleop collection pacing and keep RTX sensors rendering between control steps."""

    def __init__(self, hz: float):
        if hz <= 0.0:
            raise ValueError("--step_hz must be positive.")
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class ManualDecisionController:
    """Keyboard callbacks for manual recovery rollout decisions."""

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


def _warmup_after_restore(env, num_steps: int) -> None:
    for _ in range(max(num_steps, 0)):
        env.unwrapped.sim.render()


def _recovery_state_paths() -> list[Path]:
    state_dir = Path(args_cli.recovery_state_dir)
    if not state_dir.exists():
        raise FileNotFoundError(f"Recovery state directory does not exist: {state_dir}")
    paths = sorted(state_dir.glob(args_cli.recovery_state_glob))
    if not paths:
        raise FileNotFoundError(f"No recovery states matched {args_cli.recovery_state_glob} in {state_dir}")
    return paths


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect successful MLP recovery episodes from saved LiftCube stuck states."""
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
            raise RuntimeError("--use_pretrained_checkpoint is not supported by this IsaacLab install.")
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)
    raw_env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)

    _replace_lerobot_recorder(raw_env, env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    state_paths = _recovery_state_paths()
    initial_episode_count = _get_lerobot_episode_count(env) if args_cli.resume_dataset else 0
    target_successes = len(state_paths) if args_cli.num_episodes <= 0 else min(args_cli.num_episodes, len(state_paths))
    last_success_count = env.unwrapped.recorder_manager.exported_successful_episode_count
    rate_limiter = RateLimiter(args_cli.step_hz)
    manual_controller = ManualDecisionController() if args_cli.manual_decision else None
    collection_complete = False

    print(f"[INFO] Loaded {len(state_paths)} recovery state(s). Target successful episodes: {target_successes}.")
    if abs(args_cli.step_hz - args_cli.fps) > 1e-6:
        print(
            f"[WARN] --step_hz ({args_cli.step_hz:g}) differs from --fps ({args_cli.fps}). "
            "Videos and dataset timestamps will not reflect wall-clock collection speed."
        )
    if manual_controller is not None:
        print(
            "[INFO] Manual recovery decision mode enabled. "
            "Press B to start, N to save, R to discard current recovery rollout."
        )

    try:
        for state_index, state_path in enumerate(state_paths, start=1):
            if initial_episode_count + last_success_count >= initial_episode_count + target_successes:
                collection_complete = True
                break
            if not simulation_app.is_running():
                break

            print(f"[Recovery] Trying state {state_index}/{len(state_paths)}: {state_path}")
            state = load_lift_cube_state(state_path, device=env.unwrapped.device)
            _set_manual_success(env, False)
            obs, _ = env.reset()
            restore_lift_cube_state(env.unwrapped, state)
            _warmup_after_restore(env, args_cli.post_restore_warmup_steps)
            obs = env.get_observations()
            if manual_controller is not None:
                manual_controller.reset_flags()

            success_detected = False
            post_success_steps = 0
            recovered = False
            discarded = False
            if manual_controller is not None:
                print(f"[Recovery] Restored {state_path.name}. Press B to start or R to discard.")
                while simulation_app.is_running() and not manual_controller.start_recording:
                    if manual_controller.reset_failed:
                        print(f"[Recovery] Manual discard requested for {state_path.name}.")
                        _set_manual_success(env, False)
                        obs, _ = env.reset()
                        manual_controller.reset_flags()
                        discarded = True
                        break
                    env.unwrapped.sim.render()
                    time.sleep(0.01)
                if discarded or not simulation_app.is_running():
                    continue
                print(f"[Recovery] Start recording recovery rollout from {state_path.name}.")
                manual_controller.start_recording = False
                rate_limiter.last_time = time.time()

            for step_idx in range(args_cli.max_recovery_steps):
                if manual_controller is not None:
                    if manual_controller.mark_success:
                        print(f"[Recovery] Manual success marked for {state_path.name}; exporting recovery rollout.")
                        _set_manual_success(env, True)
                        obs, _ = env.reset()
                        _set_manual_success(env, False)
                        recovered = True
                        manual_controller.reset_flags()
                        break
                    if manual_controller.reset_failed:
                        print(f"[Recovery] Manual discard requested for {state_path.name}.")
                        _set_manual_success(env, False)
                        obs, _ = env.reset()
                        manual_controller.reset_flags()
                        break

                with torch.no_grad():
                    actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                rate_limiter.sleep(env.unwrapped)

                if manual_controller is not None:
                    continue

                is_success = bool(_detect_lift_cube_success(env)[0].item())
                if is_success and not success_detected:
                    success_detected = True
                    post_success_steps = 0
                    print(f"[Recovery] Success detected for {state_path.name}; recording post-success steps.")
                if success_detected:
                    post_success_steps += 1
                    if post_success_steps >= args_cli.post_success_steps:
                        _set_manual_success(env, True)
                        obs, _ = env.reset()
                        _set_manual_success(env, False)
                        recovered = True
                        break

            if recovered:
                success_count = env.unwrapped.recorder_manager.exported_successful_episode_count
                if success_count > last_success_count:
                    last_success_count = success_count
                    total_success_count = initial_episode_count + success_count
                    print(
                        f"[Recovery] Recorded recovery episode {total_success_count}/"
                        f"{initial_episode_count + target_successes} from {state_path.name}."
                    )
            else:
                print(f"[Recovery] Failed to recover from {state_path.name}; discarding rollout.")
                _set_manual_success(env, False)
                obs, _ = env.reset()

        if initial_episode_count + last_success_count >= initial_episode_count + target_successes:
            collection_complete = True
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

