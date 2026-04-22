#!/usr/bin/env python3
"""SO101 leader teleop + dataset recording wrapper for manipulation tasks."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_EXTENSION_SRC = _PROJECT_ROOT / "isaac_so_arm101" / "src"
if _EXTENSION_SRC.exists() and str(_EXTENSION_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SRC))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Teleoperate manipulation SO101 tasks with leader-arm and record datasets."
)
def _parse_bool_flag(value: str | None) -> bool:
    if value is None:
        return True
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true/false, 1/0, yes/no."
    )


parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-Play-v0", help="Gym task ID.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (teleop: 1 recommended).")
parser.add_argument("--teleop_device", type=str, default="so101leader", choices=["so101leader"])
parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Leader arm serial port.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--record", action="store_true", help="Enable recording.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="HDF5 dataset output path.")
parser.add_argument(
    "--resume",
    "--resume-dataset",
    dest="resume",
    nargs="?",
    const=True,
    default=False,
    type=_parse_bool_flag,
    help="Resume recording to existing dataset. Accepts: --resume, --resume true, --resume=false.",
)
parser.add_argument(
    "--overwrite",
    dest="overwrite",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Overwrite existing HDF5 output by default when not resuming.",
)
parser.add_argument("--num_demos", type=int, default=0, help="Number of successful demos to collect (0 = infinite).")
parser.add_argument("--recalibrate", action="store_true", help="Force SO101 leader calibration.")
parser.add_argument("--quality", action="store_true", help="Enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="Record directly into LeRobot format.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="LeRobot dataset repo id user/name.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="LeRobot dataset fps metadata.")
parser.add_argument(
    "--auto-create-hf-repo",
    dest="auto_create_hf_repo",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Auto-create Hugging Face dataset repo when missing in LeRobot mode.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import traceback

import isaac_so_arm101.tasks.lift  # noqa: F401
import isaac_so_arm101.tasks.reach  # noqa: F401
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.datasets.episode_data import EpisodeData
from isaaclab.utils.datasets.hdf5_dataset_file_handler import HDF5DatasetFileHandler
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import lerobot.datasets.utils as _lerobot_utils

from teleop_adapter import (
    LEADER_JOINT_ORDER,
    check_so101_jointpos_compatibility,
    collect_camera_frames,
    collect_joint_state_and_action,
    make_preprocess_device_action,
)
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


# ---------------------------------------------------------------------------
# LeRobot shape validation patch
# ---------------------------------------------------------------------------
# Some LeRobot versions compare ndarray.shape (tuple) against feature["shape"] (list),
# which can raise false mismatches like "(6,) != [6]".
def _patched_validate_feature_numpy_array(name, expected_dtype, expected_shape, value):
    import numpy as _np

    error_message = ""
    if isinstance(value, _np.ndarray):
        if value.dtype != _np.dtype(expected_dtype):
            error_message += (
                f"The feature '{name}' of dtype '{value.dtype}' is not of the "
                f"expected dtype '{expected_dtype}'.\n"
            )
        if tuple(value.shape) != tuple(expected_shape):
            error_message += (
                f"The feature '{name}' of shape '{value.shape}' does not have "
                f"the expected shape '{expected_shape}'.\n"
            )
    else:
        error_message += (
            f"The feature '{name}' is not a 'np.ndarray'. Expected type is "
            f"'{expected_dtype}', but type '{type(value)}' provided instead.\n"
        )
    return error_message


_lerobot_utils.validate_feature_numpy_array = _patched_validate_feature_numpy_array


class RateLimiter:
    """Convenience helper to enforce loop frequency."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / float(hz)
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup = self.last_time + self.sleep_duration
        while time.time() < next_wakeup:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        while self.last_time < time.time():
            self.last_time += self.sleep_duration


def _manual_terminate(env, success: bool):
    if hasattr(env, "termination_manager"):
        env.termination_manager.set_term_cfg(
            "success",
            TerminationTermCfg(
                func=lambda inner_env: torch.full(
                    (inner_env.num_envs,),
                    bool(success),
                    dtype=torch.bool,
                    device=inner_env.device,
                )
            ),
        )
        env.termination_manager.compute()


def _discover_camera_feature_spec(env) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    scene = getattr(env, "scene", None)
    sensors = getattr(scene, "sensors", None) if scene is not None else None
    if sensors is None:
        return specs
    for sensor_name, sensor in sensors.items():
        data = getattr(sensor, "data", None)
        output = getattr(data, "output", None)
        if not isinstance(output, dict) or "rgb" not in output:
            continue
        rgb = output["rgb"]
        if rgb is None or len(rgb.shape) < 4:
            continue
        height = int(rgb.shape[1])
        width = int(rgb.shape[2])
        lower = sensor_name.lower()
        if "top" in lower:
            key = "observation.images.top"
        elif "wrist" in lower:
            key = "observation.images.wrist"
        elif "side" in lower:
            key = "observation.images.side"
        else:
            key = f"observation.images.{sensor_name}"
        specs[key] = {"dtype": "video", "shape": [3, height, width], "names": ["channels", "height", "width"]}
    return specs


def _resolve_lerobot_dataset(task: str, env):
    if not args_cli.lerobot_dataset_repo_id:
        raise ValueError("--lerobot_dataset_repo_id is required with --use_lerobot_recorder")
    features = {
        "observation.state": {"dtype": "float32", "shape": [len(LEADER_JOINT_ORDER)]},
        "action": {"dtype": "float32", "shape": [len(LEADER_JOINT_ORDER)]},
    }
    camera_specs = _discover_camera_feature_spec(env)
    features.update(camera_specs)
    if not camera_specs:
        raise RuntimeError("No RGB camera sensors found in task. LeRobot recorder requires camera streams.")

    # LeRobot stores local datasets under ~/.cache/huggingface/lerobot/<repo_id>.
    # Apply overwrite/resume semantics consistently with HDF5 mode.
    local_lerobot_root = Path.home() / ".cache" / "huggingface" / "lerobot" / args_cli.lerobot_dataset_repo_id
    if not args_cli.resume:
        if local_lerobot_root.exists():
            if args_cli.overwrite:
                shutil.rmtree(local_lerobot_root)
                print(f"[INFO] Overwriting existing local LeRobot dataset: {local_lerobot_root}")
            else:
                raise FileExistsError(
                    "Local LeRobot dataset already exists: "
                    f"{local_lerobot_root}. Use --resume (or --resume true) to append, "
                    "or --overwrite to replace."
                )

    if args_cli.auto_create_hf_repo:
        try:
            HfApi().create_repo(
                repo_id=args_cli.lerobot_dataset_repo_id,
                repo_type="dataset",
                exist_ok=True,
            )
            print(f"[INFO] Ensured HF dataset repo exists: {args_cli.lerobot_dataset_repo_id}")
        except Exception as err:
            print(
                "[WARN] Could not auto-create/check HF dataset repo "
                f"{args_cli.lerobot_dataset_repo_id}: {err}"
            )

    try:
        if args_cli.resume:
            dataset = LeRobotDataset(repo_id=args_cli.lerobot_dataset_repo_id)
        else:
            dataset = LeRobotDataset.create(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
                robot_type="so101",
                features=features,
            )
    except Exception as err:
        raise RuntimeError(
            "Failed to initialize LeRobot dataset. "
            f"Parsed resume={args_cli.resume}. "
            "If dataset already exists locally, rerun with --resume (or --resume true). "
            f"Original error: {err}"
        ) from err
    print(
        f"[INFO] LeRobot dataset ready: repo_id={args_cli.lerobot_dataset_repo_id} "
        f"(resume={args_cli.resume})"
    )
    return dataset


def _resolve_hdf5_handler(task: str):
    dataset_path = args_cli.dataset_file
    if args_cli.resume:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file does not exist for --resume: {dataset_path}")
        handler = HDF5DatasetFileHandler()
        handler.open(dataset_path, mode="a")
    else:
        if os.path.exists(dataset_path):
            if args_cli.overwrite:
                os.remove(dataset_path)
                print(f"[INFO] Overwriting existing dataset file: {dataset_path}")
            else:
                raise FileExistsError(
                    f"Dataset file already exists: {dataset_path}. Use --resume, --overwrite, or change --dataset_file."
                )
        handler = HDF5DatasetFileHandler()
        handler.create(dataset_path, env_name=task)
    print(f"[INFO] HDF5 dataset ready: {dataset_path} (resume={args_cli.resume})")
    return handler


def _disable_auto_terminations(env_cfg):
    if hasattr(env_cfg, "terminations"):
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "object_dropping"):
            env_cfg.terminations.object_dropping = None
        if not hasattr(env_cfg.terminations, "success"):
            setattr(env_cfg.terminations, "success", None)
        env_cfg.terminations.success = TerminationTermCfg(
            func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        )


def _align_action_cfg_with_leisaac_so101(env_cfg):
    """Match LeIsaac teleop semantics: absolute joint-position actions, scale=1."""
    actions = getattr(env_cfg, "actions", None)
    if actions is None:
        return

    arm_action = getattr(actions, "arm_action", None)
    if arm_action is not None:
        if hasattr(arm_action, "scale"):
            arm_action.scale = 1.0
        if hasattr(arm_action, "use_default_offset"):
            arm_action.use_default_offset = False

    # Manipulation Lift tasks typically use BinaryJointPositionAction for gripper.
    # LeIsaac so101leader uses JointPositionAction for gripper, so we align to that.
    try:
        actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    except Exception:
        # If task does not support replacing this action term, keep existing cfg.
        pass


def main():
    if args_cli.use_lerobot_recorder:
        # HDF5 path is not used in LeRobot mode.
        print("[INFO] Using LeRobot recording mode (HDF5 path ignored).")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    _disable_auto_terminations(env_cfg)
    _align_action_cfg_with_leisaac_so101(env_cfg)
    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    compat = check_so101_jointpos_compatibility(env, robot_name="robot")
    if not compat.ok:
        raise RuntimeError(
            "Task is not compatible with SO101 leader wrapper.\n"
            f"Reason: {compat.message}\n"
            f"arm_action joints={list(compat.arm_joint_names)} "
            f"gripper_action joints={list(compat.gripper_joint_names)} "
            f"total_action_dim={compat.action_dim}"
        )
    print(f"[INFO] Compatibility: {compat.message}")

    # Patch preprocessing hook expected by SO101Leader device.
    preprocess_fn = make_preprocess_device_action(env, robot_name="robot")
    env.cfg.preprocess_device_action = MethodType(preprocess_fn, env.cfg)

    try:
        from leisaac.devices import SO101Leader
    except Exception as err:
        raise RuntimeError(
            "SO101 leader teleop requires leisaac package in this Python environment."
        ) from err

    teleop = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
    teleop.display_controls()

    should_reset = False
    should_mark_success = False

    def _reset_fail():
        nonlocal should_reset, should_mark_success
        should_reset = True
        should_mark_success = False

    def _reset_success():
        nonlocal should_reset, should_mark_success
        should_reset = True
        should_mark_success = True

    teleop.add_callback("R", _reset_fail)
    teleop.add_callback("N", _reset_success)

    hdf5_handler = None
    lerobot_dataset = None
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            lerobot_dataset = _resolve_lerobot_dataset(args_cli.task, env)
        else:
            hdf5_handler = _resolve_hdf5_handler(args_cli.task)

    current_episode = EpisodeData()
    current_episode_frames = 0
    successful_demos = 0
    total_saved_episodes = 0
    recording_active = False
    interrupted = False

    def _finalize_episode(success: bool):
        nonlocal current_episode, current_episode_frames, successful_demos, total_saved_episodes
        if not args_cli.record:
            current_episode = EpisodeData()
            current_episode_frames = 0
            return

        if args_cli.use_lerobot_recorder:
            if current_episode_frames == 0:
                lerobot_dataset.clear_episode_buffer()
                print("[INFO] Episode skipped (no frames).")
            else:
                if success:
                    lerobot_dataset.save_episode(parallel_encoding=False)
                    successful_demos += 1
                    total_saved_episodes += 1
                    print(f"[INFO] Saved successful LeRobot episode #{successful_demos}.")
                else:
                    lerobot_dataset.clear_episode_buffer()
                    print("[INFO] Failed episode discarded (LeRobot mode saves success only).")
        else:
            if current_episode_frames == 0:
                print("[INFO] Episode skipped (no frames).")
            else:
                current_episode.success = bool(success)
                current_episode.pre_export()
                hdf5_handler.write_episode(current_episode)
                hdf5_handler.flush()
                total_saved_episodes += 1
                if success:
                    successful_demos += 1
                print(
                    f"[INFO] Saved HDF5 episode #{total_saved_episodes} "
                    f"(success={bool(success)})."
                )

        current_episode = EpisodeData()
        current_episode_frames = 0

    def _record_step(action: torch.Tensor):
        nonlocal current_episode_frames
        if not args_cli.record:
            return

        state_deg, action_deg = collect_joint_state_and_action(env, action, robot_name="robot")
        camera_frames = collect_camera_frames(env)
        task_text = args_cli.task

        if args_cli.use_lerobot_recorder:
            frame = {
                "observation.state": state_deg,
                "action": action_deg,
                "task": task_text,
            }
            frame.update(camera_frames)
            lerobot_dataset.add_frame(frame)
            current_episode_frames += 1
            return

        # HDF5 mode
        current_episode.add("actions", action[0].detach().cpu())
        current_episode.add("observation/state_deg", torch.from_numpy(state_deg))
        for key, image in camera_frames.items():
            np_img = np.asarray(image, dtype=np.uint8)
            current_episode.add(key.replace(".", "/"), torch.from_numpy(np_img))
        current_episode_frames += 1

    rate_limiter = RateLimiter(args_cli.step_hz)
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()
    teleop.reset()

    def _signal_handler(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt detected, shutting down...")

    original_sigint = signal.signal(signal.SIGINT, _signal_handler)

    try:
        while simulation_app.is_running() and not interrupted:
            with torch.inference_mode():
                if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
                    dynamic_reset_gripper_effort_limit_sim(env, args_cli.teleop_device)
                action_or_state = teleop.advance()
                if should_reset:
                    _finalize_episode(success=should_mark_success)
                    _manual_terminate(env, success=should_mark_success)
                    env.reset()
                    teleop.reset()
                    should_reset = False
                    should_mark_success = False
                    recording_active = False
                    if args_cli.num_demos > 0 and successful_demos >= args_cli.num_demos:
                        print(f"[INFO] Collected requested successful demos: {successful_demos}.")
                        break
                elif action_or_state is None:
                    env.render()
                elif isinstance(action_or_state, dict):
                    # Reset dict is handled by callbacks; keep this branch as a safety no-op.
                    pass
                else:
                    if not recording_active and args_cli.record:
                        print("[INFO] Start recording.")
                        recording_active = True
                    env.step(action_or_state)
                    _record_step(action_or_state)
                rate_limiter.sleep(env)
    except Exception as err:
        print(f"\n[ERROR] Teleop loop failed: {err}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        try:
            if args_cli.record:
                _finalize_episode(success=False)
                if args_cli.use_lerobot_recorder and lerobot_dataset is not None:
                    lerobot_dataset.finalize()
                    print("[INFO] LeRobot dataset finalized.")
                if (not args_cli.use_lerobot_recorder) and hdf5_handler is not None:
                    hdf5_handler.close()
                    print("[INFO] HDF5 dataset closed.")
        finally:
            env.close()
            simulation_app.close()


if __name__ == "__main__":
    main()
