"""Collect a LeRobot v3 dataset using a trained RSL-RL MLP agent.

Runs the agent in the Isaac Lab environment with cameras enabled, records
image observations (top + wrist + side), proprioceptive state, and the MLP's
actions into a LeRobot v3 dataset (Parquet + MP4 videos).

Control rate defaults to **30 Hz** (``LiftEnvCfg``: ``sim.dt = 1/60``, ``decimation = 2``).
Pass ``--fps`` to match dataset metadata to your env (or override for subsampling).

Usage (from the manipulation/isaac_so_arm101 directory)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/collect_lerobot_dataset.py \\
        --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
        --num_episodes 50 \\
        --output_dir datasets/so101_lift_cube \\
        --repo_id igor-saprygin/so101-lift-cube \\
        --fps 30

By default an existing ``--output_dir`` is **replaced**. Use ``--resume-dataset`` to append
to an existing LeRobot v3 folder.

Fall-restore (optional): respawn the cube when it drops instead of ending the episode; see
``manipulation/docs/DATASET_COLLECTION_FALL_RESTORE.md``.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import tempfile

# Torch / torchvision must load before AppLauncher imports isaaclab as a namespace
# package; LeRobot pulls torch but we import torchvision explicitly for register_fake.
# See manipulation/docs/torch-fix-notes.md for the full explanation.
import torch
import torchvision  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Collect a LeRobot v3 dataset from a trained RSL-RL agent."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-Lift-Cube-Play-v0",
    help="Gym task ID (use a -Play variant for corruption-free observations).",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent config entry point name.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=50,
    help="Number of episodes to collect.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="datasets/so101_lift_cube",
    help="Local directory for the dataset.",
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="igor-saprygin/so101-lift-cube",
    help="LeRobot dataset repo ID (metadata; upload only with --push_to_hub).",
)
parser.add_argument(
    "--task_description",
    type=str,
    default="Pick up the cube",
    help="Natural-language task label stored with each episode.",
)
parser.add_argument(
    "--success_only",
    action="store_true",
    default=False,
    help="Discard episodes where the cube was never lifted.",
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--push_to_hub",
    action="store_true",
    default=False,
    help="Push dataset to HuggingFace Hub after collection.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel envs (1 recommended for sequential collection).",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="FPS stored in LeRobot metadata (default 30; match sim.dt * decimation).",
)
parser.add_argument(
    "--resume-dataset",
    action="store_true",
    default=False,
    help="If output_dir exists, load the dataset and append episodes (default: replace directory).",
)
parser.add_argument(
    "--enable-fall-restore",
    action="store_true",
    default=False,
    help="Respawn the cube when it falls instead of terminating (see docs).",
)
parser.add_argument(
    "--min-fall-restore-fraction",
    type=float,
    default=-1.0,
    help="Keep episodes with fall_restores/length >= this (set -1 to disable lower bound).",
)
parser.add_argument(
    "--max-fall-restore-fraction",
    type=float,
    default=0.10,
    help="Keep episodes with fall_restores/length <= this (default 0.10; set -1 to disable upper bound).",
)
parser.add_argument(
    "--fall-restore-stall-limit",
    type=int,
    default=64,
    help="After this many consecutive ratio-rejected episodes, save the next one anyway.",
)
parser.add_argument(
    "--max-fall-restores-per-episode",
    type=int,
    default=16,
    help="Max respawns per episode before forcing time-out (MDP). Lower is stricter.",
)
parser.add_argument(
    "--keep-force-commit-episodes",
    action="store_true",
    default=False,
    help=(
        "Keep episodes that hit max_restores_per_episode (force-commit). "
        "Default is strict quality mode: discard unless saved by stall breaker."
    ),
)
parser.add_argument(
    "--fall-restore-trigger-height",
    type=float,
    default=-0.02,
    help="World-frame object z below which a respawn is triggered.",
)
parser.add_argument(
    "--clean-lift-threshold",
    type=float,
    default=0.025,
    help="Height threshold used to mark that an episode has achieved a lift.",
)
parser.add_argument(
    "--clean-drop-threshold",
    type=float,
    default=0.020,
    help=(
        "When fall-restore is disabled, if cube height goes below this after a lift, "
        "the successful episode is considered dirty and discarded."
    ),
)
parser.add_argument(
    "--target-restore-episode-fraction",
    type=float,
    default=0.20,
    help=(
        "When fall-restore is enabled, target fraction of saved episodes that include "
        "at least one restore. Set -1 to disable quota control."
    ),
)
parser.add_argument(
    "--restore-fraction-tolerance",
    type=float,
    default=0.05,
    help=(
        "Tolerance around target restore fraction. Restore episodes are rejected only "
        "when projected fraction exceeds target + tolerance."
    ),
)
parser.add_argument(
    "--restore-quota-warmup-episodes",
    type=int,
    default=20,
    help="Do not enforce restore-fraction quota before this many saved episodes.",
)
parser.add_argument(
    "--manual-review",
    action="store_true",
    default=False,
    help="Render each completed episode to MP4 and prompt accept/skip/quit.",
)
parser.add_argument(
    "--manual-review-camera-key",
    type=str,
    default="observation.images.top",
    help="Video feature key from episode buffer used to create manual-review MP4.",
)
parser.add_argument(
    "--manual-review-dir",
    type=str,
    default="",
    help="Directory for manual-review MP4s (default: temporary file per episode).",
)
parser.add_argument(
    "--manual-review-keep-mp4",
    action="store_true",
    default=False,
    help="Keep generated review MP4 files after decision (default: delete).",
)
parser.add_argument(
    "--manual-review-gui",
    action="store_true",
    default=False,
    help=(
        "Play episode preview directly in an OpenCV window and use r/c/s/q keys. "
        "Falls back to terminal prompt if GUI is unavailable."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

# Docker / SSH without X11: GLFW cannot open a display; without --headless, Kit may appear
# stuck while retrying windowing. Force headless when no display is available.
_has_display = bool(os.environ.get("DISPLAY", "").strip()) or bool(
    os.environ.get("WAYLAND_DISPLAY", "").strip()
)
if not _has_display:
    if not getattr(args_cli, "headless", False):
        print(
            "[INFO] No DISPLAY/WAYLAND_DISPLAY; forcing --headless (required for camera rendering in Docker).",
            flush=True,
        )
    args_cli.headless = True

print(
    "[INFO] Launching Isaac Sim (first run after install can take several minutes; this is normal).",
    flush=True,
)

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import cv2
from PIL import Image

# Patch isaaclab.__file__ before importing rsl_rl / tensordict.
# tensordict calls torch.compiler.allow_in_graph at module scope, which
# triggers torch._dynamo → inspect.getfile, which crashes on namespace
# packages that lack __file__. See manipulation/docs/torch-fix-notes.md.
import isaaclab as _isaaclab_ns
if not getattr(_isaaclab_ns, "__file__", None):
    _isaaclab_ns.__file__ = next(iter(_isaaclab_ns.__path__), __file__)

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks.lift  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaac_so_arm101.tasks.lift.mdp import fall_restore_episode as fre

from log_paths import resolve_checkpoint_cli_path, rsl_rl_root


# ---------------------------------------------------------------------------
# Monkey-patch lerobot shape validation bug
# ---------------------------------------------------------------------------
# validate_feature_numpy_array compares value.shape (tuple) directly against
# feature['shape'] (list), e.g. (6,) != [6] → always True → spurious error.
# Fix: normalise both sides to tuples before comparing.
import lerobot.datasets.utils as _lerobot_utils


def _patched_validate_feature_numpy_array(name, expected_dtype, expected_shape, value):
    import numpy as np
    error_message = ""
    if isinstance(value, np.ndarray):
        if value.dtype != np.dtype(expected_dtype):
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


# ---------------------------------------------------------------------------
# Fall-restore episode filter
# ---------------------------------------------------------------------------


class FallRestoreTracker:
    """Band + stall breaker for fall_restores / episode_length."""

    def __init__(self, min_frac: float, max_frac: float, stall_limit: int, keep_force_commit: bool):
        self.min_frac = min_frac
        self.max_frac = max_frac
        self.stall_limit = max(1, stall_limit)
        self.keep_force_commit = keep_force_commit
        self.stall_count = 0

    def classify(self, fall_n: int, ep_len: int, force_commit: bool) -> str:
        """Return ``keep``, ``discard``, or ``force_keep``."""
        if force_commit:
            if self.keep_force_commit:
                self.stall_count = 0
                return "keep"
            # Strict default: treat force-commit like a rejected sample.
            self.stall_count += 1
            if self.stall_count >= self.stall_limit:
                self.stall_count = 0
                return "force_keep"
            return "discard"
        if ep_len <= 0:
            self.stall_count = 0
            return "keep"
        r = fall_n / ep_len
        ok = True
        if self.min_frac >= 0.0 and r < self.min_frac:
            ok = False
        if self.max_frac >= 0.0 and r > self.max_frac:
            ok = False
        if ok:
            self.stall_count = 0
            return "keep"
        self.stall_count += 1
        if self.stall_count >= self.stall_limit:
            self.stall_count = 0
            return "force_keep"
        return "discard"


class RestoreQuotaTracker:
    """Keep restore episode share near target by capping the upper fraction."""

    def __init__(self, target_frac: float, tolerance: float, warmup_episodes: int):
        self.target_frac = target_frac
        self.tolerance = max(0.0, tolerance)
        self.warmup_episodes = max(0, warmup_episodes)
        self.saved_restore = 0
        self.saved_clean = 0

    def classify(self, is_restore_episode: bool) -> str:
        if self.target_frac < 0.0:
            return "keep"
        total_saved = self.saved_restore + self.saved_clean
        if total_saved < self.warmup_episodes:
            return "keep"
        if not is_restore_episode:
            return "keep"
        projected_restore = self.saved_restore + 1
        projected_total = total_saved + 1
        projected_frac = projected_restore / max(1, projected_total)
        upper_bound = min(1.0, self.target_frac + self.tolerance)
        if projected_frac > upper_bound:
            return "discard"
        return "keep"

    def on_saved(self, is_restore_episode: bool) -> None:
        if is_restore_episode:
            self.saved_restore += 1
        else:
            self.saved_clean += 1


def _apply_fall_restore_env_cfg(env_cfg: ManagerBasedRLEnvCfg, args) -> None:
    step_s = float(env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.events = env_cfg.events.replace(
        fall_restore_reset=EventTerm(
            func=fre.fall_restore_reset_state,
            mode="reset",
        ),
        fall_restore_interval=EventTerm(
            func=fre.fall_restore_recover_object,
            mode="interval",
            interval_range_s=(step_s, step_s),
            params={
                "trigger_height": args.fall_restore_trigger_height,
                "max_restores_per_episode": args.max_fall_restores_per_episode,
            },
        ),
    )
    if hasattr(env_cfg.terminations, "object_dropping"):
        del env_cfg.terminations.object_dropping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_image(env_unwrapped, sensor_name: str) -> Image.Image:
    """Read a TiledCamera sensor and return an RGB PIL Image."""
    sensor = env_unwrapped.scene.sensors[sensor_name]
    rgb_tensor = sensor.data.output["rgb"][0]  # [H, W, 3+] uint8 on GPU
    rgb_np: np.ndarray = rgb_tensor[:, :, :3].cpu().numpy().astype(np.uint8)
    return Image.fromarray(rgb_np)


def _extract_state(env_unwrapped) -> torch.Tensor:
    """Return joint positions in degrees — matches SmolVLA base SO-101 convention."""
    robot = env_unwrapped.scene.articulations["robot"]
    joint_pos_rad = robot.data.joint_pos[0].cpu().float()
    return torch.rad2deg(joint_pos_rad)


def _cube_is_lifted(env_unwrapped, min_height: float = 0.025) -> bool:
    obj = env_unwrapped.scene.rigid_objects["object"]
    return obj.data.root_pos_w[0, 2].item() > min_height


def _cube_height(env_unwrapped) -> float:
    obj = env_unwrapped.scene.rigid_objects["object"]
    return float(obj.data.root_pos_w[0, 2].item())


def _write_episode_preview_mp4(dataset: LeRobotDataset, camera_key: str, output_path: Path, fps: int) -> bool:
    """Write current episode buffer frames to an MP4 for manual review."""
    episode_buffer = getattr(dataset, "episode_buffer", None)
    if episode_buffer is None or camera_key not in episode_buffer:
        return False
    frame_paths = episode_buffer.get(camera_key, [])
    if not frame_paths:
        return False

    first = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        return False

    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        return False
    try:
        writer.write(first)
        for fpath in frame_paths[1:]:
            frame = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists()


def _prompt_manual_decision(
    episode_attempt: int,
    preview_path: Path,
    suggested_keep: bool,
    suggested_kind: str,
    reason: str,
    restore_count: int,
    clean_count: int,
    skipped_count: int,
) -> str:
    """Ask user whether episode should be saved as restore/clean, skipped, or quit."""
    suggestion = suggested_kind if suggested_keep else "skip"
    print(
        f"  Episode {episode_attempt}: preview={preview_path}\n"
        f"    auto_suggestion={suggestion} reason={reason}\n"
        f"    counters: restore={restore_count} clean={clean_count} skipped={skipped_count}",
        flush=True,
    )
    while True:
        resp = input("  [r]estore / [c]lean / [s]kip / [q]uit (Enter=auto suggestion): ").strip().lower()
        if resp == "":
            return suggestion
        if resp in {"r", "restore"}:
            return "restore"
        if resp in {"c", "clean"}:
            return "clean"
        if resp in {"s", "skip"}:
            return "skip"
        if resp in {"q", "quit"}:
            return "quit"
        print("  Invalid input. Use r/c/s/q (or Enter).", flush=True)


def _prompt_manual_decision_gui(
    episode_attempt: int,
    preview_path: Path,
    suggested_keep: bool,
    suggested_kind: str,
    reason: str,
    restore_count: int,
    clean_count: int,
    skipped_count: int,
) -> str:
    """Play preview in GUI and accept r/c/s/q key decisions.

    Returns:
        One of {"restore", "clean", "skip", "quit", "fallback"}.
        "fallback" means caller should use terminal prompt.
    """
    suggestion = suggested_kind if suggested_keep else "skip"
    win_name = f"Episode {episode_attempt} review"
    cap = cv2.VideoCapture(str(preview_path))
    if not cap.isOpened():
        print(f"  Episode {episode_attempt}: failed to open preview in OpenCV GUI.", flush=True)
        return "fallback"

    delay_ms = 33  # ~30 FPS
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            overlay1 = f"ep={episode_attempt} auto={suggestion} reason={reason}"
            overlay2 = (
                f"restore={restore_count} clean={clean_count} skipped={skipped_count}  "
                "keys: r=restore c=clean s=skip q=quit"
            )
            cv2.putText(frame, overlay1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, overlay2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win_name, frame)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("r"), ord("R")):
                return "restore"
            if key in (ord("c"), ord("C")):
                return "clean"
            if key in (ord("s"), ord("S")):
                return "skip"
            if key in (ord("q"), ord("Q"), 27):  # q or ESC
                return "quit"
    except cv2.error:
        return "fallback"
    finally:
        cap.release()
        try:
            cv2.destroyWindow(win_name)
        except cv2.error:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    # ------------------------------------------------------------------ env
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed

    if args_cli.enable_fall_restore:
        _apply_fall_restore_env_cfg(env_cfg, args_cli)

    # ----------------------------------------------------------- checkpoint
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    log_root = os.path.abspath(
        os.path.join(rsl_rl_root(), agent_cfg.experiment_name)
    )
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(resolve_checkpoint_cli_path(args_cli.checkpoint))
    else:
        resume_path = get_checkpoint_path(
            log_root, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    print(f"[INFO] Checkpoint: {resume_path}")

    # Disable debug visualizations so markers don't appear in camera images
    env_cfg.scene.ee_frame.debug_vis = False
    env_cfg.commands.object_pose.debug_vis = False

    # ------------------------------------------------- env + policy loader
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # -------------------------------------------------- discover dimensions
    unwrapped = env.unwrapped
    num_joints = unwrapped.scene.articulations["robot"].data.joint_pos.shape[-1]
    state_dim = num_joints  # joint positions only (degrees), matches SmolVLA base
    action_dim = env.num_actions

    cam_shape = unwrapped.scene.sensors["camera_top"].data.output["rgb"].shape
    cam_h, cam_w = int(cam_shape[1]), int(cam_shape[2])

    print(
        f"[INFO] state_dim={state_dim}  action_dim={action_dim}  "
        f"num_joints={num_joints}  cam={cam_w}x{cam_h}"
    )

    # --------------------------------------------------- LeRobot v3 dataset
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": [3, cam_h, cam_w],
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [3, cam_h, cam_w],
            "names": ["channels", "height", "width"],
        },
        "observation.images.side": {
            "dtype": "video",
            "shape": [3, cam_h, cam_w],
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
        },
    }

    env_fps = int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)))
    dataset_fps = int(args_cli.fps)
    output_dir = os.path.abspath(args_cli.output_dir)
    dataset_dir = Path(output_dir)

    if dataset_dir.exists() and not args_cli.resume_dataset:
        print(
            f"[INFO] Replacing existing dataset directory: {output_dir}",
            flush=True,
        )
        shutil.rmtree(output_dir)

    if dataset_dir.exists() and args_cli.resume_dataset:
        try:
            dataset = LeRobotDataset(
                repo_id=args_cli.repo_id,
                root=output_dir,
            )
        except Exception as exc:
            print(
                "[ERROR] Could not open existing dataset (often incomplete Parquet after a crash).\n"
                f"  Path: {output_dir}\n"
                "  Fix: remove that directory, or omit --resume-dataset to replace it.",
                flush=True,
            )
            raise RuntimeError(
                "LeRobotDataset load failed; see message above."
            ) from exc
        print(
            f"[INFO] Resuming existing dataset ({dataset.num_episodes} episodes already recorded)"
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args_cli.repo_id,
            fps=dataset_fps,
            features=features,
            robot_type="so101",
            root=output_dir,
        )

    ratio_tracker = FallRestoreTracker(
        args_cli.min_fall_restore_fraction,
        args_cli.max_fall_restore_fraction,
        args_cli.fall_restore_stall_limit,
        args_cli.keep_force_commit_episodes,
    )
    quota_tracker = RestoreQuotaTracker(
        args_cli.target_restore_episode_fraction,
        args_cli.restore_fraction_tolerance,
        args_cli.restore_quota_warmup_episodes,
    )

    print(
        f"[INFO] Dataset: {output_dir}  env_fps≈{env_fps}  metadata_fps={dataset_fps}  "
        f"episodes={args_cli.num_episodes}  "
        f"mode={'success-only' if args_cli.success_only else 'all'}  "
        f"fall_restore={args_cli.enable_fall_restore}"
    )

    # ------------------------------------------------------ rollout & record
    obs = env.get_observations()
    episodes_saved = 0
    episodes_attempted = 0
    total_frames = 0
    episode_had_lift = False
    episode_dropped_after_lift = False
    ep_steps = 0
    user_quit = False
    manual_skipped = 0

    while episodes_saved < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        top_img = _extract_image(unwrapped, "camera_top")
        wrist_img = _extract_image(unwrapped, "camera_wrist")
        side_img = _extract_image(unwrapped, "camera_side")
        state = _extract_state(unwrapped)

        cube_h = _cube_height(unwrapped)
        if cube_h > float(args_cli.clean_lift_threshold):
            episode_had_lift = True
        if (
            not args_cli.enable_fall_restore
            and episode_had_lift
            and cube_h <= float(args_cli.clean_drop_threshold)
        ):
            episode_dropped_after_lift = True

        dataset.add_frame(
            {
                "observation.images.top": top_img,
                "observation.images.wrist": wrist_img,
                "observation.images.side": side_img,
                "observation.state": state,
                "action": actions[0].cpu(),
                "task": args_cli.task_description,
            }
        )
        total_frames += 1
        ep_steps += 1

        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if dones.any():
            episodes_attempted += 1
            fall_n = 0
            force_fc = False
            if args_cli.enable_fall_restore:
                fall_n = int(fre.get_last_episode_fall_count(unwrapped)[0].item())
                force_fc = bool(fre.get_last_episode_force_commit(unwrapped)[0].item())

            ratio_decision = "keep"
            quota_decision = "keep"
            if args_cli.enable_fall_restore and (
                args_cli.min_fall_restore_fraction >= 0.0 or args_cli.max_fall_restore_fraction >= 0.0
            ):
                ratio_decision = ratio_tracker.classify(fall_n, ep_steps, force_fc)
            is_restore_episode = bool(args_cli.enable_fall_restore and fall_n > 0)
            if args_cli.enable_fall_restore:
                quota_decision = quota_tracker.classify(is_restore_episode)
            suggested_kind = "restore" if is_restore_episode else "clean"

            ep_steps = 0

            should_keep = True
            reject_reason = "passed filters"
            if args_cli.success_only and not episode_had_lift:
                should_keep = False
                reject_reason = "no lift"
            elif args_cli.success_only and (not args_cli.enable_fall_restore) and episode_dropped_after_lift:
                should_keep = False
                reject_reason = f"dropped after lift (z<={args_cli.clean_drop_threshold:.3f})"
            elif args_cli.enable_fall_restore and ratio_decision == "discard":
                should_keep = False
                reject_reason = f"fall ratio reject (falls={fall_n})"
            elif args_cli.enable_fall_restore and quota_decision == "discard":
                should_keep = False
                total_saved_so_far = quota_tracker.saved_restore + quota_tracker.saved_clean
                cur_frac = quota_tracker.saved_restore / total_saved_so_far if total_saved_so_far > 0 else 0.0
                reject_reason = f"restore quota reject (falls={fall_n}, restore_share={cur_frac:.3f})"

            if args_cli.manual_review:
                if args_cli.manual_review_dir:
                    review_dir = Path(args_cli.manual_review_dir)
                    review_dir.mkdir(parents=True, exist_ok=True)
                    review_path = review_dir / f"episode_{episodes_attempted:06d}.mp4"
                else:
                    tmp = tempfile.NamedTemporaryFile(
                        prefix=f"episode_{episodes_attempted:06d}_", suffix=".mp4", delete=False
                    )
                    tmp.close()
                    review_path = Path(tmp.name)

                preview_ok = _write_episode_preview_mp4(
                    dataset=dataset,
                    camera_key=args_cli.manual_review_camera_key,
                    output_path=review_path,
                    fps=dataset_fps,
                )
                if preview_ok:
                    decision = "fallback"
                    if args_cli.manual_review_gui and _has_display:
                        decision = _prompt_manual_decision_gui(
                            episode_attempt=episodes_attempted,
                            preview_path=review_path,
                            suggested_keep=should_keep,
                            suggested_kind=suggested_kind,
                            reason=reject_reason,
                            restore_count=quota_tracker.saved_restore,
                            clean_count=quota_tracker.saved_clean,
                            skipped_count=manual_skipped,
                        )
                    if decision == "fallback":
                        decision = _prompt_manual_decision(
                            episode_attempt=episodes_attempted,
                            preview_path=review_path,
                            suggested_keep=should_keep,
                            suggested_kind=suggested_kind,
                            reason=reject_reason,
                            restore_count=quota_tracker.saved_restore,
                            clean_count=quota_tracker.saved_clean,
                            skipped_count=manual_skipped,
                        )
                else:
                    print(
                        f"  Episode {episodes_attempted}: preview generation failed for "
                        f"camera_key={args_cli.manual_review_camera_key}; using auto decision.",
                        flush=True,
                    )
                    decision = suggested_kind if should_keep else "skip"

                if (not args_cli.manual_review_keep_mp4) and review_path.exists():
                    review_path.unlink(missing_ok=True)

                if decision == "quit":
                    dataset.clear_episode_buffer()
                    manual_skipped += 1
                    user_quit = True
                    print("  Manual review requested quit; stopping collection.", flush=True)
                elif decision == "skip":
                    dataset.clear_episode_buffer()
                    manual_skipped += 1
                    print(f"  Episode {episodes_attempted}: skipped by manual review", flush=True)
                else:
                    if not should_keep:
                        print(
                            f"  Episode {episodes_attempted}: accepted by manual override "
                            f"(auto reason: {reject_reason})",
                            flush=True,
                        )
                    dataset.save_episode()
                    episodes_saved += 1
                    saved_as_restore = decision == "restore"
                    quota_tracker.on_saved(saved_as_restore)
                    tag = "LIFTED" if episode_had_lift else "no lift"
                    fr_tag = f" falls={fall_n}" if args_cli.enable_fall_restore else ""
                    label_tag = "restore" if saved_as_restore else "clean"
                    print(
                        f"  Episode {episodes_saved}/{args_cli.num_episodes}: "
                        f"{tag}{fr_tag} (manual accept as {label_tag})"
                    )
            elif not should_keep:
                dataset.clear_episode_buffer()
                print(f"  Episode {episodes_attempted}: {reject_reason} – discarded")
            else:
                dataset.save_episode()
                episodes_saved += 1
                quota_tracker.on_saved(is_restore_episode)
                tag = "LIFTED" if episode_had_lift else "no lift"
                fr_tag = f" falls={fall_n}" if args_cli.enable_fall_restore else ""
                if ratio_decision == "force_keep":
                    print(
                        f"  Episode {episodes_saved}/{args_cli.num_episodes}: {tag}{fr_tag} (stall save)"
                    )
                else:
                    print(
                        f"  Episode {episodes_saved}/{args_cli.num_episodes}: {tag}{fr_tag}"
                    )
            episode_had_lift = False
            episode_dropped_after_lift = False
            if user_quit:
                break

    # ------------------------------------------------------------ finalize
    dataset.finalize()
    print(
        f"\n[DONE] {episodes_saved} episodes, {total_frames} frames -> {output_dir}"
    )

    if args_cli.push_to_hub:
        dataset.push_to_hub()
        print(f"[INFO] Pushed to HuggingFace Hub: {args_cli.repo_id}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
