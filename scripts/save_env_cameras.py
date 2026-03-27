#!/usr/bin/env python3
"""
Load the Isaac Lab SO-101 env with cameras enabled, run one reset, and save
the rendered camera images to PNGs (``CameraTop.png``, ``CameraSide.png``,
``CameraWrist.png`` when those observation terms exist).

Usage:
  ./isaaclab.sh -p scripts/save_env_cameras.py
  ./isaaclab.sh -p scripts/save_env_cameras.py --task Isaac-SO-ARM101-Lift-Cube-v0 --output_dir ./my_cameras
  ./isaaclab.sh -p scripts/save_env_cameras.py --num_envs 2   # save cameras for env 0 and env 1
  ./isaaclab.sh -p scripts/save_env_cameras.py --camera_usd /path/to/scene.usd   # use cameras from USD (CameraTopXform / CameraWristXform)
  ./isaaclab.sh -p scripts/save_env_cameras.py --camera_json /path/to/cameras.json # override poses from JSON

Requires: run with isaaclab.sh (Isaac Sim). Cameras are enabled automatically.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def _pick_default_camera_json(project_root: Path) -> Path:
    candidates = (
        project_root / "camera_poses.json",
        project_root / "manipulation" / "camera_poses.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_DEFAULT_CAMERA_JSON = _pick_default_camera_json(_PROJECT_ROOT)
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_EXTENSION_SRC = _PROJECT_ROOT / "isaac_so_arm101" / "src"
if _EXTENSION_SRC.exists() and str(_EXTENSION_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SRC))

parser = argparse.ArgumentParser(description="Save env camera renders to PNGs")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-v0", help="Gym task id")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (saves first env's cameras by default)")
parser.add_argument("--output_dir", type=str, default="env_camera_samples", help="Directory to save PNGs")
parser.add_argument("--env_index", type=int, default=0, help="Which env's cameras to save (0 to num_envs-1)")
parser.add_argument("--camera_usd", type=str, default=None,
                    help="Load camera pose/intrinsics from this USD (CameraTopXform/CameraWristXform, with legacy side/up fallback)")
parser.add_argument("--camera_json", type=str, default=str(_DEFAULT_CAMERA_JSON),
                    help="Load camera pose overrides from JSON (default: manipulation/camera_poses.json)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force cameras on so we get image observations
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np

import isaac_so_arm101.tasks.reach  # noqa: F401
import isaac_so_arm101.tasks.lift   # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from camera_json_loader import apply_camera_json_to_env_cfg
from camera_usd_loader import apply_camera_usd_to_env_cfg
from env_wrapper import IsaacEEWrapper

_CANONICAL_CAMERA_OBS_KEYS = {
    "CameraTop": (
        "observation.images.top",
        "observation.images_top",
    ),
    "CameraSide": (
        "observation.images.side",
        "observation.images_side",
    ),
    "CameraWrist": (
        "observation.images.wrist",
        "observation.images_wrist",
        "observation.images.up",
        "observation.images_up",
    ),
}

# Order used when saving PNGs (lift-cube tasks with task cameras).
_SAVE_CAMERA_ORDER = ("CameraTop", "CameraSide", "CameraWrist")


def _print_camera_offsets(env_cfg) -> None:
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    for key in ("camera_top", "camera_wrist", "camera_side", "camera_up"):
        if not hasattr(scene, key):
            continue
        cam_cfg = getattr(scene, key)
        offset = getattr(cam_cfg, "offset", None)
        if offset is None:
            continue
        print(
            f"[save_env_cameras] {key}: pos={tuple(offset.pos)} "
            f"rot={tuple(offset.rot)} convention={offset.convention}"
        )


def _flatten_obs(obs, prefix=""):
    """Flatten nested dict to dotted keys."""
    out = {}
    for k, v in obs.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and not (hasattr(v, "shape") or hasattr(v, "dtype")):
            out.update(_flatten_obs(v, key))
        else:
            out[key] = v
    return out


def _is_image_array(arr):
    # Accept torch tensors or numpy arrays; always work on a CPU numpy view.
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    arr = np.asarray(arr)
    if arr.ndim not in (3, 4):
        return False
    # (N,H,W,C) or (H,W,C) or (N,C,H,W) or (C,H,W)
    if arr.ndim == 4:
        return arr.shape[-1] == 3 or arr.shape[1] == 3
    return arr.shape[-1] == 3 or arr.shape[0] == 3


def _to_uint8_rgb(img: np.ndarray, env_index: int = 0) -> np.ndarray:
    """Extract one env's image as (H,W,3) uint8."""
    if hasattr(img, "detach"):
        img = img.detach()
    if hasattr(img, "cpu"):
        img = img.cpu()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[env_index]
    # (C,H,W) -> (H,W,C)
    if img.shape[0] in (1, 3) and img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _pick_camera_observation_keys(flat_obs: dict) -> dict[str, str]:
    """Pick one observation key per canonical camera name."""
    selected = {}
    for camera_name, candidates in _CANONICAL_CAMERA_OBS_KEYS.items():
        for key in candidates:
            if key in flat_obs and _is_image_array(flat_obs[key]):
                selected[camera_name] = key
                break
    return selected


def main():
    task_id = args_cli.task
    reg = gym.envs.registry
    if task_id not in reg and not task_id.endswith("-v0"):
        alt = f"{task_id.rstrip('-v0')}-v0"
        if alt in reg:
            task_id = alt

    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.camera_usd:
        apply_camera_usd_to_env_cfg(env_cfg, args_cli.camera_usd)
        print(f"[save_env_cameras] Loaded cameras from {args_cli.camera_usd}")
    if args_cli.camera_json:
        apply_camera_json_to_env_cfg(env_cfg, args_cli.camera_json)
        print(f"[save_env_cameras] Loaded camera pose overrides from {args_cli.camera_json}")
    _print_camera_offsets(env_cfg)
    env = gym.make(task_id, cfg=env_cfg)
    env = IsaacEEWrapper(
        env,
        robot_name="robot",
        ee_link_name="gripper_link",
        add_ee_to_obs=True,
    )

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[save_env_cameras] Task={task_id} num_envs={args_cli.num_envs} output_dir={out_dir.absolute()}")

    obs, _ = env.reset()
    flat = _flatten_obs(obs)
    env_index = min(args_cli.env_index, args_cli.num_envs - 1)

    try:
        from PIL import Image
    except ImportError:
        print("PIL not found. Install with: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    selected_keys = _pick_camera_observation_keys(flat)
    saved = 0
    for camera_name in _SAVE_CAMERA_ORDER:
        key = selected_keys.get(camera_name)
        if key is None:
            continue
        value = flat[key]
        img = _to_uint8_rgb(value, env_index)
        path = out_dir / f"{camera_name}.png"
        Image.fromarray(img).save(path)
        print(f"  Saved {camera_name} from '{key}' -> {path} ({img.shape[0]}x{img.shape[1]})")
        saved += 1

    env.close()
    if saved == 0:
        print("[save_env_cameras] No image observations found. Ensure the task has cameras and --enable_cameras is set.")
        print("  Observation keys:", list(flat.keys()))
    elif saved < len(_SAVE_CAMERA_ORDER):
        missing = [name for name in _SAVE_CAMERA_ORDER if name not in selected_keys]
        print(f"[save_env_cameras] Warning: missing camera observations for {missing}")
        print(f"[save_env_cameras] Available image-like keys: {[k for k, v in flat.items() if _is_image_array(v)]}")
        print(f"[save_env_cameras] Done. Saved {saved} image(s) to {out_dir.absolute()}")
    else:
        print(f"[save_env_cameras] Done. Saved {saved} image(s) to {out_dir.absolute()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
