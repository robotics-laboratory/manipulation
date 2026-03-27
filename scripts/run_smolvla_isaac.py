#!/usr/bin/env python3
"""
Run SmolVLA policy in Isaac Lab SO-101 (lift-cube or reach).
Usage:
  ./isaaclab.sh -p manipulation/scripts/run_smolvla_isaac.py --task Isaac-SO-ARM101-Lift-Cube-v0 --enable_cameras
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

# Import torch / torchvision / LeRobot *before* ``isaaclab`` (via AppLauncher). Otherwise
# ``torchvision``'s ``@torch.library.register_fake`` runs after ``isaaclab`` is loaded as a
# namespace package and ``inspect.getsource`` / ``getfile`` can raise
# ``TypeError: ... 'isaaclab' ... is a built-in module`` inside Isaac Sim's Kit Python.
try:
    import torch
    import torchvision  # noqa: F401 — ensure torchvision fake ops register early
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except ImportError as err:
    print("LeRobot/SmolVLA not installed. Install with: pip install 'lerobot[smolvla]'", file=sys.stderr)
    raise SystemExit(1) from err

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SmolVLA inference on Isaac Lab SO-101 env")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-v0", help="Gym task id")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
parser.add_argument("--policy", type=str, default="lerobot/smolvla_base", help="HuggingFace policy repo or path")
parser.add_argument("--instruction", type=str, default="Pick the cube.", help="Language instruction for the policy")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
parser.add_argument("--record_video", action="store_true", help="Record evaluation episodes to MP4 files.")
parser.add_argument(
    "--video_folder",
    type=str,
    default="output/eval_videos",
    help="Directory for saved videos (relative to project root unless absolute).",
)
parser.add_argument("--video_fps", type=int, default=30, help="FPS for saved videos.")
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Number of steps per recorded clip (default: max_steps).",
)
parser.add_argument("--robot_name", type=str, default="robot", help="Robot articulation name in scene")
parser.add_argument("--ee_link_name", type=str, default="gripper_link", help="End-effector link name (SO-101)")
parser.add_argument("--no_ee_in_obs", action="store_true", help="Do not add ee_* terms to obs dict")
parser.add_argument(
    "--observation_state_size",
    type=int,
    default=6,
    help="Observation state vector length to match model normalization.",
)
parser.add_argument(
    "--no_dataset_joint_action_space",
    action="store_true",
    help="Disable SO101 absolute-joint action compatibility mode.",
)
parser.add_argument(
    "--clip_actions",
    action="store_true",
    help="Clip policy actions to [-1, 1] before stepping env.",
)
parser.add_argument(
    "--gripper_binary_threshold",
    type=float,
    default=None,
    help=(
        "Optional threshold to force binary gripper conversion. "
        "Default None keeps continuous output and relies on env sign-based binarization."
    ),
)
parser.add_argument(
    "--policy-action-in-degrees",
    dest="policy_action_in_degrees",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Convert first 5 arm joint targets from degrees to radians (LeRobot SO-101 / SmolVLA finetunes). "
        "Disable with --no-policy-action-in-degrees if the policy already outputs radians."
    ),
)
parser.add_argument(
    "--rename_map",
    type=str,
    default=None,
    help='JSON map env_key->policy_key, e.g. {"observation.images.top":"observation.images.camera1"}',
)
parser.add_argument(
    "--empty_cameras",
    type=int,
    default=None,
    help="Override empty camera slots. Default: use policy config (e.g. smolvla_base uses 0).",
)
parser.add_argument("--camera_usd", type=str, default=None, help="Load camera pose/intrinsics from USD.")
parser.add_argument(
    "--camera_json",
    type=str,
    default=str(_DEFAULT_CAMERA_JSON),
    help="Load camera pose overrides from JSON.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np

import isaac_so_arm101.tasks.lift  # noqa: F401
import isaac_so_arm101.tasks.reach  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from adapters import isaac_obs_to_policy_frame, policy_action_to_env
from camera_json_loader import apply_camera_json_to_env_cfg
from camera_usd_loader import apply_camera_usd_to_env_cfg
from env_wrapper import IsaacEEWrapper

_CANONICAL_CAMERA_KEYS = {
    "observation.images.top": (
        "observation.images.top",
        "observation.images_top",
        "observation.images.side",
        "observation.images_side",
    ),
    "observation.images.wrist": (
        "observation.images.wrist",
        "observation.images_wrist",
        "observation.images.up",
        "observation.images_up",
    ),
}

_SO101_STATE_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
_SO101_ARM_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def _policy_image_keys(policy) -> list[str]:
    """Read expected visual feature keys from policy config."""
    keys: list[str] = []
    image_features = getattr(policy.config, "image_features", None)
    if isinstance(image_features, dict):
        keys.extend(list(image_features.keys()))
    elif isinstance(image_features, (list, tuple)):
        keys.extend([str(k) for k in image_features])

    if not keys:
        input_features = getattr(policy.config, "input_features", None)
        if isinstance(input_features, dict):
            for key, feat in input_features.items():
                ftype = getattr(feat, "type", None)
                if ftype is not None and str(ftype).upper().endswith("VISUAL"):
                    keys.append(key)
                elif isinstance(feat, dict) and str(feat.get("type", "")).upper().endswith("VISUAL"):
                    keys.append(key)
    # Keep only image obs keys in deterministic order.
    keys = [k for k in keys if k.startswith("observation.images.")]
    return sorted(set(keys))


def _default_rename_map_for_policy(policy) -> dict[str, str] | None:
    """Map env top/wrist cameras to whichever keys the current policy expects."""
    keys = _policy_image_keys(policy)
    if not keys:
        return None

    # Candidate destinations by preference.
    top_candidates = ("observation.images.top", "observation.images.side", "observation.images.camera1")
    wrist_candidates = ("observation.images.wrist", "observation.images.up", "observation.images.camera2")

    top_dst = next((k for k in top_candidates if k in keys), None)
    wrist_dst = next((k for k in wrist_candidates if k in keys), None)

    # Fallback: first two keys from policy if semantic names absent.
    if top_dst is None and len(keys) >= 1:
        top_dst = keys[0]
    if wrist_dst is None and len(keys) >= 2:
        wrist_dst = keys[1] if keys[1] != top_dst else (keys[2] if len(keys) >= 3 else None)

    out: dict[str, str] = {}
    if top_dst is not None:
        out["observation.images.top"] = top_dst
    if wrist_dst is not None:
        out["observation.images.wrist"] = wrist_dst
    return out or None


def _get_env_action_joint_order(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    action_manager = getattr(base, "action_manager", None)
    if action_manager is None:
        return None, None
    arm_joint_names = None
    gripper_joint_names = None
    try:
        arm_term = action_manager.get_term("arm_action")
        arm_joint_names = tuple(getattr(arm_term, "_joint_names", None) or ())
    except Exception:
        pass
    try:
        gripper_term = action_manager.get_term("gripper_action")
        gripper_joint_names = tuple(getattr(gripper_term, "_joint_names", None) or ())
    except Exception:
        pass
    return arm_joint_names, gripper_joint_names


def _remap_policy_action_to_env_order(
    action: np.ndarray,
    env_arm_joint_order: tuple[str, ...] | None,
    gripper_binary_threshold: float | None = None,
) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).flatten()
    if a.size < 6:
        return a
    mapped = a.copy()
    if env_arm_joint_order and len(env_arm_joint_order) == len(_SO101_ARM_JOINT_ORDER):
        name_to_idx = {name: i for i, name in enumerate(_SO101_ARM_JOINT_ORDER)}
        for env_i, joint_name in enumerate(env_arm_joint_order):
            if joint_name in name_to_idx:
                mapped[env_i] = a[name_to_idx[joint_name]]
    # Keep continuous gripper action by default. The env's BinaryJointPositionAction
    # already converts by sign (<0 close, >=0 open). Optional thresholding is kept
    # for experiments only.
    if gripper_binary_threshold is not None:
        mapped[5] = 1.0 if float(a[5]) > gripper_binary_threshold else -1.0
    else:
        mapped[5] = float(a[5])
    return mapped


def _extract_dataset_joint_state(env, robot_name: str | None) -> np.ndarray | None:
    base = env
    while hasattr(base, "env"):
        base = base.env
    scene = getattr(base, "scene", None)
    if scene is None:
        return None
    articulations = getattr(scene, "articulations", None)
    if articulations is None:
        return None
    art = articulations.get(robot_name) if hasattr(articulations, "get") and robot_name else None
    if art is None and hasattr(articulations, "__len__") and len(articulations) > 0:
        art = articulations[0]
    if art is None:
        return None
    joint_names = list(getattr(art, "joint_names", []))
    if not joint_names:
        return None
    idx_by_name = {name: i for i, name in enumerate(joint_names)}
    if any(name not in idx_by_name for name in _SO101_STATE_JOINT_ORDER):
        return None
    joint_pos = getattr(getattr(art, "data", None), "joint_pos", None)
    if joint_pos is None:
        return None
    joint_pos = joint_pos[0] if hasattr(joint_pos, "shape") and len(joint_pos.shape) >= 2 else joint_pos
    joint_pos_np = joint_pos.detach().cpu().numpy() if hasattr(joint_pos, "detach") else np.asarray(joint_pos)
    return np.asarray([joint_pos_np[idx_by_name[name]] for name in _SO101_STATE_JOINT_ORDER], dtype=np.float32)


def _normalize_camera_obs_keys(obs: dict[str, object]) -> None:
    for canonical_key, aliases in _CANONICAL_CAMERA_KEYS.items():
        if canonical_key not in obs:
            for key in aliases:
                if key in obs:
                    obs[canonical_key] = obs[key]
                    break
        for key in aliases:
            if key != canonical_key and key in obs:
                del obs[key]


def main():
    def _find_cameras_and_dt(env):
        base = env
        while hasattr(base, "env"):
            base = base.env
        dt = None
        if hasattr(base, "sim") and base.sim is not None:
            try:
                dt = base.sim.get_physics_dt()
            except Exception:
                pass
        if dt is None:
            dt = 1.0 / 60.0
        cameras = {}
        if hasattr(base, "scene") and hasattr(base.scene, "sensors"):
            for name, sensor in base.scene.sensors.items():
                if hasattr(sensor, "data") and hasattr(sensor.data, "output"):
                    out = getattr(sensor.data, "output", None)
                    if isinstance(out, dict) and "rgb" in out:
                        cameras[name] = sensor
        return cameras, dt

    def _camera_obs_key(sensor_name: str) -> str:
        lower = sensor_name.lower()
        if "wrist" in lower or "up" in lower:
            return "observation.images.wrist"
        if "top" in lower or "side" in lower:
            return "observation.images.top"
        return f"observation.images.{sensor_name}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Policy] Loading from: {args_cli.policy}")
    policy = SmolVLAPolicy.from_pretrained(args_cli.policy).to(device).eval()
    # Some community fine-tunes contain weights/config but not full processor
    # artifacts (e.g. policy_preprocessor.json). Fall back to pretrained base.
    processor_sources = [args_cli.policy]
    pretrained_path = getattr(policy.config, "pretrained_path", None)
    if pretrained_path and pretrained_path not in processor_sources:
        processor_sources.append(pretrained_path)
    if "lerobot/smolvla_base" not in processor_sources:
        processor_sources.append("lerobot/smolvla_base")

    preprocess = None
    postprocess = None
    last_err: Exception | None = None
    for source in processor_sources:
        try:
            preprocess, postprocess = make_pre_post_processors(
                policy.config,
                source,
                preprocessor_overrides={"device_processor": {"device": str(device)}},
            )
            if source != args_cli.policy:
                print(f"[Policy] Processor config fallback source: {source}")
            break
        except FileNotFoundError as err:
            last_err = err
            continue
    if preprocess is None or postprocess is None:
        raise FileNotFoundError(
            "Failed to load policy processors from all sources: "
            + ", ".join(processor_sources)
        ) from last_err
    expected_img_keys = _policy_image_keys(policy)
    if expected_img_keys:
        print(f"[Policy] Expected image keys: {expected_img_keys}")
    resolved_empty_cameras = (
        int(args_cli.empty_cameras)
        if args_cli.empty_cameras is not None
        else int(getattr(policy.config, "empty_cameras", 0))
    )
    print(f"[Policy] empty_cameras={resolved_empty_cameras} (resolved)")

    task_id = args_cli.task
    reg = gym.envs.registry
    if task_id not in reg and not task_id.endswith("-v0"):
        alt = f"{task_id.rstrip('-v0')}-v0"
        if alt in reg:
            task_id = alt

    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    if not args_cli.no_dataset_joint_action_space:
        arm_action_cfg = getattr(getattr(env_cfg, "actions", None), "arm_action", None)
        if arm_action_cfg is not None:
            if hasattr(arm_action_cfg, "scale"):
                arm_action_cfg.scale = 1.0
            if hasattr(arm_action_cfg, "use_default_offset"):
                arm_action_cfg.use_default_offset = False
            print("[Actions] Dataset compatibility: arm_action.scale=1.0, use_default_offset=False")
    if args_cli.camera_usd:
        apply_camera_usd_to_env_cfg(env_cfg, args_cli.camera_usd)
    if args_cli.camera_json:
        apply_camera_json_to_env_cfg(env_cfg, args_cli.camera_json)

    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = args_cli.max_steps * step_dt
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.record_video else None)
    if args_cli.record_video:
        video_dir = Path(args_cli.video_folder)
        if not video_dir.is_absolute():
            video_dir = _PROJECT_ROOT / video_dir
        video_length = args_cli.video_length if args_cli.video_length is not None else args_cli.max_steps
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda ep: ep < args_cli.episodes,
            video_length=video_length,
            name_prefix="smolvla_eval",
            disable_logger=True,
            fps=args_cli.video_fps,
        )
        print(f"[Video] Recording enabled. Saving to: {video_dir}")
    env = IsaacEEWrapper(
        env,
        robot_name=args_cli.robot_name,
        ee_link_name=args_cli.ee_link_name,
        add_ee_to_obs=not args_cli.no_ee_in_obs,
    )

    cameras, sim_dt = _find_cameras_and_dt(env)
    if cameras:
        print(f"[Camera] found RGB sensors: {list(cameras.keys())}")
    action_shape = tuple(env.action_space.shape)
    num_envs = args_cli.num_envs
    env_arm_joint_order, env_gripper_joint_order = _get_env_action_joint_order(env)
    if env_arm_joint_order:
        print(f"[Actions] Env arm joint order: {env_arm_joint_order}")
    if env_gripper_joint_order:
        print(f"[Actions] Env gripper joint order: {env_gripper_joint_order}")

    for ep in range(args_cli.episodes):
        obs, _ = env.reset()
        step = 0
        while step < args_cli.max_steps:
            if isinstance(obs, dict):
                single_obs = {}
                for key, value in obs.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_key = f"{key}.{sub_key}"
                            if hasattr(sub_value, "shape") and sub_value.shape[:1] == (num_envs,):
                                single_obs[flat_key] = sub_value[0]
                            else:
                                single_obs[flat_key] = sub_value
                    else:
                        if hasattr(value, "shape") and value.shape[:1] == (num_envs,):
                            single_obs[key] = value[0]
                        else:
                            single_obs[key] = value
            else:
                single_obs = {"obs": obs[0] if obs.shape[:1] == (num_envs,) else obs}

            dataset_state = _extract_dataset_joint_state(env, args_cli.robot_name)
            if dataset_state is not None:
                single_obs["observation.state"] = dataset_state

            for sensor_name, sensor in cameras.items():
                sensor.update(dt=sim_dt)
                if "rgb" not in sensor.data.output or sensor.data.output["rgb"].shape[0] == 0:
                    continue
                single_obs[_camera_obs_key(sensor_name)] = sensor.data.output["rgb"][0]
            _normalize_camera_obs_keys(single_obs)

            rename_map = None
            if args_cli.rename_map:
                rename_map = json.loads(args_cli.rename_map)
            else:
                rename_map = _default_rename_map_for_policy(policy)

            frame = isaac_obs_to_policy_frame(
                single_obs,
                language_instruction=args_cli.instruction,
                observation_state_size=args_cli.observation_state_size,
                rename_map=rename_map,
                empty_cameras=resolved_empty_cameras,
            )

            if ep == 0 and step == 0:
                print(f"[Images] Env observation keys: {list(single_obs.keys())}")
                print(f"[Images] rename_map: {rename_map}")
                if "observation.state" in single_obs:
                    state_np = np.asarray(single_obs["observation.state"]).flatten()
                    print(f"[State] observation.state: shape={state_np.shape} values={np.array2string(state_np, precision=4)}")
                frame_cam_keys = sorted(k for k in frame.keys() if k.startswith("observation.images."))
                for cam_key in frame_cam_keys:
                    if cam_key not in frame:
                        print(f"[Images] {cam_key}: missing")
                        continue
                    x = np.asarray(frame[cam_key])
                    print(
                        f"[Images] {cam_key}: shape={x.shape} min={x.min():.4f} "
                        f"max={x.max():.4f} mean={x.mean():.4f} zeros={100*(x==0).mean():.1f}%"
                    )

            batch = preprocess(frame)
            for key, value in batch.items():
                if isinstance(value, np.ndarray):
                    batch[key] = torch.as_tensor(value, device=device)
                elif isinstance(value, torch.Tensor) and value.device != device:
                    batch[key] = value.to(device)

            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            action = action.cpu().numpy() if hasattr(action, "cpu") else np.asarray(action)
            action = np.asarray(action, dtype=np.float32).flatten()
            # LeRobot SO-101 datasets use degrees for joint targets; Isaac JointPositionAction expects radians.
            if args_cli.policy_action_in_degrees and action.shape[0] >= 5:
                action = action.copy()
                action[:5] = action[:5] * (np.pi / 180.0)
            action = _remap_policy_action_to_env_order(
                action,
                env_arm_joint_order=env_arm_joint_order,
                gripper_binary_threshold=args_cli.gripper_binary_threshold,
            )
            if ep == 0 and step == 0:
                if args_cli.policy_action_in_degrees:
                    print("[Policy] Arm joint commands converted from degrees to radians (first 5 dims).")
                a = np.asarray(action).flatten()
                print(f"[Policy] First-step action stats: min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")

            env_action = policy_action_to_env(action, env_action_space_shape=action_shape, clip=args_cli.clip_actions)
            if env_action.ndim == 1:
                env_action = np.broadcast_to(env_action, (num_envs, env_action.shape[0])).copy()

            inner = env
            while hasattr(inner, "env"):
                inner = inner.env
            env_action_t = torch.as_tensor(env_action, device=inner.device, dtype=torch.float32)
            obs, _, terminated, truncated, _ = env.step(env_action_t)
            step += 1
            done = (terminated.any() if hasattr(terminated, "any") else terminated) or (
                truncated.any() if hasattr(truncated, "any") else truncated
            )
            if done:
                break
        print(f"Episode {ep + 1}/{args_cli.episodes} done ({step} steps).")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
