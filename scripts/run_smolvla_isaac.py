#!/usr/bin/env python3
"""
Run SmolVLA policy in Isaac Lab SO-101 (lift-cube or reach).
Usage: ./isaaclab.sh -p /path/to/scripts/run_smolvla_isaac.py --task Isaac-SO-ARM101-Lift-Cube-v0
For the Lift task with real camera images, add: --enable_cameras
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_EXTENSION_SRC = _PROJECT_ROOT / "isaac_so_arm101" / "src"
if _EXTENSION_SRC.exists() and str(_EXTENSION_SRC) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_SRC))

# add argparse arguments
parser = argparse.ArgumentParser(description="SmolVLA inference on Isaac Lab SO-101 env")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Lift-Cube-v0", help="Gym task id")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
parser.add_argument("--policy", type=str, default="lerobot/smolvla_base", help="HuggingFace policy repo or path")
parser.add_argument("--instruction", type=str, default="Pick the cube.", help="Language instruction for the policy")
parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
parser.add_argument("--robot_name", type=str, default="robot", help="Robot articulation name in scene")
parser.add_argument("--ee_link_name", type=str, default="gripper_link", help="End-effector link name (SO-101: gripper_link)")
parser.add_argument("--no_ee_in_obs", action="store_true", help="Do not add ee_pos/ee_quat/delta to obs dict")
parser.add_argument("--observation_state_size", type=int, default=6,
                    help="Observation state vector length to match model normalization (default 6 for svla_so101_pickplace)")
parser.add_argument("--rename_map", type=str, default=None,
                    help='JSON map env_key->policy_key, e.g. \'{"observation.images.side": "observation.images.camera1", '
                    '"observation.images.up": "observation.images.camera2"}\'')
parser.add_argument("--empty_cameras", type=int, default=1,
                    help="Number of trailing policy camera slots to fill with zeros (default 1 for SmolVLA side+up+empty)")
parser.add_argument("--camera_usd", type=str, default=None,
                    help="Load camera pose/intrinsics from this USD (cameras under CameraSideXform and CameraUpXform)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app (must run before any isaaclab/pxr imports)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np

import isaac_so_arm101.tasks.reach  # noqa: F401
import isaac_so_arm101.tasks.lift   # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from adapters import isaac_obs_to_policy_frame, policy_action_to_env
from camera_usd_loader import apply_camera_usd_to_env_cfg
from env_wrapper import IsaacEEWrapper


def main():
    def _find_camera_and_dt(env):
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
        camera = None
        if hasattr(base, "scene") and hasattr(base.scene, "sensors"):
            for name, sensor in base.scene.sensors.items():
                if hasattr(sensor, "data") and hasattr(sensor.data, "output"):
                    if isinstance(getattr(sensor.data, "output", None), dict) and "rgb" in sensor.data.output:
                        camera = sensor
                        break
        return camera, dt

    # Load policy and processors (LeRobot)
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as e:
        print("LeRobot/SmolVLA not installed. Install with: pip install 'lerobot[smolvla]'", file=sys.stderr)
        raise SystemExit(1) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Policy] Loading from: {args_cli.policy}")
    policy = SmolVLAPolicy.from_pretrained(args_cli.policy).to(device).eval()
    # Weight fingerprint: ensure weights are loaded (same repo => same norm; different => different)
    with torch.inference_mode():
        total_norm = 0.0
        n_params = 0
        for p in policy.parameters():
            total_norm += p.float().norm().item() ** 2
            n_params += 1
            if n_params >= 10:
                break
        weight_fingerprint = total_norm ** 0.5
    print(f"[Policy] Loaded. Weight fingerprint (norm of first 10 params): {weight_fingerprint:.6f}")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args_cli.policy,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    task_id = args_cli.task
    reg = gym.envs.registry
    if task_id not in reg and not task_id.endswith("-v0"):
        alt = f"{task_id.rstrip('-v0')}-v0"
        if alt in reg:
            task_id = alt

    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.camera_usd:
        apply_camera_usd_to_env_cfg(env_cfg, args_cli.camera_usd)
        print(f"[Cameras] Loaded from {args_cli.camera_usd}")
    # Override episode length so the env allows max_steps (env otherwise truncates at episode_length_s)
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = args_cli.max_steps * step_dt
    env = gym.make(task_id, cfg=env_cfg)

    env = IsaacEEWrapper(
        env,
        robot_name=args_cli.robot_name,
        ee_link_name=args_cli.ee_link_name,
        add_ee_to_obs=not args_cli.no_ee_in_obs,
    )

    camera, sim_dt = _find_camera_and_dt(env)
    if camera is not None:
        print("[Camera] found.")

    action_shape = tuple(env.action_space.shape)
    num_envs = args_cli.num_envs

    for ep in range(args_cli.episodes):
        obs, info = env.reset()
        step = 0
        while step < args_cli.max_steps:
            if isinstance(obs, dict):
                single_obs = {}
                for k, v in obs.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            single_obs[f"{k}.{sub_k}"] = sub_v[0] if hasattr(sub_v, "shape") and sub_v.shape[:1] == (num_envs,) else sub_v
                    else:
                        single_obs[k] = v[0] if hasattr(v, "shape") and v.shape[:1] == (num_envs,) else v
            else:
                single_obs = {"obs": obs[0] if obs.shape[:1] == (num_envs,) else obs}
            rename_map = None
            if args_cli.rename_map:
                rename_map = json.loads(args_cli.rename_map)
            elif args_cli.empty_cameras >= 1:
                # Default: map lift task camera keys (observation.images_side/up) to policy camera1/camera2
                rename_map = {
                    "observation.images_side": "observation.images.camera1",
                    "observation.images_up": "observation.images.camera2",
                }
            frame = isaac_obs_to_policy_frame(
                single_obs,
                language_instruction=args_cli.instruction,
                observation_state_size=args_cli.observation_state_size,
                rename_map=rename_map,
                empty_cameras=args_cli.empty_cameras,
            )
            if ep == 0 and step == 0:
                print(f"[Images] Env observation keys: {list(single_obs.keys())}")
                for key in ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3"):
                    if key not in frame:
                        print(f"[Images] {key}: missing")
                        continue
                    x = np.asarray(frame[key])
                    print(f"[Images] {key}: shape={x.shape} min={x.min():.4f} max={x.max():.4f} mean={x.mean():.4f} zeros={100 * (x == 0).mean():.1f}%")
                # Warn if env provides no images (e.g. Lift task has no cameras by default)
                all_zero = all(
                    np.asarray(frame.get(k, np.zeros(1))).size > 0 and np.asarray(frame[k]).mean() == 0.0
                    for k in ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")
                )
                if all_zero:
                    print("[Images] WARNING: All camera slots are zeros. This env does not provide image observations."
                          " Add camera sensors and observation terms to the task (see Isaac Lab Camera docs),"
                          " or use an env that includes cameras.")
            batch = preprocess(frame)
            for k, v in batch.items():
                if isinstance(v, np.ndarray):
                    batch[k] = torch.as_tensor(v, device=device)
                elif isinstance(v, torch.Tensor) and v.device != device:
                    batch[k] = v.to(device)
            if ep == 0 and step == 0:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 3:
                        if "image" in k.lower() or "pixel" in k.lower() or "camera" in k.lower() or v.numel() > 1e6:
                            a = v.detach().float().cpu().numpy()
                            print(f"[Batch] {k}: shape={a.shape} min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            action = action.cpu().numpy() if hasattr(action, "cpu") else np.asarray(action)
            if ep == 0 and step == 0:
                a = np.asarray(action).flatten()
                print(f"[Policy] First-step action stats: min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f} (near zero => check model/obs)")
            env_action = policy_action_to_env(action, env_action_space_shape=action_shape, clip=True)
            if env_action.ndim == 1:
                env_action = np.broadcast_to(env_action, (num_envs, env_action.shape[0])).copy()
            inner = env
            while hasattr(inner, "env"):
                inner = inner.env
            env_action_t = torch.as_tensor(env_action, device=inner.device, dtype=torch.float32)
            obs, reward, terminated, truncated, info = env.step(env_action_t)
            step += 1
            if camera is not None:
                camera.update(dt=sim_dt)
                if "rgb" in camera.data.output and camera.data.output["rgb"].shape[0] > 0:
                    rgb_np = camera.data.output["rgb"][0].cpu().numpy()
                    if ep == 0 and step == 1:
                        print(f"[Camera] RGB shape={rgb_np.shape}")
            if (terminated.any() if hasattr(terminated, "any") else terminated) or (truncated.any() if hasattr(truncated, "any") else truncated):
                break
        print(f"Episode {ep + 1}/{args_cli.episodes} done ({step} steps).")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
