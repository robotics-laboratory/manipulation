"""Collect a LeRobot v3 dataset using a trained RSL-RL MLP agent.

Runs the agent in the Isaac Lab environment with cameras enabled, records
image observations (top + wrist), proprioceptive state, and the MLP's
actions into a LeRobot v3 dataset (Parquet + MP4 videos).

Usage (from the manipulation/isaac_so_arm101 directory)::

    isaaclab -p src/isaac_so_arm101/scripts/rsl_rl/collect_lerobot_dataset.py \\
        --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \\
        --num_episodes 50 \\
        --output_dir datasets/so101_lift_cube \\
        --repo_id local/so101-lift-cube
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# LeRobot (and torchvision it pulls in) must be imported BEFORE AppLauncher to
# avoid a TypeError crash in inspect.getfile when torch.library.register_fake
# walks the call stack and encounters the isaaclab namespace package.
# See manipulation/docs/torch-fix-notes.md for the full explanation.
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
    default="local/so101-lift-cube",
    help="LeRobot dataset repo ID (used for metadata; no upload unless --push_to_hub).",
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

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks.lift  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from log_paths import rsl_rl_root


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

    # ----------------------------------------------------------- checkpoint
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    log_root = os.path.abspath(
        os.path.join(rsl_rl_root(), agent_cfg.experiment_name)
    )
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
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
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
        },
        "action": {
            "dtype": "float32",
            "shape": [action_dim],
        },
    }

    fps = int(1.0 / (env_cfg.sim.dt * env_cfg.decimation))
    output_dir = os.path.abspath(args_cli.output_dir)
    dataset_dir = Path(output_dir)
    if dataset_dir.exists():
        # Resume: load the existing dataset and keep appending
        dataset = LeRobotDataset(
            repo_id=args_cli.repo_id,
            root=output_dir,
        )
        print(f"[INFO] Resuming existing dataset ({dataset.num_episodes} episodes already recorded)")
    else:
        # Fresh start: create from scratch
        dataset = LeRobotDataset.create(
            repo_id=args_cli.repo_id,
            fps=fps,
            features=features,
            robot_type="so101",
            root=output_dir,
        )

    print(
        f"[INFO] Dataset: {output_dir}  fps={fps}  "
        f"episodes={args_cli.num_episodes}  "
        f"mode={'success-only' if args_cli.success_only else 'all'}"
    )

    # ------------------------------------------------------ rollout & record
    obs = env.get_observations()
    episodes_saved = 0
    episodes_attempted = 0
    total_frames = 0
    episode_had_lift = False

    while episodes_saved < args_cli.num_episodes and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)

        top_img = _extract_image(unwrapped, "camera_top")
        wrist_img = _extract_image(unwrapped, "camera_wrist")
        state = _extract_state(unwrapped)

        if _cube_is_lifted(unwrapped):
            episode_had_lift = True

        dataset.add_frame(
            {
                "observation.images.top": top_img,
                "observation.images.wrist": wrist_img,
                "observation.state": state,
                "action": actions[0].cpu(),
                "task": args_cli.task_description,
            }
        )
        total_frames += 1

        with torch.inference_mode():
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        if dones.any():
            episodes_attempted += 1
            if args_cli.success_only and not episode_had_lift:
                dataset.clear_episode_buffer()
                print(
                    f"  Episode {episodes_attempted}: FAILED (no lift) – discarded"
                )
            else:
                dataset.save_episode()
                episodes_saved += 1
                tag = "LIFTED" if episode_had_lift else "no lift"
                print(
                    f"  Episode {episodes_saved}/{args_cli.num_episodes}: {tag}"
                )
            episode_had_lift = False

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
