"""In-process LeIsaac rollout loop for SO-101 LiftCube with an OpenPI/FAST policy.

This intentionally does not use RLinf workers or IsaacLab subprocess wrappers.  The
simulator, observation conversion, and environment stepping all stay in this process
so CUDA tensors never cross a multiprocessing queue.

The script can run in two modes:

* ``--random_policy``: smoke-test the LeIsaac task and rollout recording path.
* default: query an OpenPI websocket policy server, e.g. one serving pi0-FAST.

The saved ``.npz`` files are the handoff point for the online RL learner.  The
standard OpenPI inference server returns decoded action chunks; for PPO/GRPO over
FAST tokens, extend the policy server to also return action token ids and token
logprobs, and this script will save those arrays when present.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Online-rollout scaffold for OpenPI pi0-FAST on LeIsaac SO-101 LiftCube."
)
parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=64)
parser.add_argument("--episode_length_s", type=float, default=5.0)
parser.add_argument("--action_horizon", type=int, default=10)
parser.add_argument("--prompt", type=str, default="Lift the red cube up.")
parser.add_argument("--camera_size", type=int, default=224)
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=8000)
parser.add_argument("--policy_timeout_ms", type=int, default=15000)
parser.add_argument("--policy_api_key", type=str, default=None)
parser.add_argument(
    "--obs_schema",
    choices=("openpi", "leisaac"),
    default="openpi",
    help=(
        "Observation keys sent to the policy server. 'openpi' uses "
        "observation/image, observation/wrist_image, observation/state; "
        "'leisaac' uses images/front, images/wrist, state."
    ),
)
parser.add_argument(
    "--random_policy",
    action="store_true",
    help="Use random SO-101 motor-unit actions instead of an OpenPI policy server.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="rollouts/pi_fast_so101",
    help="Directory for compressed rollout files and summary JSONL.",
)
parser.add_argument(
    "--save_images",
    action="store_true",
    help="Store policy-resolution front/wrist images in rollout files.",
)
parser.add_argument(
    "--learner_command",
    type=str,
    default=None,
    help=(
        "Optional command to run after each saved episode. The command receives "
        "ROLLOUT_PATH and ROLLOUT_DIR in its environment."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.policy.base import WebsocketServicePolicy
from leisaac.policy.openpi import image_tools
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

import leisaac  # noqa: F401  # Registers LeIsaac gym environments.


FAST_EXTRA_KEYS = (
    "action_tokens",
    "action_token_logprobs",
    "action_token_mask",
    "token_logprobs",
    "tokens",
    "logprobs",
)


@dataclass
class PolicyOutput:
    lerobot_actions: np.ndarray
    leisaac_actions: np.ndarray
    extras: dict[str, np.ndarray]


class OpenPIFastClient(WebsocketServicePolicy):
    """Thin OpenPI websocket client that keeps raw policy outputs for RL logging."""

    def get_action(self, *_args, **_kwargs) -> torch.Tensor:
        raise NotImplementedError("Use infer_policy() so raw OpenPI metadata is preserved.")

    def infer_policy(self, obs: dict[str, Any]) -> PolicyOutput:
        result = self.infer(obs)
        if "actions" not in result:
            raise KeyError(f"OpenPI server response is missing 'actions': {result.keys()}")

        lerobot_actions = np.asarray(result["actions"], dtype=np.float32)
        if lerobot_actions.ndim != 2 or lerobot_actions.shape[-1] != 6:
            raise ValueError(
                "Expected OpenPI actions with shape [horizon, 6] for SO-101, "
                f"got {lerobot_actions.shape}."
            )

        leisaac_actions = convert_lerobot_action_to_leisaac(lerobot_actions).astype(
            np.float32
        )
        extras = {
            key: np.asarray(result[key])
            for key in FAST_EXTRA_KEYS
            if key in result and result[key] is not None
        }
        return PolicyOutput(
            lerobot_actions=lerobot_actions,
            leisaac_actions=leisaac_actions,
            extras=extras,
        )


class RandomSO101Policy:
    """Random motor-unit action chunks for validating the env path."""

    def __init__(self, action_horizon: int):
        self.action_horizon = action_horizon

    def infer_policy(self, _obs: dict[str, Any]) -> PolicyOutput:
        actions = np.empty((self.action_horizon, 6), dtype=np.float32)
        actions[:, :5] = np.random.uniform(-25.0, 25.0, size=(self.action_horizon, 5))
        actions[:, 5] = np.random.uniform(40.0, 100.0, size=(self.action_horizon,))
        leisaac_actions = convert_lerobot_action_to_leisaac(actions).astype(np.float32)
        return PolicyOutput(
            lerobot_actions=actions,
            leisaac_actions=leisaac_actions,
            extras={},
        )


def configure_absolute_so101_actions(env_cfg: Any, task_type: str) -> None:
    """Make SO-101 leader actions absolute joint targets, matching OpenPI outputs."""

    if task_type != "so101leader":
        raise ValueError(f"This runner is for SO-101 leader tasks, got {task_type!r}.")
    for action_name in ("arm_action", "gripper_action"):
        action_cfg = getattr(env_cfg.actions, action_name, None)
        if action_cfg is None:
            continue
        if hasattr(action_cfg, "use_default_offset"):
            action_cfg.use_default_offset = False
        if hasattr(action_cfg, "scale"):
            action_cfg.scale = 1.0


def resize_camera_cfg(env_cfg: Any, camera_size: int) -> None:
    for camera_name in ("front", "wrist"):
        camera_cfg = getattr(env_cfg.scene, camera_name, None)
        if camera_cfg is None:
            continue
        camera_cfg.height = camera_size
        camera_cfg.width = camera_size


def first_env_item(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] == 1:
        value = value[0]
    return value


def policy_image(value: Any, camera_size: int) -> np.ndarray:
    image = first_env_item(value)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    image = image_tools.convert_to_uint8(image)
    return image_tools.resize_with_pad(image, camera_size, camera_size)


def policy_state(joint_pos: Any) -> np.ndarray:
    joint_pos_np = first_env_item(joint_pos)[None, :]
    return convert_leisaac_action_to_lerobot(joint_pos_np).squeeze(0).astype(np.float32)


def build_policy_obs(
    policy_obs: dict[str, Any],
    prompt: str,
    camera_size: int,
    obs_schema: str,
) -> dict[str, Any]:
    front = policy_image(policy_obs["front"], camera_size)
    wrist = policy_image(policy_obs["wrist"], camera_size)
    state = policy_state(policy_obs["joint_pos"]).astype(np.float64)
    if obs_schema == "openpi":
        return {
            "observation/image": front,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": prompt,
        }
    if obs_schema == "leisaac":
        return {
            "images/front": front,
            "images/wrist": wrist,
            "state": state,
            "prompt": prompt,
        }
    raise ValueError(f"Unsupported obs schema: {obs_schema}")


def append_obs_buffers(
    buffers: dict[str, list[np.ndarray]],
    policy_obs: dict[str, Any],
    *,
    camera_size: int,
    save_images: bool,
) -> None:
    buffers["states"].append(policy_state(policy_obs["joint_pos"]))
    if "ee_frame_state" in policy_obs:
        buffers["ee_frame_states"].append(first_env_item(policy_obs["ee_frame_state"]))
    if save_images:
        buffers["front_images"].append(policy_image(policy_obs["front"], camera_size))
        buffers["wrist_images"].append(policy_image(policy_obs["wrist"], camera_size))


def stack_or_empty(items: list[np.ndarray], shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if items:
        return np.stack(items)
    return np.empty(shape, dtype=dtype)


def pack_policy_extra(values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.stack(values)
    except ValueError:
        # FAST/BPE token sequences may be ragged if the policy server exposes them.
        return np.asarray(values, dtype=object)


def save_episode(
    output_dir: Path,
    episode_idx: int,
    buffers: dict[str, list[np.ndarray]],
    meta: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / f"episode_{episode_idx:06d}.npz"

    arrays = {
        "states": stack_or_empty(buffers["states"], (0, 6)),
        "actions_lerobot": stack_or_empty(buffers["actions_lerobot"], (0, 6)),
        "actions_leisaac": stack_or_empty(buffers["actions_leisaac"], (0, 6)),
        "rewards": np.asarray(buffers["rewards"], dtype=np.float32),
        "terminations": np.asarray(buffers["terminations"], dtype=np.bool_),
        "truncations": np.asarray(buffers["truncations"], dtype=np.bool_),
        "chunk_start_steps": np.asarray(buffers["chunk_start_steps"], dtype=np.int32),
        "chunk_lengths": np.asarray(buffers["chunk_lengths"], dtype=np.int32),
        "meta_json": np.asarray(json.dumps(meta), dtype=np.str_),
    }
    if buffers["ee_frame_states"]:
        arrays["ee_frame_states"] = np.stack(buffers["ee_frame_states"])
    if buffers["front_images"]:
        arrays["front_images"] = np.stack(buffers["front_images"])
        arrays["wrist_images"] = np.stack(buffers["wrist_images"])
    for key, values in buffers["policy_extras"].items():
        if values:
            arrays[key] = pack_policy_extra(values)

    np.savez_compressed(rollout_path, **arrays)
    return rollout_path


def maybe_run_learner(command: str | None, rollout_path: Path, output_dir: Path) -> None:
    if not command:
        return
    env = os.environ.copy()
    env["ROLLOUT_PATH"] = str(rollout_path)
    env["ROLLOUT_DIR"] = str(output_dir)
    subprocess.run(shlex.split(command), check=True, env=env)


def make_env() -> tuple[Any, str]:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    configure_absolute_so101_actions(env_cfg, task_type)
    resize_camera_cfg(env_cfg, args_cli.camera_size)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.recorders = None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped
    return env, task_type


def make_policy() -> OpenPIFastClient | RandomSO101Policy:
    if args_cli.random_policy:
        return RandomSO101Policy(args_cli.action_horizon)
    return OpenPIFastClient(
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        timeout_ms=args_cli.policy_timeout_ms,
        api_key=args_cli.policy_api_key,
    )


def main() -> None:
    output_dir = Path(args_cli.output_dir).expanduser()
    summary_path = output_dir / "summary.jsonl"
    env, task_type = make_env()
    policy = make_policy()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for episode_idx in range(args_cli.episodes):
            obs, _ = env.reset(seed=args_cli.seed)
            buffers: dict[str, Any] = {
                "states": [],
                "ee_frame_states": [],
                "front_images": [],
                "wrist_images": [],
                "actions_lerobot": [],
                "actions_leisaac": [],
                "rewards": [],
                "terminations": [],
                "truncations": [],
                "chunk_start_steps": [],
                "chunk_lengths": [],
                "policy_extras": {key: [] for key in FAST_EXTRA_KEYS},
            }

            total_reward = 0.0
            step_count = 0
            success = False
            timeout = False

            while simulation_app.is_running() and step_count < args_cli.max_steps:
                policy_obs = obs["policy"]
                openpi_obs = build_policy_obs(
                    policy_obs,
                    args_cli.prompt,
                    args_cli.camera_size,
                    args_cli.obs_schema,
                )
                policy_output = policy.infer_policy(openpi_obs)
                chunk_len = min(args_cli.action_horizon, len(policy_output.leisaac_actions))
                chunk_start_step = step_count
                executed_chunk_steps = 0

                for action_idx in range(chunk_len):
                    if step_count >= args_cli.max_steps:
                        break
                    append_obs_buffers(
                        buffers,
                        policy_obs,
                        camera_size=args_cli.camera_size,
                        save_images=args_cli.save_images,
                    )
                    buffers["actions_lerobot"].append(
                        policy_output.lerobot_actions[action_idx]
                    )
                    buffers["actions_leisaac"].append(
                        policy_output.leisaac_actions[action_idx]
                    )

                    action = torch.as_tensor(
                        policy_output.leisaac_actions[action_idx][None, :],
                        dtype=torch.float32,
                        device=env.device,
                    )
                    if getattr(env.cfg, "dynamic_reset_gripper_effort_limit", False):
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)

                    obs, reward, terminated, truncated, _ = env.step(action)
                    reward_value = float(first_env_item(reward).reshape(-1)[0])
                    terminated_value = bool(first_env_item(terminated).reshape(-1)[0])
                    truncated_value = bool(first_env_item(truncated).reshape(-1)[0])

                    buffers["rewards"].append(np.asarray(reward_value, dtype=np.float32))
                    buffers["terminations"].append(
                        np.asarray(terminated_value, dtype=np.bool_)
                    )
                    buffers["truncations"].append(np.asarray(truncated_value, dtype=np.bool_))

                    total_reward += reward_value
                    step_count += 1
                    executed_chunk_steps += 1
                    policy_obs = obs["policy"]

                    if terminated_value or truncated_value:
                        success = terminated_value
                        timeout = truncated_value
                        break

                if executed_chunk_steps > 0:
                    buffers["chunk_start_steps"].append(np.asarray(chunk_start_step, dtype=np.int32))
                    buffers["chunk_lengths"].append(np.asarray(executed_chunk_steps, dtype=np.int32))
                    for key, value in policy_output.extras.items():
                        if key in buffers["policy_extras"]:
                            buffers["policy_extras"][key].append(np.asarray(value))

                if success or timeout:
                    break

            meta = {
                "episode": episode_idx,
                "task": args_cli.task,
                "prompt": args_cli.prompt,
                "steps": step_count,
                "return": total_reward,
                "success": success,
                "timeout": timeout,
                "random_policy": args_cli.random_policy,
                "saved_images": args_cli.save_images,
                "obs_schema": args_cli.obs_schema,
            }
            rollout_path = save_episode(output_dir, episode_idx, buffers, meta)
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**meta, "rollout_path": str(rollout_path)}) + "\n")
            print(
                f"[episode {episode_idx}] steps={step_count} "
                f"return={total_reward:.4f} success={success} timeout={timeout} "
                f"saved={rollout_path}"
            )
            maybe_run_learner(args_cli.learner_command, rollout_path, output_dir)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
