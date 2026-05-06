"""Run VLA inference and capture stuck LiftCube states for offline MLP recovery collection."""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture LiftCube stuck states during VLA inference.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode timeout in seconds.")
parser.add_argument("--eval_rounds", type=int, default=0, help="Number of evaluation rounds. 0 runs forever.")
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, gr00tn1.6, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
parser.add_argument(
    "--policy_must_go",
    action="store_true",
    default=False,
    help="Force LeRobot policy server inference for every observation.",
)
parser.add_argument("--capture_dir", type=str, default="output/recovery_states/lift_cube_vla_stuck")
parser.add_argument("--capture_run_name", type=str, default=None, help="Optional subdirectory name for this capture run.")
parser.add_argument("--capture_prefix", type=str, default="state", help="Prefix for saved recovery state files.")
parser.add_argument(
    "--no_capture_mlp_obs",
    action="store_false",
    dest="capture_mlp_obs",
    default=True,
    help="Do not store diagnostic MLP policy observation terms in recovery state files.",
)
parser.add_argument("--record_video", action="store_true", default=False, help="Record evaluation video.")
parser.add_argument("--video", action="store_true", default=False, help="Alias for --record_video.")
parser.add_argument("--video_record", action="store_true", default=False, help="Alias for --record_video.")
parser.add_argument("--video_folder", type=str, default="videos/policy_inference")
parser.add_argument("--video_fps", type=int, default=30, help="FPS used for encoded evaluation videos.")
parser.add_argument("--video_length", type=int, default=2000, help="Recorded video length in env steps.")
parser.add_argument("--video_single_file", action="store_true", default=True)
parser.add_argument("--video_max_episodes", type=int, default=0)
parser.add_argument("--video_name_prefix", type=str, default="vla-stuck-capture")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import sys
import time
from pathlib import Path

import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type

import leisaac  # noqa: F401

_MANIPULATION_ROOT = Path(__file__).resolve().parents[4]
_MANIPULATION_SRC = _MANIPULATION_ROOT / "src"
if str(_MANIPULATION_SRC) not in sys.path:
    sys.path.append(str(_MANIPULATION_SRC))

from isaac_so_arm101.scripts.rsl_rl.recovery_state_utils import (  # noqa: E402
    capture_lift_cube_state,
    save_lift_cube_state,
)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        if hz <= 0:
            raise ValueError("--step_hz must be positive.")
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time += self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    """Keyboard controller for manual reset and stuck-state capture."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self.reset_state = False
        self.capture_state = False

    def __del__(self):
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False
        self.capture_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
            elif event.input.name == "C":
                self.capture_state = True
        return True


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    if model_type in ["gr00tn1.5", "gr00tn1.6", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    raise ValueError(f"Model type {model_type} not supported")


def configure_lerobot_absolute_joint_actions(env_cfg, task_type: str) -> None:
    """Apply LeRobot actions as absolute joint targets after motor-to-joint conversion."""
    if task_type != "so101leader":
        return
    for action_name in ("arm_action", "gripper_action"):
        action_cfg = getattr(env_cfg.actions, action_name, None)
        if action_cfg is not None and hasattr(action_cfg, "use_default_offset"):
            action_cfg.use_default_offset = False
            action_cfg.scale = 1.0
    print("[INFO] LeRobot policy actions use absolute joint targets (scale=1.0, no default offset).")


def _capture_output_dir() -> Path:
    capture_dir = Path(args_cli.capture_dir)
    run_name = args_cli.capture_run_name or time.strftime("%Y-%m-%d_%H-%M-%S")
    return capture_dir / run_name


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    if "lerobot" in args_cli.policy_type:
        configure_lerobot_absolute_joint_actions(env_cfg, task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.episode_length_s <= 0 and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    do_record_video = args_cli.record_video or args_cli.video or args_cli.video_record
    render_mode = "rgb_array" if do_record_video else None
    sim_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if do_record_video:
        if args_cli.video_single_file:
            sim_env = gym.wrappers.RecordVideo(
                sim_env,
                video_folder=args_cli.video_folder,
                step_trigger=lambda step_id: step_id == 0,
                video_length=args_cli.video_length,
                fps=args_cli.video_fps,
                name_prefix=args_cli.video_name_prefix,
                disable_logger=True,
            )
        else:
            sim_env = gym.wrappers.RecordVideo(
                sim_env,
                video_folder=args_cli.video_folder,
                episode_trigger=lambda episode_id: args_cli.video_max_episodes <= 0
                or episode_id < args_cli.video_max_episodes,
                video_length=args_cli.video_length,
                fps=args_cli.video_fps,
                name_prefix=args_cli.video_name_prefix,
                disable_logger=True,
            )
    env: ManagerBasedRLEnv = sim_env.unwrapped

    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00tServicePolicyClient

        if task_type != "so101leader":
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")
        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=["single_arm", "gripper"],
        )
    elif args_cli.policy_type == "gr00tn1.6":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00t16ServicePolicyClient

        if task_type != "so101leader":
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.6 policy yet.")
        policy = Gr00t16ServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=["single_arm", "gripper"],
        )
    elif "lerobot" in args_cli.policy_type:
        from isaaclab.sensors import Camera
        from leisaac.policy import LeRobotServicePolicyClient

        model_type = "lerobot"
        policy_type = args_cli.policy_type.split("-")[1]
        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={
                key: sensor.image_shape for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)
            },
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            force_must_go=args_cli.policy_must_go,
            device=args_cli.device,
        )
    elif args_cli.policy_type == "openpi":
        from isaaclab.sensors import Camera
        from leisaac.policy import OpenPIServicePolicyClient

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            task_type=task_type,
        )
    else:
        raise ValueError(f"Unsupported policy type: {args_cli.policy_type}")

    output_dir = _capture_output_dir()
    print("[INFO] Press C to capture the current stuck state; press R to reset the episode.")
    print(f"[INFO] Capturing recovery states to: {output_dir}")
    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()
    obs_dict, _ = sim_env.reset()
    controller.reset()

    def reset_policy_client():
        if hasattr(policy, "reset"):
            policy.reset()

    def maybe_capture(episode_idx: int, episode_step: int, last_action: torch.Tensor | None, capture_idx: int) -> int:
        if not controller.capture_state:
            return capture_idx
        state = capture_lift_cube_state(
            env,
            task=args_cli.task,
            instruction=args_cli.policy_language_instruction,
            episode_index=episode_idx,
            episode_step=episode_step,
            last_action=last_action,
            include_mlp_obs=args_cli.capture_mlp_obs,
        )
        path = save_lift_cube_state(state, output_dir, args_cli.capture_prefix, capture_idx)
        print(f"[Capture] Saved stuck state #{capture_idx}: {path}")
        controller.capture_state = False
        return capture_idx + 1

    reset_policy_client()
    max_episode_count = args_cli.eval_rounds
    success_count, episode_count, capture_count = 0, 1, 1

    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        episode_step = 0
        last_action = None
        while simulation_app.is_running():
            with torch.no_grad():
                if controller.reset_state:
                    print(f"[Evaluation] Episode {episode_count} manually marked failed/reset with R.")
                    controller.reset()
                    obs_dict, _ = sim_env.reset()
                    reset_policy_client()
                    episode_count += 1
                    break

                capture_count = maybe_capture(episode_count, episode_step, last_action, capture_count)
                policy_obs = preprocess_obs_dict(
                    obs_dict["policy"], model_type, args_cli.policy_language_instruction
                )
                actions = policy.get_action(policy_obs).to(env.device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = sim_env.step(action)
                    episode_step += 1
                    last_action = action.detach().cpu().clone()
                    capture_count = maybe_capture(episode_count, episode_step, last_action, capture_count)
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                episode_count += 1
                success_count += 1
                obs_dict, _ = sim_env.reset()
                reset_policy_client()
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                episode_count += 1
                obs_dict, _ = sim_env.reset()
                reset_policy_client()
                break
        if episode_count > 1:
            print(f"[Evaluation] now success rate: {success_count / (episode_count - 1)} [{success_count}/{episode_count - 1}]")

    if max_episode_count > 0:
        print(f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} [{success_count}/{max_episode_count}]")
    sim_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

