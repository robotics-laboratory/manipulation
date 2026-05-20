#!/usr/bin/env python3
"""Run IsaacLabEureka on the LeIsaac direct LiftCube task.

This wrapper keeps IsaacLabEureka unmodified. It registers the LeIsaac task in
Eureka's task config and patches the worker environment creation so subprocesses
import ``leisaac`` before calling Isaac Lab's registry helpers.
"""

from __future__ import annotations

import argparse
import os

TASKS_CFG_PATCH = {
    "LeIsaac-SO101-LiftCube-Eureka-Direct-v0": {
        "description": (
            "lift the red cube from the table with the SO-101 arm in a stable, human-like motion. "
            "The reward should encourage reaching and grasping the cube, lifting it above the base-height "
            "success threshold, keeping the cube stable while lifted, avoiding wrist-flip exploits, and "
            "finishing with a controlled hold at the target height."
        ),
        "success_metric": "self._eureka_success_metric(env_ids)",
        "success_metric_to_win": 1.0,
        "success_metric_tolerance": 0.05,
    }
}


def _validate_llm_credentials() -> None:
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_azure_key = bool(os.environ.get("AZURE_OPENAI_API_KEY"))
    has_azure_endpoint = bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))

    if has_openai:
        return
    if has_azure_key and has_azure_endpoint:
        return

    raise RuntimeError(
        "Missing LLM credentials for IsaacLabEureka. Set OPENAI_API_KEY, "
        "or set AZURE_OPENAI_API_KEY together with AZURE_OPENAI_ENDPOINT."
    )


def _patch_isaaclab_eureka() -> None:
    from isaaclab_eureka.config import TASKS_CFG
    from isaaclab_eureka.managers import EurekaTaskManager
    from isaaclab_eureka.utils import get_freest_gpu

    TASKS_CFG.update(TASKS_CFG_PATCH)

    def _create_leisaac_environment(self):
        from isaaclab.app import AppLauncher

        # IMPORTANT: workers run in separate processes; never reuse a SimulationApp
        # object created in the parent process.
        if self._device == "cuda":
            device_id = get_freest_gpu()
            self._device = f"cuda:{device_id}"
        app_launcher = AppLauncher(headless=True, device=self._device)
        self._simulation_app = app_launcher.app

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import leisaac  # noqa: F401
        from isaaclab.envs import DirectRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg: DirectRLEnvCfg = parse_env_cfg(self._task)
        env_cfg.sim.device = self._device
        env_cfg.seed = self._env_seed
        self._env = gym.make(self._task, cfg=env_cfg)

    EurekaTaskManager._create_environment = _create_leisaac_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LiftCube rewards with IsaacLabEureka.")
    parser.add_argument("--task", type=str, default="LeIsaac-SO101-LiftCube-Eureka-Direct-v0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env_seed", type=int, default=42)
    parser.add_argument("--max_eureka_iterations", type=int, default=5)
    parser.add_argument("--max_training_iterations", type=int, default=100)
    parser.add_argument("--feedback_subsampling", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpt_model", type=str, default="gpt-4")
    parser.add_argument("--num_parallel_runs", type=int, default=1)
    parser.add_argument("--rl_library", type=str, default="rsl_rl", choices=["rsl_rl", "rl_games"])
    args = parser.parse_args()

    if args.rl_library != "rsl_rl":
        raise ValueError("LeIsaac LiftCube Eureka currently registers only an RSL-RL config.")
    if os.name == "nt" and args.num_parallel_runs > 1:
        args.num_parallel_runs = 1

    _validate_llm_credentials()
    _patch_isaaclab_eureka()

    from isaaclab_eureka.eureka import Eureka

    eureka = Eureka(
        task=args.task,
        rl_library=args.rl_library,
        num_parallel_runs=args.num_parallel_runs,
        device=args.device,
        env_seed=args.env_seed,
        max_training_iterations=args.max_training_iterations,
        feedback_subsampling=args.feedback_subsampling,
        temperature=args.temperature,
        gpt_model=args.gpt_model,
    )
    eureka.run(max_eureka_iterations=args.max_eureka_iterations)


if __name__ == "__main__":
    main()
