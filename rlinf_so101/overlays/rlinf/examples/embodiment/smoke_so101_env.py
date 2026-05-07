# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls


def _build_cfg(config_path: str, num_envs: int):
    cfg = OmegaConf.load(config_path)
    cfg.total_num_envs = num_envs
    cfg.init_params.num_envs = num_envs
    return cfg


def _sample_so101_actions(device: torch.device, num_envs: int) -> torch.Tensor:
    actions = torch.empty((num_envs, 6), device=device, dtype=torch.float32)
    actions[:, :5].uniform_(-0.25, 0.25)
    actions[:, 5].uniform_(0.4, 1.0)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test SO-101 LiftCube via RLinf.")
    default_cfg = (
        Path(__file__).resolve().parent / "config/env/so101_lift_cube.yaml"
    )
    parser.add_argument("--config", default=str(default_cfg))
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    cfg = _build_cfg(args.config, args.num_envs)
    env_cls = get_env_cls(cfg.env_type, cfg)
    env = env_cls(
        cfg=cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, _ = env.reset()
        print(
            "reset ok:",
            f"main_images={tuple(obs['main_images'].shape)}",
            f"wrist_images={tuple(obs['wrist_images'].shape)}",
            f"states={tuple(obs['states'].shape)}",
        )
        for step in range(args.steps):
            actions = _sample_so101_actions(env.device, args.num_envs)
            obs, reward, terminations, truncations, _ = env.step(actions)
            if step == 0 or (step + 1) % 10 == 0:
                print(
                    f"step {step + 1}:",
                    f"reward={reward.detach().cpu().tolist()}",
                    f"terminated={terminations.detach().cpu().tolist()}",
                    f"truncated={truncations.detach().cpu().tolist()}",
                )
        print("SO-101 random-action smoke test passed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
