#!/usr/bin/env python3
"""Diagnostic: load the trajectory dataset and print stats without launching Isaac Sim.

Usage (no sim needed):
    python manipulation/isaac_so_arm101/scripts/debug_trajectory_dataset.py [--file PATH]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect a teacher trajectory .pt dataset.")
    parser.add_argument(
        "--file",
        type=str,
        default=os.environ.get(
            "ISAAC_SO_ARM101_TRAJECTORY_FILE",
            os.path.join("isaac_so_arm101", "logs", "rsl_rl", "teacher_trajectories", "so101_lift_cube_teacher.pt"),
        ),
        help="Path to the .pt trajectory file.",
    )
    args = parser.parse_args()

    resolved = Path(args.file).expanduser().resolve()
    print(f"Resolved path : {resolved}")
    print(f"Exists        : {resolved.exists()}")
    if not resolved.exists():
        print("\n*** FILE NOT FOUND — this is why markers are not drawn! ***")
        print("The visualize_teacher_trajectory event silently returns when")
        print("_trajectory_guidance_state has not been created (because TrajectoryStore")
        print("would raise FileNotFoundError in the reward/event that creates it).")
        print()
        print("Check the _DEFAULT_TRAJECTORY_FILE in guided_env_cfg.py or set env var:")
        print("  export ISAAC_SO_ARM101_TRAJECTORY_FILE=/path/to/teacher.pt")
        alt = list(Path(".").rglob("*teacher*.pt"))
        if alt:
            print(f"\nCandidate files found on disk:")
            for p in alt:
                print(f"  {p}")
        return

    raw = torch.load(resolved, map_location="cpu", weights_only=False)

    print(f"\n--- Dataset keys ---")
    for k, v in sorted(raw.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k:35s}  shape={tuple(v.shape)}  dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k:35s}  (dict with {len(v)} entries)")
        else:
            print(f"  {k:35s}  {type(v).__name__} = {v}")

    ee = raw["ee_trajectories"]
    lengths = raw["trajectory_lengths"]
    N, T, D = ee.shape
    print(f"\n--- Trajectory stats ---")
    print(f"  Num trajectories (N)      : {N}")
    print(f"  Max timesteps (T)         : {T}")
    print(f"  EE dim (D)                : {D}")
    print(f"  Lengths  min={lengths.min().item()}  max={lengths.max().item()}  mean={lengths.float().mean().item():.1f}")

    if "success" in raw:
        succ = raw["success"]
        print(f"  Successful trajectories   : {succ.sum().item()}/{succ.shape[0]}")

    print(f"\n--- Coordinate ranges (first waypoints, world frame) ---")
    first_pts = ee[:, 0, :]
    for ax, name in enumerate(["X", "Y", "Z"]):
        lo, hi = first_pts[:, ax].min().item(), first_pts[:, ax].max().item()
        print(f"  {name}: [{lo:.4f}, {hi:.4f}]")

    if "initial_object_pos" in raw:
        obj = raw["initial_object_pos"]
        print(f"\n--- Object position ranges (world frame) ---")
        for ax, name in enumerate(["X", "Y", "Z"]):
            lo, hi = obj[:, ax].min().item(), obj[:, ax].max().item()
            print(f"  {name}: [{lo:.4f}, {hi:.4f}]")

    if "initial_ee_pos_local" in raw:
        ee_loc = raw["initial_ee_pos_local"]
        print(f"\n--- Initial EE local ranges ---")
        for ax, name in enumerate(["X", "Y", "Z"]):
            lo, hi = ee_loc[:, ax].min().item(), ee_loc[:, ax].max().item()
            print(f"  {name}: [{lo:.4f}, {hi:.4f}]")
        print("  (use_ee_local will be True)")
    else:
        print("\n  No 'initial_ee_pos_local' key → use_ee_local=False (legacy dataset)")

    if "initial_object_pos_local" in raw:
        opl = raw["initial_object_pos_local"]
        obj_w = raw["initial_object_pos"]
        inferred_origins = obj_w - opl
        print(f"\n--- Inferred recording env origins (obj_world - obj_local) ---")
        for ax, name in enumerate(["X", "Y", "Z"]):
            lo, hi = inferred_origins[:, ax].min().item(), inferred_origins[:, ax].max().item()
            print(f"  {name}: [{lo:.4f}, {hi:.4f}]")
        print("  (use_local_layout will be True)")
    else:
        print("\n  No 'initial_object_pos_local' → use_local_layout=False (legacy dataset)")

    if "meta" in raw and isinstance(raw["meta"], dict):
        print(f"\n--- Meta ---")
        for mk, mv in sorted(raw["meta"].items()):
            print(f"  {mk}: {mv}")

    print("\n✅  Dataset loaded successfully. If markers still don't show, run the env diagnostic script next.")


if __name__ == "__main__":
    main()
