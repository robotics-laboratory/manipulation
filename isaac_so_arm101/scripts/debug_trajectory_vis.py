#!/usr/bin/env python3
"""Diagnostic: launch a guided env for a few steps and inspect trajectory visualization state.

Run with Isaac Sim (GUI mode recommended so you can see the viewport):

    python manipulation/isaac_so_arm101/scripts/debug_trajectory_vis.py \
        --task Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0 \
        --num_envs 2

Or with a fixed-layout guided task:

    python manipulation/isaac_so_arm101/scripts/debug_trajectory_vis.py \
        --task Isaac-SO-ARM101-FixedLayout-Guided-Lift-Cube-Sparse-v0 \
        --num_envs 2

Add --headless to run without a GUI (just prints diagnostics).
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug trajectory visualization.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Guided-Lift-Cube-Sparse-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=20, help="Number of env steps to run.")
parser.add_argument("--marker_radius", type=float, default=0.008, help="Override marker radius for debugging (larger = easier to see).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after app launch ──
import torch
import gymnasium as gym

import isaac_so_arm101.tasks  # noqa: F401  — register gym envs

# ── helpers ──

def _check_state(env, step_label: str):
    """Print diagnostic info about the trajectory guidance / visualization state."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env

    state = getattr(base, "_trajectory_guidance_state", None)
    vis_state = getattr(base, "_teacher_traj_vis_state", None)

    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC @ {step_label}")
    print(f"{'='*60}")

    # 1. Check trajectory guidance state
    if state is None:
        print("  [FAIL] _trajectory_guidance_state is None")
        print("         → The reward/event that creates it never ran, or the")
        print("           trajectory file was not found (FileNotFoundError).")
        print("         → Check the trajectory_file path in guided_env_cfg.py")
        return
    print(f"  [OK]   _trajectory_guidance_state exists")
    print(f"         trajectory_file = {state.get('trajectory_file')}")
    store = state.get("store")
    if store is not None:
        print(f"         store.N = {store.initial_object_pos.shape[0]} trajectories")
        print(f"         store.use_ee_local     = {store.use_ee_local}")
        print(f"         store.use_local_layout = {store.use_local_layout}")
    traj_idx = state.get("traj_indices")
    if traj_idx is not None:
        print(f"         traj_indices[:4] = {traj_idx[:4].tolist()}")
        neg = (traj_idx < 0).sum().item()
        print(f"         negative indices = {neg}/{traj_idx.shape[0]}")
    origin_delta = state.get("origin_delta")
    if origin_delta is not None:
        print(f"         origin_delta[0]  = {origin_delta[0].tolist()}")

    # 2. Check visualization state
    if vis_state is None:
        print(f"\n  [FAIL] _teacher_traj_vis_state is None")
        print(f"         → visualize_teacher_trajectory never created markers.")
        print(f"         Possible causes:")
        print(f"           a) The interval event never fired (is_global_time timing?)")
        print(f"           b) state was None when the interval event ran")
        print(f"           c) all traj_indices[:max_envs_to_draw] < 0")
        return

    print(f"\n  [OK]   _teacher_traj_vis_state exists")
    markers = vis_state.get("markers")
    last_key = vis_state.get("last_traj_key")
    print(f"         last_traj_key  = {last_key}")
    if markers is not None:
        print(f"         markers.prim_path = {markers.prim_path}")
        print(f"         markers.count     = {markers.count}")
        print(f"         markers.visible   = {markers.is_visible()}")
    else:
        print(f"  [FAIL] markers object is None inside vis_state!")

    # 3. Manually compute what waypoints would be drawn
    if store is not None and traj_idx is not None and origin_delta is not None:
        n_draw = min(1, base.num_envs)
        all_pts = []
        for i in range(n_draw):
            idx = int(traj_idx[i].item())
            if idx < 0:
                continue
            tlen = int(store.trajectory_lengths[idx].item())
            pts = store.ee_trajectories[idx, :tlen:2, :]
            pts_w = pts + origin_delta[i].unsqueeze(0)
            all_pts.append(pts_w)
        if all_pts:
            all_pts_t = torch.cat(all_pts, dim=0)
            print(f"\n  Waypoints that should be drawn: {all_pts_t.shape[0]} points")
            print(f"    X range: [{all_pts_t[:,0].min().item():.4f}, {all_pts_t[:,0].max().item():.4f}]")
            print(f"    Y range: [{all_pts_t[:,1].min().item():.4f}, {all_pts_t[:,1].max().item():.4f}]")
            print(f"    Z range: [{all_pts_t[:,2].min().item():.4f}, {all_pts_t[:,2].max().item():.4f}]")
            print(f"    First pt : {all_pts_t[0].tolist()}")
            print(f"    Last pt  : {all_pts_t[-1].tolist()}")
        else:
            print(f"\n  [WARN] No valid waypoints to draw (all indices < 0 for drawn envs)")

    # 4. Env origins
    env_origins = base.scene.env_origins[:, :3]
    print(f"\n  env_origins[0] = {env_origins[0].tolist()}")
    if base.num_envs > 1:
        print(f"  env_origins[1] = {env_origins[1].tolist()}")

    # 5. Check event manager for interval events
    em = base.event_manager
    if "interval" in em.available_modes:
        print(f"\n  Interval events registered:")
        for idx, name in enumerate(em._mode_term_names["interval"]):
            cfg = em._mode_term_cfgs["interval"][idx]
            tl = em._interval_term_time_left[idx]
            print(f"    [{idx}] {name}  interval={cfg.interval_range_s}  "
                  f"global_time={cfg.is_global_time}  time_left={tl.item():.6f}")
    else:
        print(f"\n  [WARN] No 'interval' events registered in event manager!")
        print(f"         Available modes: {list(em.available_modes)}")


def main():
    env_cfg_entry = gym.spec(args_cli.task).kwargs.get("env_cfg_entry_point")
    print(f"\nTask          : {args_cli.task}")
    print(f"Env config    : {env_cfg_entry}")
    print(f"Num envs      : {args_cli.num_envs}")
    print(f"Steps         : {args_cli.steps}")
    print(f"Marker radius : {args_cli.marker_radius}")

    env = gym.make(args_cli.task, num_envs=args_cli.num_envs)
    base = env.unwrapped if hasattr(env, "unwrapped") else env

    print(f"\nEvents config class: {type(base.cfg.events).__name__}")
    print(f"Rewards config class: {type(base.cfg.rewards).__name__}")

    # Optionally override marker radius for better visibility
    if args_cli.marker_radius != 0.008:
        print(f"  (marker_radius override not applied at config level; "
              f"modify GuidedEEAlignEventCfg for permanent change)")

    print("\n── env.reset() ──")
    obs, info = env.reset()
    _check_state(base, "after reset")

    for step_i in range(args_cli.steps):
        action = torch.zeros(args_cli.num_envs, env.action_space.shape[-1], device=base.device)
        obs, rew, terminated, truncated, info = env.step(action)

        if step_i in (0, 1, 4, args_cli.steps - 1):
            _check_state(base, f"step {step_i + 1}")

        if simulation_app.is_running() and not args_cli.headless:
            simulation_app.update()

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    state = getattr(base, "_trajectory_guidance_state", None)
    vis_state = getattr(base, "_teacher_traj_vis_state", None)
    if state is None:
        print("  ❌ _trajectory_guidance_state never created → check trajectory file path")
    elif vis_state is None:
        print("  ❌ Markers never created → visualize_teacher_trajectory returned early every time")
        print("     Check traj_indices (all < 0?) or event registration")
    elif vis_state.get("markers") and vis_state["markers"].is_visible():
        mk = vis_state["markers"]
        print(f"  ✅ Markers exist: {mk.count} instances at {mk.prim_path}")
        print(f"     If you still see nothing in the viewport:")
        print(f"       1) Zoom in — marker radius is only {args_cli.marker_radius*1000:.0f}mm")
        print(f"       2) The prim might be hidden in the Stage tree → expand /World/Visuals/")
        print(f"       3) Camera might not be looking at the markers")
        print(f"       4) Try increasing marker_radius in guided_env_cfg.py (e.g. 0.01)")
    else:
        print("  ⚠️  Markers created but not visible (is_visible=False)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
