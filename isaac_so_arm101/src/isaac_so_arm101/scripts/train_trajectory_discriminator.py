#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a trajectory discriminator for discriminator-guided exploration.

This is an offline approximation of the paper's discriminator loop:
  - positive samples: teacher (sketch-to-3D/executed) EE transitions
  - negative samples: mismatched EE transitions paired with the same task g

The resulting discriminator is used at RL time as:
  r_hat = r + lambda * log D(p_t, delta_p_t, g)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Avoid isaac_so_arm101.tasks package auto-import (gym registration via isaaclab_tasks),
# which requires a full Isaac Sim / pxr stack. This script is offline (torch + dataset only).
os.environ.setdefault("ISAAC_SO_ARM101_SKIP_TASK_AUTOIMPORT", "1")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from isaac_so_arm101.scripts.rsl_rl import log_paths
from isaac_so_arm101.tasks.lift.trajectory_discriminator import TrajectoryDiscriminator, TrajectoryDiscriminatorCfg
from isaac_so_arm101.tasks.lift.trajectory_store import TrajectoryStore


def _sample_transitions(store: TrajectoryStore, num_samples: int, generator: torch.Generator):
    """Sample (idx, t) pairs where t>=1 so delta is defined."""
    lengths = store.trajectory_lengths
    valid = lengths > 1
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        raise ValueError("Teacher dataset contains no trajectories with length > 1.")

    i = valid_idx[torch.randint(low=0, high=valid_idx.numel(), size=(num_samples,), generator=generator)]
    max_t = lengths[i] - 1  # >= 1
    # t = 1 + floor(U(0,1) * max_t)
    u = torch.rand(num_samples, generator=generator)
    t = 1 + torch.floor(u * max_t.to(dtype=torch.float32)).to(dtype=torch.long)
    return i, t


def main():
    parser = argparse.ArgumentParser(description="Train trajectory discriminator for guided exploration.")
    parser.add_argument(
        "--teacher_file",
        type=str,
        default=os.path.join(log_paths.rsl_rl_root(), "teacher_trajectories", "so101_lift_cube_teacher.pt"),
        help="Path to the teacher trajectory dataset (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(log_paths.rsl_rl_root(), "teacher_trajectories", "trajectory_discriminator_lift_cube.pt"),
        help="Where to save the discriminator model + normalization stats.",
    )
    parser.add_argument("--g_mode", type=str, default="object_only", choices=["object_only", "object_goal"])
    parser.add_argument("--num_samples", type=int, default=200_000, help="Number of positive transitions.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dims", type=str, default="256,128", help="Comma-separated hidden dims.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="cpu/cuda")

    args = parser.parse_args()

    teacher_path = Path(args.teacher_file).expanduser().resolve()
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher dataset not found: {teacher_path}")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))

    store = TrajectoryStore(path=str(teacher_path), device="cpu")

    # Sample positive and negative transitions.
    i_pos, t_pos = _sample_transitions(store, args.num_samples, generator)
    # Negative transitions are sampled independently but paired with the *same* g from i_pos.
    i_neg, t_neg = _sample_transitions(store, args.num_samples, generator)

    ee = store.ee_trajectories  # (N, T, 3)
    p_pos = ee[i_pos, t_pos, :]  # (M, 3)
    p_pos_prev = ee[i_pos, t_pos - 1, :]
    delta_pos = p_pos - p_pos_prev

    p_neg = ee[i_neg, t_neg, :]
    p_neg_prev = ee[i_neg, t_neg - 1, :]
    delta_neg = p_neg - p_neg_prev

    object_pos = store.initial_object_pos
    goal_pos = store.goal_pos

    if args.g_mode == "object_goal":
        g_pos = torch.cat([object_pos[i_pos], goal_pos[i_pos]], dim=1)  # (M, 6)
        g_dim = 6
    else:
        g_pos = object_pos[i_pos]  # (M, 3)
        g_dim = 3

    # Normalization stats from positives.
    p_mean = p_pos.mean(dim=0)
    p_std = p_pos.std(dim=0, unbiased=False).clamp(min=1e-6)
    delta_mean = delta_pos.mean(dim=0)
    delta_std = delta_pos.std(dim=0, unbiased=False).clamp(min=1e-6)
    g_mean = g_pos.mean(dim=0)
    g_std = g_pos.std(dim=0, unbiased=False).clamp(min=1e-6)

    # Build train tensors (positives then negatives).
    p_train = torch.cat([p_pos, p_neg], dim=0).to(device=device)
    delta_train = torch.cat([delta_pos, delta_neg], dim=0).to(device=device)
    g_train = torch.cat([g_pos, g_pos], dim=0).to(device=device)  # mismatch happens in (p, delta)
    y_train = torch.cat(
        [torch.ones(args.num_samples, device=device), torch.zeros(args.num_samples, device=device)], dim=0
    ).to(device=device)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    model = TrajectoryDiscriminator(
        cfg=TrajectoryDiscriminatorCfg(p_dim=3, delta_dim=3, g_dim=g_dim, hidden_dims=hidden_dims)
    ).to(device=device)
    model.p_mean.copy_(p_mean.to(device=device))
    model.p_std.copy_(p_std.to(device=device))
    model.delta_mean.copy_(delta_mean.to(device=device))
    model.delta_std.copy_(delta_std.to(device=device))
    model.g_mean.copy_(g_mean.to(device=device))
    model.g_std.copy_(g_std.to(device=device))
    model.train()

    ds = TensorDataset(p_train, delta_train, g_train, y_train)
    # pin_memory only applies to CPU tensors (speeds H2D copy). Dataset is already on `device`.
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for p_b, delta_b, g_b, y_b in dl:
            logits = model(p_b, delta_b, g_b)
            loss = F.binary_cross_entropy_with_logits(logits, y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(dl))
        print(f"[epoch {epoch+1}/{args.epochs}] loss={avg_loss:.6f}")

    model.eval()
    payload = model.to_dict()
    payload.update({"g_mode": args.g_mode})
    torch.save(payload, str(out_path))
    print(f"[INFO] Saved discriminator to: {out_path}")


if __name__ == "__main__":
    main()

