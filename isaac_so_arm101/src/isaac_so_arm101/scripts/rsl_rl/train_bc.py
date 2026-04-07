"""Train a simple MLP policy with Behavioural Cloning from a collected BC dataset.

Trains on the .npz episode files produced by collect_bc_dataset.py.
Runs entirely outside of Isaac Sim — no AppLauncher needed.

Usage::

    python src/isaac_so_arm101/scripts/rsl_rl/train_bc.py \\
        --dataset_dir datasets/bc_lift_cube \\
        --output_dir outputs/bc/lift_cube \\
        --epochs 100 \\
        --batch_size 256 \\
        --lr 3e-4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BCEpisodeDataset(Dataset):
    """Loads all episode .npz files from a directory into memory."""

    def __init__(self, dataset_dir: str):
        files = sorted(Path(dataset_dir).glob("episode_*.npz"))
        if not files:
            raise FileNotFoundError(f"No episode_*.npz files found in {dataset_dir}")

        all_obs, all_acts = [], []
        for f in files:
            data = np.load(f)
            all_obs.append(data["observations"].astype(np.float32))
            all_acts.append(data["actions"].astype(np.float32))

        self.observations = torch.from_numpy(np.concatenate(all_obs, axis=0))
        self.actions = torch.from_numpy(np.concatenate(all_acts, axis=0))
        print(f"[Dataset] {len(files)} episodes, {len(self.observations)} frames")
        print(f"[Dataset] obs shape: {tuple(self.observations.shape)}  action shape: {tuple(self.actions.shape)}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class BCPolicy(nn.Module):
    """Simple MLP: obs → action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256, layers: int = 3):
        super().__init__()
        dims = [obs_dim] + [hidden] * layers + [action_dim]
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.ELU())
        self.net = nn.Sequential(*modules)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    dataset = BCEpisodeDataset(args.dataset_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    obs_dim = dataset.observations.shape[-1]
    action_dim = dataset.actions.shape[-1]

    model = BCPolicy(obs_dim, action_dim, hidden=args.hidden, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train] obs_dim={obs_dim}  action_dim={action_dim}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"[Train] epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    print(f"[Train] Saving checkpoints to: {output_dir}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            pred = model(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(obs_batch)

        scheduler.step()
        avg_loss = total_loss / len(dataset)

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": best_loss},
                       output_dir / "best_model.pt")

        if epoch % args.save_freq == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": avg_loss},
                       output_dir / f"model_{epoch:05d}.pt")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(), "loss": avg_loss},
               output_dir / "model_final.pt")
    print(f"\n[DONE] Best loss: {best_loss:.6f}  Saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BC policy from collected episodes.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory with episode_*.npz files.")
    parser.add_argument("--output_dir", type=str, default="outputs/bc/lift_cube", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size.")
    parser.add_argument("--layers", type=int, default=3, help="Number of hidden layers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_interval", type=int, default=10, help="Print loss every N epochs.")
    parser.add_argument("--save_freq", type=int, default=50, help="Save checkpoint every N epochs.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
