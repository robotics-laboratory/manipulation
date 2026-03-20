from __future__ import annotations

from pathlib import Path

import torch


class TrajectoryStore:
    """Stores teacher trajectories and provides batched matching/lookups."""

    def __init__(self, path: str, device: str = "cuda"):
        file_path = Path(path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Trajectory dataset not found: {file_path}")

        raw = torch.load(file_path, map_location="cpu")
        required_keys = {"initial_object_pos", "goal_pos", "ee_trajectories", "trajectory_lengths"}
        missing = required_keys.difference(raw.keys())
        if missing:
            raise KeyError(f"Trajectory dataset missing keys: {sorted(missing)}")

        # Prefer successful trajectories if available.
        if "success" in raw:
            success_mask = torch.as_tensor(raw["success"], dtype=torch.bool)
            if success_mask.any():
                select_idx = success_mask.nonzero(as_tuple=False).squeeze(-1)
            else:
                select_idx = torch.arange(success_mask.shape[0], dtype=torch.long)
        else:
            select_idx = torch.arange(raw["initial_object_pos"].shape[0], dtype=torch.long)

        self.device = torch.device(device)
        self.initial_object_pos = torch.as_tensor(raw["initial_object_pos"], dtype=torch.float32)[select_idx].to(self.device)
        self.goal_pos = torch.as_tensor(raw["goal_pos"], dtype=torch.float32)[select_idx].to(self.device)
        self.ee_trajectories = torch.as_tensor(raw["ee_trajectories"], dtype=torch.float32)[select_idx].to(self.device)
        self.trajectory_lengths = torch.as_tensor(raw["trajectory_lengths"], dtype=torch.long)[select_idx].to(self.device)

        if self.initial_object_pos.ndim != 2 or self.initial_object_pos.shape[1] != 3:
            raise ValueError("initial_object_pos must have shape (N, 3)")
        if self.goal_pos.ndim != 2 or self.goal_pos.shape[1] != 3:
            raise ValueError("goal_pos must have shape (N, 3)")
        if self.ee_trajectories.ndim != 3 or self.ee_trajectories.shape[2] != 3:
            raise ValueError("ee_trajectories must have shape (N, T, 3)")
        if self.trajectory_lengths.ndim != 1:
            raise ValueError("trajectory_lengths must have shape (N,)")
        if self.initial_object_pos.shape[0] == 0:
            raise ValueError("Trajectory dataset is empty after filtering")

    def _build_features(self, object_pos: torch.Tensor, goal_pos: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "object_goal":
            return torch.cat([object_pos, goal_pos], dim=1)
        if mode == "object_only":
            return object_pos
        raise ValueError(f"Unknown match mode: {mode}")

    def match(
        self,
        object_pos: torch.Tensor,
        goal_pos: torch.Tensor,
        mode: str = "object_goal",
        exact_tol: float = 1.0e-3,
    ) -> torch.Tensor:
        """Match trajectories by reset condition with exact-first fallback-nearest."""
        object_pos = object_pos.to(self.device, dtype=torch.float32)
        goal_pos = goal_pos.to(self.device, dtype=torch.float32)
        if object_pos.ndim != 2 or object_pos.shape[1] != 3:
            raise ValueError("object_pos must have shape (B, 3)")
        if goal_pos.ndim != 2 or goal_pos.shape[1] != 3:
            raise ValueError("goal_pos must have shape (B, 3)")
        if object_pos.shape[0] != goal_pos.shape[0]:
            raise ValueError("object_pos and goal_pos batch sizes must match")

        query = self._build_features(object_pos, goal_pos, mode=mode)
        keys = self._build_features(self.initial_object_pos, self.goal_pos, mode=mode)
        dists = torch.cdist(query, keys, p=2.0)
        _, nearest_idx = torch.min(dists, dim=1)

        if exact_tol <= 0.0:
            return nearest_idx.to(dtype=torch.long)

        out = nearest_idx.clone()
        exact_threshold = float(exact_tol)
        for b in range(query.shape[0]):
            candidates = torch.nonzero(dists[b] <= exact_threshold, as_tuple=False).squeeze(-1)
            if candidates.numel() > 0:
                choice = candidates[torch.randint(low=0, high=candidates.numel(), size=(1,), device=self.device)]
                out[b] = choice
        return out.to(dtype=torch.long)

    def get_ee_pos(self, traj_indices: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get teacher EE positions at given timesteps (clamped by trajectory lengths)."""
        traj_indices = traj_indices.to(self.device, dtype=torch.long).flatten()
        timesteps = timesteps.to(self.device, dtype=torch.long).flatten()
        if traj_indices.shape[0] != timesteps.shape[0]:
            raise ValueError("traj_indices and timesteps must have the same batch size")

        lengths = self.trajectory_lengths[traj_indices].clamp(min=1)
        clamped_t = torch.clamp(timesteps, min=0)
        clamped_t = torch.minimum(clamped_t, lengths - 1)
        return self.ee_trajectories[traj_indices, clamped_t, :]
