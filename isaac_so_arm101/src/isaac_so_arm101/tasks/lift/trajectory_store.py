from __future__ import annotations

from pathlib import Path

import torch


def project_point_to_polyline_detailed(
    p: torch.Tensor, pts: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project ``p`` onto polyline ``pts``; return progress, lateral, arc length to closest point, total length.

    Returns:
        ``progress`` in ``[0, 1]``, ``lateral_dist``, ``arc_to_closest`` (meters along polyline),
        ``total_path_length`` (sum of segment lengths).
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must have shape (T, 3)")
    T = pts.shape[0]
    if T == 0:
        raise ValueError("empty polyline")
    if T == 1:
        lateral = torch.norm(p - pts[0])
        z = torch.zeros((), device=p.device, dtype=p.dtype)
        return z, lateral, z, torch.tensor(eps, device=p.device, dtype=p.dtype)

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    ab_len_sq = (ab * ab).sum(dim=-1).clamp(min=eps)
    ap = p - a
    t = (ap * ab).sum(dim=-1) / ab_len_sq
    t = t.clamp(0.0, 1.0)
    closest = a + t.unsqueeze(-1) * ab
    lateral_seg = torch.norm(p - closest, dim=-1)
    seg_lens = torch.norm(ab, dim=-1)

    cum_at_vertex = torch.zeros(T, device=p.device, dtype=p.dtype)
    cum_at_vertex[1:] = torch.cumsum(seg_lens, dim=0)
    arc_to = cum_at_vertex[:-1] + t * seg_lens

    best_idx = int(lateral_seg.argmin().item())
    total_len = seg_lens.sum().clamp(min=eps)
    arc_closest = arc_to[best_idx]
    progress = (arc_closest / total_len).clamp(0.0, 1.0)
    return progress, lateral_seg[best_idx], arc_closest, total_len


def project_point_to_polyline_progress(
    p: torch.Tensor, pts: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a point onto a 3D polyline and return normalized arc-length progress and lateral distance."""
    prog, lat, _arc, _tot = project_point_to_polyline_detailed(p, pts, eps=eps)
    return prog, lat


def project_points_to_polyline_detailed_batched(
    p: torch.Tensor, pts: torch.Tensor, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a batch of points onto the **same** polyline (vectorized, no Python loop over batch).

    Args:
        p: ``(B, 3)`` points.
        pts: ``(T, 3)`` polyline vertices.

    Returns:
        ``progress``, ``lateral``, ``arc_to_closest``, ``total_len`` each ``(B,)`` (``total_len`` is constant per row).
    """
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p must have shape (B, 3)")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must have shape (T, 3)")
    T = pts.shape[0]
    device, dtype = p.device, p.dtype
    B = p.shape[0]
    if T == 0:
        raise ValueError("empty polyline")
    if T == 1:
        lateral = torch.norm(p - pts[0].unsqueeze(0), dim=-1)
        z = torch.zeros(B, device=device, dtype=dtype)
        tl = torch.full((B,), eps, device=device, dtype=dtype)
        return z, lateral, z, tl

    a = pts[:-1]
    b = pts[1:]
    ab = b - a
    ab_len_sq = (ab * ab).sum(dim=-1).clamp(min=eps)
    seg_lens = torch.norm(ab, dim=-1)

    cum_at_vertex = torch.zeros(T, device=device, dtype=dtype)
    cum_at_vertex[1:] = torch.cumsum(seg_lens, dim=0)

    ap = p.unsqueeze(1) - a.unsqueeze(0)
    t = (ap * ab.unsqueeze(0)).sum(dim=-1) / ab_len_sq.unsqueeze(0)
    t = t.clamp(0.0, 1.0)
    closest = a.unsqueeze(0) + t.unsqueeze(-1) * ab.unsqueeze(0)
    lateral_seg = torch.norm(p.unsqueeze(1) - closest, dim=-1)

    best_idx = lateral_seg.argmin(dim=1)

    arc_to = cum_at_vertex[:-1].unsqueeze(0) + t * seg_lens.unsqueeze(0)
    arc_closest = torch.gather(arc_to, 1, best_idx.unsqueeze(1)).squeeze(1)

    total_len = seg_lens.sum().clamp(min=eps)
    progress = (arc_closest / total_len).clamp(0.0, 1.0)
    lateral = torch.gather(lateral_seg, 1, best_idx.unsqueeze(1)).squeeze(1)
    total_b = total_len.expand(B)
    return progress, lateral, arc_closest, total_b


def project_points_to_polyline_windowed(
    p: torch.Tensor,
    pts: torch.Tensor,
    current_seg: torch.Tensor,
    window: int = 20,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Windowed polyline projection — each point searches only nearby segments.

    Instead of projecting onto the *entire* polyline (which lets the agent skip
    trajectory sections that loop back spatially), we restrict the search to
    ``[current_seg - window, current_seg + window]`` for each batch element.

    Args:
        p: ``(B, 3)`` query points (student EE positions).
        pts: ``(T, 3)`` polyline vertices (single trajectory, same for all batch elements).
        current_seg: ``(B,)`` long tensor — current segment index per env.
        window: half-width of the search window (in segments).

    Returns:
        ``progress (B,)``, ``lateral (B,)``, ``arc_to_closest (B,)``,
        ``total_len (B,)``, ``best_seg (B,)`` — the winning segment index
        (absolute, in ``[0, S-1]`` where ``S = T - 1``),
        ``lateral_xy (B,)`` — XY-only distance to the same closest point (ignores vertical offset).
    """
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("p must have shape (B, 3)")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must have shape (T, 3)")
    T = pts.shape[0]
    B = p.shape[0]
    device, dtype = p.device, p.dtype
    S = T - 1  # number of segments

    if T == 0:
        raise ValueError("empty polyline")
    if T == 1:
        lateral = torch.norm(p - pts[0].unsqueeze(0), dim=-1)
        lateral_xy = torch.norm(p[:, :2] - pts[0, :2].unsqueeze(0), dim=-1)
        z = torch.zeros(B, device=device, dtype=dtype)
        tl = torch.full((B,), eps, device=device, dtype=dtype)
        return z, lateral, z, tl, torch.zeros(B, device=device, dtype=torch.long), lateral_xy

    a = pts[:-1]  # (S, 3)
    b = pts[1:]   # (S, 3)
    ab = b - a
    seg_lens = torch.norm(ab, dim=-1)  # (S,)
    ab_len_sq = (ab * ab).sum(dim=-1).clamp(min=eps)  # (S,)

    cum_at_vertex = torch.zeros(T, device=device, dtype=dtype)
    cum_at_vertex[1:] = torch.cumsum(seg_lens, dim=0)
    total_len_scalar = seg_lens.sum().clamp(min=eps)

    lo = (current_seg - window).clamp(min=0)         # (B,)
    hi = (current_seg + window).clamp(max=S - 1)     # (B,)
    win_size = (hi - lo + 1).max().item()             # uniform pad width

    seg_range = torch.arange(win_size, device=device).unsqueeze(0) + lo.unsqueeze(1)  # (B, W)
    seg_range = seg_range.clamp(0, S - 1)

    a_win = a[seg_range]          # (B, W, 3)
    ab_win = ab[seg_range]        # (B, W, 3)
    ab_lsq_win = ab_len_sq[seg_range]  # (B, W)
    sl_win = seg_lens[seg_range]  # (B, W)
    cum_win = cum_at_vertex[:-1][seg_range]  # (B, W)

    ap = p.unsqueeze(1) - a_win                        # (B, W, 3)
    t = (ap * ab_win).sum(dim=-1) / ab_lsq_win         # (B, W)
    t = t.clamp(0.0, 1.0)
    closest = a_win + t.unsqueeze(-1) * ab_win          # (B, W, 3)
    lateral_seg = torch.norm(p.unsqueeze(1) - closest, dim=-1)  # (B, W)
    lateral_xy_seg = torch.norm(p.unsqueeze(1)[..., :2] - closest[..., :2], dim=-1)  # (B, W)

    valid_mask = torch.arange(win_size, device=device).unsqueeze(0) <= (hi - lo).unsqueeze(1)
    lateral_seg = torch.where(valid_mask, lateral_seg, torch.full_like(lateral_seg, 1e9))
    lateral_xy_seg = torch.where(valid_mask, lateral_xy_seg, torch.full_like(lateral_xy_seg, 1e9))

    win_best = lateral_seg.argmin(dim=1)  # (B,) index within window
    best_seg_abs = torch.gather(seg_range, 1, win_best.unsqueeze(1)).squeeze(1)  # (B,)

    arc_to = cum_win + t * sl_win  # (B, W)
    arc_closest = torch.gather(arc_to, 1, win_best.unsqueeze(1)).squeeze(1)  # (B,)
    lateral = torch.gather(lateral_seg, 1, win_best.unsqueeze(1)).squeeze(1)  # (B,)
    lateral_xy = torch.gather(lateral_xy_seg, 1, win_best.unsqueeze(1)).squeeze(1)  # (B,)

    progress = (arc_closest / total_len_scalar).clamp(0.0, 1.0)
    total_b = total_len_scalar.expand(B)
    return progress, lateral, arc_closest, total_b, best_seg_abs, lateral_xy


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

        # Env-origin–invariant layout (recommended for multi-env training / dataset reset events).
        if "initial_object_pos_local" in raw and "goal_pos_local" in raw:
            self.initial_object_pos_local = torch.as_tensor(raw["initial_object_pos_local"], dtype=torch.float32)[
                select_idx
            ].to(self.device)
            self.goal_pos_local = torch.as_tensor(raw["goal_pos_local"], dtype=torch.float32)[select_idx].to(self.device)
            self.use_local_layout = True
        else:
            self.initial_object_pos_local = self.initial_object_pos.clone()
            self.goal_pos_local = self.goal_pos.clone()
            self.use_local_layout = False

        self.ee_trajectories = torch.as_tensor(raw["ee_trajectories"], dtype=torch.float32)[select_idx].to(self.device)
        self.trajectory_lengths = torch.as_tensor(raw["trajectory_lengths"], dtype=torch.long)[select_idx].to(self.device)

        if "gripper_trajectories" in raw:
            gt = torch.as_tensor(raw["gripper_trajectories"], dtype=torch.float32)[select_idx].to(self.device)
            if gt.ndim != 2:
                raise ValueError("gripper_trajectories must have shape (N, T)")
            self.gripper_trajectories = gt
            self.has_gripper = True
        else:
            self.gripper_trajectories = torch.zeros(
                (self.ee_trajectories.shape[0], self.ee_trajectories.shape[1]), dtype=torch.float32, device=self.device
            )
            self.has_gripper = False

        # First EE sample of each recorded episode, env-local (for IK align / world targets).
        if "initial_ee_pos_local" in raw:
            self.initial_ee_pos_local = torch.as_tensor(raw["initial_ee_pos_local"], dtype=torch.float32)[
                select_idx
            ].to(self.device)
            self.use_ee_local = True
        else:
            # Legacy: first waypoint of each trajectory (world frame); matching without origin subtract.
            self.initial_ee_pos_local = self.ee_trajectories[:, 0, :].clone()
            self.use_ee_local = False

        if self.initial_object_pos.ndim != 2 or self.initial_object_pos.shape[1] != 3:
            raise ValueError("initial_object_pos must have shape (N, 3)")
        if self.goal_pos.ndim != 2 or self.goal_pos.shape[1] != 3:
            raise ValueError("goal_pos must have shape (N, 3)")
        if self.initial_object_pos_local.shape != self.initial_object_pos.shape:
            raise ValueError("initial_object_pos_local must match initial_object_pos shape")
        if self.goal_pos_local.shape != self.goal_pos.shape:
            raise ValueError("goal_pos_local must match goal_pos shape")
        if self.ee_trajectories.ndim != 3 or self.ee_trajectories.shape[2] != 3:
            raise ValueError("ee_trajectories must have shape (N, T, 3)")
        if self.trajectory_lengths.ndim != 1:
            raise ValueError("trajectory_lengths must have shape (N,)")
        if self.initial_object_pos.shape[0] == 0:
            raise ValueError("Trajectory dataset is empty after filtering")
        if self.initial_ee_pos_local.ndim != 2 or self.initial_ee_pos_local.shape[1] != 3:
            raise ValueError("initial_ee_pos_local must have shape (N, 3)")

    def _build_features(self, object_pos: torch.Tensor, goal_pos: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "object_goal":
            return torch.cat([object_pos, goal_pos], dim=1)
        if mode == "object_only":
            return object_pos
        raise ValueError(f"Unknown match mode: {mode}")

    def _layout_keys(self, mode: str) -> torch.Tensor:
        """Feature rows for dataset trajectories (local coords when ``use_local_layout``)."""
        return self._build_features(self.initial_object_pos_local, self.goal_pos_local, mode=mode)

    def get_first_teacher_ee_world(
        self, traj_indices: torch.Tensor, env_origins: torch.Tensor
    ) -> torch.Tensor:
        """World-frame first EE waypoint (same convention as ``collect_trajectories`` / ``ee_frame``)."""
        idx = traj_indices.to(self.device, dtype=torch.long).flatten()
        env_origins = env_origins.to(self.device, dtype=torch.float32)
        if env_origins.ndim != 2 or env_origins.shape[1] < 3:
            raise ValueError("env_origins must have shape (B, 3)")
        if idx.shape[0] != env_origins.shape[0]:
            raise ValueError("traj_indices and env_origins batch sizes must match")
        if self.use_ee_local:
            return self.initial_ee_pos_local[idx] + env_origins[:, :3]
        return self.ee_trajectories[idx, 0, :]

    def match(
        self,
        object_pos: torch.Tensor | None = None,
        goal_pos: torch.Tensor | None = None,
        mode: str = "object_goal",
        exact_tol: float = 1.0e-3,
        env_origins: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match trajectories by reset condition with exact-first fallback-nearest.

        ``object_goal`` / ``object_only``: use cube (and optionally goal) features; when the dataset
        has ``*_local`` keys, pass ``env_origins`` (shape ``(B, 3)``) so queries match stored locals.
        """
        if object_pos is None or goal_pos is None:
            raise ValueError(f"mode {mode!r} requires object_pos and goal_pos")
        object_pos = object_pos.to(self.device, dtype=torch.float32)
        goal_pos = goal_pos.to(self.device, dtype=torch.float32)
        if object_pos.ndim != 2 or object_pos.shape[1] != 3:
            raise ValueError("object_pos must have shape (B, 3)")
        if goal_pos.ndim != 2 or goal_pos.shape[1] != 3:
            raise ValueError("goal_pos must have shape (B, 3)")
        if object_pos.shape[0] != goal_pos.shape[0]:
            raise ValueError("object_pos and goal_pos batch sizes must match")

        if env_origins is not None and self.use_local_layout:
            origins = env_origins.to(self.device, dtype=torch.float32)
            if origins.ndim != 2 or origins.shape[1] < 3:
                raise ValueError("env_origins must have shape (B, 3) or (B, >=3)")
            object_pos = object_pos - origins[:, :3]
            goal_pos = goal_pos - origins[:, :3]

        query = self._build_features(object_pos, goal_pos, mode=mode)
        keys = self._layout_keys(mode=mode)

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

    def get_gripper(self, traj_indices: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Teacher gripper joint values at timesteps (same padding rules as :meth:`get_ee_pos`)."""
        traj_indices = traj_indices.to(self.device, dtype=torch.long).flatten()
        timesteps = timesteps.to(self.device, dtype=torch.long).flatten()
        if traj_indices.shape[0] != timesteps.shape[0]:
            raise ValueError("traj_indices and timesteps must have the same batch size")

        lengths = self.trajectory_lengths[traj_indices].clamp(min=1)
        clamped_t = torch.clamp(timesteps, min=0)
        clamped_t = torch.minimum(clamped_t, lengths - 1)
        return self.gripper_trajectories[traj_indices, clamped_t]

    def project_ee_to_progress_windowed(
        self,
        student_ee: torch.Tensor,
        traj_indices: torch.Tensor,
        current_seg: torch.Tensor,
        window: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Windowed projection: search only ``[current_seg ± window]`` segments.

        Only supports the fast path where all envs share the same trajectory index.
        Falls back to a per-env loop otherwise.

        Returns:
            ``progress (B,)``, ``lateral (B,)``, ``arc_len (B,)``,
            ``total_len (B,)``, ``best_seg (B,)``, ``lateral_xy (B,)`` — winning segment index.
        """
        student_ee = student_ee.to(self.device, dtype=torch.float32)
        traj_indices = traj_indices.to(self.device, dtype=torch.long).flatten()
        current_seg = current_seg.to(self.device, dtype=torch.long).flatten()
        bsz = student_ee.shape[0]

        if bsz > 0 and torch.all(traj_indices == traj_indices[0]):
            idx = int(traj_indices[0].item())
            tlen = int(self.trajectory_lengths[idx].item())
            pts = self.ee_trajectories[idx, :tlen, :]
            return project_points_to_polyline_windowed(student_ee, pts, current_seg, window=window)

        progress_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        lateral_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        lateral_xy_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        arc_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        total_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        seg_out = torch.empty(bsz, device=self.device, dtype=torch.long)
        for b in range(bsz):
            idx = int(traj_indices[b].item())
            tlen = int(self.trajectory_lengths[idx].item())
            pts = self.ee_trajectories[idx, :tlen, :]
            cs = current_seg[b:b + 1]
            prog, lat, arc_c, tot, bseg, lat_xy = project_points_to_polyline_windowed(
                student_ee[b:b + 1], pts, cs, window=window,
            )
            progress_out[b] = prog[0]
            lateral_out[b] = lat[0]
            lateral_xy_out[b] = lat_xy[0]
            arc_out[b] = arc_c[0]
            total_out[b] = tot[0]
            seg_out[b] = bseg[0]
        return progress_out, lateral_out, arc_out, total_out, seg_out, lateral_xy_out

    def project_ee_to_progress(
        self, student_ee: torch.Tensor, traj_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project student EE positions onto matched teacher polylines.

        For each environment, uses the polyline defined by ``ee_trajectories[idx, :T]`` where
        ``T = trajectory_lengths[idx]``. The closest point on the polyline (over all segments)
        defines **normalized arc-length progress** in ``[0, 1]`` and **lateral** 3D distance.

        Args:
            student_ee: ``(B, 3)`` world-frame EE positions.
            traj_indices: ``(B,)`` trajectory indices in ``[0, N)``.

        Returns:
            ``progress``: ``(B,)`` in ``[0, 1]``; ``lateral_dist``: ``(B,)`` Euclidean distances.
        """
        student_ee = student_ee.to(self.device, dtype=torch.float32)
        traj_indices = traj_indices.to(self.device, dtype=torch.long).flatten()
        if student_ee.ndim != 2 or student_ee.shape[1] != 3:
            raise ValueError("student_ee must have shape (B, 3)")
        if traj_indices.shape[0] != student_ee.shape[0]:
            raise ValueError("traj_indices and student_ee batch sizes must match")

        bsz = student_ee.shape[0]
        if bsz > 0 and torch.all(traj_indices == traj_indices[0]):
            idx = int(traj_indices[0].item())
            tlen = int(self.trajectory_lengths[idx].item())
            pts = self.ee_trajectories[idx, :tlen, :]
            prog, lat, _, _ = project_points_to_polyline_detailed_batched(student_ee, pts)
            return prog, lat

        progress_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        lateral_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        for b in range(bsz):
            idx = int(traj_indices[b].item())
            tlen = int(self.trajectory_lengths[idx].item())
            pts = self.ee_trajectories[idx, :tlen, :]
            prog, lat = project_point_to_polyline_progress(student_ee[b], pts)
            progress_out[b] = prog
            lateral_out[b] = lat
        return progress_out, lateral_out

    def project_ee_to_progress_detailed(
        self, student_ee: torch.Tensor, traj_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like :meth:`project_ee_to_progress` but also returns arc length along path and total path length."""
        student_ee = student_ee.to(self.device, dtype=torch.float32)
        traj_indices = traj_indices.to(self.device, dtype=torch.long).flatten()
        if student_ee.ndim != 2 or student_ee.shape[1] != 3:
            raise ValueError("student_ee must have shape (B, 3)")
        if traj_indices.shape[0] != student_ee.shape[0]:
            raise ValueError("traj_indices and student_ee batch sizes must match")

        bsz = student_ee.shape[0]
        if bsz > 0 and torch.all(traj_indices == traj_indices[0]):
            idx = int(traj_indices[0].item())
            tlen = int(self.trajectory_lengths[idx].item())
            pts = self.ee_trajectories[idx, :tlen, :]
            return project_points_to_polyline_detailed_batched(student_ee, pts)

        progress_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        lateral_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        arc_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        total_out = torch.empty(bsz, device=self.device, dtype=torch.float32)
        for b in range(bsz):
            idx = int(traj_indices[b].item())
            tlen = int(self.trajectory_lengths[idx].item())
            pts = self.ee_trajectories[idx, :tlen, :]
            prog, lat, arc_c, tot = project_point_to_polyline_detailed(student_ee[b], pts)
            progress_out[b] = prog
            lateral_out[b] = lat
            arc_out[b] = arc_c
            total_out[b] = tot
        return progress_out, lateral_out, arc_out, total_out
