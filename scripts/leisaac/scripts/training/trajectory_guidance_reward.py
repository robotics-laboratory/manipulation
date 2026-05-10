from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GuidanceRewardCoefficients:
    """Weights/scales for trajectory guidance reward terms."""

    progress_weight: float = 1.0
    xy_weight: float = 0.25
    gripper_weight: float = 0.25
    xy_scale: float = 0.08
    gripper_scale: float = 15.0
    progress_power: float = 1.0


@dataclass
class ReferenceTrajectory:
    traj_id: str
    ee_pos_w: np.ndarray  # [T, 3]
    gripper_state: np.ndarray  # [T]
    initial_cube_pose_w: np.ndarray  # [7]
    meta: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.ee_pos_w.shape[0])


@dataclass
class TrajectoryGuidanceState:
    traj: ReferenceTrajectory
    prev_progress: float = 0.0


@dataclass
class GuidanceStepMetrics:
    reward: float
    progress: float
    delta_progress: float
    xy_error: float
    gripper_error: float
    closest_index: int


def _parse_meta(meta_value: Any) -> dict[str, Any]:
    if meta_value is None:
        return {}
    if isinstance(meta_value, np.ndarray):
        if meta_value.shape == ():
            meta_value = meta_value.item()
        elif meta_value.size == 1:
            meta_value = meta_value.reshape(()).item()
    if isinstance(meta_value, bytes):
        meta_value = meta_value.decode("utf-8")
    if isinstance(meta_value, str):
        try:
            return json.loads(meta_value)
        except json.JSONDecodeError:
            return {"raw_meta": meta_value}
    if isinstance(meta_value, dict):
        return meta_value
    return {"raw_meta": str(meta_value)}


def load_reference_trajectories(traj_db: str | Path) -> list[ReferenceTrajectory]:
    """Load reference trajectories from a directory of ``trajectory_*.npz`` files."""
    traj_dir = Path(traj_db).expanduser()
    if not traj_dir.exists():
        raise FileNotFoundError(f"Trajectory DB path does not exist: {traj_dir}")
    paths = sorted(traj_dir.glob("trajectory_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No trajectory_*.npz files found in: {traj_dir}")

    trajectories: list[ReferenceTrajectory] = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        ee_pos_w = np.asarray(data["ee_pos_w"], dtype=np.float32)
        if ee_pos_w.ndim != 2 or ee_pos_w.shape[1] < 2:
            raise ValueError(f"Invalid ee_pos_w shape in {path}: {ee_pos_w.shape}")
        gripper_state = np.asarray(data["gripper_state"], dtype=np.float32).reshape(-1)
        initial_cube_pose_w = np.asarray(data["initial_cube_pose_w"], dtype=np.float32).reshape(-1)
        if initial_cube_pose_w.shape[0] != 7:
            raise ValueError(
                f"Expected initial_cube_pose_w shape (7,), got {initial_cube_pose_w.shape} in {path}"
            )
        if gripper_state.shape[0] != ee_pos_w.shape[0]:
            min_len = min(gripper_state.shape[0], ee_pos_w.shape[0])
            ee_pos_w = ee_pos_w[:min_len]
            gripper_state = gripper_state[:min_len]
        meta = _parse_meta(data["meta_json"] if "meta_json" in data else None)
        trajectories.append(
            ReferenceTrajectory(
                traj_id=path.stem,
                ee_pos_w=ee_pos_w[:, :3],
                gripper_state=gripper_state,
                initial_cube_pose_w=initial_cube_pose_w,
                meta=meta,
            )
        )
    return trajectories


def select_reference_trajectory(
    trajectories: list[ReferenceTrajectory],
    episode_idx: int,
    *,
    strategy: str,
    rng: np.random.Generator,
) -> ReferenceTrajectory:
    if not trajectories:
        raise ValueError("No trajectories available for selection.")
    if strategy == "round_robin":
        return trajectories[episode_idx % len(trajectories)]
    if strategy == "random":
        return trajectories[int(rng.integers(len(trajectories)))]
    raise ValueError(f"Unsupported trajectory sampling strategy: {strategy}")


def _closest_index_xy(ref_xy: np.ndarray, current_xy: np.ndarray, min_index: int) -> tuple[int, float]:
    if min_index >= len(ref_xy):
        min_index = len(ref_xy) - 1
    tail = ref_xy[min_index:]
    if tail.size == 0:
        idx = len(ref_xy) - 1
        dist = float(np.linalg.norm(ref_xy[idx] - current_xy))
        return idx, dist
    dists = np.linalg.norm(tail - current_xy[None, :], axis=1)
    rel_idx = int(np.argmin(dists))
    idx = min_index + rel_idx
    return idx, float(dists[rel_idx])


def compute_guidance_reward(
    state: TrajectoryGuidanceState,
    *,
    current_ee_pos_w: np.ndarray,
    current_gripper_state: float,
    coeffs: GuidanceRewardCoefficients,
) -> GuidanceStepMetrics:
    """Compute guidance reward from progress, XY error, and gripper alignment."""
    traj = state.traj
    ref_xy = traj.ee_pos_w[:, :2]
    current_xy = np.asarray(current_ee_pos_w, dtype=np.float32)[:2]

    prev_idx = int(round(state.prev_progress * max(traj.length - 1, 1)))
    closest_idx, xy_error = _closest_index_xy(ref_xy, current_xy, prev_idx)
    progress = float(closest_idx / max(traj.length - 1, 1))
    delta_progress = max(progress - state.prev_progress, 0.0)

    ref_gripper = float(traj.gripper_state[closest_idx])
    gripper_error = abs(float(current_gripper_state) - ref_gripper)

    progress_term = coeffs.progress_weight * (delta_progress**coeffs.progress_power)
    xy_term = coeffs.xy_weight * np.exp(-xy_error / max(coeffs.xy_scale, 1e-6))
    gripper_term = coeffs.gripper_weight * np.exp(-gripper_error / max(coeffs.gripper_scale, 1e-6))
    reward = float(progress_term + xy_term + gripper_term)

    state.prev_progress = max(state.prev_progress, progress)
    return GuidanceStepMetrics(
        reward=reward,
        progress=progress,
        delta_progress=delta_progress,
        xy_error=xy_error,
        gripper_error=gripper_error,
        closest_index=closest_idx,
    )
