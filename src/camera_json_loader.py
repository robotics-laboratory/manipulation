#!/usr/bin/env python3
"""
Load camera pose overrides from a JSON file and apply to env_cfg scene cameras.

Expected JSON shape:
{
  "camera_top": {
    "translate": [x, y, z],
    "orient": [rx_deg, ry_deg, rz_deg],
    "euler_mode": "ui"
  },
  "camera_wrist": {
    "translate": [x, y, z],
    "orient": [rx_deg, ry_deg, rz_deg]
  }
}

Notes:
- "orient" with 3 values is interpreted as XYZ Euler angles in degrees.
- "orient" with 4 values is interpreted as quaternion (w, x, y, z).
- Optional "euler_mode" (when orient has 3 values):
  - "xyz_extrinsic" (default): q = qz * qy * qx
  - "ui" or "xyz_intrinsic": q = qx * qy * qz (matches Isaac Transform panel entry order)
- Optional "convention" per camera can be "opengl", "ros", or "world".
  For Isaac Lab camera offsets, "opengl" is typically the correct convention.
- For reproducible camera poses, prefer quaternion orientation in JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _qmul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _euler_xyz_deg_to_quat_wxyz(euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert XYZ Euler degrees to quaternion (w, x, y, z)."""
    rx, ry, rz = (math.radians(v) for v in euler_deg)
    qx = (math.cos(rx / 2.0), math.sin(rx / 2.0), 0.0, 0.0)
    qy = (math.cos(ry / 2.0), 0.0, math.sin(ry / 2.0), 0.0)
    qz = (math.cos(rz / 2.0), 0.0, 0.0, math.sin(rz / 2.0))
    # Equivalent to applying XYZ Euler rotations (extrinsic): q = qz * qy * qx
    return _qmul(_qmul(qz, qy), qx)


def _euler_xyz_deg_to_quat_wxyz_intrinsic(euler_deg: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert XYZ Euler degrees to quaternion (w, x, y, z), intrinsic order (q = qx * qy * qz)."""
    rx, ry, rz = (math.radians(v) for v in euler_deg)
    qx = (math.cos(rx / 2.0), math.sin(rx / 2.0), 0.0, 0.0)
    qy = (math.cos(ry / 2.0), 0.0, math.sin(ry / 2.0), 0.0)
    qz = (math.cos(rz / 2.0), 0.0, 0.0, math.sin(rz / 2.0))
    return _qmul(_qmul(qx, qy), qz)


def _to_xyz(vec, field_name: str) -> tuple[float, float, float]:
    if not isinstance(vec, (list, tuple)) or len(vec) != 3:
        raise ValueError(f"'{field_name}' must be a list/tuple of 3 numbers")
    return float(vec[0]), float(vec[1]), float(vec[2])


def _to_quat_wxyz(
    orient,
    field_name: str,
    euler_mode: str = "xyz_extrinsic",
) -> tuple[float, float, float, float]:
    if not isinstance(orient, (list, tuple)):
        raise ValueError(f"'{field_name}' must be a list/tuple")
    if len(orient) == 3:
        euler_xyz = _to_xyz(orient, field_name)
        if euler_mode in {"ui", "xyz_intrinsic"}:
            return _euler_xyz_deg_to_quat_wxyz_intrinsic(euler_xyz)
        if euler_mode == "xyz_extrinsic":
            return _euler_xyz_deg_to_quat_wxyz(euler_xyz)
        raise ValueError(f"'{field_name}' euler_mode must be one of: xyz_extrinsic, xyz_intrinsic, ui")
    if len(orient) == 4:
        return float(orient[0]), float(orient[1]), float(orient[2]), float(orient[3])
    raise ValueError(f"'{field_name}' must contain either 3 Euler values or 4 quaternion values")


def _apply_single_camera(scene, camera_key: str, camera_cfg_data: dict):
    scene_attr = camera_key
    if not hasattr(scene, scene_attr):
        # Backward compatibility aliases
        if camera_key == "camera_top" and hasattr(scene, "camera_side"):
            scene_attr = "camera_side"
        elif camera_key == "camera_wrist" and hasattr(scene, "camera_up"):
            scene_attr = "camera_up"
        else:
            return False

    if not isinstance(camera_cfg_data, dict):
        raise ValueError(f"'{camera_key}' must be an object")

    translate = camera_cfg_data.get("translate", camera_cfg_data.get("pos"))
    orient = camera_cfg_data.get("orient", camera_cfg_data.get("rot"))
    euler_mode = camera_cfg_data.get("euler_mode", "xyz_extrinsic")
    if translate is None or orient is None:
        raise ValueError(f"'{camera_key}' must include 'translate' and 'orient'")

    pos = _to_xyz(translate, f"{camera_key}.translate")
    quat_wxyz = _to_quat_wxyz(orient, f"{camera_key}.orient", euler_mode=euler_mode)
    convention = camera_cfg_data.get("convention", "opengl")
    if convention not in {"opengl", "ros", "world"}:
        raise ValueError(f"'{camera_key}.convention' must be one of: opengl, ros, world")

    camera_cfg = getattr(scene, scene_attr)
    camera_cfg.offset.pos = pos
    camera_cfg.offset.rot = quat_wxyz
    camera_cfg.offset.convention = convention
    setattr(scene, scene_attr, camera_cfg)
    return True


def apply_camera_json_to_env_cfg(env_cfg, json_path: str | Path):
    """
    Apply camera pose overrides from JSON to env_cfg.scene.

    Recognized camera keys:
    - camera_top
    - camera_wrist
    """
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Camera JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Camera JSON root must be an object")

    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return

    applied = 0
    for key in ("camera_top", "camera_wrist"):
        cam_cfg_data = data.get(key)
        if cam_cfg_data is None:
            continue
        if _apply_single_camera(scene, key, cam_cfg_data):
            applied += 1

    if applied == 0:
        raise ValueError("No compatible camera entries found in JSON (expected camera_top and/or camera_wrist)")
