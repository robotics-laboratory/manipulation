"""
Adapters: Isaac Lab obs/actions <-> SmolVLA policy I/O.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")
IMAGE_SHAPE = (3, 256, 256)
STATE_KEY = "observation.state"
LANGUAGE_KEY = "language_instruction"
TASK_KEY = "task"


def _to_numpy(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _resize_to_chw(img: np.ndarray, target_hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    img = _to_numpy(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        img = np.transpose(img, (2, 0, 1))
    h, w = img.shape[1], img.shape[2]
    if (h, w) != target_hw:
        y = np.linspace(0, h - 1, target_hw[0]).astype(np.int32)
        x = np.linspace(0, w - 1, target_hw[1]).astype(np.int32)
        img = img[:, y, :][:, :, x]
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def _gather_images(obs: dict[str, Any]) -> list[np.ndarray]:
    keys = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.images.camera3",
        "observation.images.side",
        "observation.images.up",
        "observation.images.top",
        "observation.images.wrist",
        "observation.images_top",
        "observation.images_wrist",
        "observation.images_side",
        "observation.images_up",
        "rgb",
        "image",
    )
    images = []
    for key in keys:
        if key not in obs or len(images) >= 3:
            continue
        value = _to_numpy(obs[key])
        if (value.ndim == 2 or (value.ndim == 3 and min(value.shape) >= 2)) and value.size > 0:
            images.append(value)
    return images


def _empty_image_batch() -> np.ndarray:
    return np.expand_dims(np.zeros((3, IMAGE_SHAPE[1], IMAGE_SHAPE[2]), dtype=np.float32), axis=0)


# When rename_map uses dataset-style keys (e.g. ``up``) but Isaac env exposes
# ``top``/``wrist`` for the same physical views, resolve to whichever key exists.
_RENAME_SRC_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "observation.images.up": (
        "observation.images.up",
        "observation.images_up",
        "observation.images.top",
        "observation.images_top",
    ),
    "observation.images.top": (
        "observation.images.top",
        "observation.images_top",
        "observation.images.up",
        "observation.images_up",
    ),
    "observation.images.side": (
        "observation.images.side",
        "observation.images_side",
        "observation.images.wrist",
        "observation.images_wrist",
    ),
    "observation.images.wrist": (
        "observation.images.wrist",
        "observation.images_wrist",
        "observation.images.side",
        "observation.images_side",
    ),
}


def _resolve_rename_src_obs_key(obs: dict[str, Any], src_key: str) -> str | None:
    """Return the observation key to read for ``src_key``, or None if no image found."""
    if src_key in obs:
        return src_key
    for candidate in _RENAME_SRC_EQUIVALENTS.get(src_key, ()):
        if candidate in obs:
            return candidate
    return None


def _fallback_tensor_for_missing_cameras(
    frame: dict[str, Any],
    present_camera_slots: list[str],
    strategy: str,
) -> np.ndarray:
    """Pick a (1,3,H,W) tensor to copy into missing camera slots."""
    if strategy == "zeros" or not present_camera_slots:
        return _empty_image_batch()
    if strategy == "last":
        return np.array(frame[present_camera_slots[-1]], copy=True)
    if strategy == "first":
        return np.array(frame[present_camera_slots[0]], copy=True)
    if strategy == "camera1":
        k = "observation.images.camera1"
        if k in frame:
            return np.array(frame[k], copy=True)
        return np.array(frame[present_camera_slots[0]], copy=True)
    if strategy == "camera2":
        k = "observation.images.camera2"
        if k in frame:
            return np.array(frame[k], copy=True)
        return np.array(frame[present_camera_slots[-1]], copy=True)
    return np.array(frame[present_camera_slots[-1]], copy=True)


def isaac_obs_to_policy_frame(
    obs: dict[str, Any],
    language_instruction: str = "Pick the cube.",
    image_key_map: dict[str, str] | None = None,
    state_keys: list[str] | None = None,
    observation_state_size: int | None = None,
    rename_map: dict[str, str] | None = None,
    empty_cameras: int = 0,
    missing_camera_fill: str = "first",
) -> dict[str, Any]:
    """
    Build a policy frame from Isaac env observation.

    rename_map maps env observation key -> policy key.

    missing_camera_fill: when fewer than three views are mapped, how to fill
    ``observation.images.camera*`` gaps (``first`` ≈ duplicate top/camera1 for
    cam2/3 — common for 2-camera SO-101 finetunes; ``last`` = legacy behavior).
    """
    frame = {LANGUAGE_KEY: language_instruction, TASK_KEY: language_instruction}
    image_key_map = image_key_map or {}
    rename_map = rename_map or {}

    if rename_map:
        for src_key, dst_key in rename_map.items():
            resolved = _resolve_rename_src_obs_key(obs, src_key)
            if resolved is None:
                continue
            img = _resize_to_chw(_to_numpy(obs[resolved]), (IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            frame[dst_key] = np.expand_dims(img.astype(np.float32), axis=0)

        has_any_image_key = any(k.startswith("observation.images.") for k in frame.keys())
        present_camera_slots = [k for k in CAMERA_KEYS if k in frame]

        # Base SmolVLA expects camera1/2/3 and often uses empty_cameras=0.
        # If we have some camera slots but not all, fill missing slots (duplicate
        # another view or zeros — see ``missing_camera_fill``).
        if empty_cameras <= 0 and present_camera_slots:
            fallback = _fallback_tensor_for_missing_cameras(
                frame, present_camera_slots, missing_camera_fill
            )
            for key in CAMERA_KEYS:
                if key not in frame:
                    frame[key] = np.array(fallback, copy=True)

        # If nothing image-like made it into the frame, keep one empty slot to avoid
        # complete failure in downstream preprocessors.
        if not has_any_image_key:
            frame[CAMERA_KEYS[0]] = _empty_image_batch()
    else:
        images = _gather_images(obs)
        if not images:
            images = [np.zeros((3, IMAGE_SHAPE[1], IMAGE_SHAPE[2]), dtype=np.float32)]
        n_real = max(0, len(CAMERA_KEYS) - empty_cameras)
        resized = []
        for i in range(len(CAMERA_KEYS)):
            if i < n_real and i < len(images):
                img = _resize_to_chw(images[i], (IMAGE_SHAPE[1], IMAGE_SHAPE[2]))
            else:
                img = np.zeros((3, IMAGE_SHAPE[1], IMAGE_SHAPE[2]), dtype=np.float32)
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)
            resized.append(img.astype(np.float32))
        for key, img in zip(CAMERA_KEYS, resized):
            frame[key] = np.expand_dims(img, axis=0)
        for src, dst in image_key_map.items():
            if src in frame and dst != src:
                frame[dst] = frame.pop(src)

    default_state_keys = (
        "observation.state",
        "joint_pos",
        "joint_positions",
        "proprio",
        "obs",
        "policy",
        "ee_pos",
        "ee_pos_delta",
        "ee_quat",
    )
    keys = state_keys if state_keys else default_state_keys
    state_parts = [_to_numpy(obs[k]).flatten() for k in keys if k in obs]
    state = np.concatenate(state_parts).astype(np.float32) if state_parts else np.zeros(0, dtype=np.float32)
    if observation_state_size is not None:
        n = observation_state_size
        if state.shape[0] >= n:
            state = state[:n].astype(np.float32)
        else:
            state = np.pad(state, (0, n - state.shape[0]), mode="constant", constant_values=0.0).astype(np.float32)
    frame[STATE_KEY] = np.expand_dims(state, axis=0)
    return frame


def policy_action_to_env(
    action: np.ndarray,
    env_action_space_shape: tuple[int, ...] | None = None,
    clip: bool = True,
    scale: float | None = None,
) -> np.ndarray:
    action = np.asarray(action).flatten()
    if env_action_space_shape is not None:
        n = int(np.prod(env_action_space_shape))
        action = action[:n] if action.shape[0] >= n else np.pad(action, (0, n - action.shape[0]))
    if clip:
        action = np.clip(action, -1.0, 1.0)
    if scale is not None:
        action = action * scale
    return action.astype(np.float32)
