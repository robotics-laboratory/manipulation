#!/usr/bin/env python3
"""
Load top/wrist camera config from a USD file at runtime.
Expects cameras nested under prims named "CameraTopXform" and "CameraWristXform".
Used to override env scene cameras when running with --camera_usd /path/to/scene.usd.
"""

from __future__ import annotations

from pathlib import Path


def _find_camera_under_prim(stage, prim):
    """Return first UsdGeom.Camera under prim (prim or any descendant), or None."""
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Camera):
        return prim
    for child in prim.GetChildren():
        found = _find_camera_under_prim(stage, child)
        if found is not None:
            return found
    return None


def _get_local_pose(prim):
    """Return (position_xyz, quat_wxyz) from prim local transform."""
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    local_tf = xform.GetLocalTransformation()
    local = local_tf[0] if isinstance(local_tf, tuple) else local_tf
    pos = local.ExtractTranslation()
    rot = local.ExtractRotation().GetQuat()
    # Gf.Quat: GetReal() = w, GetImaginary() = (x,y,z)
    quat_wxyz = (
        rot.GetReal(),
        rot.GetImaginary()[0],
        rot.GetImaginary()[1],
        rot.GetImaginary()[2],
    )
    return (pos[0], pos[1], pos[2]), quat_wxyz


def _get_camera_intrinsics(prim):
    """Return dict with focal_length, horizontal_aperture, vertical_aperture, clipping_range."""
    from pxr import UsdGeom

    cam = UsdGeom.Camera(prim)
    focal = cam.GetFocalLengthAttr()
    ha = cam.GetHorizontalApertureAttr()
    va = cam.GetVerticalApertureAttr()
    clip = cam.GetClippingRangeAttr()
    return {
        "focal_length": focal.Get() if focal else 24.0,
        "horizontal_aperture": ha.Get() if ha else 20.955,
        "vertical_aperture": va.Get() if va else 15.716,  # 20.955 * 3/4
        "clipping_range": tuple(clip.Get()) if clip else (0.1, 1.0e5),
    }


def load_camera_config_from_usd(
    usd_path: str | Path,
    top_xform_name: str = "CameraTopXform",
    wrist_xform_name: str = "CameraWristXform",
    width: int = 640,
    height: int = 480,
):
    """
    Open the USD and find cameras under prims named top_xform_name and wrist_xform_name.
    Falls back to legacy xform names (CameraSideXform/CameraUpXform) if needed.
    Returns (camera_top_cfg, camera_wrist_cfg) as TiledCameraCfg instances, or (None, None) if not found.
    """
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import TiledCameraCfg

    usd_path = Path(usd_path).resolve()
    if not usd_path.exists():
        raise FileNotFoundError(f"Camera USD not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    def find_camera_by_xform_name(name):
        for prim in stage.Traverse():
            if prim.GetName() == name:
                cam = _find_camera_under_prim(stage, prim)
                if cam is not None:
                    return cam
        return None

    top_cam = find_camera_by_xform_name(top_xform_name) or find_camera_by_xform_name("CameraSideXform")
    wrist_cam = find_camera_by_xform_name(wrist_xform_name) or find_camera_by_xform_name("CameraUpXform")

    def make_cfg(cam_prim, env_prim_path_key):
        if cam_prim is None:
            return None
        pos, quat_wxyz = _get_local_pose(cam_prim)
        intrinsics = _get_camera_intrinsics(cam_prim)
        return TiledCameraCfg(
            prim_path=env_prim_path_key,
            offset=TiledCameraCfg.OffsetCfg(
                pos=pos,
                rot=quat_wxyz,
                # USD camera prim rotation is authored in OpenGL camera convention.
                convention="opengl",
            ),
            data_types=["rgb"],
            width=width,
            height=height,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=intrinsics["focal_length"],
                focus_distance=400.0,
                horizontal_aperture=intrinsics["horizontal_aperture"],
                clipping_range=intrinsics["clipping_range"],
            ),
        )

    camera_top_cfg = make_cfg(top_cam, "{ENV_REGEX_NS}/CameraTop")
    camera_wrist_cfg = make_cfg(wrist_cam, "{ENV_REGEX_NS}/Robot/gripper_link/CameraWrist")
    return camera_top_cfg, camera_wrist_cfg


def apply_camera_usd_to_env_cfg(env_cfg, usd_path: str | Path, **kwargs):
    """
    If env_cfg has scene.camera_top/camera_wrist, override them from the given USD.
    Also supports legacy scene.camera_side/camera_up names.
    kwargs are passed to load_camera_config_from_usd (e.g. width, height).
    """
    camera_top_cfg, camera_wrist_cfg = load_camera_config_from_usd(usd_path, **kwargs)
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    if camera_top_cfg is not None:
        if hasattr(scene, "camera_top"):
            scene.camera_top = camera_top_cfg
        elif hasattr(scene, "camera_side"):
            scene.camera_side = camera_top_cfg
    if camera_wrist_cfg is not None:
        if hasattr(scene, "camera_wrist"):
            scene.camera_wrist = camera_wrist_cfg
        elif hasattr(scene, "camera_up"):
            scene.camera_up = camera_wrist_cfg
