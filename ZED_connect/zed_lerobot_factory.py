from .zed_sdk_camera import ZEDSDKCamera
from lerobot_zed_camera import LeRobotZEDCamera

_SHARED_ZED = ZEDSDKCamera()

def create_zed_cameras_config():
    """Генерация конфига для LeRobot"""
    return {
        "cameras": {
            "laparoscope_left": {
                "type": "custom",
                "class": LeRobotZEDCamera,
                "kwargs": {"zed_manager": _SHARED_ZED, "mode": "left"}
            },
            "laparoscope_right": {
                "type": "custom",
                "class": LeRobotZEDCamera,
                "kwargs": {"zed_manager": _SHARED_ZED, "mode": "right"}
            },
            # Глубина
            # "laparoscope_depth": {
            #     "type": "custom",
            #     "class": LeRobotZEDCamera,
            #     "kwargs": {"zed_manager": _SHARED_ZED, "mode": "depth"}
            # }
        }
    }