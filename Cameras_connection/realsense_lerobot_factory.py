from .realsense_sdk_camera import RealSenseSDKCamera
from typing import Tuple
import numpy as np

# Общий менеджер камеры
_SHARED_REALSENSE = RealSenseSDKCamera()

class LeRobotRealsenseCamera:
    """Адаптер для LeRobot"""
    def __init__(self, rs_manager, mode='color'):
        self.manager = rs_manager
        self.mode = mode # 'color' или 'depth'

    def connect(self):
        return self.manager.connect()

    def read(self) -> Tuple[np.ndarray, dict]:
        # 'color' работает как триггер захвата (так же, как 'left' у ZED)
        if self.mode == 'color':
            self.manager.grab()
            frame = self.manager.get_color()
        else: # mode == 'depth'
            frame = self.manager.get_depth()
            
        metadata = {'mode': self.mode}
        return frame, metadata

def create_realsense_camera(**kwargs):
    mode = kwargs.get("config", {}).get("rs_mode", "color")
    return LeRobotRealsenseCamera(rs_manager=_SHARED_REALSENSE, mode=mode)