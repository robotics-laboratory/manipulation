import torch
from typing import Tuple

class LeRobotZEDCamera:
    """Адаптер, который делает ZED совместимым с LeRobot"""
    def __init__(self, zed_manager, mode='left'):
        self.manager = zed_manager # Общий объект ZEDSDKCamera
        self.mode = mode # 'left', 'right' или 'depth'

    def connect(self):
        return self.manager.connect()

    def read(self) -> Tuple[np.ndarray, dict]:
        # В режиме синглтона один из потоков вызывает grab()
        if self.mode == 'left':
            self.manager.grab()
            frame = self.manager.get_left()
        elif self.mode == 'right':
            frame = self.manager.get_right()
        else: # depth
            frame = self.manager.get_depth()
            
        metadata = {'mode': self.mode}
        return frame, metadata