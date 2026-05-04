import numpy as np
import pyzed.sl as sl

class ZEDSDKCamera:
    """Базовый класс для работы с ZED SDK"""
    def __init__(self, resolution="HD720", fps=30):
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        self.init_params.camera_fps = fps
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        
        # Настройки для ближнего диапазона --
        self.init_params.depth_minimum_distance = 200 
        self.init_params.depth_maximum_distance = 500
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL # Лучшее качество
        
        self.runtime_params = sl.RuntimeParameters()
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.depth_map = sl.Mat()

    def connect(self):
        if not self.zed.is_opened():
            err = self.zed.open(self.init_params)
            return err == sl.ERROR_CODE.SUCCESS
        return True

    def grab(self):
        """Захват текущего состояния всех сенсоров"""
        if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.left_image, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.right_image, sl.VIEW.RIGHT)
            self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
            return True
        return False

    def get_left(self): return self.left_image.get_data()[:, :, :3] # BGR
    def get_right(self): return self.right_image.get_data()[:, :, :3]
    def get_depth(self): return self.depth_map.get_data()