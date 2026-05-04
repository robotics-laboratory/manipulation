import pyrealsense2 as rs
import numpy as np

class RealSenseSDKCamera:
    """Управление железом RealSense D435"""
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Включаем потоки цвета и глубины
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Инструмент для выравнивания карты глубины под цветную картинку
        self.align = rs.align(rs.stream.color)
        
        self.color_image = None
        self.depth_image = None
        self.is_connected = False

    def connect(self):
        if not self.is_connected:
            self.pipeline.start(self.config)
            self.is_connected = True
        return True

    def grab(self):
        """Захватывает новые кадры и синхронизирует их"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return False
            
        self.color_image = np.asanyarray(color_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())
        return True

    def get_color(self): 
        return self.color_image  # В формате BGR

    def get_depth(self): 
        return self.depth_image  # В формате Z16 (uint16)