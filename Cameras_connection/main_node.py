#как я понял этот файл нужен чтобы dora нормально работала
import os
import cv2
import numpy as np
from dora import Node

from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    node = Node()
    config_path = os.getenv("LEROBOT_CONFIG_PATH", "configs/your_zed_robot.yaml")
    
    print("Подключение к роботу и лидеру...")
    robot_config = SO101FollowerConfig(port="/dev/ttyUSB0")
    teleop_config = SO101LeaderConfig(port="/dev/ttyUSB1")
    
    robot = SO101Follower(robot_config)
    teleop_device = SO101Leader(teleop_config)
    
    robot.connect()
    teleop_device.connect()
    
    print("Создание датасета LeRobot...")
    dataset = LeRobotDataset.create(
        repo_id="so101_stereo_dataset",
        fps=30,
        features={
            "observation.images.laparoscope_left": {"dtype": "uint8", "shape": (720, 1280, 3)},
            "observation.images.wrist_color": {"dtype": "uint8", "shape": (480, 640, 3)},
            "observation.images.wrist_depth": {"dtype": "uint16", "shape": (480, 640, 1)},
            "observation.state": {"dtype": "float32", "shape": (6,)},
            "action": {"dtype": "float32", "shape": (6,)}
        }
    )

    is_recording = False
    episode_frames = 0
    print("\n--- ГОТОВО К РАБОТЕ ---")
    print("Нажми 'r' в окне камеры для СТАРТА/СТОПА записи эпизода.")
    print("Нажми 'q' для ВЫХОДА.\n")

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            
            observation = robot.get_observation()
            
            action = teleop_device.get_action()
            
            robot.send_action(action)
            
            if "laparoscope_left" in observation:
                cv2.imshow("ZED Left", observation["laparoscope_left"])
            if "wrist_color" in observation:
                cv2.imshow("Realsense Wrist", observation["wrist_color"])
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                if not is_recording:
                    is_recording = True
                    episode_frames = 0
                    print(f"🔴 ЗАПИСЬ ЭПИЗОДА {dataset.num_episodes} НАЧАТА...")
                else:
                    is_recording = False
                    dataset.save_episode() # Сохраняем собранные кадры в эпизод
                    print(f"⏹ ЭПИЗОД СОХРАНЕН! Всего кадров: {episode_frames}")
                    
            elif key == ord('q'):
                print("Выход...")
                break

            if is_recording:
                frame_dict = {
                    "observation.images.laparoscope_left": observation.get("laparoscope_left"),
                    "observation.images.wrist_color": observation.get("wrist_color"),
                    "observation.images.wrist_depth": observation.get("wrist_depth", np.zeros((480, 640, 1), dtype=np.uint16)),
                    "observation.state": observation.get("state"),
                    "action": action
                }
                dataset.add_frame(frame_dict)
                episode_frames += 1

        elif event["type"] == "STOP":
            break

    robot.disconnect()
    teleop_device.disconnect()
    cv2.destroyAllWindows()
    
    if is_recording:
        dataset.save_episode()
    
    dataset.consolidate()
    print("Датасет успешно сохранен на диске!")

if __name__ == "__main__":
    main()