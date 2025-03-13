import os
import time
import cv2
import numpy as np

# 确保保存图像的文件夹存在
PHOTO_DIR = "/home/tiga/tiga_ws/carla/PythonAPI/ad_model/photo"
os.makedirs(PHOTO_DIR, exist_ok=True)

import carla
import numpy as np

class SensorManager:
    """负责车辆各类传感器的初始化和数据获取"""
    def __init__(self, vehicle, world, controller):
        self.vehicle = vehicle
        self.world = world
        self.controller = controller
        self.sensors = []
        self.last_save_time = 0
        self.frame_counter = 0
        self.should_save_image = False  # 添加标志控制是否保存图片
        
        # 初始化传感器
        self._setup_sensors()
    
    def _setup_sensors(self):
        """设置传感器"""
        # 添加RGB相机
        self._setup_camera()
        print(f"已设置 {len(self.sensors)} 个传感器")
    
    def _setup_camera(self):
        """设置前视相机"""
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.controller.config.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.controller.config.camera_height))
        camera_bp.set_attribute('fov', str(self.controller.config.camera_fov))
        
        # 设置相机位置
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.7), carla.Rotation(pitch=-15))
        
        # 生成相机并附加到车辆
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(lambda image: self._process_image(image))
        self.sensors.append(camera)
        
        print("前视相机已设置")
    
    def _process_image(self, image):
        """处理摄像头图像"""
        try:
            # 转换图像格式
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # 只保留RGB通道
            
            # 保存到控制器
            self.controller.front_image = array
            
            # 只在需要时保存图片（接近交叉路口时）
            if self.should_save_image:
                current_time = time.time()
                save_interval = 0.5  # 每0.5秒保存一次
                
                if current_time - self.last_save_time >= save_interval:
                    self.last_save_time = current_time
                    self.frame_counter += 1
                    
                    # 生成文件名
                    timestamp = int(current_time * 1000)
                    filename = f"junction_{timestamp}.png"
                    filepath = os.path.join(PHOTO_DIR, filename)
                    
                    # 保存图像
                    cv2.imwrite(filepath, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
                    
                    # 限制图片数量
                    self._manage_photos(100)
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            traceback.print_exc()
    
    def _manage_photos(self, max_count):
        """管理照片数量，保持照片数量在指定范围内"""
        try:
            if not os.path.exists(PHOTO_DIR):
                return
                
            # 获取所有照片并按名称排序（名称中包含时间戳）
            photos = sorted(os.listdir(PHOTO_DIR))
            
            # 如果照片数量超过最大值，删除旧照片
            if len(photos) > max_count:
                # 需要删除的数量
                to_delete = len(photos) - max_count
                
                for i in range(to_delete):
                    old_photo = os.path.join(PHOTO_DIR, photos[i])
                    try:
                        os.remove(old_photo)
                        # print(f"删除旧照片: {old_photo}")
                    except Exception as e:
                        print(f"删除照片失败: {e}")
                        
        except Exception as e:
            print(f"管理照片时出错: {e}")
    
    def cleanup(self):
        """清理传感器"""
        for sensor in self.sensors:
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        self.sensors.clear()
        print("传感器已清理")
    
    def set_save_image(self, should_save):
        """设置是否保存图片"""
        self.should_save_image = should_save