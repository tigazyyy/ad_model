import carla
import math
import traceback
import cv2
import numpy as np
import time

class JunctionDetector:
    """交叉路口检测器"""
    
    def __init__(self, world, config):
        """初始化交叉路口检测器
        
        Args:
            world: CARLA世界对象
            config: 配置对象
        """
        self.world = world
        # 修改这里，使用正确的配置路径
        self.junction_distance_threshold = config.junction_detection["distance_threshold"]
        self.junction_confidence = 0.0  # 交叉路口置信度
        self.in_junction_area = False  # 是否在交叉路口区域
        print(f"交叉路口检测器已初始化，检测距离: {self.junction_distance_threshold}米")
        self.current_junction = None
        
        # 图像检测参数
        self.use_image_detection = True  # 是否使用图像检测
        self.image_detection_threshold = 0.5  # 降低图像检测阈值，提高灵敏度
        self.last_image_detections = []  # 保存最近的图像检测结果
        self.detection_history_size = 3  # 减少历史检测结果数量，更快响应变化
        
        # 添加拍照标志和计时器
        self.should_take_photo = False
        self.last_photo_time = 0
        self.photo_interval = 1.0  # 拍照间隔（秒）
        self.photo_count = 0
        self.max_photos = 3  # 每个路口最多拍摄的照片数
        
        # 添加调试信息
        self.debug_info = {
            'map_detection': False,
            'image_detection': False,
            'traffic_light_detected': False,
            'junction_features_detected': False,
            'distance_to_junction': float('inf')
        }
    
    def detect_traffic_lights(self, image):
        """检测图像中的交通信号灯
        
        Args:
            image: 输入图像
            
        Returns:
            bool: 如果检测到交通信号灯则返回True
        """
        try:
            if image is None:
                return False
                
            # 转换为HSV以便更好地检测颜色
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 定义交通信号灯颜色范围（红、黄、绿）
            # 红色在HSV中有两个范围
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            # 黄色
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # 绿色
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            
            # 创建掩码
            red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # 合并掩码
            combined_mask = cv2.bitwise_or(red_mask, yellow_mask)
            combined_mask = cv2.bitwise_or(combined_mask, green_mask)
            
            # 关注图像上半部分，交通信号灯通常出现在那里
            height, width = image.shape[:2]
            upper_region = combined_mask[0:int(height/2), :]
            
            # 计算非零像素
            light_pixels = np.count_nonzero(upper_region)
            
            # 计算光像素与上半区域总像素的比例
            ratio = light_pixels / (upper_region.shape[0] * upper_region.shape[1])
            
            # 降低阈值，提高检测灵敏度
            detected = ratio > 0.003  # 降低阈值，原来是0.005
            
            # 保存调试信息
            self.debug_info['traffic_light_detected'] = detected
            
            return detected
            
        except Exception as e:
            print(f"交通信号灯检测错误: {e}")
            traceback.print_exc()
            return False
    
    def detect_junction(self, vehicle):
        """检测车辆是否接近或位于交叉路口"""
        try:
            if not vehicle:
                return False, None
            
            # 获取车辆位置
            vehicle_location = vehicle.get_location()
            waypoint = self.world.get_map().get_waypoint(vehicle_location)
            
            # 检查是否已在交叉路口内
            if waypoint.is_junction:
                self.in_junction_area = True
                self.junction_confidence = 1.0
                self.current_junction = waypoint.get_junction()
                self.debug_info['map_detection'] = True
                self.debug_info['distance_to_junction'] = 0.0
                return True, self.current_junction
            
            # 检查前方是否有交叉路口
            min_distance = float('inf')
            junction_found = False
            
            # 使用多个距离检查点，提高检测灵敏度
            for distance in [5.0, 10.0, 15.0, 20.0, self.junction_distance_threshold]:
                next_waypoints = waypoint.next(distance)
                for next_wp in next_waypoints:
                    if next_wp.is_junction:
                        junction_found = True
                        if distance < min_distance:
                            min_distance = distance
                            # 根据距离计算置信度
                            self.junction_confidence = 1.0 - (distance / self.junction_distance_threshold)
                            self.in_junction_area = self.junction_confidence > 0.5  # 降低阈值，原来是0.7
                            self.current_junction = next_wp.get_junction()
            
            # 保存调试信息
            self.debug_info['map_detection'] = junction_found
            self.debug_info['distance_to_junction'] = min_distance if junction_found else float('inf')
            
            if junction_found:
                return True, self.current_junction
            
            # 重置状态
            self.in_junction_area = False
            self.junction_confidence = 0.0
            self.current_junction = None
            return False, None
            
        except Exception as e:
            print(f"检测交叉路口异常: {e}")
            traceback.print_exc()
            return False, None
    
    def is_approaching_junction(self, image=None):
        """综合判断是否接近交叉路口，结合地图和图像信息"""
        try:
            # 首先使用地图信息判断
            vehicle = None
            for actor in self.world.get_actors():
                if actor.type_id.startswith('vehicle') and actor.is_alive:
                    # 假设第一个找到的车辆是玩家车辆
                    vehicle = actor
                    break
            
            map_detection = False
            if vehicle:
                map_detection, _ = self.detect_junction(vehicle)
            
            # 如果启用图像检测且有图像输入
            image_detection = False
            if self.use_image_detection and image is not None:
                # 首先检查交通信号灯
                traffic_light_detected = self.detect_traffic_lights(image)
                
                # 然后使用原始图像检测
                junction_detected = self.detect_junction_from_image(image)
                
                # 交通信号灯或交叉路口特征都表示存在交叉路口
                image_detection = traffic_light_detected or junction_detected
                    
                # 更新历史检测结果
                self.last_image_detections.append(image_detection)
                if len(self.last_image_detections) > self.detection_history_size:
                    self.last_image_detections.pop(0)
                
                # 使用多数投票法确定最终结果
                image_detection_count = sum(self.last_image_detections)
                image_detection = image_detection_count > (self.detection_history_size / 2)
                
                # 保存调试信息
                self.debug_info['image_detection'] = image_detection
            
            # 综合判断：地图检测或图像检测任一为真，则认为接近交叉路口
            is_approaching = map_detection or image_detection
            
            # 更新拍照标志
            current_time = time.time()
            if is_approaching and (current_time - self.last_photo_time > self.photo_interval) and self.photo_count < self.max_photos:
                self.should_take_photo = True
                self.last_photo_time = current_time
                self.photo_count += 1
                print(f"检测到交叉路口，准备拍照 ({self.photo_count}/{self.max_photos})")
            elif not is_approaching:
                # 重置拍照计数器
                self.photo_count = 0
            
            return is_approaching
        
        except Exception as e:
            print(f"判断是否接近交叉路口时出错: {e}")
            traceback.print_exc()
            return False
    
    def detect_junction_from_image(self, image):
        """从图像中检测交叉路口
        
        使用计算机视觉技术从图像中检测交叉路口特征，如:
        1. 道路标线变化
        2. 道路宽度变化
        3. 交通信号灯
        """
        try:
            # 图像预处理
            height, width = image.shape[:2]
            
            # 1. 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 2. 应用高斯模糊
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 3. Canny边缘检测
            edges = cv2.Canny(blur, 50, 150)
            
            # 4. 设置ROI区域 - 关注道路前方区域
            roi_height = height * 0.5  # ROI高度为图像高度的50%
            roi_top = int(height * 0.3)  # ROI顶部位置
            roi_bottom = int(roi_top + roi_height)  # ROI底部位置
            
            # 创建ROI掩码
            mask = np.zeros_like(edges)
            roi_vertices = np.array([
                [0, roi_bottom],
                [0, roi_top],
                [width, roi_top],
                [width, roi_bottom]
            ], np.int32)
            cv2.fillPoly(mask, [roi_vertices], 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # 5. 霍夫变换检测直线
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=20)
            
            # 6. 分析线段特征
            if lines is not None:
                # 计算水平线和垂直线的数量
                horizontal_lines = 0
                vertical_lines = 0
                
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        # 计算线段角度
                        if x2 - x1 == 0:  # 避免除零错误
                            angle = 90  # 垂直线
                        else:
                            angle = abs(math.degrees(math.atan((y2 - y1) / (x2 - x1))))
                        
                        # 分类线段
                        if angle < 30:  # 接近水平的线段
                            horizontal_lines += 1
                        elif angle > 60:  # 接近垂直的线段
                            vertical_lines += 1
                
                # 7. 判断是否为交叉路口
                # 交叉路口通常有更多的垂直线（如停止线、人行横道）和水平线（如交叉道路）
                total_lines = len(lines)
                horizontal_ratio = horizontal_lines / total_lines if total_lines > 0 else 0
                vertical_ratio = vertical_lines / total_lines if total_lines > 0 else 0
                
                # 降低阈值，提高检测灵敏度
                is_junction = (vertical_ratio > 0.25 and horizontal_ratio > 0.15)  # 原来是0.3和0.2
                
                # 保存调试信息
                self.debug_info['junction_features_detected'] = is_junction
                
                return is_junction
            
            return False
            
        except Exception as e:
            print(f"从图像检测交叉路口异常: {e}")
            traceback.print_exc()
            return False
    
    def should_capture_photo(self):
        """检查是否应该拍照"""
        if self.should_take_photo:
            self.should_take_photo = False  # 重置标志
            return True
        return False
    
    def reset_photo_counter(self):
        """重置拍照计数器"""
        self.photo_count = 0
        
    def get_debug_info(self):
        """获取调试信息"""
        return self.debug_info
    
    def get_junction_waypoints(self, junction=None):
        """获取交叉路口的所有可能路径"""
        try:
            # 如果没有提供junction参数，使用当前保存的junction
            if junction is None:
                junction = self.current_junction
                
            if not junction:
                return []
            
            # 获取交叉路口的所有路径
            waypoints = []
            for wp_pair in junction.get_waypoints(carla.LaneType.Driving):
                waypoints.append((wp_pair[0], wp_pair[1]))  # 路口入口和出口
            
            return waypoints
            
        except Exception as e:
            print(f"获取交叉路口航点时出错: {e}")
            return []
    
    def analyze_junction_directions(self, vehicle):
        """分析交叉路口可能的方向"""
        try:
            if not self.current_junction or not self.in_junction_area:
                return []
            
            # 获取车辆朝向
            vehicle_transform = vehicle.get_transform()
            vehicle_yaw = vehicle_transform.rotation.yaw
            
            # 获取可能的路径
            junction_paths = self.get_junction_waypoints()
            
            # 分析每条路径相对于当前车辆的方向
            directions = []
            for entry, exit in junction_paths:
                # 计算出口相对于入口的方向
                exit_yaw = exit.transform.rotation.yaw
                
                # 计算角度差
                yaw_diff = (exit_yaw - vehicle_yaw + 180) % 360 - 180
                
                if -45 <= yaw_diff <= 45:
                    directions.append({
                        "type": "前",
                        "yaw": exit_yaw,
                        "diff": yaw_diff,
                        "waypoint": exit
                    })
                elif 45 < yaw_diff <= 135:
                    directions.append({
                        "type": "右",
                        "yaw": exit_yaw,
                        "diff": yaw_diff,
                        "waypoint": exit
                    })
                elif -135 <= yaw_diff < -45:
                    directions.append({
                        "type": "左",
                        "yaw": exit_yaw,
                        "diff": yaw_diff,
                        "waypoint": exit
                    })
                else:
                    directions.append({
                        "type": "掉头",
                        "yaw": exit_yaw,
                        "diff": yaw_diff,
                        "waypoint": exit
                    })
            
            return directions
            
        except Exception as e:
            print(f"分析交叉路口方向时出错: {e}")
            return []
    
    def create_debug_image(self, vehicle, image):
        """创建交叉路口调试图像"""
        # 此处实现可以根据需要添加
        return image