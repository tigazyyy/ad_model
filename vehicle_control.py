import pygame
import carla
import cv2
import numpy as np
import math
import time
import traceback
import sys
import os
import json
import socket
import threading

from lane_detection import LaneDetector
from lane_detector import AdvancedLaneDetector, SimpleOpenCVLaneDetector  # 导入新的简单OpenCV车道线检测器
from junction_detect import JunctionDetector
from utils.pid_controller import PIDController, LateralPIDController, LongitudinalPIDController
from sensor_manager import SensorManager
from model_client import OllamaClient
from trajectory_planner import TrajectoryPlanner
from display import Display
from mpc_controller import MPCController  # 导入MPC控制器

class VehicleController:
    """车辆控制器"""
    
    def __init__(self, client, world, config):
        """初始化车辆控制器
        
        Args:
            client: Carla客户端
            world: Carla世界
            config: 配置参数
        """
        self.client = client
        self.world = world
        self.config = config
        
        # 初始化车辆和传感器
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.front_image = None
        self.debug_image = None
        
        # 初始化控制参数
        self.target_speed = config.default_speed  # 使用config.default_speed
        self.max_throttle = 0.75  # 默认值
        self.max_brake = 0.3  # 默认值
        self.max_steering = 0.8  # 默认值
        
        # 初始化控制模式
        self.control_mode = "AUTO"  # AUTO, MODEL, MANUAL, JUNCTION
        self.previous_mode = "AUTO"
        self.view_mode = "FOLLOW"  # FOLLOW, BIRDVIEW, FIXED
        
        # 初始化状态变量
        self.current_status = {
            'speed': 0.0,
            'lane_center': 0.0,
            'lane_confidence': 0.0,
            'collision': False,
            'lane_invasion': False
        }
        
        # 初始化路口检测相关变量
        self.is_at_junction = False
        self.junction_detector = None
        self.junction_decision = None
        self.last_steer = 0.0
        
        # 初始化交叉路口状态
        self.junction_state = {
            'in_junction': False,
            'trajectory_generated': False,
            'trajectory_completed': True,
            'direction': 'straight',
            'stopped_at_junction': False,  # 是否在路口已停止
            'waiting_for_model': False,    # 是否正在等待模型响应
            'model_queried': False,        # 是否已经查询过模型
            'model_response': None,        # 模型响应
            'safe_to_proceed': False       # 是否安全通过
        }
        
        # 初始化车道检测器
        self.lane_detector = None
        
        # 初始化PID控制器
        self.pid_controllers = None
        
        # 初始化MPC控制器
        self.mpc_controller = None
        
        # 初始化模型客户端
        self.model_client = None
        
        # 初始化显示
        self.display = None
        
        # 初始化轨迹规划器
        self.trajectory_planner = None
        
        # 初始化标志
        self.initialized = False
        self.running = False
        self.debug_mode = config.visualization["enable_debug"] if hasattr(config, "visualization") else True
        self.show_trajectory = True  # 默认开启轨迹可视化
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化系统"""
        try:
            print("开始初始化...")
            
            # 初始化pygame
            pygame.init()
            self.screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption("Autonomous Driving")
            
            # 初始化HUD显示
            self.hud = Display()
            
            # 使用支持中文的字体
            try:
                self.font = pygame.font.Font("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 24)
            except:
                try:
                    self.font = pygame.font.Font("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 24)
                except:
                    print("警告：找不到中文字体，界面可能显示乱码")
                    self.font = pygame.font.SysFont("Arial", 24)
            
            # 生成车辆
            self._spawn_vehicle()
            
            # 初始化传感器
            self.sensor_manager = SensorManager(self.vehicle, self.world, self)
            
            # 初始化车道线检测器
            # 使用新的SimpleOpenCVLaneDetector
            self.lane_detector = SimpleOpenCVLaneDetector(self.vehicle, self.world, self.config)
            
            # 初始化交叉路口检测器
            self.junction_detector = JunctionDetector(self.world, self.config)
            
            # 初始化模型客户端
            self.model_client = OllamaClient(self.config)
            
            # 初始化轨迹规划器
            self.trajectory_planner = TrajectoryPlanner()

            # 初始化MPC控制器
            self.mpc_controller = MPCController(self.config)
            
            # 初始化PID控制器
            self.pid_controllers = {
                'throttle': PIDController(1.0, 0.1, 0.05),  # 油门PID
                'steer': PIDController(1.0, 0.0, 0.1),      # 转向PID
                'brake': PIDController(1.0, 0.1, 0.05)      # 刹车PID
            }
            
            # 初始化转向滤波器
            self.steer_filter = []
            self.steer_filter_size = 5
            
            # 初始化调试打印相关变量
            self.last_print_time = time.time()
            self.print_interval = 0.5  # 打印间隔（秒）
            
            # 修改车辆物理属性，提高转向响应和移动性能
            self._modify_vehicle_physics()
            
            # 设置观察者相机
            self.spectator = self.world.get_spectator()
            
            # 标记为初始化完成
            self.initialized = True
            print("初始化完成")
            
            # 在initialize方法中更新这些点
            self.src_points = np.float32([
                [520, 480],  # 左上 - 向下移动
                [760, 480],  # 右上 - 向下移动
                [1200, 680], # 右下 - 扩大宽度
                [100, 680]   # 左下 - 扩大宽度
            ])
            
        except Exception as e:
            print(f"初始化失败: {e}")
            traceback.print_exc()
            self.cleanup()
    
    def _spawn_vehicle(self):
        """生成车辆"""
        try:
            # 获取车辆蓝图
            blueprint = self.world.get_blueprint_library().find(self.config.vehicle_type)
            if blueprint is None:
                print(f"无法找到车辆蓝图: {self.config.vehicle_type}")
                return False
            
            # 获取生成点
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                print("地图中没有生成点")
                return False
            
            # 尝试多个生成点
            for i in range(len(spawn_points)):
                try_point_id = (self.config.spawn_point_id + i) % len(spawn_points)
                spawn_point = spawn_points[try_point_id]
                
                # 检查生成点是否在车道上
                waypoint = self.world.get_map().get_waypoint(spawn_point.location)
                if waypoint.lane_type == carla.LaneType.Driving:
                    # 调整生成点的朝向以对齐车道
                    spawn_point.rotation = waypoint.transform.rotation
                    
                    try:
                        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
                        print(f"车辆成功生成于: {spawn_point.location}, 朝向: {spawn_point.rotation}")
                        
                        # 等待一帧确保车辆生成
                        self.world.tick()
                        
                        # 修改车辆物理属性，使其更容易转向
                        self._modify_vehicle_physics()
                        
                        return True
                    except Exception as e:
                        print(f"在生成点 {try_point_id} 生成失败: {e}")
                        continue
            
            print("无法找到合适的生成点")
            return False
            
        except Exception as e:
            print(f"生成车辆时出错: {e}")
            traceback.print_exc()
            return False
    
    def _modify_vehicle_physics(self):
        """修改车辆物理特性，使车辆更容易控制"""
        try:
            # 获取物理控制参数
            physics_control = self.vehicle.get_physics_control()
            
            # 增强刹车力
            physics_control.brakes_force = 1.0   # 增强刹车力
            physics_control.mass = 1800          # 增加质量提高稳定性
            physics_control.damping_rate_full_throttle = 0.1  # 降低加速率
            physics_control.damping_rate_zero_throttle_clutch_engaged = 0.3  # 增加阻尼
            physics_control.damping_rate_zero_throttle_clutch_disengaged = 0.3  # 增加阻尼
            
            # 配置车轮物理特性
            for wheel in physics_control.wheels:
                wheel.damping_rate = 2.0      # 增加车轮阻尼
                wheel.max_brake_torque = 5000.0  # 增强车轮制动力矩
                wheel.max_handbrake_torque = 10000.0  # 增强手刹
                wheel.tire_friction = 5.0      # 增加轮胎摩擦力
            
            # 应用修改后的物理控制参数
            self.vehicle.apply_physics_control(physics_control)
            print("已修改车辆物理特性")
            
        except Exception as e:
            print(f"修改车辆物理特性失败: {e}")
            traceback.print_exc()

    def _calculate_steering(self, center_offset, confidence=1.0):
        """计算转向角度
        
        Args:
            center_offset: 车道中心偏移量 (-1.0到1.0)，负值表示车辆偏左，正值表示车辆偏右
            confidence: 车道线检测的置信度 (0.0到1.0)
            
        Returns:
            float: 转向角度 (-1.0到1.0)，负值表示向左转，正值表示向右转
        """
        try:
            # 缩放系数，控制转向灵敏度
            scale = 0.8  # 可调整，较大的值会使转向更加灵敏
            
            # 根据偏移量计算初始转向角度 - 偏左为负（应右转），偏右为正（应左转）
            # 反向转向以纠正偏移
            initial_steering = -center_offset * scale
            
            # 根据车道检测的置信度调整转向幅度
            # 低置信度时减小转向幅度，避免错误转向
            confidence_adjusted_steering = initial_steering * confidence
            
            # 平滑转向 - 使用移动平均
            if not hasattr(self, 'steer_history'):
                self.steer_history = []
            
            # 添加当前转向到历史记录
            self.steer_history.append(confidence_adjusted_steering)
            
            # 保持历史记录长度
            max_history = 5
            if len(self.steer_history) > max_history:
                self.steer_history = self.steer_history[-max_history:]
            
            # 计算平均转向
            smoothed_steering = sum(self.steer_history) / len(self.steer_history)
            
            # 限制转向范围
            final_steering = max(-self.max_steering, min(self.max_steering, smoothed_steering))
            
            # 记录最后的转向值，用于安全停车等场景
            self.last_steer = final_steering
            
            # 调试输出
            if self.debug_mode:
                self._print_debug_info(f"中心偏移: {center_offset:.3f}, 置信度: {confidence:.2f}, 初始转向: {initial_steering:.3f}, 最终转向: {final_steering:.3f}")
            
            return final_steering
            
        except Exception as e:
            print(f"计算转向角度异常: {e}")
            traceback.print_exc()
            
            # 错误情况下返回安全的小转向值或最后的已知转向值
            if hasattr(self, 'last_steer'):
                return self.last_steer * 0.5  # 返回最后转向的一半，逐渐减小转向角
            else:
                return 0.0  # 默认直行
    
    def run(self):
        """运行控制器"""
        if not self.initialized:
            print("系统未初始化，无法运行")
            return
        
        print("按ESC退出，1/2/3切换控制模式，V切换视角，D切换调试输出，T切换轨迹可视化")
        
        try:
            # 创建调试窗口
            cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Lane Detection", 640, 360)
            
            clock = pygame.time.Clock()
            
            # 主循环
            while True:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return
                        elif event.key == pygame.K_1:
                            self.control_mode = "MANUAL"
                            print("切换到手动控制模式")
                        elif event.key == pygame.K_2:
                            self.control_mode = "AUTO"
                            print("切换到自动控制模式")
                        elif event.key == pygame.K_3:
                            self.control_mode = "MODEL"
                            print("切换到模型控制模式")
                        elif event.key == pygame.K_4:
                            self.control_mode = "JUNCTION"
                            print("切换到交叉路口模式")
                        elif event.key == pygame.K_v:
                            # 切换视角
                            if self.view_mode == "FOLLOW":
                                self.view_mode = "BIRDVIEW"
                            elif self.view_mode == "BIRDVIEW":
                                self.view_mode = "FIXED"
                            else:
                                self.view_mode = "FOLLOW"
                            print(f"切换视角: {self.view_mode}")
                        elif event.key == pygame.K_d:
                            self.debug_mode = not self.debug_mode
                            print(f"调试模式: {'开启' if self.debug_mode else '关闭'}")
                        elif event.key == pygame.K_t:
                            self.show_trajectory = not self.show_trajectory
                            print(f"轨迹可视化: {'开启' if self.show_trajectory else '关闭'}")
                
                # 获取当前速度
                velocity = self.vehicle.get_velocity()
                speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5  # km/h
                self.current_speed = speed
                
                # 更新控制
                self.update_control(self.front_image)
                
                # 渲染界面
                self.render()
                
                # 更新观察者相机位置
                self._update_spectator()
                
                # 世界tick
                self.world.tick()
                
                # 更新调试窗口
                if hasattr(self, 'lane_detector') and self.lane_detector is not None and self.front_image is not None:
                    _, debug_img = self.lane_detector.detect_lanes(self.front_image)
                    if debug_img is not None and isinstance(debug_img, np.ndarray):
                        cv2.imshow("Lane Detection", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)
                
                # 控制帧率
                clock.tick(30)
                
            # 关闭调试窗口
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"运行时出错: {e}")
            traceback.print_exc()
            cv2.destroyAllWindows()
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        try:
            print("开始清理资源...")
            
            # 销毁传感器
            if hasattr(self, 'sensors') and self.sensors:
                for sensor in self.sensors:
                    if sensor is not None and sensor.is_alive:
                        sensor.destroy()
                print("传感器已销毁")
            
            # 销毁车辆
            if hasattr(self, 'vehicle') and self.vehicle is not None and self.vehicle.is_alive:
                self.vehicle.destroy()
                print("车辆已销毁")
            
            # 关闭pygame
            pygame.quit()
            print("pygame已关闭")
            
            print("资源清理完成")
        except Exception as e:
            print(f"清理资源时出错: {e}")
            traceback.print_exc()
    
    def render(self):
        """渲染界面"""
        try:
            # 清空屏幕
            self.screen.fill((0, 0, 0))
            
            # 显示前视图像
            if hasattr(self, 'front_image') and self.front_image is not None:
                # 将图像调整为屏幕大小，铺满整个屏幕
                resized_image = cv2.resize(self.front_image, (self.config.width, self.config.height))
                
                # 将OpenCV图像转换为pygame表面
                image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                image_surface = pygame.surfarray.make_surface(image_rgb.swapaxes(0, 1))
                self.screen.blit(image_surface, (0, 0))
                
                # 可视化车道线检测结果
                if hasattr(self.lane_detector, '_debug_image') and self.lane_detector._debug_image is not None:
                    # 调整大小为屏幕右上角的小窗口
                    debug_width = self.config.width // 4  # 屏幕宽度的1/4
                    debug_height = self.config.height // 4  # 屏幕高度的1/4
                    debug_image = cv2.resize(self.lane_detector._debug_image, (debug_width, debug_height))
                    
                    # 转为pygame表面
                    debug_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                    debug_surface = pygame.surfarray.make_surface(debug_rgb.swapaxes(0, 1))
                    
                    # 在右上角显示，留出一定边距
                    self.screen.blit(debug_surface, (self.config.width - debug_width - 10, 10))
            
            # 使用HUD显示状态信息
            if hasattr(self, 'current_status'):
                self.hud.render(self.screen, self.current_status)
            
            # 更新显示
            pygame.display.flip()
            
        except Exception as e:
            print(f"渲染界面时出错: {e}")
            traceback.print_exc()
    
    def update_control(self, image):
        """更新车辆控制
        
        Args:
            image: 前视摄像头图像
            
        Returns:
            carla.VehicleControl: 控制命令
        """
        try:
            # 保存前视图像
            self.front_image = image
            
            # 自动检测是否接近交叉路口 - 使用junction_detector进行检测
            is_near_junction, junction = self.junction_detector.detect_junction(self.vehicle)
            
            # 输出调试信息 - 确保检测到交叉路口时有明确的日志
            if is_near_junction:
                print(f"【检测到交叉路口】距离约: {self.junction_detector.debug_info['distance_to_junction']:.1f}米")
            
            # 更新交叉路口状态
            if is_near_junction and not self.junction_state['in_junction']:
                print("【进入交叉路口模式】准备停车、询问大模型")
                self.junction_state['in_junction'] = True
                self.junction_state['trajectory_completed'] = False
                
                # 如果接近交叉路口且之前不在交叉路口模式，切换到交叉路口模式
                if self.control_mode != "JUNCTION":
                    self.previous_mode = self.control_mode
                    self.control_mode = "JUNCTION"
                    # 重置交叉路口相关的状态
                    self.junction_state['stopped_at_junction'] = False
                    self.junction_state['waiting_for_model'] = False
                    self.junction_state['model_queried'] = False
                    self.junction_state['model_response'] = None
                    self.junction_state['safe_to_proceed'] = False
            
            # 如果离开交叉路口，重置状态
            if not is_near_junction and self.junction_state['in_junction'] and self.junction_state['safe_to_proceed']:
                if self.mpc_controller.is_trajectory_completed():
                    print("【离开交叉路口】恢复普通驾驶模式")
                    self.junction_state['in_junction'] = False
                    self.junction_state['trajectory_completed'] = True
                    self.junction_state['trajectory_generated'] = False
                    self.junction_state['stopped_at_junction'] = False
                    self.junction_state['model_queried'] = False
                    self.junction_state['waiting_for_model'] = False
                    self.junction_state['model_response'] = None
                    self.junction_state['safe_to_proceed'] = False
                    
                    # 恢复到之前的控制模式
                    self.control_mode = self.previous_mode
            
            # 根据控制模式进行控制
            if self.control_mode == "MANUAL":
                # 获取键盘输入
                keys = pygame.key.get_pressed()
                return self._manual_control(keys)
                
            elif self.control_mode == "AUTO":
                # 自动驾驶控制
                return self._auto_control(image)
                
            elif self.control_mode == "MODEL":
                # 基于模型的控制
                speed = self._get_speed()
                return self._model_control(image, speed)
                
            elif self.control_mode == "JUNCTION":
                # 交叉路口特殊控制 - 确保这个方法正确处理停车和询问模型
                return self._handle_junction_with_model(image, junction)
                
            else:
                # 默认控制
                return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                
        except Exception as e:
            print(f"更新控制异常: {e}")
            traceback.print_exc()
            return self._safe_stop(f"控制异常: {e}")
    
    def _manual_control(self, keys):
        """处理手动控制"""
        control = carla.VehicleControl()
        
        # 油门
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            control.throttle = 0.7
        else:
            control.throttle = 0.0
        
        # 刹车
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            control.brake = 1.0
        else:
            control.brake = 0.0
        
        # 转向
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            control.steer = -0.5
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            control.steer = 0.5
        else:
            control.steer = 0.0
        
        # 手刹
        control.hand_brake = keys[pygame.K_SPACE]
        
        return control

    def _auto_control(self, image=None):
        """自动控制模式
        
        Args:
            image: 前视图像
        
        Returns:
            carla.VehicleControl: 车辆控制指令
        """
        try:
            if image is None or self.vehicle is None:
                return carla.VehicleControl()
            
            # 获取当前速度
            current_speed = self._get_speed()
            
            # 检测是否接近交叉路口
            approaching_junction = self.junction_detector.is_approaching_junction(image)
            
            # 检查是否应该拍照
            if self.junction_detector.should_capture_photo():
                # 保存交叉路口图像
                photo_path = os.path.join(os.path.dirname(__file__), "photo", f"junction_{int(time.time())}.jpg")
                os.makedirs(os.path.dirname(photo_path), exist_ok=True)
                cv2.imwrite(photo_path, image)
                print(f"已保存交叉路口图像: {photo_path}")
                
                # 向大模型发送图像进行决策
                junction_status = {
                    'image_path': photo_path,
                    'question': 'is_junction'
                }
                
                # 异步获取大模型决策
                threading.Thread(target=self._get_junction_decision, args=(junction_status,)).start()
            
            # 如果接近交叉路口，切换到交叉路口控制模式
            if approaching_junction and not self.is_at_junction:
                print("接近交叉路口，准备进入交叉路口控制模式")
                self.is_at_junction = True
                self.control_mode = "JUNCTION"
                return self._junction_control()
            
            # 如果已经离开交叉路口，重置状态
            if not approaching_junction and self.is_at_junction:
                print("已离开交叉路口")
                self.is_at_junction = False
                self.junction_detector.reset_photo_counter()
            
            # 检测车道线
            lane_info, debug_image = self.lane_detector.detect_lanes(image)
            
            # 如果没有检测到车道线，使用默认控制
            if lane_info is None:
                print("未检测到车道线，使用默认控制")
                return carla.VehicleControl(throttle=0.5, steer=0.0)
            
            # 获取车道中心偏移和置信度
            center_offset = lane_info.get('center_offset', 0.0)
            confidence = lane_info.get('confidence', 0.0)
            
            # 打印调试信息
            if self.debug_mode:
                print(f"车道偏移: {center_offset:.3f}, 置信度: {confidence:.2f}")
            
            # 更新当前状态
            self.current_status.update({
                'speed': current_speed,
                'lane_center': center_offset,
                'lane_confidence': confidence,
                'lane_status': {
                    'approaching_junction': approaching_junction
                }
            })
            
            # 计算转向角
            steering = self._calculate_steering(center_offset, confidence)
            
            # 计算油门和刹车
            throttle, brake = self._calculate_throttle_brake(current_speed, self.target_speed)
            
            # 创建控制命令
            control = carla.VehicleControl()
            control.throttle = throttle
            control.steer = steering
            control.brake = brake
            control.hand_brake = False
            control.reverse = False
            
            return control
            
        except Exception as e:
            print(f"自动控制异常: {e}")
            traceback.print_exc()
            return carla.VehicleControl()

    def _get_junction_decision(self, status):
        """获取交叉路口决策
        
        Args:
            status: 包含图像路径和问题的字典
        """
        try:
            # 调用大模型获取决策
            response = self.model_client.ask_simple_question(status)
            
            if response and isinstance(response, dict):
                is_junction = response.get('is_junction', False)
                reason = response.get('reason', '未提供理由')
                
                print(f"大模型判断结果: {'是' if is_junction else '不是'}交叉路口")
                print(f"理由: {reason}")
                
                # 如果确认是交叉路口，获取方向决策
                if is_junction:
                    # 分析可能的方向
                    directions = self._analyze_junction_directions()
                    
                    # 构建决策请求
                    decision_status = {
                        'image_path': status['image_path'],
                        'directions': directions
                    }
                    
                    # 获取方向决策
                    direction_decision = self.model_client.get_junction_decision(decision_status)
                    
                    if direction_decision:
                        self.junction_decision = direction_decision
                        print(f"交叉路口决策: {direction_decision.get('direction', '直行')}")
                        print(f"理由: {direction_decision.get('reason', '未提供理由')}")
            
        except Exception as e:
            print(f"获取交叉路口决策异常: {e}")
            traceback.print_exc()

    def _analyze_junction_directions(self):
        """分析交叉路口可能的方向"""
        try:
            # 获取当前交叉路口
            junction = self.junction_detector.current_junction
            
            if junction is None:
                return [{'type': '直行', 'confidence': 1.0}]
            
            # 获取可能的方向
            directions = []
            
            # 默认添加直行
            directions.append({'type': '直行', 'confidence': 1.0})
            
            # 检查是否可以左转
            left_turn_possible = False
            
            # 检查是否可以右转
            right_turn_possible = False
            
            # 简单判断：如果是交叉路口，假设可以左转和右转
            if junction:
                left_turn_possible = True
                right_turn_possible = True
            
            # 添加可能的方向
            if left_turn_possible:
                directions.append({'type': '左转', 'confidence': 0.8})
            
            if right_turn_possible:
                directions.append({'type': '右转', 'confidence': 0.8})
            
            return directions
            
        except Exception as e:
            print(f"分析交叉路口方向异常: {e}")
            traceback.print_exc()
            return [{'type': '直行', 'confidence': 1.0}]

    def _junction_control(self):
        """交叉路口控制模式"""
        try:
            # 如果没有决策，等待决策
            if self.junction_decision is None:
                print("等待交叉路口决策...")
                # 减速等待
                control = carla.VehicleControl()
                control.throttle = 0.2
                control.brake = 0.3
                control.steer = 0.0
                return control
            
            # 获取决策方向
            direction = self.junction_decision.get('direction', '直行')
            
            # 根据方向设置控制参数
            if direction == '左转':
                control = carla.VehicleControl()
                control.throttle = 0.4
                control.steer = -0.5
                control.brake = 0.0
                print("执行左转")
                return control
            elif direction == '右转':
                control = carla.VehicleControl()
                control.throttle = 0.4
                control.steer = 0.5
                control.brake = 0.0
                print("执行右转")
                return control
            else:  # 直行
                control = carla.VehicleControl()
                control.throttle = 0.5
                control.steer = 0.0
                control.brake = 0.0
                print("执行直行")
                return control
                
        except Exception as e:
            print(f"交叉路口控制异常: {e}")
            traceback.print_exc()
            
            # 返回安全的默认控制
            control = carla.VehicleControl()
            control.throttle = 0.3
            control.steer = 0.0
            control.brake = 0.0
            return control

    def _model_control(self, image, speed):
        """基于大模型的控制"""
        try:
            # 检测是否接近交叉路口
            near_junction, junction = self.junction_detector.detect_junction(self.vehicle)
            
            if near_junction:
                print(f"检测到交叉路口，准备生成轨迹")
                
                # 保存前视图像用于决策
                front_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo", "junction_image.png")
                if image is not None and image.size > 0:
                    cv2.imwrite(front_image_path, image)
                    print(f"已保存交叉路口图像: {front_image_path}")
                else:
                    print("警告：图像为空，无法保存交叉路口图像")
                
                # 获取可能的方向
                waypoints = self.junction_detector.get_junction_waypoints(junction)
                directions = []
                
                if not waypoints:
                    print("警告：未找到交叉路口航点")
                    # 使用默认直行控制
                    return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
                
                current_transform = self.vehicle.get_transform()
                current_rotation = current_transform.rotation.yaw
                
                print(f"当前车辆朝向: {current_rotation}度")
                
                # 分析可能的方向
                for entry_wp, exit_wp in waypoints:
                    exit_yaw = exit_wp.transform.rotation.yaw
                    rel_yaw = (exit_yaw - current_rotation + 180) % 360 - 180
                    
                    direction = {}
                    if -45 <= rel_yaw <= 45:
                        direction["type"] = "直行"
                    elif 45 < rel_yaw <= 135:
                        direction["type"] = "右转"
                    elif -135 <= rel_yaw < -45:
                        direction["type"] = "左转"
                    else:
                        direction["type"] = "掉头"
                    
                    direction["waypoint"] = exit_wp
                    direction["rel_yaw"] = rel_yaw
                    directions.append(direction)
                    print(f"可能方向: {direction['type']}, 相对角度: {rel_yaw}度")
                
                # 过滤出唯一的方向类型
                unique_directions = []
                direction_types = set()
                for direction in directions:
                    if direction["type"] not in direction_types:
                        direction_types.add(direction["type"])
                        unique_directions.append(direction)
                
                print(f"唯一方向类型: {[d['type'] for d in unique_directions]}")
                
                # 如果有可选方向，询问模型
                if unique_directions:
                    print("请求大模型决策交叉路口方向...")
                    
                    # 创建一个硬编码的决策，用于测试
                    hardcoded_decision = {
                        "direction": "左转",
                        "reason": "测试左转功能"
                    }
                    
                    # 首先尝试从模型获取决策
                    decision = self.model_client.get_junction_decision(unique_directions)
                    print(f"大模型原始响应: {decision}")
                    
                    # 如果模型没有返回有效决策，使用硬编码决策
                    if not decision or "direction" not in decision:
                        print("模型未返回有效决策，使用硬编码决策")
                        decision = hardcoded_decision
                    
                    if decision and "direction" in decision:
                        chosen_direction = decision["direction"]
                        reason = decision.get('reason', '未提供')
                        print(f"模型决策: {chosen_direction}, 原因: {reason}")
                        
                        # 找到对应的方向对象
                        selected_direction = None
                        for direction in unique_directions:
                            # 标准化方向值进行比较
                            if ((chosen_direction == '左转' or chosen_direction.lower() == 'left') and direction["type"] == "左转") or \
                               ((chosen_direction == '右转' or chosen_direction.lower() == 'right') and direction["type"] == "右转") or \
                               ((chosen_direction == '直行' or chosen_direction.lower() == 'straight') and direction["type"] == "直行"):
                                selected_direction = direction
                                break
                        
                        # 如果找不到对应的方向，使用第一个方向
                        if not selected_direction and unique_directions:
                            selected_direction = unique_directions[0]
                            print(f"未找到匹配的方向 {chosen_direction}，使用默认方向: {selected_direction['type']}")
                        
                        if selected_direction:
                            print(f"选择方向: {selected_direction['type']}")
                            # 根据决策生成轨迹
                            trajectory = self.trajectory_planner.generate_trajectory(selected_direction["type"], self.vehicle)
                            
                            if not trajectory:
                                print(f"警告：无法为 {selected_direction['type']} 生成轨迹")
                                # 使用简单的控制命令
                                if selected_direction["type"] == "左转":
                                    return carla.VehicleControl(throttle=0.5, steer=-0.7, brake=0.0)
                                elif selected_direction["type"] == "右转":
                                    return carla.VehicleControl(throttle=0.5, steer=0.7, brake=0.0)
                                else:  # 直行
                                    return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
                            
                            # 保存轨迹用于后续跟踪
                            self.current_trajectory = trajectory
                            self.trajectory_index = 0
                            
                            # 根据轨迹生成控制命令
                            return self._follow_trajectory(speed)
                
                # 如果没有决策或决策失败，使用简单控制
                print("无法获取有效的交叉路口决策，使用默认控制")
                return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
            else:
                # 如果有轨迹正在执行，继续跟踪
                if self.current_trajectory and self.trajectory_index < len(self.current_trajectory):
                    return self._follow_trajectory(speed)
                
                # 不在交叉路口，使用车道线控制
                return self._auto_control()
                
        except Exception as e:
            print(f"模型控制出错: {e}")
            traceback.print_exc()
            # 返回安全的默认控制
            return carla.VehicleControl(throttle=0.3, steer=0.0, brake=0.0)

    def reset_junction_state(self):
        """重置交叉路口状态，防止重复处理"""
        try:
            # 删除临时属性
            if hasattr(self, 'junction_process_time'):
                delattr(self, 'junction_process_time')
            if hasattr(self, 'proceed_start_time'):
                delattr(self, 'proceed_start_time')
                
            # 重置交叉路口状态字典
            self.junction_state = {
                'in_junction': False,
                'trajectory_generated': False,
                'trajectory_completed': True,
                'direction': 'straight',
                'stopped_at_junction': False,
                'waiting_for_model': False,
                'model_queried': False,
                'model_response': None,
                'safe_to_proceed': False
            }
            
            # 打印重置消息
            print("【状态重置】交叉路口状态已重置")
            
            # 恢复控制模式
            if self.control_mode == "JUNCTION":
                self.control_mode = self.previous_mode if hasattr(self, 'previous_mode') else "AUTO"
                print(f"【控制模式】已从JUNCTION恢复为{self.control_mode}")
                
        except Exception as e:
            print(f"【错误】重置交叉路口状态失败: {e}")
            traceback.print_exc()

    def _print_debug_info(self, message):
        """带时间间隔的调试信息打印"""
        if self.debug_mode:
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                print(message)
                self.last_print_time = current_time
    
    def _get_speed(self):
        """获取当前速度（km/h）"""
        if self.vehicle is None:
            return 0.0
        
        velocity = self.vehicle.get_velocity()
        # 计算速度大小（三维向量的模）
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        # 转换为km/h
        return speed_ms * 3.6
    
    def _update_spectator(self):
        """更新观察者相机位置，跟随车辆"""
        if not self.vehicle:
            return
        
        try:
            # 获取车辆位置和方向
            vehicle_transform = self.vehicle.get_transform()
            
            # 创建跟随相机变换
            if self.view_mode == "FOLLOW":
                # 车辆后上方，跟随视角
                camera_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(
                        x=-6.0 * math.cos(math.radians(vehicle_transform.rotation.yaw)),
                        y=-6.0 * math.sin(math.radians(vehicle_transform.rotation.yaw)),
                        z=3.0),
                    carla.Rotation(
                        pitch=-15,
                        yaw=vehicle_transform.rotation.yaw,
                        roll=vehicle_transform.rotation.roll)
                )
            elif self.view_mode == "BIRDVIEW":
                # 鸟瞰视角
                camera_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(z=50.0),
                    carla.Rotation(pitch=-90.0, yaw=vehicle_transform.rotation.yaw)
                )
            else:  # FIXED
                # 固定视角，不跟随
                return
            
            # 更新观察者相机位置
            self.spectator.set_transform(camera_transform)
            
        except Exception as e:
            print(f"更新观察者相机位置失败: {e}")
            traceback.print_exc()
    
    def _safe_stop(self, reason="未知原因"):
        """安全停车逻辑
        
        在检测失败或异常情况下，执行安全停车
        """
        print(f"执行安全停车程序: {reason}")
        
        # 获取当前速度
        current_speed = self._get_speed()
        
        # 根据当前速度决定刹车力度
        if current_speed > 20:  # 高速
            brake = 0.8  # 强刹车
        elif current_speed > 10:  # 中速
            brake = 0.5  # 中等刹车
        else:  # 低速
            brake = 0.3  # 轻刹车
        
        # 保持方向盘直行或轻微修正
        steer = 0.0
        if hasattr(self, 'last_steer'):
            # 逐渐减小转向，但保持一定的方向修正
            steer = self.last_steer * 0.5
        
        # 创建控制命令
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = 0.0
        control.brake = float(brake)
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        
        # 更新状态
        self.current_status['steer'] = steer
        self.current_status['throttle'] = 0.0
        self.current_status['brake'] = brake
        self.current_status['safe_stop_active'] = True
        self.current_status['safe_stop_reason'] = reason
        
        return control

    def _calculate_throttle_brake(self, current_speed, target_speed):
        """计算油门和刹车值"""
        # 速度误差
        speed_error = target_speed - current_speed
        
        # 调试输出
        print(f"目标速度: {target_speed:.1f} km/h, 当前速度: {current_speed:.1f} km/h, 速度误差: {speed_error:.1f} km/h")
        
        # 使用PID控制器计算油门/刹车
        if speed_error > 0:
            # 需要加速
            throttle_value = self.pid_controllers['throttle'].step(speed_error)
            # 确保throttle_value是单个浮点数
            if isinstance(throttle_value, tuple):
                throttle = throttle_value[0]  # 假设第一个元素是控制值
            else:
                throttle = throttle_value
            brake = 0.0
            
            # 确保油门值足够大，以克服静摩擦
            if current_speed < 5.0:  # 低速或静止状态
                throttle = 1.0  # 提供最大油门以克服静摩擦
                print("低速状态，使用最大油门值以克服静摩擦")
                
                # 尝试直接设置车辆速度以克服静摩擦
                try:
                    forward_vector = self.vehicle.get_transform().get_forward_vector()
                    self.vehicle.set_target_velocity(carla.Vector3D(
                        forward_vector.x * 5.0,
                        forward_vector.y * 5.0,
                        forward_vector.z * 5.0
                    ))
                    print("已设置初始速度以克服静摩擦")
                except Exception as e:
                    print(f"设置初始速度时出错: {e}")
        else:
            # 需要减速
            throttle = 0.0
            # 根据速度误差的绝对值计算刹车力度
            brake_value = self.pid_controllers['throttle'].step(-speed_error)
            # 确保brake_value是单个浮点数
            if isinstance(brake_value, tuple):
                brake = brake_value[0]  # 假设第一个元素是控制值
            else:
                brake = brake_value
        
        # 限制油门和刹车的范围
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        
        print(f"计算得到的控制值 - 油门: {throttle:.2f}, 刹车: {brake:.2f}")
        
        return throttle, brake
        
    def _follow_trajectory(self, target_speed):
        """根据MPC轨迹生成控制指令"""
        try:
            # 获取当前速度
            current_speed = self._get_speed()
            
            # 目标速度最低为5km/h确保车辆能动起来
            if target_speed < 5.0:
                target_speed = 5.0
                
            print(f"【跟踪轨迹】目标速度: {target_speed:.1f} km/h, 当前速度: {current_speed:.1f} km/h")
            
            # 如果车辆静止，给予强力初始推动
            if current_speed < 0.5:
                print("【启动推动】车辆静止，提供强力初始推动")
                return carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False)
            
            # 计算速度误差
            speed_error = target_speed - current_speed
            
            # 计算油门和刹车
            if speed_error > 0:
                # 需要加速
                throttle = min(0.8, speed_error * 0.1)  # 根据速度误差调整油门
                brake = 0.0
            else:
                # 需要减速
                throttle = 0.0
                brake = min(0.5, -speed_error * 0.05)  # 根据速度误差调整刹车
            
            # 获取转向角 - 从MPC控制器获取
            steer = self.mpc_controller.get_steering()
            
            # 确保转向角在合理范围内
            steer = max(-0.8, min(0.8, steer))
            
            # 限制油门和刹车
            throttle = max(0.0, min(1.0, throttle))
            brake = max(0.0, min(1.0, brake))
            
            print(f"【控制输出】油门: {throttle:.2f}, 刹车: {brake:.2f}, 方向: {steer:.2f}")
            
            # 返回控制命令
            return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=False)
            
        except Exception as e:
            print(f"【错误】跟踪轨迹时出错: {e}")
            traceback.print_exc()
            # 出错时提供默认控制 - 确保继续前进
            return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
    
    def _handle_junction_with_model(self, image, junction):
        """处理交叉路口，使用大模型决策
        
        Args:
            image: 前视摄像头图像
            junction: 交叉路口对象
            
        Returns:
            carla.VehicleControl: 控制命令
        """
        try:
            # 获取当前车速
            current_speed = self._get_speed()
            
            # 增加更详细的日志和调试信息
            print(f"【交叉路口控制】当前速度: {current_speed:.1f} km/h, 停车状态: {self.junction_state['stopped_at_junction']}, 等待模型: {self.junction_state['waiting_for_model']}, 已查询模型: {self.junction_state['model_queried']}")
            
            # 检查是否为真正的交叉路口（有多个出口选择）
            if junction is None:
                print("不是真正的交叉路口，继续正常行驶")
                return self._auto_control(image)
            
            # 用于跟踪进度的状态变量
            if not hasattr(self, 'junction_process_time'):
                self.junction_process_time = time.time()
                self.junction_timeout = 20.0  # 20秒超时
            
            # 如果在某个状态停留太久，强制进入下一个状态
            current_time = time.time()
            time_in_state = current_time - self.junction_process_time
            
            # 打印当前状态停留时间
            print(f"【状态时间】当前状态已持续: {time_in_state:.1f}秒")
            
            # 1. 如果车辆还未停止，则强制减速停车
            if not self.junction_state['stopped_at_junction']:
                # 如果速度已经接近0，认为已停止或超时5秒强制进入下一阶段
                if current_speed < 0.5 or time_in_state > 5.0:  # 减少阈值到0.5km/h视为停止或超时5秒强制进入下一阶段
                    print("【已停车】车辆已在交叉路口停止，准备查询大模型")
                    self.junction_state['stopped_at_junction'] = True
                    self.junction_process_time = current_time  # 重置计时器
                    
                    # 强制拍照，确保有图像可用
                    front_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo", f"junction_image_{int(time.time())}.png")
                    if image is not None and image.size > 0:
                        cv2.imwrite(front_image_path, image)
                        print(f"【已保存图像】交叉路口图像: {front_image_path}")
                    else:
                        print("警告：图像为空，无法保存交叉路口图像")
                        # 创建一个测试图像
                        test_img = np.ones((720, 1280, 3), dtype=np.uint8) * 255  # 白色图像
                        cv2.putText(test_img, "NO CAMERA IMAGE", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                        test_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo", "test_image.png")
                        cv2.imwrite(test_img_path, test_img)
                        front_image_path = test_img_path
                        print(f"【已创建测试图像】: {test_img_path}")
                    
                    # 立即设置为等待模型查询状态
                    self.junction_state['waiting_for_model'] = True
                    self.junction_state['model_queried'] = False
                    
                    # 返回停车控制（保持刹车）
                    return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
                else:
                    # 强制减速更激进 - 增加刹车强度并启用手刹
                    print(f"【减速中】强制减速至停车，当前速度: {current_speed:.1f} km/h")
                    
                    # 使用急刹车 - 刹车强度1.0并启用手刹
                    control = carla.VehicleControl()
                    control.throttle = 0.0
                    control.steer = 0.0
                    control.brake = 1.0
                    control.hand_brake = True  # 启用手刹
                    
                    # 如果速度仍然太高，尝试使用物理方法强制减速
                    if current_speed > 5.0 and hasattr(self.vehicle, 'set_target_velocity'):
                        try:
                            # 尝试直接设置速度为零
                            self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                            print("【强制停车】直接设置车速为零")
                        except Exception as e:
                            print(f"【警告】无法设置目标速度: {e}")
                    
                    return control
            
            # 2. 如果已停止，且需要查询模型
            if self.junction_state['stopped_at_junction'] and self.junction_state['waiting_for_model'] and not self.junction_state['model_queried']:
                # 如果已经等待太久，强制进入查询阶段
                if time_in_state > 10.0 and not self.junction_state['model_queried']:
                    print("【强制查询】等待时间过长，强制进入模型查询阶段")
                
                print("【查询大模型】正在查询Ollama模型决策...")
                
                # 更新进程时间
                self.junction_process_time = current_time
                
                # 获取可能的方向
                possible_directions = []
                if junction:
                    waypoints = self.junction_detector.get_junction_waypoints(junction)
                    exit_count = self.junction_detector._count_junction_exits(junction)
                    print(f"【交叉路口信息】有 {exit_count} 个可能的出口")
                    possible_directions = self._analyze_junction_directions()
                    print(f"【可能方向】: {[d['type'] for d in possible_directions]}")
                else:
                    possible_directions = [{'type': '直行', 'confidence': 1.0}]
                
                # 构建查询状态
                query_status = {
                    'image_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo", "junction_image.png"),
                    'directions': possible_directions,
                    'exit_count': exit_count if 'exit_count' in locals() else 1
                }
                
                # 尝试使用最新保存的图像
                photo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo")
                try:
                    photo_files = [f for f in os.listdir(photo_dir) if f.endswith('.png') or f.endswith('.jpg')]
                    if photo_files:
                        newest_photo = max([os.path.join(photo_dir, f) for f in photo_files], key=os.path.getmtime)
                        query_status['image_path'] = newest_photo
                        print(f"【使用最新图像】: {newest_photo}")
                except Exception as e:
                    print(f"【警告】查找最新图像失败: {e}")
                
                # 确保查询状态中包含图像路径
                if not os.path.exists(query_status['image_path']):
                    print(f"【警告】找不到交叉路口图像: {query_status['image_path']}")
                    # 尝试使用当前图像保存一份
                    if image is not None and image.size > 0:
                        new_image_path = os.path.join(photo_dir, f"emergency_image_{int(time.time())}.png")
                        cv2.imwrite(new_image_path, image)
                        query_status['image_path'] = new_image_path
                        print(f"【已重新保存图像】: {new_image_path}")
                
                # 调用模型客户端获取决策
                model_response = self.model_client.get_junction_decision(query_status)
                print(f"【大模型响应】: {model_response}")
                
                # 更新状态
                self.junction_state['model_queried'] = True
                self.junction_state['waiting_for_model'] = False
                self.junction_state['model_response'] = model_response
                
                # 重置进程时间，立即进入下一阶段 - 处理模型响应
                self.junction_process_time = current_time
                
                # 仍然保持停车状态，等待下一步
                return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
            
            # 3. 如果已查询模型，处理模型响应
            if self.junction_state['model_queried'] and not self.junction_state['safe_to_proceed']:
                print("【生成轨迹】处理模型响应，准备生成轨迹...")
                
                # 更新进程时间
                self.junction_process_time = current_time
                
                model_response = self.junction_state['model_response']
                
                # 如果没有有效的模型响应，创建一个默认的
                if not model_response:
                    print("【警告】没有有效的模型响应，使用默认直行")
                    model_response = {
                        'direction': '直行',
                        'reason': '没有获取到有效的模型响应，默认直行',
                        'confidence': 0.5
                    }
                    self.junction_state['model_response'] = model_response
                
                if 'direction' in model_response:
                    direction = model_response['direction']
                    
                    # 将方向转换为MPC控制器可理解的格式，并确保方向正确
                    mpc_direction = 'straight'
                    print(f"【原始方向】模型返回的原始方向: '{direction}'")
                    
                    # 增强方向识别，支持多种表达方式
                    direction_lower = direction.lower() if isinstance(direction, str) else ""
                    
                    # 检查方向关键词
                    if any(keyword in direction_lower for keyword in ['左', 'left', '左转']):
                        mpc_direction = 'left'
                        print("【方向选择】模型决策为左转，设置MPC方向为left")
                    elif any(keyword in direction_lower for keyword in ['右', 'right', '右转']):
                        mpc_direction = 'right'
                        print("【方向选择】模型决策为右转，设置MPC方向为right")
                    elif any(keyword in direction_lower for keyword in ['直', 'straight', '直行', 'forward']):
                        mpc_direction = 'straight'
                        print("【方向选择】模型决策为直行，设置MPC方向为straight")
                    else:
                        print(f"【方向警告】无法识别的方向 '{direction}'，默认使用直行")
                    
                    print(f"【生成轨迹】根据大模型决策，选择方向: {direction}，MPC将生成{mpc_direction}轨迹")
                    
                    # 保存决策方向，用于后续匹配验证
                    self.junction_state['direction'] = mpc_direction
                    
                    # 生成MPC轨迹，使用强制方向参数确保生成正确的轨迹
                    try:
                        print(f"【开始生成】调用MPC生成{mpc_direction}方向的轨迹...")
                        # 对左转使用明确的左转方法，右转使用明确的右转方法
                        if mpc_direction == 'left':
                            print("【强制左转】使用明确的左转轨迹生成方法")
                            trajectory = self.mpc_controller.generate_explicit_left_turn(self.vehicle)
                        elif mpc_direction == 'right':
                            print("【强制右转】使用明确的右转轨迹生成方法")
                            trajectory = self.mpc_controller.generate_explicit_right_turn(self.vehicle)
                        else:
                            trajectory = self.mpc_controller.generate_junction_trajectory(
                                self.vehicle, mpc_direction, junction_center=None
                            )
                        
                        # 验证轨迹方向是否正确
                        if trajectory and len(trajectory) > 10:
                            # 获取轨迹的起点和中点
                            start_point = trajectory[0]
                            mid_point = trajectory[10]
                            
                            # 计算方向变化向量
                            direction_vector = (mid_point[0] - start_point[0], mid_point[1] - start_point[1])
                            
                            # 获取车辆初始朝向
                            vehicle_transform = self.vehicle.get_transform()
                            forward_vector = vehicle_transform.get_forward_vector()
                            right_vector = vehicle_transform.get_right_vector()
                            
                            # 计算前向向量和轨迹方向向量的点积和叉积
                            dot_product = forward_vector.x * direction_vector[0] + forward_vector.y * direction_vector[1]
                            cross_product = forward_vector.x * direction_vector[1] - forward_vector.y * direction_vector[0]
                            
                            # 判断实际轨迹方向（相对于车辆）
                            actual_direction = ""
                            if abs(cross_product) < 0.5 * abs(dot_product):  # 大致平行于前向向量
                                actual_direction = "直行"
                            elif cross_product < 0:  # 叉积为负，在左侧
                                actual_direction = "左转"
                            else:  # 叉积为正，在右侧
                                actual_direction = "右转"
                            
                            # 检查是否与期望方向匹配
                            expected_direction = {
                                'left': '左转',
                                'right': '右转', 
                                'straight': '直行'
                            }.get(mpc_direction, "未知")
                            
                            print(f"【轨迹验证】期望方向: {expected_direction}, 实际方向: {actual_direction}")
                            
                            # 如果方向不匹配，重新生成轨迹
                            if expected_direction != actual_direction:
                                print(f"【警告】轨迹方向不匹配! 期望{expected_direction}但得到{actual_direction}")
                                
                                if expected_direction == '左转':
                                    print("【重新生成】强制生成左转轨迹")
                                    # 尝试使用更强的强制参数重新生成左转轨迹
                                    trajectory = self.mpc_controller.generate_explicit_left_turn(self.vehicle)
                                elif expected_direction == '右转':
                                    print("【重新生成】强制生成右转轨迹")
                                    # 尝试使用更强的强制参数重新生成右转轨迹
                                    trajectory = self.mpc_controller.generate_explicit_right_turn(self.vehicle)
                                
                                print(f"【重新生成完成】生成了 {len(trajectory) if trajectory else 0} 个轨迹点")
                        
                        if trajectory and len(trajectory) > 0:
                            print(f"【成功生成轨迹】包含 {len(trajectory)} 个点")
                            self.junction_state['trajectory_generated'] = True
                            self.junction_state['safe_to_proceed'] = True
                            # 可视化轨迹
                            if self.show_trajectory:
                                try:
                                    self.mpc_controller.visualize_trajectory_in_carla(self.world, self.vehicle, lifetime=5.0)
                                except Exception as traj_vis_error:
                                    print(f"【警告】轨迹可视化失败: {traj_vis_error}")
                        else:
                            print("【警告】生成轨迹失败，使用默认直行控制")
                            # 默认情况下直行
                            self.junction_state['safe_to_proceed'] = True
                    except Exception as traj_error:
                        print(f"【错误】生成轨迹异常: {traj_error}")
                        traceback.print_exc()
                        # 出错时也设置为可以通过
                        self.junction_state['safe_to_proceed'] = True
                else:
                    print("【警告】模型响应中没有direction字段，使用默认直行")
                    self.junction_state['safe_to_proceed'] = True
                
                # 重置进程时间，立即进入下一阶段 - 执行轨迹
                self.junction_process_time = current_time
                
                # 仍然保持停车状态，等待下一步
                return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
            
            # 4. 如果已生成轨迹且安全通过，使用MPC跟踪轨迹
            if self.junction_state['safe_to_proceed']:
                print("【准备通过】安全状态已确认，开始执行轨迹通过交叉路口")
                
                # 延迟1秒再开始移动，给系统一些缓冲时间
                if not hasattr(self, 'proceed_start_time'):
                    self.proceed_start_time = time.time()
                    print("【启动倒计时】准备在1秒后启动...")
                    # 确保手刹已释放
                    return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False)
                
                # 确保停车至少1秒后再出发
                if time.time() - self.proceed_start_time < 1.0:
                    print(f"【倒计时】还有 {1.0 - (time.time() - self.proceed_start_time):.1f} 秒启动...")
                    # 确保手刹已释放
                    return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False)
                
                # 强制给车一个初始推动
                if not hasattr(self, 'initial_push'):
                    self.initial_push = True
                    print("【启动】给予车辆初始推动力...")
                    return carla.VehicleControl(throttle=1.0, steer=0.0, brake=0.0, hand_brake=False)
                
                if not self.mpc_controller.is_trajectory_completed():
                    # 获取下一个轨迹点
                    try:
                        next_point = self.mpc_controller.get_next_trajectory_point()
                        
                        if next_point:
                            # 使用MPC控制器计算控制量
                            # 获取下一个轨迹点的速度
                            target_speed = next_point[3] * 3.6  # m/s转km/h
                            # 确保目标速度至少有5km/h
                            target_speed = max(5.0, target_speed)
                            
                            # 解决MPC问题获取最优控制
                            control = self._follow_trajectory(target_speed)
                            print(f"【沿轨迹行驶】目标速度: {target_speed:.1f} km/h")
                            return control
                        else:
                            print("【轨迹完成】恢复普通驾驶")
                            # 轨迹完成
                            self.mpc_controller.reset_trajectory()
                            self.junction_state['trajectory_completed'] = True
                            # 重置所有决策状态，防止重复处理
                            self.reset_junction_state()
                            return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
                    except Exception as follow_error:
                        print(f"【错误】跟踪轨迹异常: {follow_error}")
                        # 出错时也完成轨迹
                        self.mpc_controller.reset_trajectory()
                        self.junction_state['trajectory_completed'] = True
                        # 重置所有决策状态，防止重复处理
                        self.reset_junction_state()
                        return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
                else:
                    print("【轨迹已完成】恢复普通驾驶")
                    # 轨迹完成
                    self.junction_state['trajectory_completed'] = True
                    # 重置所有决策状态，防止重复处理
                    self.reset_junction_state()
                    return carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
            
            # 默认控制 - 保持停车状态
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True)
            
        except Exception as e:
            print(f"【严重错误】处理交叉路口异常: {e}")
            traceback.print_exc()
            # 出现异常时，重置所有状态，恢复正常驾驶
            self.junction_state['in_junction'] = False
            self.junction_state['stopped_at_junction'] = False
            self.junction_state['waiting_for_model'] = False
            self.junction_state['model_queried'] = False
            self.junction_state['safe_to_proceed'] = False
            self.junction_state['trajectory_completed'] = True
            self.control_mode = "AUTO"  # 恢复自动模式
            return self._safe_stop(f"处理交叉路口异常: {e}")