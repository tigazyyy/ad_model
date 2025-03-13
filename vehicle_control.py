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
            'direction': 'straight'
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
        """修改车辆物理属性，提高转向响应和移动性能"""
        try:
            physics_control = self.vehicle.get_physics_control()
            
            # 增加转向角度
            physics_control.max_steer_angle = 70.0  # 最大转向角度 (度)
            
            # 调整车轮参数 - 大幅减小摩擦力
            for wheel in physics_control.wheels:
                wheel.tire_friction = 5.0  # 增加轮胎摩擦力，提高牵引力（原为3.0）
                wheel.damping_rate = 0.1  # 进一步减小阻尼率，使转向更灵敏（原为0.25）
                wheel.max_steer_angle = 70.0  # 最大转向角度 (度)
                wheel.radius = wheel.radius * 1.2  # 进一步增大轮胎半径（原为1.1）
            
            # 调整质量和空气动力学
            physics_control.mass = 500  # 进一步减轻车辆质量 (kg)，原为800
            physics_control.drag_coefficient = 0.05  # 进一步减小空气阻力系数，原为0.1
            
            # 调整发动机参数 - 大幅增加低速扭矩
            physics_control.torque_curve = [
                carla.Vector2D(0, 1000),     # 大幅增加起步扭矩（原为600）
                carla.Vector2D(1000, 1000),  # 大幅增加低转速扭矩（原为700）
                carla.Vector2D(2000, 900),
                carla.Vector2D(3000, 800),
                carla.Vector2D(4000, 750),
                carla.Vector2D(5000, 700)
            ]
            physics_control.max_rpm = 6000.0  # 增加最大转速（原为5000）
            physics_control.moi = 0.3  # 进一步减小转动惯量（原为0.5）
            physics_control.damping_rate_full_throttle = 0.01  # 进一步减小全油门阻尼（原为0.05）
            physics_control.damping_rate_zero_throttle_clutch_engaged = 0.05  # 原为0.1
            physics_control.damping_rate_zero_throttle_clutch_disengaged = 0.05  # 原为0.15
            
            # 调整传动系统
            physics_control.use_gear_autobox = True
            physics_control.gear_switch_time = 0.05  # 减小换挡时间（原为0.1）
            physics_control.clutch_strength = 20.0  # 增加离合器强度（原为15.0）
            
            # 调整差速器 - 移除对VehicleDifferentialType的引用
            # physics_control.differential_type = carla.VehicleDifferentialType.LimitedSlip_4W
            physics_control.differential_ratio = 2.5  # 调整差速比，提高低速扭矩（原为3.5）
            
            # 应用修改
            self.vehicle.apply_physics_control(physics_control)
            
            # 修改世界的物理设置 - 减小重力
            try:
                settings = self.world.get_settings()
                if hasattr(settings, 'fixed_delta_seconds'):
                    # 确保使用固定时间步长
                    settings.fixed_delta_seconds = 0.05
                
                # 应用设置
                self.world.apply_settings(settings)
                
                # 尝试直接设置车辆速度以克服静摩擦
                self.vehicle.set_target_velocity(carla.Vector3D(5.0, 0.0, 0.0))  # 设置一个初始速度
                
                print("已修改车辆物理属性和世界设置，提高移动性能")
            except Exception as e:
                print(f"修改世界设置时出错: {e}")
            
        except Exception as e:
            print(f"修改车辆物理属性时出错: {e}")
            traceback.print_exc()
    
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
        """更新车辆控制"""
        try:
            # 保存前视图像
            self.front_image = image
            
            # 获取键盘输入
            keys = pygame.key.get_pressed()
            
            # 检查控制模式切换
            if keys[pygame.K_1]:
                self.control_mode = "MANUAL"
                print("切换到手动控制模式")
            elif keys[pygame.K_2]:
                self.control_mode = "AUTO"
                print("切换到自动控制模式")
            elif keys[pygame.K_3]:
                self.control_mode = "MODEL"
                print("切换到模型控制模式")
            elif keys[pygame.K_4]:
                self.control_mode = "JUNCTION"
                print("切换到交叉路口模式")
            
            # 检查视角模式切换
            if keys[pygame.K_F1]:
                self.view_mode = "FOLLOW"
                print("切换到跟随视角")
            elif keys[pygame.K_F2]:
                self.view_mode = "BIRDVIEW"
                print("切换到鸟瞰视角")
            elif keys[pygame.K_F3]:
                self.view_mode = "FIXED"
                print("切换到固定视角")
            
            # 检查轨迹可视化开关
            if keys[pygame.K_t]:
                self.show_trajectory = not self.show_trajectory
                print(f"轨迹可视化: {'开启' if self.show_trajectory else '关闭'}")
            
            # 根据控制模式更新控制
            if self.control_mode == "MANUAL":
                control = self._manual_control(keys)
            elif self.control_mode == "AUTO":
                control = self._auto_control(image)
            elif self.control_mode == "MODEL":
                # 获取当前速度
                speed = self._get_speed()
                control = self._model_control(image, speed)
            elif self.control_mode == "JUNCTION":
                control = self._junction_control()
            else:
                control = self._auto_control()
            
            # 更新视角
            self._update_spectator()
            
            return control
            
        except Exception as e:
            print(f"更新控制时出错: {e}")
            traceback.print_exc()
            return carla.VehicleControl()
    
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
            lane_info = self.lane_detector.detect_lane(image)
            
            # 如果没有检测到车道线，使用默认控制
            if lane_info is None:
                return carla.VehicleControl(throttle=0.5, steer=0.0)
            
            # 获取车道中心偏移
            center_offset = lane_info.get('center_offset', 0.0)
            confidence = lane_info.get('confidence', 0.0)
            
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
            steering = self._calculate_steering(center_offset)
            
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
    
    def _follow_trajectory(self, speed):
        """跟踪轨迹的PID控制器"""
        try:
            if not self.current_trajectory or self.trajectory_index >= len(self.current_trajectory):
                # 轨迹已完成，重置
                self.current_trajectory = []
                self.trajectory_index = 0
                return carla.VehicleControl(throttle=0.3, steer=0.0)
            
            # 获取当前目标点
            target_point = self.current_trajectory[self.trajectory_index]
            
            # 获取车辆当前位置和朝向
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # 计算目标点相对于车辆的位置
            # 如果目标点是相对坐标
            if isinstance(target_point, tuple) and len(target_point) == 2:
                # 将相对坐标转换为世界坐标
                forward_vector = vehicle_transform.get_forward_vector()
                right_vector = vehicle_transform.get_right_vector()
                
                target_world_location = carla.Location(
                    x=vehicle_location.x + forward_vector.x * target_point[1] + right_vector.x * target_point[0],
                    y=vehicle_location.y + forward_vector.y * target_point[1] + right_vector.y * target_point[0],
                    z=vehicle_location.z
                )
            else:
                # 目标点已经是世界坐标
                target_world_location = target_point
            
            # 计算目标点相对于车辆的方向向量
            direction_vector = carla.Vector3D(
                x=target_world_location.x - vehicle_location.x,
                y=target_world_location.y - vehicle_location.y,
                z=0
            )
            
            # 计算目标点与车辆的距离
            distance = math.sqrt(direction_vector.x**2 + direction_vector.y**2)
            
            # 如果距离小于阈值，移动到下一个目标点
            if distance < 2.0:
                self.trajectory_index += 1
                if self.trajectory_index < len(self.current_trajectory):
                    return self._follow_trajectory(speed)
                else:
                    # 轨迹完成，重置
                    self.current_trajectory = []
                    self.trajectory_index = 0
                    return carla.VehicleControl(throttle=0.3, steer=0.0)
            
            # 计算目标点相对于车辆前向的角度
            forward_vector = vehicle_transform.get_forward_vector()
            forward_vector_normalized = carla.Vector3D(
                x=forward_vector.x / math.sqrt(forward_vector.x**2 + forward_vector.y**2),
                y=forward_vector.y / math.sqrt(forward_vector.x**2 + forward_vector.y**2),
                z=0
            )
            
            direction_vector_normalized = carla.Vector3D(
                x=direction_vector.x / distance,
                y=direction_vector.y / distance,
                z=0
            )
            
            # 计算点积和叉积
            dot_product = forward_vector_normalized.x * direction_vector_normalized.x + \
                          forward_vector_normalized.y * direction_vector_normalized.y
            cross_product = forward_vector_normalized.x * direction_vector_normalized.y - \
                           forward_vector_normalized.y * direction_vector_normalized.x
            
            # 计算角度
            angle = math.acos(max(-1.0, min(1.0, dot_product)))
            if cross_product < 0:
                angle = -angle
            
            # 使用PID控制器计算转向
            steering = self.pid_controllers['steer'].step(angle)
            
            # 限制转向范围
            steering = max(-0.8, min(0.8, steering))
            
            # 根据角度和距离调整速度
            if abs(angle) > 0.5:  # 大约30度
                throttle = 0.2  # 转弯时减速
            else:
                throttle = 0.4
            
            # 创建控制命令
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steering,
                brake=0.0
            )
            
            return control
            
        except Exception as e:
            print(f"跟踪轨迹时出错: {e}")
            return carla.VehicleControl(throttle=0.0, brake=0.5)

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

    def _calculate_curve_radius(self, center_curve):
        """计算曲线半径
        
        Args:
            center_curve: 中心曲线点列表 [(x, y), ...]
            
        Returns:
            曲线半径 (米)
        """
        try:
            # 如果点数不足，返回一个大半径（近似直线）
            if len(center_curve) < 3:
                return 1000.0
            
            # 选择三个点来计算曲率
            p1 = center_curve[0]
            p2 = center_curve[len(center_curve) // 2]
            p3 = center_curve[-1]
            
            # 计算三点之间的距离
            d1 = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            d2 = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
            d3 = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
            
            # 使用海伦公式计算三角形面积
            s = (d1 + d2 + d3) / 2
            area = np.sqrt(s * (s - d1) * (s - d2) * (s - d3))
            
            # 计算曲率半径
            radius = (d1 * d2 * d3) / (4 * area) if area > 0 else 1000.0
            
            # 限制半径范围
            radius = max(1.0, min(1000.0, radius))
            
            return radius
            
        except Exception as e:
            print(f"计算曲线半径异常: {e}")
            # 出现异常时，返回一个大半径
            return 1000.0

    def _calculate_steering(self, lane_center):
        """根据车道中心计算转向角"""
        try:
            # 车道中心是归一化的偏移量，范围在[-1, 1]之间
            # 负值表示车辆在车道右侧，需要向左转
            # 正值表示车辆在车道左侧，需要向右转
            
            # 增加转向灵敏度，原来是0.7
            steering = -lane_center * 1.0  
            
            # 应用非线性响应，使小偏移时转向更灵敏，大偏移时转向更平稳
            if abs(lane_center) < 0.3:
                # 小偏移时增加灵敏度
                steering = -lane_center * 1.2
            else:
                # 大偏移时保持平稳
                steering = -lane_center * 0.9
            
            # 应用平滑处理
            if hasattr(self, 'steer_filter'):
                # 添加到滤波器
                self.steer_filter.append(steering)
                if len(self.steer_filter) > self.steer_filter_size:
                    self.steer_filter.pop(0)
                
                # 计算平均值
                steering = sum(self.steer_filter) / len(self.steer_filter)
            
            # 保存当前转向值
            self.last_steer = steering
            
            # 限制最大转向角度
            steering = max(-0.7, min(0.7, steering))
            
            return steering
            
        except Exception as e:
            print(f"计算转向角异常: {e}")
            traceback.print_exc()
            return 0.0

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

