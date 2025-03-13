class Config:
    """配置类"""
    def __init__(self):
        # 显示配置
        self.width = 1280
        self.height = 720
        
        # 车辆配置
        self.vehicle_type = "vehicle.tesla.model3"
        self.spawn_point_id = 0
        
        # 传感器配置
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_fov = 110
        
        # 控制配置
        self.default_speed = 10  # km/h
        self.pid_params = {
            "throttle": {"Kp": 1.0, "Ki": 0.1, "Kd": 0.0},
            "steering": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.05}
        }
        
        # 车道线检测配置
        self.lane_detection = {
            "canny_low": 50,
            "canny_high": 150,
            "roi_top_ratio": 0.6,  # ROI区域顶部位置比例
        }
        
        # 交叉路口检测配置
        self.junction_detection = {
            "distance_threshold": 20.0,  # 检测交叉路口的距离阈值
        }
        
        # 大模型配置
        self.model = {
            "type": "ollama",
            "model_name": "car",
            "timeout": 30.0
        }
        
        # 可视化配置
        self.visualization = {
            "enable_debug": True,
            "window_width": 1280,
            "window_height": 720
        }
        
        # MPC控制器配置
        self.mpc = {
            "dt": 0.1,                # 时间步长
            "N": 20,                  # 预测步数
            "L": 2.5,                 # 车辆轴距
            "max_acceleration": 1.0,  # 最大加速度 (m/s^2)
            "max_deceleration": -1.0, # 最大减速度 (m/s^2)
            "max_steer_rate": 0.5,    # 最大转向角速度 (rad/s)
            "max_steer": 30.0,        # 最大转向角 (度)
            "target_speed": 20.0,     # 目标速度 (km/h)
            "lookahead_distance": 5.0 # 前视距离 (m)
        }
        
        # 交叉路口轨迹配置
        self.junction_trajectory = {
            "left_turn": {
                "points": [(0, 0), (2, 0), (5, 2), (8, 5), (10, 8), (12, 10)],
                "speed": 15.0  # km/h
            },
            "right_turn": {
                "points": [(0, 0), (2, 0), (5, -2), (8, -5), (10, -8), (12, -10)],
                "speed": 15.0  # km/h
            },
            "straight": {
                "points": [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0)],
                "speed": 20.0  # km/h
            }
        }