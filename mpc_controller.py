import numpy as np
import cvxpy as cp
import math
import carla
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

class MPCController:
    """模型预测控制器，用于在交叉路口生成轨迹"""
    
    def __init__(self, config):
        """初始化MPC控制器
        
        Args:
            config: 配置对象
        """
        # MPC参数
        self.dt = 0.1  # 时间步长
        self.N = 20    # 预测步数
        self.L = 2.5   # 车辆轴距
        
        # 权重参数
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5])  # 状态权重 [x, y, v, yaw]
        self.R = np.diag([0.01, 0.1])           # 控制权重 [加速度, 转向角速度]
        self.Rd = np.diag([0.01, 0.1])          # 控制变化率权重
        
        # 约束参数
        self.max_acceleration = 1.0   # 最大加速度 (m/s^2)
        self.max_deceleration = -1.0  # 最大减速度 (m/s^2)
        self.max_steer_rate = 0.5     # 最大转向角速度 (rad/s)
        self.max_steer = np.deg2rad(30.0)  # 最大转向角 (rad)
        
        # 轨迹参数 - 降低目标速度
        self.target_speed = 15.0 / 3.6  # 目标速度 (m/s)，约15km/h
        
        # 存储生成的轨迹
        self.trajectory = []
        self.current_trajectory_index = 0
        self.trajectory_completed = True
        
        # 交叉路口轨迹模板
        self.junction_templates = {
            'left': self._create_left_turn_template(),
            'right': self._create_right_turn_template(),
            'straight': self._create_straight_template()
        }
    
    def _create_left_turn_template(self):
        """创建左转轨迹模板"""
        # 左转轨迹点 (相对坐标) - 更平滑的曲线
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0, 0, 0, 0.5, 1, 2, 3, 4, 5, 6, 7]
        return np.column_stack((x, y))
    
    def _create_right_turn_template(self):
        """创建右转轨迹模板"""
        # 右转轨迹点 (相对坐标) - 更平滑的曲线
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [0, 0, 0, -0.5, -1, -2, -3, -4, -5, -6, -7]
        return np.column_stack((x, y))
    
    def _create_straight_template(self):
        """创建直行轨迹模板"""
        # 直行轨迹点 (相对坐标) - 更多的点
        x = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return np.column_stack((x, y))
    
    def generate_junction_trajectory(self, vehicle, direction, junction_center=None):
        """生成交叉路口轨迹
        
        Args:
            vehicle: 车辆对象
            direction: 方向 ('left', 'right', 'straight')
            junction_center: 交叉路口中心点 (可选)
            
        Returns:
            生成的轨迹点列表 [(x, y, yaw, speed), ...]
        """
        # 获取车辆当前状态
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        vehicle_yaw = np.deg2rad(vehicle_rotation.yaw)
        vehicle_speed = vehicle.get_velocity()
        vehicle_speed_scalar = np.sqrt(vehicle_speed.x**2 + vehicle_speed.y**2 + vehicle_speed.z**2)
        
        # 确保初始速度不为零
        if vehicle_speed_scalar < 0.1:
            vehicle_speed_scalar = 2.0  # 默认初始速度为2m/s (约7.2km/h)
        
        # 如果没有提供交叉路口中心点，使用前方较近的距离作为参考点
        if junction_center is None:
            forward_vector = vehicle_transform.get_forward_vector()
            junction_center = carla.Location(
                x=vehicle_location.x + forward_vector.x * 10,  # 减少距离到10米
                y=vehicle_location.y + forward_vector.y * 10,
                z=vehicle_location.z
            )
        
        # 获取轨迹模板
        template = self.junction_templates[direction]
        
        # 将模板转换为全局坐标
        cos_yaw = np.cos(vehicle_yaw)
        sin_yaw = np.sin(vehicle_yaw)
        
        global_trajectory = []
        for point in template:
            # 旋转并平移到车辆坐标系
            x_global = point[0] * cos_yaw - point[1] * sin_yaw + vehicle_location.x
            y_global = point[0] * sin_yaw + point[1] * cos_yaw + vehicle_location.y
            
            # 计算该点的航向角 (简化处理，实际应根据轨迹曲率计算)
            if len(global_trajectory) > 0:
                prev_x, prev_y = global_trajectory[-1][0], global_trajectory[-1][1]
                yaw = np.arctan2(y_global - prev_y, x_global - prev_x)
            else:
                yaw = vehicle_yaw
            
            # 计算该点的速度 (简化处理，实际应根据曲率调整)
            # 起始点使用当前速度，然后逐渐增加到目标速度
            if len(global_trajectory) == 0:
                speed = vehicle_speed_scalar
            else:
                # 逐渐增加速度，但不超过目标速度
                prev_speed = global_trajectory[-1][3]
                speed = min(prev_speed + 0.5, self.target_speed)  # 每个点最多增加0.5m/s
            
            global_trajectory.append((x_global, y_global, yaw, speed))
        
        # 使用样条插值生成更平滑的轨迹
        if len(global_trajectory) > 3:
            x = [p[0] for p in global_trajectory]
            y = [p[1] for p in global_trajectory]
            
            # 样条插值
            tck, u = splprep([x, y], s=0, k=3)
            u_new = np.linspace(0, 1, 50)  # 生成50个点
            x_new, y_new = splev(u_new, tck)
            
            # 计算新轨迹点的航向角和速度
            smooth_trajectory = []
            for i in range(len(x_new)):
                if i > 0:
                    yaw = np.arctan2(y_new[i] - y_new[i-1], x_new[i] - x_new[i-1])
                else:
                    yaw = vehicle_yaw
                
                # 根据曲率调整速度
                if i > 0 and i < len(x_new) - 1:
                    # 简单的曲率计算 (三点法)
                    dx1 = x_new[i] - x_new[i-1]
                    dy1 = y_new[i] - y_new[i-1]
                    dx2 = x_new[i+1] - x_new[i]
                    dy2 = y_new[i+1] - y_new[i]
                    
                    # 计算方向变化
                    angle1 = np.arctan2(dy1, dx1)
                    angle2 = np.arctan2(dy2, dx2)
                    angle_diff = abs(angle2 - angle1)
                    while angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    # 根据角度变化调整速度
                    speed = self.target_speed * (1 - 0.5 * angle_diff / np.pi)
                    speed = max(5.0 / 3.6, speed)  # 最低速度5km/h
                else:
                    speed = self.target_speed
                
                smooth_trajectory.append((x_new[i], y_new[i], yaw, speed))
            
            self.trajectory = smooth_trajectory
        else:
            self.trajectory = global_trajectory
        
        self.current_trajectory_index = 0
        self.trajectory_completed = False
        
        return self.trajectory
    
    def solve_mpc(self, vehicle, reference_trajectory):
        """求解MPC问题
        
        Args:
            vehicle: 车辆对象
            reference_trajectory: 参考轨迹
            
        Returns:
            最优控制序列 [加速度, 转向角速度]
        """
        # 获取车辆当前状态
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation
        vehicle_yaw = np.deg2rad(vehicle_rotation.yaw)
        vehicle_velocity = vehicle.get_velocity()
        vehicle_speed = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
        
        # 当前状态 [x, y, v, yaw]
        current_state = np.array([
            vehicle_location.x,
            vehicle_location.y,
            vehicle_speed,
            vehicle_yaw
        ])
        
        # 定义MPC问题
        x = cp.Variable((4, self.N+1))  # 状态变量 [x, y, v, yaw]
        u = cp.Variable((2, self.N))    # 控制变量 [加速度, 转向角速度]
        
        # 目标函数
        cost = 0
        for i in range(self.N):
            # 参考轨迹点
            if i < len(reference_trajectory):
                ref_x, ref_y, ref_yaw, ref_speed = reference_trajectory[i]
                ref = np.array([ref_x, ref_y, ref_speed, ref_yaw])
            else:
                ref = np.array([
                    reference_trajectory[-1][0],
                    reference_trajectory[-1][1],
                    reference_trajectory[-1][3],
                    reference_trajectory[-1][2]
                ])
            
            # 状态跟踪误差
            cost += cp.quad_form(x[:, i] - ref, self.Q)
            
            # 控制代价
            cost += cp.quad_form(u[:, i], self.R)
            
            # 控制变化率代价
            if i > 0:
                cost += cp.quad_form(u[:, i] - u[:, i-1], self.Rd)
        
        # 约束条件
        constraints = [x[:, 0] == current_state]  # 初始状态约束
        
        for i in range(self.N):
            # 运动学模型约束 (自行车模型)
            constraints += [
                x[0, i+1] == x[0, i] + x[2, i] * cp.cos(x[3, i]) * self.dt,
                x[1, i+1] == x[1, i] + x[2, i] * cp.sin(x[3, i]) * self.dt,
                x[2, i+1] == x[2, i] + u[0, i] * self.dt,
                x[3, i+1] == x[3, i] + x[2, i] * cp.tan(u[1, i]) / self.L * self.dt
            ]
            
            # 控制约束
            constraints += [
                u[0, i] >= self.max_deceleration,
                u[0, i] <= self.max_acceleration,
                u[1, i] >= -self.max_steer,
                u[1, i] <= self.max_steer
            ]
            
            # 速度约束
            constraints += [
                x[2, i] >= 0,  # 速度非负
                x[2, i] <= self.target_speed * 1.2  # 最大速度
            ]
        
        # 定义并求解问题
        problem = cp.Problem(cp.Minimize(cost), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
                # 返回最优控制序列的第一个值
                return np.array([u[0, 0].value, u[1, 0].value])
            else:
                print(f"MPC求解失败: {problem.status}")
                return np.array([0.0, 0.0])
        except Exception as e:
            print(f"MPC求解异常: {e}")
            return np.array([0.0, 0.0])
    
    def get_next_trajectory_point(self):
        """获取下一个轨迹点
        
        Returns:
            轨迹点 (x, y, yaw, speed) 或 None (如果轨迹已完成)
        """
        if self.trajectory_completed or len(self.trajectory) == 0:
            return None
        
        if self.current_trajectory_index < len(self.trajectory):
            point = self.trajectory[self.current_trajectory_index]
            self.current_trajectory_index += 1
            return point
        else:
            self.trajectory_completed = True
            return None
    
    def is_trajectory_completed(self):
        """检查轨迹是否已完成"""
        return self.trajectory_completed
    
    def reset_trajectory(self):
        """重置轨迹"""
        self.trajectory = []
        self.current_trajectory_index = 0
        self.trajectory_completed = True
    
    def get_target_point(self, vehicle, lookahead_distance=5.0):
        """获取前方指定距离处的目标点
        
        Args:
            vehicle: 车辆对象
            lookahead_distance: 前视距离 (米)
            
        Returns:
            目标点 (x, y, yaw, speed) 或 None
        """
        if self.trajectory_completed or len(self.trajectory) == 0:
            return None
        
        # 获取车辆当前位置
        vehicle_location = vehicle.get_transform().location
        
        # 找到距离车辆前方lookahead_distance的轨迹点
        min_dist = float('inf')
        target_point = None
        target_index = self.current_trajectory_index
        
        for i in range(self.current_trajectory_index, len(self.trajectory)):
            point = self.trajectory[i]
            dist = np.sqrt((point[0] - vehicle_location.x)**2 + (point[1] - vehicle_location.y)**2)
            
            if abs(dist - lookahead_distance) < min_dist:
                min_dist = abs(dist - lookahead_distance)
                target_point = point
                target_index = i
        
        # 更新当前轨迹索引
        if target_index > self.current_trajectory_index:
            self.current_trajectory_index = target_index
        
        return target_point
    
    def visualize_trajectory(self, vehicle=None):
        """可视化轨迹
        
        Args:
            vehicle: 车辆对象 (可选)
        """
        if len(self.trajectory) == 0:
            print("没有轨迹可视化")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 绘制轨迹
        x = [point[0] for point in self.trajectory]
        y = [point[1] for point in self.trajectory]
        plt.plot(x, y, 'b-', label='Trajectory')
        
        # 标记当前位置
        if self.current_trajectory_index < len(self.trajectory):
            current_point = self.trajectory[self.current_trajectory_index]
            plt.plot(current_point[0], current_point[1], 'ro', label='Current Position')
        
        # 绘制车辆位置
        if vehicle is not None:
            vehicle_location = vehicle.get_transform().location
            plt.plot(vehicle_location.x, vehicle_location.y, 'go', label='Vehicle')
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('MPC Trajectory')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        
        plt.savefig('trajectory.png')
        plt.close()
        
    def visualize_trajectory_in_carla(self, world, vehicle=None, lifetime=0.1):
        """在CARLA环境中可视化轨迹
        
        Args:
            world: CARLA世界对象
            vehicle: 车辆对象 (可选)
            lifetime: 可视化持续时间 (秒)
        """
        if len(self.trajectory) == 0:
            print("没有轨迹可视化")
            return
        
        # 获取调试帮助器
        debug = world.debug
        
        # 轨迹点颜色 - 降低亮度
        trajectory_color = carla.Color(0, 0, 150)  # 深蓝色
        current_point_color = carla.Color(150, 0, 0)  # 深红色
        
        # 绘制轨迹线 - 降低线条粗细
        for i in range(len(self.trajectory) - 1):
            start_point = carla.Location(x=self.trajectory[i][0], y=self.trajectory[i][1], z=vehicle.get_transform().location.z + 0.2)
            end_point = carla.Location(x=self.trajectory[i+1][0], y=self.trajectory[i+1][1], z=vehicle.get_transform().location.z + 0.2)
            
            # 如果是当前轨迹点，使用红色
            if i == self.current_trajectory_index:
                debug.draw_line(start_point, end_point, thickness=0.1, color=current_point_color, life_time=lifetime)
            else:
                debug.draw_line(start_point, end_point, thickness=0.05, color=trajectory_color, life_time=lifetime)
        
        # 标记轨迹点 - 减少点的大小和数量
        for i in range(0, len(self.trajectory), 5):  # 每5个点标记一次
            location = carla.Location(x=self.trajectory[i][0], y=self.trajectory[i][1], z=vehicle.get_transform().location.z + 0.2)
            
            # 如果是当前轨迹点，使用红色并增大尺寸
            if i == self.current_trajectory_index:
                debug.draw_point(location, size=0.05, color=current_point_color, life_time=lifetime)
            else:
                debug.draw_point(location, size=0.03, color=trajectory_color, life_time=lifetime)
            
            # 只在关键点显示序号
            if i % 10 == 0:
                debug.draw_string(location, str(i), draw_shadow=False, color=carla.Color(150, 150, 150), life_time=lifetime)
        
        # 只在关键点显示速度信息
        for i in range(0, len(self.trajectory), 10):
            location = carla.Location(x=self.trajectory[i][0], y=self.trajectory[i][1], z=vehicle.get_transform().location.z + 0.5)
            speed_text = f"{self.trajectory[i][3] * 3.6:.1f}"
            debug.draw_string(location, speed_text, draw_shadow=False, color=carla.Color(0, 150, 0), life_time=lifetime)
