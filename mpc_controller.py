import numpy as np
import cvxpy as cp
import math
import carla
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import traceback

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
        
        # 增加调试标志
        self.debug = True
        
        print("MPC控制器初始化完成")
    
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
    
    def generate_junction_trajectory(self, vehicle, direction='straight'):
        """生成交叉路口轨迹"""
        try:
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # 获取车辆的前向向量和右向向量
            forward_vector = vehicle_transform.get_forward_vector()
            right_vector = vehicle_transform.get_right_vector()
            
            # 根据方向选择轨迹模板
            if direction == 'left':
                template = self.junction_templates['left']
            elif direction == 'right':
                template = self.junction_templates['right']
            else:  # straight
                template = self.junction_templates['straight']
            
            # 生成实际轨迹点
            trajectory_points = []
            
            # 转换模板点到全局坐标系
            for point in template:
                # 计算全局坐标
                x = vehicle_location.x + forward_vector.x * point[0] + right_vector.x * point[1]
                y = vehicle_location.y + forward_vector.y * point[0] + right_vector.y * point[1]
                
                # 计算朝向角度
                if direction == 'left':
                    yaw = vehicle_rotation.yaw - (point[0] / template[-1][0]) * 90
                elif direction == 'right':
                    yaw = vehicle_rotation.yaw + (point[0] / template[-1][0]) * 90
                else:
                    yaw = vehicle_rotation.yaw
                
                # 计算目标速度（转弯时降低速度）
                if direction in ['left', 'right']:
                    target_speed = max(5.0, self.target_speed * (1.0 - 0.5 * (point[0] / template[-1][0])))
                else:
                    target_speed = self.target_speed
                
                trajectory_points.append((x, y, yaw, target_speed))
            
            # 使用样条插值平滑轨迹
            if len(trajectory_points) >= 4:
                x = [p[0] for p in trajectory_points]
                y = [p[1] for p in trajectory_points]
                yaw = [p[2] for p in trajectory_points]
                speed = [p[3] for p in trajectory_points]
                
                # 生成参数化的样条曲线
                tck, u = splprep([x, y], s=0.1, k=3)
                u_new = np.linspace(0, 1, num=50)
                x_new, y_new = splev(u_new, tck)
                
                # 重新生成平滑的轨迹点
                smooth_trajectory = []
                for i in range(len(x_new)):
                    # 插值计算朝向和速度
                    t = i / (len(x_new) - 1)
                    yaw_interp = np.interp(t, np.linspace(0, 1, len(yaw)), yaw)
                    speed_interp = np.interp(t, np.linspace(0, 1, len(speed)), speed)
                    smooth_trajectory.append((x_new[i], y_new[i], yaw_interp, speed_interp))
                
                trajectory_points = smooth_trajectory
            
            # 保存轨迹
            self.trajectory = trajectory_points
            self.current_trajectory_index = 0
            self.trajectory_completed = False
            
            print(f"【MPC】已生成{direction}转轨迹，共{len(trajectory_points)}个点")
            return trajectory_points
            
        except Exception as e:
            print(f"生成交叉路口轨迹失败: {e}")
            traceback.print_exc()
            return None
    
    def generate_explicit_right_turn(self, vehicle):
        """强制生成确定的右转轨迹
        
        Args:
            vehicle: 车辆对象
            
        Returns:
            trajectory: 生成的轨迹点列表
        """
        try:
            print("【MPC】强制生成明确的右转轨迹")
            
            # 获取车辆当前状态
            vehicle_location = vehicle.get_location()
            vehicle_transform = vehicle.get_transform()
            vehicle_rotation = vehicle_transform.rotation
            yaw_rad = math.radians(vehicle_rotation.yaw)
            
            # 获取车辆的前向向量和右向向量
            forward_vector = vehicle_transform.get_forward_vector()
            right_vector = vehicle_transform.get_right_vector()
            
            # 右转轨迹参数
            trajectory_length = 30  # 轨迹总长度
            turning_start = 8       # 从这个距离开始转弯
            straight_speed = 10.0   # 直行速度 (km/h)
            turn_speed = 8.0        # 转弯速度 (km/h)
            arc_radius = 6.0        # 转弯半径 - 较小半径，更显著的右转
            
            # 轨迹模板 - 右转时保持在车辆右侧
            trajectory_template = []
            
            # 计算弧形路径中心点 - 确保在车辆右侧
            # 使用正的右向量表示右侧方向
            arc_center_x = vehicle_location.x + forward_vector.x * turning_start + right_vector.x * arc_radius
            arc_center_y = vehicle_location.y + forward_vector.y * turning_start + right_vector.y * arc_radius
            
            print(f"【MPC右转】弧形中心点: ({arc_center_x:.2f}, {arc_center_y:.2f})")
            print(f"【MPC右转】车辆位置: ({vehicle_location.x:.2f}, {vehicle_location.y:.2f})")
            print(f"【MPC右转】车辆朝向: {vehicle_rotation.yaw:.2f}度")
            
            # 创建更明确的右转轨迹
            for i in range(trajectory_length):
                progress = i / trajectory_length
                distance = i * 1.0  # 每个点间隔1米
                
                if distance < turning_start:
                    # 直行段 - 使用前向向量
                    x = vehicle_location.x + distance * forward_vector.x
                    y = vehicle_location.y + distance * forward_vector.y
                    heading = vehicle_rotation.yaw
                    target_speed = straight_speed
                else:
                    # 转弯段 - 逆时针旋转90度（在车辆右侧）
                    turning_progress = (distance - turning_start) / (trajectory_length - turning_start)
                    turning_angle = turning_progress * math.pi/2  # 90度
                    
                    # 计算旋转后的位置 - 绕弧中心旋转
                    angle = turning_angle
                    x = arc_center_x - arc_radius * math.sin(angle)
                    y = arc_center_y + arc_radius * math.cos(angle)
                    
                    # 计算朝向 - 逐渐旋转到右转90度
                    heading = (vehicle_rotation.yaw + math.degrees(angle)) % 360
                    
                    # 减速转弯
                    target_speed = max(5.0, turn_speed - 3.0 * math.sin(turning_progress * math.pi/2))
                
                # 添加到轨迹模板
                trajectory_template.append((x, y, heading, target_speed / 3.6))  # 转换为m/s
            
            # 保存轨迹
            self.trajectory = trajectory_template
            self.trajectory_index = 0
            
            # 强制设置右转方向盘角度
            self.current_steering = 0.5  # 更强的右转方向
            
            # 打印轨迹点，确认确实是右转
            if len(self.trajectory) > 0:
                start_point = self.trajectory[0]
                mid_point = self.trajectory[min(10, len(self.trajectory)-1)]
                end_point = self.trajectory[-1]
                
                print(f"【右转轨迹起点】: ({start_point[0]:.2f}, {start_point[1]:.2f}), 朝向: {start_point[2]:.2f}度")
                print(f"【右转轨迹中点】: ({mid_point[0]:.2f}, {mid_point[1]:.2f}), 朝向: {mid_point[2]:.2f}度")
                print(f"【右转轨迹终点】: ({end_point[0]:.2f}, {end_point[1]:.2f}), 朝向: {end_point[2]:.2f}度")
                
                # 计算方向向量，验证是否为右转
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                # 计算点积和叉积，判断轨迹方向
                dot_product = forward_vector.x * dx + forward_vector.y * dy
                cross_product = forward_vector.x * dy - forward_vector.y * dx
                
                if cross_product > 0:
                    print("【MPC】验证成功：生成的是右转轨迹")
                else:
                    print("【MPC警告】轨迹方向验证失败，可能不是右转")
            
            return self.trajectory
            
        except Exception as e:
            print(f"【错误】强制生成右转轨迹失败: {e}")
            traceback.print_exc()
            return []
    
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
        """获取下一个轨迹点"""
        if self.is_trajectory_completed():
            return None
            
        current_point = self.trajectory[self.current_trajectory_index]
        self.current_trajectory_index += 1
        
        # 如果到达轨迹末尾，标记轨迹已完成
        if self.current_trajectory_index >= len(self.trajectory):
            self.trajectory_completed = True
            
        return current_point
    
    def is_trajectory_completed(self):
        """检查轨迹是否已完成"""
        return self.trajectory_completed or len(self.trajectory) == 0 or self.current_trajectory_index >= len(self.trajectory)
    
    def reset_trajectory(self):
        """重置轨迹"""
        self.trajectory = []
        self.current_trajectory_index = 0
        self.trajectory_completed = True
        print("【MPC】已重置轨迹")
    
    def get_remaining_trajectory(self):
        """获取剩余的轨迹点"""
        if self.is_trajectory_completed():
            return []
            
        return self.trajectory[self.current_trajectory_index:]
    
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
    
    def get_steering(self):
        """获取当前MPC计算的方向盘转角"""
        if not hasattr(self, 'current_steering') or self.current_steering is None:
            return 0.0
        return self.current_steering
    
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
    
    def visualize_trajectory_in_carla(self, world, vehicle, lifetime=2.0):
        """在CARLA中可视化轨迹"""
        try:
            if self.trajectory is None or len(self.trajectory) == 0:
                print("【警告】没有轨迹可以可视化")
                return
            
            # 获取车辆高度
            vehicle_z = vehicle.get_location().z
            
            # 轨迹颜色 - 使用更加柔和的颜色
            trajectory_color = carla.Color(r=15, g=80, b=30, a=80)  # 更暗更透明的绿色
            
            # 只显示关键轨迹点，减少视觉干扰
            for i in range(1, len(self.trajectory), 2):  # 每隔一个点显示
                prev_point = self.trajectory[i-1]
                curr_point = self.trajectory[i]
                
                # 创建轨迹中心线，贴近地面
                prev_loc = carla.Location(x=prev_point[0], y=prev_point[1], z=vehicle_z + 0.03)
                curr_loc = carla.Location(x=curr_point[0], y=curr_point[1], z=vehicle_z + 0.03)
                
                # 使用车道线风格，但更细更暗
                world.debug.draw_line(
                    prev_loc, curr_loc,
                    thickness=0.03,  # 更细的线
                    color=trajectory_color,
                    life_time=lifetime,
                    persistent_lines=False
                )
                
                # 在转弯点处添加箭头指示
                if i % 6 == 0:  # 每6个点添加一个方向指示
                    # 计算方向向量
                    direction = carla.Vector3D(
                        x=curr_point[0] - prev_point[0],
                        y=curr_point[1] - prev_point[1],
                        z=0
                    )
                    
                    # 标准化方向向量
                    length = math.sqrt(direction.x**2 + direction.y**2)
                    if length > 0:
                        direction.x /= length
                        direction.y /= length
                    
                    # 计算垂直于方向的向量
                    perpendicular = carla.Vector3D(
                        x=-direction.y,
                        y=direction.x,
                        z=0
                    )
                    
                    # 创建小型箭头
                    center = carla.Location(x=curr_point[0], y=curr_point[1], z=vehicle_z + 0.03)
                    
                    # 箭头尺寸
                    width = 0.3
                    
                    # 绘制简单的小箭头
                    points = [
                        carla.Location(
                            x=center.x - direction.x * width * 0.5,
                            y=center.y - direction.y * width * 0.5,
                            z=center.z
                        ),
                        carla.Location(
                            x=center.x + direction.x * width,
                            y=center.y + direction.y * width,
                            z=center.z
                        )
                    ]
                    
                    # 绘制箭头主干
                    world.debug.draw_line(
                        points[0], points[1],
                        thickness=0.03,
                        color=carla.Color(r=15, g=100, b=15, a=100),  # 更柔和的绿色
                        life_time=lifetime
                    )
            
            print(f"【轨迹可视化】已显示 {len(self.trajectory)} 个轨迹点，使用柔和的车道线风格")
            
        except Exception as e:
            print(f"【错误】轨迹可视化失败: {e}")
            traceback.print_exc() 