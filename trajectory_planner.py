import carla
import math
import numpy as np

class TrajectoryPlanner:
    """根据大模型返回的方向生成一条简单轨迹供PID跟随"""
    
    def __init__(self):
        self.waypoint_distance = 2.0  # 轨迹点之间的距离
        self.turn_radius = 6.0  # 转弯半径
    
    def generate_trajectory(self, direction, vehicle):
        """根据方向生成轨迹
        
        Args:
            direction: 方向，如"左转"、"右转"、"直行"
            vehicle: 车辆对象
        
        Returns:
            轨迹点列表，每个点是相对于车辆的(x, y)坐标
        """
        try:
            # 获取车辆当前位置和朝向
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation
            
            # 根据方向生成不同的轨迹
            if direction == "左转":
                return self._generate_left_turn_trajectory(vehicle_transform)
            elif direction == "右转":
                return self._generate_right_turn_trajectory(vehicle_transform)
            else:  # 直行或其他
                return self._generate_straight_trajectory(vehicle_transform)
                
        except Exception as e:
            print(f"生成轨迹时出错: {e}")
            # 返回一个简单的默认轨迹
            return [(0, 0), (0, 5), (0, 10)]
    
    def _generate_left_turn_trajectory(self, vehicle_transform):
        """生成左转轨迹"""
        # 左转轨迹是一系列点，从车辆当前位置开始，先直行一段距离，然后左转90度
        trajectory = []
        
        # 添加起点（车辆当前位置）
        trajectory.append((0, 0))
        
        # 先直行一段距离（5米）
        for i in range(1, 6):
            trajectory.append((0, i * self.waypoint_distance))
        
        # 然后左转90度，使用圆弧
        radius = self.turn_radius
        num_points = 10  # 圆弧上的点数
        for i in range(num_points + 1):
            angle = i * (math.pi / 2) / num_points  # 从0到90度
            x = -radius * math.sin(angle)
            y = radius * (1 - math.cos(angle)) + 5 * self.waypoint_distance
            trajectory.append((x, y))
        
        # 转弯后再直行一段距离
        last_x, last_y = trajectory[-1]
        for i in range(1, 4):
            trajectory.append((last_x - i * self.waypoint_distance, last_y))
        
        return trajectory
    
    def _generate_right_turn_trajectory(self, vehicle_transform):
        """生成右转轨迹"""
        # 右转轨迹是一系列点，从车辆当前位置开始，先直行一段距离，然后右转90度
        trajectory = []
        
        # 添加起点（车辆当前位置）
        trajectory.append((0, 0))
        
        # 先直行一段距离（5米）
        for i in range(1, 6):
            trajectory.append((0, i * self.waypoint_distance))
        
        # 然后右转90度，使用圆弧
        radius = self.turn_radius
        num_points = 10  # 圆弧上的点数
        for i in range(num_points + 1):
            angle = i * (math.pi / 2) / num_points  # 从0到90度
            x = radius * math.sin(angle)
            y = radius * (1 - math.cos(angle)) + 5 * self.waypoint_distance
            trajectory.append((x, y))
        
        # 转弯后再直行一段距离
        last_x, last_y = trajectory[-1]
        for i in range(1, 4):
            trajectory.append((last_x + i * self.waypoint_distance, last_y))
        
        return trajectory
    
    def _generate_straight_trajectory(self, vehicle_transform):
        """生成直行轨迹"""
        # 直行轨迹是一系列沿车辆前方的点
        trajectory = []
        
        # 添加起点（车辆当前位置）
        trajectory.append((0, 0))
        
        # 直行20米，每隔waypoint_distance添加一个点
        for i in range(1, int(20 / self.waypoint_distance) + 1):
            trajectory.append((0, i * self.waypoint_distance))
        
        return trajectory
    
    def visualize_trajectory(self, trajectory, vehicle, world):
        """可视化轨迹（用于调试）"""
        # 获取车辆变换
        vehicle_transform = vehicle.get_transform()
        
        # 清除之前的可视化
        world.debug.draw_string(vehicle_transform.location, "X", color=carla.Color(255, 0, 0), life_time=0.1)
        
        # 绘制轨迹点
        for i, point in enumerate(trajectory):
            if isinstance(point, tuple) and len(point) == 2:
                # 相对坐标
                x, y = point
                # 转换为世界坐标
                forward_vector = vehicle_transform.get_forward_vector()
                right_vector = vehicle_transform.get_right_vector()
                
                world_point = carla.Location(
                    x=vehicle_transform.location.x + forward_vector.x * y + right_vector.x * x,
                    y=vehicle_transform.location.y + forward_vector.y * y + right_vector.y * x,
                    z=vehicle_transform.location.z
                )
            else:
                # 已经是世界坐标
                world_point = point
            
            # 绘制点
            color = carla.Color(0, 255, 0) if i == 0 else carla.Color(0, 0, 255)
            world.debug.draw_point(world_point, size=0.1, color=color, life_time=0.1)
            
            # 连接相邻点
            if i > 0:
                prev_point = trajectory[i-1]
                if isinstance(prev_point, tuple) and len(prev_point) == 2:
                    # 相对坐标
                    prev_x, prev_y = prev_point
                    # 转换为世界坐标
                    prev_world_point = carla.Location(
                        x=vehicle_transform.location.x + forward_vector.x * prev_y + right_vector.x * prev_x,
                        y=vehicle_transform.location.y + forward_vector.y * prev_y + right_vector.y * prev_x,
                        z=vehicle_transform.location.z
                    )
                else:
                    # 已经是世界坐标
                    prev_world_point = prev_point
                
                world.debug.draw_line(
                    prev_world_point, world_point,
                    thickness=0.1, color=carla.Color(255, 255, 0),
                    life_time=0.1
                )
