import numpy as np
import carla
import math

class PathPlanner:
    """用于路径规划的类"""
    def __init__(self):
        self.waypoint_separation = 2.0  # 路径点之间的距离
        self.max_path_length = 100.0    # 最大路径长度
    
    def plan_route(self, start, goal):
        """基本的路径规划方法"""
        # 创建一个直线路径作为简单实现
        path = []
        
        # 计算方向向量
        direction = np.array([goal.x - start.x, goal.y - start.y])
        distance = np.linalg.norm(direction)
        
        # 如果距离为0，返回空路径
        if distance < 0.1:
            return []
            
        # 标准化方向向量
        direction = direction / distance
        
        # 生成路径点
        num_points = min(int(distance / self.waypoint_separation), int(self.max_path_length / self.waypoint_separation))
        
        for i in range(num_points + 1):
            step_distance = i * self.waypoint_separation
            if step_distance > distance:
                break
                
            point_x = start.x + direction[0] * step_distance
            point_y = start.y + direction[1] * step_distance
            point_z = start.z  # 假设相同的z坐标
            
            path.append(carla.Location(x=point_x, y=point_y, z=point_z))
            
        return path

    def generate_curved_path(self, start_point, end_point, control_point, num_points=20):
        """生成二次贝塞尔曲线路径"""
        path = []
        
        for i in range(num_points):
            t = i / (num_points - 1)
            # 二次贝塞尔曲线公式: (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
            x = (1-t)**2 * start_point.x + 2*(1-t)*t * control_point.x + t**2 * end_point.x
            y = (1-t)**2 * start_point.y + 2*(1-t)*t * control_point.y + t**2 * end_point.y
            z = start_point.z  # 假设相同的高度
            
            path.append(carla.Location(x=x, y=y, z=z))
        
        return path
    
    def create_junction_path(self, vehicle, junction, direction):
        """创建交叉路口路径"""
        # 获取车辆位置和朝向
        vehicle_loc = vehicle.get_location()
        vehicle_transform = vehicle.get_transform()
        
        # 获取交叉路口中心
        junction_loc = junction.get_location() if hasattr(junction, 'get_location') else junction
        
        # 计算方向向量
        forward_vec = vehicle_transform.get_forward_vector()
        
        # 交叉路口中心距离
        distance = math.sqrt((junction_loc.x - vehicle_loc.x)**2 + 
                            (junction_loc.y - vehicle_loc.y)**2 + 
                            (junction_loc.z - vehicle_loc.z)**2)