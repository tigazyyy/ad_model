import cv2
import numpy as np
import math
import traceback

class LaneDetector:
    """车道线检测器"""
    
    def __init__(self, vehicle=None, world=None, config=None):
        self.vehicle = vehicle
        self.world = world
        self.config = config
        self.last_left_line = None
        self.last_right_line = None
        self.line_memory_frames = 5  # 记忆帧数
        self.left_line_history = []
        self.right_line_history = []
        # 添加车道中心历史记录，用于平滑
        self.lane_center_history = []
        self.lane_center_memory_frames = 5
    
    def detect_lanes(self, image):
        """检测车道线"""
        try:
            # 创建调试图像
            debug_image = np.copy(image)
            height, width = image.shape[:2]
            
            # 转换为HSV颜色空间，更好地检测白色和黄色车道线
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # 定义白色和黄色的HSV范围
            lower_white = np.array([0, 0, 210])  # 提高亮度阈值
            upper_white = np.array([180, 25, 255])
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            
            # 创建白色和黄色的掩码
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 合并掩码
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # 应用高斯模糊
            blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            
            # 应用Canny边缘检测
            edges = cv2.Canny(blurred, 50, 150)
            
            # 定义感兴趣区域 - 更适合弯道
            mask = np.zeros_like(edges)
            # 更宽的梯形区域，覆盖更多的道路
            polygon = np.array([[
                (width * 0.2, height),           # 左下
                (width * 0.8, height),           # 右下
                (width * 0.55, height * 0.6),    # 右上
                (width * 0.45, height * 0.6)     # 左上
            ]], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # 修改霍夫变换参数
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi/180,
                threshold=30,        # 增加阈值，减少误检测
                minLineLength=40,    # 增加最小线长
                maxLineGap=50        # 减小最大间隙
            )
            
            # 添加形状过滤
            def filter_lines(lines):
                filtered_lines = []
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        
                        # 避免除零错误
                        if x2 == x1:
                            continue
                        
                        slope = (y2 - y1) / (x2 - x1)
                        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        
                        # 1. 长度过滤：车道线应该足够长
                        if length < 40:  # 增加最小长度阈值
                            continue
                            
                        # 2. 斜率过滤：车道线应该有合适的斜率
                        if abs(slope) < 0.3 or abs(slope) > 2.0:
                            continue
                        
                        # 3. 位置过滤：忽略图像中间区域的线段
                        center_zone = image.shape[1] * 0.1  # 中间区域宽度
                        center_x = image.shape[1] / 2
                        if abs(x1 - center_x) < center_zone and abs(x2 - center_x) < center_zone:
                            continue
                        
                        # 4. 方向一致性检查：确保线段方向合理
                        if y1 > y2:  # 确保y1在下方，y2在上方
                            x1, y1, x2, y2 = x2, y2, x1, y1
                        
                        filtered_lines.append((line[0], slope, length))
                
                return filtered_lines
            
            # 应用过滤
            left_lines = []
            right_lines = []
            
            if lines is not None:
                # 首先过滤所有线段
                filtered_lines = filter_lines(lines)
                
                # 然后分类为左右车道线
                for line, slope, length in filtered_lines:
                    x1, y1, x2, y2 = line
                    
                    # 根据斜率和位置区分左右车道线
                    if slope < 0 and x1 < width * 0.6:  # 左车道线
                        left_lines.append((line, slope, length))
                    elif slope > 0 and x1 > width * 0.4:  # 右车道线
                        right_lines.append((line, slope, length))
            
            # 添加车道线验证
            def validate_lane_lines(left_lines, right_lines):
                """验证检测到的车道线是否合理"""
                if not left_lines and not right_lines:
                    return False
                    
                # 检查车道宽度是否合理
                if left_lines and right_lines:
                    # 计算车道宽度
                    left_x = sum(x1 for (x1, y1, x2, y2), _, _ in left_lines) / len(left_lines)
                    right_x = sum(x1 for (x1, y1, x2, y2), _, _ in right_lines) / len(right_lines)
                    lane_width = right_x - left_x
                    
                    # 车道宽度应在合理范围内
                    expected_width = image.shape[1] * 0.5  # 预期车道宽度
                    if not (0.3 * expected_width <= lane_width <= 0.8 * expected_width):
                        return False
                
                return True
            
            # 验证检测结果
            if not validate_lane_lines(left_lines, right_lines):
                # 如果验证失败，使用历史数据
                left_line = self.last_left_line
                right_line = self.last_right_line
            else:
                # 获取左右车道线参数
                left_line = self.get_line_params(left_lines, height, width, is_left=True)
                right_line = self.get_line_params(right_lines, height, width, is_left=False)
            
            # 更新历史记录
            if left_line is not None:
                self.left_line_history.append(left_line)
                if len(self.left_line_history) > self.line_memory_frames:
                    self.left_line_history.pop(0)
                self.last_left_line = left_line
            elif self.last_left_line is not None:
                # 如果当前帧没有检测到，使用历史记录
                left_line = self.last_left_line
            
            if right_line is not None:
                self.right_line_history.append(right_line)
                if len(self.right_line_history) > self.line_memory_frames:
                    self.right_line_history.pop(0)
                self.last_right_line = right_line
            elif self.last_right_line is not None:
                # 如果当前帧没有检测到，使用历史记录
                right_line = self.last_right_line
            
            # 平滑历史记录中的线
            if self.left_line_history:
                left_line = self.smooth_lines(self.left_line_history)
            
            if self.right_line_history:
                right_line = self.smooth_lines(self.right_line_history)
            
            # 获取车道信息
            lane_info = {
                'has_lines': left_line is not None or right_line is not None,
                'left_lane': left_line,
                'right_lane': right_line,
                'is_curve': False,
                'curve_direction': 'straight'
            }
            
            # 检测弯道
            if left_line and right_line:
                left_slope, _ = left_line
                right_slope, _ = right_line
                
                # 斜率差异大说明是弯道
                slope_diff = abs(abs(left_slope) - abs(right_slope))
                if slope_diff > 0.2:
                    lane_info['is_curve'] = True
                    # 判断弯道方向
                    if abs(left_slope) > abs(right_slope):
                        lane_info['curve_direction'] = 'left'
                    else:
                        lane_info['curve_direction'] = 'right'
            elif left_line:
                # 只有左车道线，可能是右弯
                left_slope, _ = left_line
                if abs(left_slope) > 0.5:
                    lane_info['is_curve'] = True
                    lane_info['curve_direction'] = 'right'
            elif right_line:
                # 只有右车道线，可能是左弯
                right_slope, _ = right_line
                if abs(right_slope) > 0.5:
                    lane_info['is_curve'] = True
                    lane_info['curve_direction'] = 'left'
            
            # 计算车道中心
            lane_center = None
            if left_line and right_line:
                # 在多个高度计算车道中心，增加更多采样点
                centers = []
                for h_ratio in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                    h = height * h_ratio
                    left_x = self.get_x_from_line(left_line, h)
                    right_x = self.get_x_from_line(right_line, h)
                    if 0 <= left_x < width and 0 <= right_x < width:
                        # 添加权重，较近的点权重更大
                        weight = 2.0 - h_ratio  # 较低的点获得更大的权重
                        centers.append((left_x + right_x) / 2 * weight)
                
                if centers:
                    # 计算加权平均
                    total_weight = sum([2.0 - (0.6 + i * 0.05) for i in range(len(centers))])
                    lane_center = sum(centers) / total_weight
            elif left_line:
                # 只检测到左车道线，估计中心
                left_x = self.get_x_from_line(left_line, height * 0.8)
                lane_center = left_x + width * 0.25  # 增加偏移量
            elif right_line:
                # 只检测到右车道线，估计中心
                right_x = self.get_x_from_line(right_line, height * 0.8)
                lane_center = right_x - width * 0.25  # 增加偏移量
            
            # 平滑车道中心
            if lane_center is not None:
                self.lane_center_history.append(lane_center)
                if len(self.lane_center_history) > self.lane_center_memory_frames:
                    self.lane_center_history.pop(0)
                
                # 计算平滑后的车道中心
                if self.lane_center_history:
                    lane_center = sum(self.lane_center_history) / len(self.lane_center_history)
            
            lane_info['lane_center'] = lane_center
            
            # 创建彩色可视化图像 - 类似于你提供的参考图像
            visualization = np.zeros_like(image)
            
            # 绘制感兴趣区域
            cv2.fillPoly(visualization, [polygon], (50, 50, 50))
            
            # 绘制车道线
            if left_line:
                self.draw_colored_lane_line(visualization, left_line, (0, 0, 255), height, width=10)  # 红色左车道线
            if right_line:
                self.draw_colored_lane_line(visualization, right_line, (0, 255, 0), height, width=10)  # 绿色右车道线
            
            # 绘制车道中心线
            if lane_info['lane_center'] is not None:
                center_x = int(lane_info['lane_center'])
                cv2.line(visualization, (center_x, height), (center_x, int(height * 0.6)), (255, 0, 0), 5)  # 蓝色中心线
                
                # 绘制方向箭头
                arrow_length = 50
                arrow_width = 20
                if lane_info['is_curve']:
                    if lane_info['curve_direction'] == 'left':
                        # 左转箭头
                        pt1 = (center_x, int(height * 0.7))
                        pt2 = (center_x - arrow_length, int(height * 0.7))
                        cv2.arrowedLine(visualization, pt1, pt2, (255, 0, 255), 3, tipLength=0.5)  # 紫色箭头
                    else:
                        # 右转箭头
                        pt1 = (center_x, int(height * 0.7))
                        pt2 = (center_x + arrow_length, int(height * 0.7))
                        cv2.arrowedLine(visualization, pt1, pt2, (255, 0, 255), 3, tipLength=0.5)  # 紫色箭头
                else:
                    # 直行箭头
                    pt1 = (center_x, int(height * 0.8))
                    pt2 = (center_x, int(height * 0.7) - arrow_length)
                    cv2.arrowedLine(visualization, pt1, pt2, (255, 255, 0), 3, tipLength=0.5)  # 黄色箭头
            
            # 添加文本信息
            if lane_info['is_curve']:
                direction_text = "左转弯" if lane_info['curve_direction'] == 'left' else "右转弯"
                cv2.putText(visualization, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 将可视化图像与原图混合
            alpha = 0.7  # 透明度
            beta = 1.0 - alpha
            blended = cv2.addWeighted(image, beta, visualization, alpha, 0)
            
            # 添加控制信息
            if hasattr(self, 'vehicle') and self.vehicle:
                velocity = self.vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
                cv2.putText(blended, f"速度: {speed:.1f} km/h", (10, height - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return blended, lane_info
            
        except Exception as e:
            print(f"车道线检测出错: {e}")
            traceback.print_exc()
            return image, None
    
    def get_line_params(self, lines, height, width, is_left=True):
        """获取车道线参数"""
        if not lines:
            return None
        
        # 按线段长度排序，优先使用长线段
        lines.sort(key=lambda x: x[2], reverse=True)
        
        # 取前几条最长的线
        top_lines = lines[:min(5, len(lines))]
        
        # 计算加权平均斜率和截距
        total_weight = 0
        slope_sum = 0
        intercept_sum = 0
        
        for line, slope, length in top_lines:
            x1, y1, x2, y2 = line
            
            # 使用线段长度作为权重
            weight = length
            
            # 计算截距 b = y - mx
            intercept = y1 - slope * x1
            
            slope_sum += slope * weight
            intercept_sum += intercept * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        avg_slope = slope_sum / total_weight
        avg_intercept = intercept_sum / total_weight
        
        # 对于极端情况进行修正
        if is_left and avg_slope > 0:
            avg_slope = -0.5  # 左车道线斜率应为负
        elif not is_left and avg_slope < 0:
            avg_slope = 0.5   # 右车道线斜率应为正
        
        return (avg_slope, avg_intercept)
    
    def smooth_lines(self, line_history):
        """平滑多帧的车道线"""
        if not line_history:
            return None
        
        # 使用指数加权平均，最近的帧权重更高
        total_weight = 0
        slope_sum = 0
        intercept_sum = 0
        
        for i, (slope, intercept) in enumerate(line_history):
            # 越新的帧权重越高
            weight = 1.0 + i * 0.5
            slope_sum += slope * weight
            intercept_sum += intercept * weight
            total_weight += weight
        
        avg_slope = slope_sum / total_weight
        avg_intercept = intercept_sum / total_weight
        
        return (avg_slope, avg_intercept)
    
    def get_x_from_line(self, line_params, y):
        """根据直线参数和y坐标计算x坐标"""
        slope, intercept = line_params
        # y = mx + b => x = (y - b) / m
        return (y - intercept) / slope if slope != 0 else 0
    
    def draw_colored_lane_line(self, image, line_params, color, height=None, width=5):
        """在图像上绘制彩色车道线"""
        if height is None:
            height = image.shape[0]
            
        slope, intercept = line_params
        
        # 计算多个点以绘制平滑曲线
        num_points = 20
        y_step = (height - height * 0.6) / num_points
        
        points = []
        for i in range(num_points + 1):
            y = height - i * y_step
            if slope != 0:
                x = int((y - intercept) / slope)
                # 确保x在图像范围内
                if 0 <= x < image.shape[1]:
                    points.append((x, int(y)))
        
        # 绘制平滑曲线
        if len(points) > 1:
            # 使用多边形填充而不是线条
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], False, color, width)
            
            # 添加虚线效果
            for i in range(0, len(points) - 1, 2):
                if i + 1 < len(points):
                    cv2.line(image, points[i], points[i+1], (255, 255, 255), width//2)
    
    def get_lane_distances(self, image_center, height_ratio=0.8):
        """计算到左右车道线的距离"""
        height = self.image.shape[0]
        left_distance = float('inf')
        right_distance = float('inf')
        
        if self.last_left_line:
            left_x = self.get_x_from_line(self.last_left_line, height * height_ratio)
            if 0 <= left_x < self.image.shape[1]:
                left_distance = abs(image_center - left_x)
        
        if self.last_right_line:
            right_x = self.get_x_from_line(self.last_right_line, height * height_ratio)
            if 0 <= right_x < self.image.shape[1]:
                right_distance = abs(right_x - image_center)
        
        return left_distance, right_distance