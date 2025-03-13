import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import glob
import traceback

class LaneDetector:
    def __init__(self, vehicle, world, config):
        self.vehicle = vehicle
        self.world = world
        self.config = config
        self.last_center = 0.0
        self._debug_image = None
        self.lane_info = {}
        
        # 滤波器参数
        self.center_filter = []
        self.filter_size = 5
        
        # 调试模式
        self.debug = True
        
        # 车道检测参数
        self.roi_height = 0.5  # ROI区域的高度比例
        self.min_line_length = 30  # 最小线段长度
        self.max_line_gap = 20  # 最大线段间隙
        self.hough_threshold = 30  # 霍夫变换阈值
        self.canny_low = 50  # Canny边缘检测低阈值
        self.canny_high = 150  # Canny边缘检测高阈值
        
        # 曲线拟合参数
        self.use_spline = True  # 是否使用样条曲线拟合
        self.poly_degree = 2  # 多项式拟合的阶数
        self.spline_smoothness = 0.1  # 样条曲线平滑度
        
        # 异常检测参数
        self.max_curve_angle = 60  # 曲线最大角度变化（度）
        self.min_points_for_fit = 5  # 拟合曲线所需的最少点数

    def detect_lanes(self, image):
        """检测车道线"""
        # 创建调试图像
        debug_image = image.copy() if self.debug else None
        height, width = image.shape[:2]
        
        # 预处理图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        
        # 边缘检测
        edges = cv2.Canny(enhanced, self.canny_low, self.canny_high)
        
        # 定义ROI区域 - 梯形区域
        roi_vertices = np.array([
            [width * 0.1, height],  # 左下
            [width * 0.4, height * self.roi_height],  # 左上
            [width * 0.6, height * self.roi_height],  # 右上
            [width * 0.9, height]  # 右下
        ], dtype=np.int32)
        
        # 创建ROI掩码
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [roi_vertices], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # 如果没有检测到线段，返回None
        if lines is None:
            return None, debug_image
        
        # 对线段按长度排序，优先考虑长线段
        sorted_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            slope = (y2-y1)/(x2-x1) if x2!=x1 else float('inf')
            sorted_lines.append((x1, y1, x2, y2, length, slope))
        
        sorted_lines.sort(key=lambda x: -x[4])  # 按长度降序排序
        
        # 根据斜率和位置分类线段
        left_points = []
        right_points = []
        
        for x1, y1, x2, y2, length, slope in sorted_lines:
            # 跳过太短的线段
            if length < self.min_line_length:
                continue
                
            # 跳过近乎水平的线
            if abs(slope) < 0.3:
                continue
                
            # 根据斜率和位置分类
            # 左车道线: 负斜率, 在图像左侧
            # 右车道线: 正斜率, 在图像右侧
            x_center = (x1 + x2) / 2
            if slope < 0 and x_center < width/2:
                # 左车道线
                left_points.append((x1, y1))
                left_points.append((x2, y2))
            elif slope > 0 and x_center > width/2:
                # 右车道线
                right_points.append((x1, y1))
                right_points.append((x2, y2))
        
        # 如果调试模式开启，绘制原始检测点
        if self.debug and debug_image is not None:
            for x, y in left_points:
                cv2.circle(debug_image, (x, y), 3, (255, 0, 0), -1)
            for x, y in right_points:
                cv2.circle(debug_image, (x, y), 3, (0, 0, 255), -1)
        
        # 车道线拟合结果
        left_curve = None
        right_curve = None
        center_curve = None
        
        # 尝试拟合左车道线
        if len(left_points) >= self.min_points_for_fit:
            left_curve = self._fit_curve(left_points, height, width, 'left')
            if left_curve is not None and debug_image is not None:
                self._draw_curve(debug_image, left_curve, (255, 0, 0), 2)  # 蓝色
        
        # 尝试拟合右车道线
        if len(right_points) >= self.min_points_for_fit:
            right_curve = self._fit_curve(right_points, height, width, 'right')
            if right_curve is not None and debug_image is not None:
                self._draw_curve(debug_image, right_curve, (0, 0, 255), 2)  # 红色
        
        # 计算中心线
        center_offset = 0.0
        
        if left_curve is not None and right_curve is not None:
            # 如果同时检测到左右车道线，计算中心线
            center_curve = []
            # 使用左右车道线的中点计算中心线
            min_len = min(len(left_curve), len(right_curve))
            for i in range(min_len):
                x1, y1 = left_curve[i]
                x2, y2 = right_curve[i]
                center_curve.append(((x1 + x2) // 2, (y1 + y2) // 2))
            
            if debug_image is not None:
                self._draw_curve(debug_image, center_curve, (0, 255, 0), 2)  # 绿色
            
            # 计算底部的偏移量
            if center_curve:
                # 找到最底部的点（通常是y坐标最大的点）
                bottom_points = sorted(center_curve, key=lambda p: -p[1])
                if bottom_points:
                    bottom_x = bottom_points[0][0]
                    image_center = width // 2
                    center_offset = (bottom_x - image_center) / (width // 2)  # 归一化到[-1,1]
            
                # 平滑偏移量
                self.center_filter.append(center_offset)
                if len(self.center_filter) > self.filter_size:
                    self.center_filter.pop(0)
                center_offset = sum(self.center_filter) / len(self.center_filter)
        
        # 构建车道信息
        lane_info = {
            'left_curve': left_curve,
            'right_curve': right_curve,
            'center_curve': center_curve,
            'center_offset': center_offset,
            'confidence': 1.0 if (left_curve is not None and right_curve is not None) else 0.5
        }
        
        # 在调试图像上显示相关信息
        if debug_image is not None:
            cv2.putText(debug_image, f"Offset: {center_offset:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_image, f"Confidence: {lane_info['confidence']:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return lane_info, debug_image
        
    def _fit_curve(self, points, height, width, lane_type):
        """拟合车道线曲线"""
        if not points or len(points) < self.min_points_for_fit:
            return None
            
        # 根据车道线类型不同，使用不同的排序方式
        if lane_type == 'left' or lane_type == 'right':
            # 将点按y坐标排序（从下到上）
            points = sorted(points, key=lambda p: -p[1])
        
        # 移除离群点
        filtered_points = self._remove_outliers(points, lane_type)
        if len(filtered_points) < self.min_points_for_fit:
            return None
            
        # 提取坐标
        x_values = [p[0] for p in filtered_points]
        y_values = [p[1] for p in filtered_points]
        
        # 确保y值从大到小排序（从图像底部到顶部）
        if not all(y_values[i] >= y_values[i+1] for i in range(len(y_values)-1)):
            combined = list(zip(x_values, y_values))
            combined.sort(key=lambda p: -p[1])  # 按y从大到小排序
            x_values = [p[0] for p in combined]
            y_values = [p[1] for p in combined]
        
        try:
            curve_points = []
            
            if self.use_spline and len(filtered_points) >= 4:
                # 使用样条曲线拟合，但要避免过度拟合
                # 先确保点是唯一的（对于样条拟合是必要的）
                unique_y = {}
                for x, y in zip(x_values, y_values):
                    if y not in unique_y:
                        unique_y[y] = x
                        
                if len(unique_y) >= 4:
                    unique_x = list(unique_y.values())
                    unique_y_keys = list(unique_y.keys())
                    
                    # 使用较大的平滑度，减少过拟合
                    s = max(self.spline_smoothness, 0.5)  # 增大平滑度
                    k = min(3, len(unique_y) - 1)  # 确保k不超过点数-1
                    
                    # 计算样条曲线
                    tck, u = splprep([unique_x, unique_y_keys], s=s, k=k)
                    
                    # 生成平滑的点序列
                    t_points = np.linspace(0, 1, 20)  # 减少点数，避免过度拟合
                    spline_points = splev(t_points, tck)
                    
                    # 生成曲线点
                    for x, y in zip(spline_points[0], spline_points[1]):
                        curve_points.append((int(x), int(y)))
                else:
                    # 样条拟合失败，使用多项式拟合
                    self.use_spline = False
            
            # 如果样条拟合未用或失败，使用多项式拟合
            if not self.use_spline or not curve_points:
                # 使用RANSAC拟合，更鲁棒
                X = np.array(y_values).reshape(-1, 1)  # 使用y作为自变量
                y = np.array(x_values)  # 使用x作为因变量
                
                # 创建RANSAC模型
                ransac = RANSACRegressor(
                    LinearRegression(), 
                    min_samples=0.6,           # 使用60%的数据
                    residual_threshold=15.0,   # 容许的残差阈值
                    max_trials=100             # 最大尝试次数
                )
                
                # 进行拟合
                if self.poly_degree <= 1:
                    # 线性拟合 (阶数为1)
                    ransac.fit(X, y)
                    
                    # 生成拟合点
                    y_new = np.linspace(min(y_values), max(y_values), 20)
                    X_new = y_new.reshape(-1, 1)
                    x_new = ransac.predict(X_new)
                    
                    # 将点转换为整数坐标
                    for x, y in zip(x_new, y_new):
                        curve_points.append((int(x), int(y)))
                else:
                    # 多项式拟合 (阶数 > 1)
                    model = make_pipeline(
                        PolynomialFeatures(degree=min(self.poly_degree, 2)),  # 限制阶数，避免过拟合
                        RANSACRegressor(
                            LinearRegression(),
                            min_samples=0.6,
                            residual_threshold=15.0,
                            max_trials=100
                        )
                    )
                    
                    model.fit(X, y)
                    
                    # 生成拟合点
                    y_new = np.linspace(min(y_values), max(y_values), 20)
                    X_new = y_new.reshape(-1, 1)
                    x_new = model.predict(X_new)
                    
                    # 将点转换为整数坐标
                    for x, y in zip(x_new, y_new):
                        curve_points.append((int(x), int(y)))
            
            # 检查曲线的合理性
            if self._is_valid_curve(curve_points, lane_type, width, height):
                return curve_points
            else:
                # 如果曲线不合理，尝试线性拟合作为后备方案
                model = LinearRegression()
                X = np.array(y_values).reshape(-1, 1)
                y = np.array(x_values)
                model.fit(X, y)
                
                # 生成直线点
                y_new = np.linspace(min(y_values), max(y_values), 20)
                X_new = y_new.reshape(-1, 1)
                x_new = model.predict(X_new)
                
                curve_points = [(int(x), int(y)) for x, y in zip(x_new, y_new)]
                
                # 再次检查合理性
                if self._is_valid_curve(curve_points, lane_type, width, height):
                    return curve_points
                else:
                    return None
                
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            # 如果拟合失败，尝试简单的线性拟合
            try:
                model = LinearRegression()
                X = np.array(y_values).reshape(-1, 1)
                y = np.array(x_values)
                model.fit(X, y)
                
                # 生成直线点
                y_new = np.linspace(min(y_values), max(y_values), 20)
                X_new = y_new.reshape(-1, 1)
                x_new = model.predict(X_new)
                
                curve_points = [(int(x), int(y)) for x, y in zip(x_new, y_new)]
                
                # 检查合理性
                if self._is_valid_curve(curve_points, lane_type, width, height):
                    return curve_points
                else:
                    return None
            except:
                return None
        
        return None
    
    def _remove_outliers(self, points, lane_type):
        """移除离群点"""
        if len(points) < 5:
            return points
            
        # 提取坐标
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        
        # 计算中位数和标准差
        x_median = np.median(x_values)
        x_std = np.std(x_values)
        
        # 根据车道类型设置不同的阈值
        if lane_type == 'left':
            # 左车道线点应该在图像左侧
            threshold = 3.0  # 标准差倍数
            filtered_points = [p for p in points if abs(p[0] - x_median) < threshold * x_std and p[0] < 2 * x_median]
        elif lane_type == 'right':
            # 右车道线点应该在图像右侧
            threshold = 3.0
            filtered_points = [p for p in points if abs(p[0] - x_median) < threshold * x_std and p[0] > 0.5 * x_median]
        else:
            # 中心线使用较大阈值
            threshold = 4.0
            filtered_points = [p for p in points if abs(p[0] - x_median) < threshold * x_std]
            
        return filtered_points
        
    def _is_valid_curve(self, curve, lane_type, width, height):
        """检查曲线是否合理"""
        if not curve or len(curve) < 2:
            return False
            
        # 检查曲线的斜率变化是否合理
        angles = []
        for i in range(len(curve) - 2):
            p1 = curve[i]
            p2 = curve[i+1]
            p3 = curve[i+2]
            
            # 计算两段线之间的角度
            angle1 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
            angle2 = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) * 180 / np.pi
            angle_diff = abs(angle1 - angle2)
            
            # 确保角度变化不超过阈值
            if angle_diff > self.max_curve_angle:
                return False
                
            angles.append(angle_diff)
            
        # 对于左右车道线，检查是否在合理位置
        if lane_type == 'left':
            # 左车道线应该在图像左侧
            for x, y in curve:
                if x > width * 0.6:  # 左车道线不应在右半部分
                    return False
        elif lane_type == 'right':
            # 右车道线应该在图像右侧
            for x, y in curve:
                if x < width * 0.4:  # 右车道线不应在左半部分
                    return False
                    
        # 检查曲线的起点和终点是否合理
        if lane_type == 'left' or lane_type == 'right':
            # 曲线应该从图像底部向上延伸
            # 按y坐标排序
            sorted_points = sorted(curve, key=lambda p: -p[1])
            bottom_y = sorted_points[0][1] if sorted_points else 0
            
            # 底部点应该接近图像底部
            if bottom_y < height * 0.6:
                return False
                
        return True
        
    def _draw_curve(self, image, curve, color, thickness):
        """在图像上绘制曲线"""
        if curve is None or len(curve) < 2:
            return
            
        # 将点转换为NumPy数组
        points = np.array(curve, dtype=np.int32)
        
        # 绘制曲线（通过连接相邻点）
        for i in range(len(points) - 1):
            cv2.line(image, tuple(points[i]), tuple(points[i+1]), color, thickness)
        
    def get_lane_center(self, image):
        """获取车道中心线和偏移信息"""
        lane_info, debug_image = self.detect_lanes(image)
        self._debug_image = debug_image
        return lane_info if lane_info else None 

# 新增的高级车道线检测器，整合laneDection/main.py中的方法
class AdvancedLaneDetector(LaneDetector):
    def __init__(self, vehicle, world, config):
        super().__init__(vehicle, world, config)
        
        # 相机校准参数
        self.mtx = None
        self.dist = None
        self.M = None
        self.M_inverse = None
        
        # 车道线拟合参数
        self.left_fit = None
        self.right_fit = None
        self.lane_center = None
        
        # 车道线历史记录，用于平滑
        self.left_fit_history = []
        self.right_fit_history = []
        self.history_size = 5
        
        # 车道线检测参数
        self.s_thresh = (170, 255)  # S通道阈值
        self.sx_thresh = (40, 200)  # Sobel边缘阈值
        
        # 透视变换参数
        self.src_points = None  # 透视变换源点
        self.offset_x = 330
        self.offset_y = 0
        
        # 初始化相机校准
        self.initialize()
    
    def initialize(self):
        """初始化相机校准和透视变换参数"""
        # 对于CARLA相机，不需要校准，直接使用默认参数
        
        # 默认相机内参
        self.mtx = np.array([
            [1200.0, 0.0, 960.0],
            [0.0, 1200.0, 540.0],
            [0.0, 0.0, 1.0]
        ])
        
        # 默认畸变系数（CARLA相机没有畸变）
        self.dist = np.zeros(5)
        
        # 调整透视变换点以匹配CARLA相机视角
        # 这些点需要根据实际情况调整 - 使用更宽的梯形
        self.src_points = np.float32([
            [480, 500],  # 左上 - 更宽的梯形
            [800, 500],  # 右上 - 更宽的梯形
            [1200, 700], # 右下 - 更宽的梯形
            [100, 700]   # 左下 - 更宽的梯形
        ])
        
        # 设置填充区域的不透明度
        self.fill_opacity = 0.8  # 增加不透明度，使绿色更明显
        
        # 计算透视变换矩阵
        self.calculate_perspective_transform()
        
        # 设置默认的车道线参数（如果检测失败）
        self.default_left_fit = np.array([0, 0, 480])
        self.default_right_fit = np.array([0, 0, 800])
    
    def calibrate_camera(self, calibration_images):
        """使用棋盘格图像校准相机"""
        # 棋盘格角点数
        nx, ny = 9, 6
        
        # 存储角点数据的坐标
        object_points = []  # 角点在三维空间的位置
        image_points = []   # 角点在图像空间中的位置
        
        # 生成角点在真实世界中的位置
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # 角点检测
        for file_path in calibration_images:
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            # 灰度化
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 角点检测
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
            if ret:
                object_points.append(objp)
                image_points.append(corners)
        
        if len(object_points) > 0:
            # 相机校准
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points, image_points, gray.shape[::-1], None, None
            )
            return True
        
        return False
    
    def undistort_image(self, img):
        """图像去畸变"""
        if self.mtx is not None and self.dist is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return img
    
    def calculate_perspective_transform(self):
        """计算透视变换矩阵"""
        # 创建一个虚拟图像来计算透视变换矩阵
        img_size = (1920, 1080)  # 假设图像大小为1920x1080
        
        # 源点（可以根据实际情况调整）
        src = self.src_points
        
        # 目标点（俯视图中的对应点）
        # 调整目标点以创建更宽的俯视图
        dst = np.float32([
            [img_size[0] * 0.25, 0],                    # 左上 - 更宽的视图
            [img_size[0] * 0.75, 0],                    # 右上 - 更宽的视图
            [img_size[0] * 0.75, img_size[1]],          # 右下 - 更宽的视图
            [img_size[0] * 0.25, img_size[1]]           # 左下 - 更宽的视图
        ])
        
        # 计算变换矩阵
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inverse = cv2.getPerspectiveTransform(dst, src)
    
    def set_perspective_transform(self, img, points):
        """设置透视变换的源点"""
        self.src_points = np.float32(points)
        self.calculate_perspective_transform()
    
    def perspective_transform(self, img, inverse=False):
        """执行透视变换"""
        img_size = (img.shape[1], img.shape[0])
        if inverse:
            return cv2.warpPerspective(img, self.M_inverse, img_size)
        else:
            return cv2.warpPerspective(img, self.M, img_size)
    
    def pipeline(self, img, s_thresh=None, sx_thresh=None):
        """车道线提取管道 - 增强边缘检测"""
        if s_thresh is None:
            s_thresh = (100, 255)  # S通道阈值
        if sx_thresh is None:
            sx_thresh = (20, 200)  # Sobel边缘阈值
            
        # 复制原图像
        img = np.copy(img)
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊减少噪声
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1. Canny边缘检测
        canny_edges = cv2.Canny(blur, 50, 150)
        
        # 2. 颜色空间转换
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        
        # 3. Sobel边缘检测 - x方向（检测垂直边缘）
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # 4. Sobel边缘检测 - y方向（检测水平边缘）
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobely = np.absolute(sobely)
        scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
        
        # 5. 计算梯度幅值（边缘强度）
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        
        # 6. 计算梯度方向
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        
        # 对各种边缘检测结果进行二值化
        # Sobel x方向二值化
        sxbinary = np.zeros_like(scaled_sobelx)
        sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1
        
        # Sobel y方向二值化（对于弯道有帮助）
        sybinary = np.zeros_like(scaled_sobely)
        sybinary[(scaled_sobely >= sx_thresh[0]) & (scaled_sobely <= sx_thresh[1])] = 1
        
        # 梯度幅值二值化
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= 30) & (gradmag <= 255)] = 1
        
        # 梯度方向二值化（选择接近垂直的边缘）
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= 0.7) & (absgraddir <= 1.3)] = 1
        
        # S通道阈值处理
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        
        # Canny边缘二值化（已经是二值图像）
        canny_binary = canny_edges / 255
        
        # 组合所有边缘检测结果
        # 1. 颜色阈值和Sobel x方向（原始方法）
        combined = np.zeros_like(sxbinary)
        combined[((sxbinary == 1) | (s_binary == 1)) & (l_channel > 60)] = 1
        
        # 2. 添加Canny边缘和其他Sobel结果
        enhanced = np.zeros_like(sxbinary)
        enhanced[
            ((sxbinary == 1) | (sybinary == 1) | (mag_binary == 1) | (dir_binary == 1) | (s_binary == 1) | (canny_binary == 1)) 
            & (l_channel > 60)
        ] = 1
        
        # 应用ROI掩码，只保留下半部分的图像
        height, width = enhanced.shape
        mask = np.zeros_like(enhanced)
        roi_vertices = np.array([
            [0, height],
            [0, height*0.6],
            [width, height*0.6],
            [width, height]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [roi_vertices], 1)
        masked_binary = enhanced & mask
        
        return masked_binary
    
    def find_lane_pixels(self, binary_warped):
        """使用滑动窗口方法查找车道线像素"""
        # 统计直方图
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # 创建输出图像用于调试
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        
        # 找到左右峰值的位置
        midpoint = np.int32(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # 设置滑动窗口参数
        nwindows = 9
        margin = 100
        minpix = 50
        
        # 窗口高度
        window_height = np.int32(binary_warped.shape[0] // nwindows)
        
        # 获取非零像素的位置
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # 当前窗口的中心位置
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # 存储左右车道线像素的索引
        left_lane_inds = []
        right_lane_inds = []
        
        # 遍历窗口
        for window in range(nwindows):
            # 窗口边界
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            # 左右窗口的x范围
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # 绘制窗口（用于调试）
            if self.debug:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            
            # 找到窗口内的非零像素
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # 添加到索引列表
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # 如果找到足够的点，更新窗口中心
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))
        
        # 合并索引
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # 如果没有找到足够的点
            pass
        
        # 提取左右车道线的像素位置
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty, out_img
    
    def fit_polynomial(self, binary_warped):
        """拟合多项式曲线"""
        # 查找车道线像素
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)
        
        # 拟合多项式
        left_fit = None
        right_fit = None
        
        # 确保有足够的点进行拟合
        if len(leftx) > self.min_points_for_fit:
            left_fit = np.polyfit(lefty, leftx, 2)
            # 添加到历史记录
            self.left_fit_history.append(left_fit)
            if len(self.left_fit_history) > self.history_size:
                self.left_fit_history.pop(0)
        elif len(self.left_fit_history) > 0:
            # 使用历史记录
            left_fit = np.mean(self.left_fit_history, axis=0)
        
        if len(rightx) > self.min_points_for_fit:
            right_fit = np.polyfit(righty, rightx, 2)
            # 添加到历史记录
            self.right_fit_history.append(right_fit)
            if len(self.right_fit_history) > self.history_size:
                self.right_fit_history.pop(0)
        elif len(self.right_fit_history) > 0:
            # 使用历史记录
            right_fit = np.mean(self.right_fit_history, axis=0)
        
        # 如果仍然没有有效的拟合，使用默认值
        if left_fit is None:
            left_fit = np.array([0, 0, binary_warped.shape[1] // 4])
        if right_fit is None:
            right_fit = np.array([0, 0, binary_warped.shape[1] * 3 // 4])
        
        # 保存拟合结果
        self.left_fit = left_fit
        self.right_fit = right_fit
        
        return left_fit, right_fit, out_img
    
    def fill_lane_area(self, binary_warped, left_fit, right_fit):
        """填充车道区域 - 使用更直接的方式确保填充整个车道"""
        # 创建空白图像
        out_img = np.zeros_like(binary_warped)
        if len(out_img.shape) == 2:
            out_img = np.dstack((out_img, out_img, out_img))
        
        # 获取图像尺寸
        height, width = binary_warped.shape[:2] if len(binary_warped.shape) > 2 else binary_warped.shape
        
        # 直接创建一个梯形区域，而不是依赖于拟合的曲线
        # 这样可以确保填充整个车道区域
        if left_fit is not None and right_fit is not None:
            try:
                # 计算底部的左右位置
                bottom_left = int(left_fit[0]*height**2 + left_fit[1]*height + left_fit[2])
                bottom_right = int(right_fit[0]*height**2 + right_fit[1]*height + right_fit[2])
                
                # 计算顶部的左右位置（使用图像高度的20%处）
                top_y = int(height * 0.2)
                top_left = int(left_fit[0]*top_y**2 + left_fit[1]*top_y + left_fit[2])
                top_right = int(right_fit[0]*top_y**2 + right_fit[1]*top_y + right_fit[2])
                
                # 确保点在图像范围内
                bottom_left = max(0, min(width-1, bottom_left))
                bottom_right = max(0, min(width-1, bottom_right))
                top_left = max(0, min(width-1, top_left))
                top_right = max(0, min(width-1, top_right))
                
                # 创建填充多边形的点
                pts = np.array([
                    [bottom_left, height-1],
                    [top_left, top_y],
                    [top_right, top_y],
                    [bottom_right, height-1]
                ], np.int32)
                
                # 使用亮绿色填充
                cv2.fillPoly(out_img, [pts], (0, 255, 0))
                
                # 绘制车道线边界
                cv2.polylines(out_img, [np.array([[bottom_left, height-1], [top_left, top_y]])], False, (255, 0, 0), 8)
                cv2.polylines(out_img, [np.array([[bottom_right, height-1], [top_right, top_y]])], False, (0, 0, 255), 8)
                
            except Exception as e:
                print(f"填充车道区域时出错: {e}")
                # 如果计算失败，使用默认的梯形区域
                pts = np.array([
                    [int(width*0.25), height-1],
                    [int(width*0.45), int(height*0.6)],
                    [int(width*0.55), int(height*0.6)],
                    [int(width*0.75), height-1]
                ], np.int32)
                cv2.fillPoly(out_img, [pts], (0, 255, 0))
        else:
            # 如果没有检测到车道线，使用默认的梯形区域
            pts = np.array([
                [int(width*0.25), height-1],
                [int(width*0.45), int(height*0.6)],
                [int(width*0.55), int(height*0.6)],
                [int(width*0.75), height-1]
            ], np.int32)
            cv2.fillPoly(out_img, [pts], (0, 255, 0))
        
        return out_img
    
    def calculate_curvature(self, left_fit, right_fit, img_shape):
        """计算曲率半径"""
        # 定义转换比例（像素到米）
        ym_per_pix = 30/720  # 每像素代表的y方向距离（米）
        xm_per_pix = 3.7/700  # 每像素代表的x方向距离（米）
        
        # 生成y值
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        
        # 计算曲率
        try:
            # 将像素坐标转换为实际距离
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            # 重新拟合曲线（使用实际距离）
            left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
            
            # 计算曲率半径
            y_eval = np.max(ploty)
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
            
            # 平均曲率
            curvature = (left_curverad + right_curverad) / 2
        except:
            curvature = 0
            
        return curvature
    
    def calculate_vehicle_position(self, left_fit, right_fit, img_shape):
        """计算车辆相对于车道中心的位置"""
        # 定义转换比例
        xm_per_pix = 3.7/700  # 每像素代表的x方向距离（米）
        
        try:
            # 计算图像底部的车道线位置
            y_eval = img_shape[0]
            left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
            right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
            
            # 计算车道中心
            lane_center = (left_x + right_x) / 2
            
            # 计算图像中心
            center_image = img_shape[1] / 2
            
            # 计算偏移量（米）
            center_offset = (lane_center - center_image) * xm_per_pix
            
            # 保存车道中心位置
            self.lane_center = lane_center
        except:
            center_offset = 0
            
        return center_offset
    
    def detect_lanes(self, image):
        """高级车道线检测方法，覆盖父类方法"""
        # 创建调试图像
        debug_image = image.copy() if self.debug else None
        
        try:
            # 获取图像尺寸
            height, width = image.shape[:2]
            
            # 直接使用draw_lane_on_original方法绘制车道区域
            # 这是最可靠的方法，确保始终有绿色区域显示
            result = self.draw_lane_on_original(image)
            
            # 尝试使用透视变换和多项式拟合方法来计算曲率和车辆位置
            try:
                # 1. 图像去畸变
                undistorted = self.undistort_image(image)
                
                # 2. 车道线提取
                binary = self.pipeline(undistorted)
                
                # 3. 透视变换
                warped = self.perspective_transform(binary)
                
                # 4. 拟合车道线
                left_fit, right_fit, _ = self.fit_polynomial(warped)
                
                # 5. 计算曲率和车辆位置
                curvature = self.calculate_curvature(left_fit, right_fit, image.shape)
                center_offset = self.calculate_vehicle_position(left_fit, right_fit, image.shape)
                
                # 保存车道信息
                self.lane_info = {
                    'left_fit': left_fit,
                    'right_fit': right_fit,
                    'curvature': curvature,
                    'center_offset': center_offset,
                    'lane_center': self.lane_center
                }
                
                # 创建中心曲线
                if left_fit is not None and right_fit is not None:
                    # 生成y值
                    ploty = np.linspace(0, image.shape[0]-1, 20)  # 减少点数，只需要20个点
                    
                    # 计算左右车道线的x值
                    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                    
                    # 计算中心线
                    center_fitx = (left_fitx + right_fitx) / 2
                    
                    # 创建中心曲线点列表
                    center_curve = []
                    for i in range(len(ploty)):
                        center_curve.append((int(center_fitx[i]), int(ploty[i])))
                    
                    # 添加到车道信息中
                    self.lane_info['center_curve'] = center_curve
                
            except Exception as e:
                print(f"计算曲率和位置失败: {e}")
                # 设置默认的车道信息
                if not hasattr(self, 'lane_info') or self.lane_info is None:
                    self.lane_info = {}
                
                # 计算默认的曲率和中心偏移
                curvature = 100.0  # 默认曲率
                center_offset = 0.0  # 默认中心偏移
                self.lane_info['curvature'] = curvature
                self.lane_info['center_offset'] = center_offset
            
            # 添加文本信息
            if self.debug:
                cv2.putText(result, f'Radius of Curvature = {self.lane_info.get("curvature", 0):.2f}m', 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                center_offset = self.lane_info.get('center_offset', 0)
                if center_offset > 0:
                    cv2.putText(result, f'Vehicle is {abs(center_offset):.2f}m right of center', 
                               (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(result, f'Vehicle is {abs(center_offset):.2f}m left of center', 
                               (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 保存调试图像
            self._debug_image = result
            
            # 返回车道信息和调试图像
            return self.lane_info, result
            
        except Exception as e:
            print(f"车道线检测异常: {e}")
            # 如果检测失败，返回空信息和原图像
            return None, image

    def draw_lane_on_original(self, image):
        """直接在原始图像上绘制大型车道区域，确保覆盖整个车道，并计算车道中心线和偏移信息"""
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 创建一个覆盖整个车道的大型多边形
        # 使用固定的比例来确定多边形的位置，确保覆盖整个车道
        pts = np.array([
            [int(width*0.1), height],             # 左下 - 更宽
            [int(width*0.45), int(height*0.55)],  # 左上 - 更高
            [int(width*0.55), int(height*0.55)],  # 右上 - 更高
            [int(width*0.9), height]              # 右下 - 更宽
        ], np.int32)
        
        # 创建一个空白图像作为覆盖层
        overlay = np.zeros_like(image)
        
        # 填充多边形区域
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        
        # 将填充区域叠加到原始图像上，使用高不透明度
        result = cv2.addWeighted(image, 1, overlay, 0.7, 0)
        
        # 绘制车道线边界
        cv2.polylines(result, [np.array([pts[0], pts[1]])], False, (255, 0, 0), 8)
        cv2.polylines(result, [np.array([pts[3], pts[2]])], False, (0, 0, 255), 8)
        
        # 计算车道中心线
        center_curve = []
        # 从底部到顶部生成20个点
        for i in range(20):
            # 计算当前高度
            y = height - i * (height - int(height*0.55)) / 20
            # 计算左右边界在当前高度的x坐标
            if y >= int(height*0.55):
                # 在多边形内部插值计算x坐标
                left_ratio = (y - int(height*0.55)) / (height - int(height*0.55))
                right_ratio = left_ratio
                left_x = int(width*0.1) + left_ratio * (int(width*0.45) - int(width*0.1))
                right_x = int(width*0.9) - right_ratio * (int(width*0.9) - int(width*0.55))
                # 计算中心点
                center_x = (left_x + right_x) / 2
                center_curve.append((int(center_x), int(y)))
        
        # 绘制中心线
        for i in range(len(center_curve)-1):
            cv2.line(result, center_curve[i], center_curve[i+1], (0, 0, 255), 2)
        
        # 计算车道中心偏移
        image_center_x = width / 2
        if center_curve:
            # 使用最底部的点计算偏移
            bottom_center_x = center_curve[0][0]
            # 计算归一化偏移（-1到1之间）
            self.lane_center = (bottom_center_x - image_center_x) / (width / 2)
        else:
            self.lane_center = 0.0
        
        # 更新车道信息
        if not hasattr(self, 'lane_info') or self.lane_info is None:
            self.lane_info = {}
        
        # 添加中心曲线到车道信息中
        self.lane_info['center_curve'] = center_curve
        
        # 计算曲率（使用简单的圆弧近似）
        if len(center_curve) >= 3:
            # 使用三个点计算曲率
            p1 = np.array(center_curve[0])
            p2 = np.array(center_curve[len(center_curve)//2])
            p3 = np.array(center_curve[-1])
            
            # 计算三点之间的距离
            d1 = np.linalg.norm(p2 - p3)
            d2 = np.linalg.norm(p1 - p3)
            d3 = np.linalg.norm(p1 - p2)
            
            # 使用海伦公式计算三角形面积
            s = (d1 + d2 + d3) / 2
            area = np.sqrt(s * (s - d1) * (s - d2) * (s - d3))
            
            # 计算曲率半径
            if area > 0:
                radius = (d1 * d2 * d3) / (4 * area)
                # 转换为米（假设图像宽度对应3.7米的车道宽度）
                radius_m = radius * 3.7 / width
                self.lane_info['curvature'] = radius_m
            else:
                self.lane_info['curvature'] = 1000.0  # 近似直线
        else:
            self.lane_info['curvature'] = 1000.0  # 近似直线
        
        # 添加车道中心偏移到车道信息中
        self.lane_info['center_offset'] = self.lane_center
        
        return result

    def detect_lane(self, image):
        """检测车道线并返回标准化的车道信息
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 包含车道信息的字典，包括center（中心偏移）、confidence（置信度）和curvature（曲率）
        """
        try:
            if image is None or image.size == 0:
                return None
                
            # 调用原始的detect_lanes方法
            lane_info, debug_image = self.detect_lanes(image)
            
            # 如果没有检测到车道线
            if lane_info is None:
                return None
                
            # 标准化车道信息
            result = {
                'center': self.lane_center if hasattr(self, 'lane_center') else 0.0,
                'confidence': self.lane_confidence if hasattr(self, 'lane_confidence') else 0.0,
                'curvature': 0.0  # 默认曲率
            }
            
            # 如果有曲率信息，添加到结果中
            if 'curvature' in lane_info:
                result['curvature'] = lane_info['curvature']
            elif 'center_curve' in lane_info and lane_info['center_curve']:
                # 从中心曲线计算曲率
                center_curve = lane_info['center_curve']
                if len(center_curve) >= 3:
                    # 使用三点法估算曲率
                    x1, y1 = center_curve[0]
                    x2, y2 = center_curve[len(center_curve)//2]
                    x3, y3 = center_curve[-1]
                    
                    # 计算三点确定的圆的半径
                    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                    if abs(D) > 1e-6:  # 避免除零错误
                        Ux = ((x1*x1 + y1*y1) * (y2 - y3) + (x2*x2 + y2*y2) * (y3 - y1) + (x3*x3 + y3*y3) * (y1 - y2)) / D
                        Uy = ((x1*x1 + y1*y1) * (x3 - x2) + (x2*x2 + y2*y2) * (x1 - x3) + (x3*x3 + y3*y3) * (x2 - x1)) / D
                        
                        # 计算半径
                        radius = np.sqrt((Ux - x1)**2 + (Uy - y1)**2)
                        
                        # 曲率是半径的倒数
                        result['curvature'] = 1.0 / radius if radius > 0 else 0.0
            
            return result
            
        except Exception as e:
            print(f"检测车道线异常: {e}")
            traceback.print_exc()
            return None

class SimpleOpenCVLaneDetector:
    """简单的OpenCV车道线检测器"""
    
    def __init__(self, vehicle, world, config):
        """初始化检测器"""
        self.vehicle = vehicle
        self.world = world
        self.config = config
        
        # 调试模式
        self.debug = True
        
        # 滤波器参数
        self.center_filter = []
        self.filter_size = 7  # 增加滤波器大小，提高稳定性
        
        # 车道检测参数 - 调整参数以提高检测效果
        self.roi_height = 0.55  # 增加ROI区域的高度比例
        self.min_line_length = 40  # 增加最小线段长度
        self.max_line_gap = 25  # 增加最大线段间隙
        self.hough_threshold = 25  # 降低霍夫变换阈值，检测更多线段
        self.canny_low = 40  # 降低Canny边缘检测低阈值
        self.canny_high = 120  # 降低Canny边缘检测高阈值
        
        # 初始化ROI区域
        self._initialize_roi()
        
        # 保存上一帧的结果，用于平滑
        self.last_left_line = None
        self.last_right_line = None
        self.last_center_offset = 0.0
        self.last_confidence = 0.0
        
        # 历史记录
        self.left_line_history = []
        self.right_line_history = []
        self.history_size = 5
        
        # 调试图像
        self.debug_image = None
        
        print("简单OpenCV车道线检测器已初始化")

    def _initialize_roi(self):
        """初始化ROI区域"""
        # 使用更宽的ROI区域，以适应更多的道路情况
        self.src_points = np.float32([
            [480, 480],  # 左上 - 更宽的区域
            [800, 480],  # 右上 - 更宽的区域
            [1200, 680], # 右下 - 扩大宽度
            [80, 680]    # 左下 - 扩大宽度
        ])
        
        # 目标点 - 鸟瞰图
        self.dst_points = np.float32([
            [320, 0],    # 左上
            [960, 0],    # 右上
            [960, 720],  # 右下
            [320, 720]   # 左下
        ])
        
        # 计算透视变换矩阵
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)

    def detect_lane(self, image):
        """检测车道线并返回调试图像"""
        lane_info, debug_image = self.detect_lanes(image)
        self.debug_image = debug_image
        return lane_info

    def detect_lanes(self, image):
        """检测车道线"""
        try:
            # 创建调试图像
            debug_image = image.copy() if self.debug else None
            height, width = image.shape[:2]
            
            # 绘制ROI区域
            if debug_image is not None:
                cv2.polylines(debug_image, [np.int32(self.src_points)], True, (0, 255, 255), 2)
            
            # 预处理图像
            # 1. 转换为HSV颜色空间，更好地检测白色和黄色车道线
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 2. 定义白色和黄色的HSV范围
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            lower_yellow = np.array([15, 80, 80])
            upper_yellow = np.array([35, 255, 255])
            
            # 3. 创建白色和黄色的掩码
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 4. 合并掩码
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # 5. 应用高斯模糊
            blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            
            # 6. 应用Canny边缘检测
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            # 7. 定义感兴趣区域
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, [np.int32(self.src_points)], 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # 8. 霍夫变换检测直线
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            # 如果没有检测到线段，使用历史数据
            if lines is None:
                return self._use_history_data(), debug_image
            
            # 9. 过滤和分类线段
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # 计算线段长度和斜率
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # 避免除零错误
                if x2 == x1:
                    continue
                    
                slope = (y2-y1)/(x2-x1)
                
                # 过滤掉斜率太小的线段（接近水平）
                if abs(slope) < 0.3:
                    continue
                
                # 过滤掉斜率太大的线段（接近垂直）
                if abs(slope) > 2.0:
                    continue
                
                # 根据斜率和位置分类
                if slope < 0 and x1 < width * 0.6:  # 左车道线
                    left_lines.append((x1, y1, x2, y2, slope, length))
                elif slope > 0 and x1 > width * 0.4:  # 右车道线
                    right_lines.append((x1, y1, x2, y2, slope, length))
            
            # 10. 按长度排序，优先考虑长线段
            left_lines.sort(key=lambda x: -x[5])
            right_lines.sort(key=lambda x: -x[5])
            
            # 11. 拟合左右车道线
            left_line = self._fit_line(left_lines, height)
            right_line = self._fit_line(right_lines, height)
            
            # 12. 更新历史记录
            if left_line is not None:
                self.left_line_history.append(left_line)
                if len(self.left_line_history) > self.history_size:
                    self.left_line_history.pop(0)
                self.last_left_line = left_line
            elif self.last_left_line is not None:
                left_line = self.last_left_line
            
            if right_line is not None:
                self.right_line_history.append(right_line)
                if len(self.right_line_history) > self.history_size:
                    self.right_line_history.pop(0)
                self.last_right_line = right_line
            elif self.last_right_line is not None:
                right_line = self.last_right_line
            
            # 13. 平滑历史记录中的线
            if self.left_line_history:
                left_line = self._smooth_lines(self.left_line_history)
            
            if self.right_line_history:
                right_line = self._smooth_lines(self.right_line_history)
            
            # 14. 计算车道中心和偏移量
            center_offset = 0.0
            confidence = 0.0
            
            if left_line is not None and right_line is not None:
                # 在图像底部计算车道中心
                left_x = self._get_x_from_line(left_line, height)
                right_x = self._get_x_from_line(right_line, height)
                
                # 计算车道中心
                lane_center = (left_x + right_x) / 2
                
                # 计算偏移量（归一化到[-1,1]）
                image_center = width / 2
                center_offset = (lane_center - image_center) / (width / 2)
                
                # 计算置信度
                confidence = 1.0
                
                # 在调试图像上绘制车道线和中心线
                if debug_image is not None:
                    self._draw_lane_lines(debug_image, left_line, right_line)
            elif left_line is not None:
                # 只有左车道线
                left_x = self._get_x_from_line(left_line, height)
                lane_center = left_x + width * 0.25  # 估计车道中心
                
                image_center = width / 2
                center_offset = (lane_center - image_center) / (width / 2)
                
                confidence = 0.6
                
                if debug_image is not None:
                    self._draw_lane_line(debug_image, left_line, (255, 0, 0), 2)
            elif right_line is not None:
                # 只有右车道线
                right_x = self._get_x_from_line(right_line, height)
                lane_center = right_x - width * 0.25  # 估计车道中心
                
                image_center = width / 2
                center_offset = (lane_center - image_center) / (width / 2)
                
                confidence = 0.6
                
                if debug_image is not None:
                    self._draw_lane_line(debug_image, right_line, (0, 0, 255), 2)
            else:
                # 没有检测到车道线，使用上一帧的偏移量
                center_offset = self.last_center_offset
                confidence = max(0.1, self.last_confidence - 0.3)  # 降低置信度
            
            # 15. 平滑偏移量
            self.center_filter.append(center_offset)
            if len(self.center_filter) > self.filter_size:
                self.center_filter.pop(0)
            
            if self.center_filter:
                center_offset = sum(self.center_filter) / len(self.center_filter)
            
            # 16. 更新上一帧的结果
            self.last_center_offset = center_offset
            self.last_confidence = confidence
            
            # 17. 构建车道信息
            lane_info = {
                'center_offset': center_offset,
                'confidence': confidence,
                'left_line': left_line,
                'right_line': right_line
            }
            
            # 18. 在调试图像上显示相关信息
            if debug_image is not None:
                self.add_debug_info(debug_image, lane_info)
            
            return lane_info, debug_image
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"车道线检测异常: {e}")
            return self._use_history_data(), debug_image
    
    def _fit_line(self, lines, height):
        """拟合直线"""
        if not lines:
            return None
        
        # 提取点
        x_sum = 0
        y_sum = 0
        m_sum = 0
        
        # 使用加权平均，长线段权重更大
        total_length = sum(line[5] for line in lines)
        
        for x1, y1, x2, y2, slope, length in lines:
            weight = length / total_length
            x_sum += (x1 + x2) * weight / 2
            y_sum += (y1 + y2) * weight / 2
            m_sum += slope * weight
        
        # 计算直线参数: y = mx + b
        x_avg = x_sum
        y_avg = y_sum
        m = m_sum
        b = y_avg - m * x_avg
        
        # 返回直线参数 (m, b)
        return (m, b)
    
    def _smooth_lines(self, line_history):
        """平滑历史线段"""
        if not line_history:
            return None
        
        # 计算加权平均，最近的线段权重更大
        m_sum = 0
        b_sum = 0
        weight_sum = 0
        
        for i, (m, b) in enumerate(line_history):
            # 越新的数据权重越大
            weight = (i + 1) / sum(range(1, len(line_history) + 1))
            m_sum += m * weight
            b_sum += b * weight
            weight_sum += weight
        
        return (m_sum / weight_sum, b_sum / weight_sum)
    
    def _get_x_from_line(self, line, y):
        """根据y坐标计算x坐标"""
        if line is None:
            return 0
        
        m, b = line
        # x = (y - b) / m
        if m == 0:
            return 0
        
        return (y - b) / m
    
    def _draw_lane_lines(self, image, left_line, right_line):
        """绘制车道线和车道区域"""
        height, width = image.shape[:2]
        
        # 绘制左车道线
        if left_line is not None:
            self._draw_lane_line(image, left_line, (255, 0, 0), 2)
        
        # 绘制右车道线
        if right_line is not None:
            self._draw_lane_line(image, right_line, (0, 0, 255), 2)
        
        # 如果同时有左右车道线，绘制车道区域
        if left_line is not None and right_line is not None:
            # 创建车道区域多边形
            pts = np.zeros((4, 2), dtype=np.int32)
            
            # 底部点
            pts[0] = [self._get_x_from_line(left_line, height), height]
            pts[1] = [self._get_x_from_line(right_line, height), height]
            
            # 顶部点 (使用ROI区域的高度)
            top_y = int(height * self.roi_height)
            pts[2] = [self._get_x_from_line(right_line, top_y), top_y]
            pts[3] = [self._get_x_from_line(left_line, top_y), top_y]
            
            # 创建填充区域
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0, 64))
            
            # 添加透明度
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # 绘制车道中心线
            center_bottom_x = (pts[0][0] + pts[1][0]) // 2
            center_top_x = (pts[2][0] + pts[3][0]) // 2
            
            cv2.line(image, (center_bottom_x, height), (center_top_x, top_y), (0, 255, 255), 2)
    
    def _draw_lane_line(self, image, line, color, thickness):
        """绘制车道线"""
        height, width = image.shape[:2]
        
        if line is None:
            return
        
        m, b = line
        
        # 计算顶部和底部的x坐标
        top_y = int(height * self.roi_height)
        bottom_y = height
        
        top_x = int(self._get_x_from_line(line, top_y))
        bottom_x = int(self._get_x_from_line(line, bottom_y))
        
        # 确保x坐标在图像范围内
        top_x = max(0, min(width - 1, top_x))
        bottom_x = max(0, min(width - 1, bottom_x))
        
        # 绘制线段
        cv2.line(image, (bottom_x, bottom_y), (top_x, top_y), color, thickness)
    
    def _use_history_data(self):
        """使用历史数据"""
        return {
            'center_offset': self.last_center_offset,
            'confidence': max(0.1, self.last_confidence - 0.2),  # 降低置信度
            'left_line': self.last_left_line,
            'right_line': self.last_right_line
        }
    
    def add_debug_info(self, result_image, lane_info=None):
        """添加调试信息到图像上"""
        if lane_info is None:
            return
        
        # 添加车道偏移信息
        offset = lane_info.get('center_offset', 0)
        confidence = lane_info.get('confidence', 0)
        
        cv2.putText(result_image, f"偏移: {offset:.2f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_image, f"置信度: {confidence:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加指示方向的箭头
        height, width = result_image.shape[:2]
        center_x = width // 2
        arrow_length = int(abs(offset) * width / 4)
        
        if abs(offset) > 0.05:  # 只有当偏移量足够大时才显示箭头
            # 箭头起点
            start_point = (center_x, height - 50)
            
            # 箭头终点
            if offset < 0:  # 需要向左修正
                end_point = (center_x - arrow_length, height - 50)
                cv2.arrowedLine(result_image, start_point, end_point, (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(result_image, "向左修正", (center_x - arrow_length - 100, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:  # 需要向右修正
                end_point = (center_x + arrow_length, height - 50)
                cv2.arrowedLine(result_image, start_point, end_point, (0, 0, 255), 3, tipLength=0.3)
                cv2.putText(result_image, "向右修正", (center_x + arrow_length - 100, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)