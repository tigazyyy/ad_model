import collections
import time

class PIDController:
    """PID控制器"""
    
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.05):
        self.K_P = K_P  # 比例系数
        self.K_I = K_I  # 积分系数
        self.K_D = K_D  # 微分系数
        self.dt = dt    # 时间间隔
        
        self.error_buffer = collections.deque(maxlen=10)
        self.last_time = time.time()
        self.integral = 0.0
        
        # 添加积分限制，防止积分饱和
        self.integral_limit = 1.0
        
        # 添加输出限制
        self.output_limit = 1.0
    
    def step(self, error):
        """更新一个时间步长的PID控制量"""
        # 更新时间间隔
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = self.dt
        self.last_time = current_time
        
        # 保存误差
        self.error_buffer.append(error)
        
        # 计算积分项，并应用积分限制
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        
        # 计算微分项
        derivative = 0
        if len(self.error_buffer) >= 2:
            derivative = (error - self.error_buffer[-2]) / dt
        
        # 计算PID控制量
        P = self.K_P * error
        I = self.K_I * self.integral
        D = self.K_D * derivative
        
        # 应用输出限制
        output = P + I + D
        output = max(-self.output_limit, min(self.output_limit, output))
        
        return output
    
    def reset(self):
        """重置控制器状态"""
        self.error_buffer.clear()
        self.integral = 0.0
        self.last_time = time.time()


class LateralPIDController(PIDController):
    """横向PID控制器，专门用于车辆转向控制"""
    
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.05):
        super().__init__(K_P, K_I, K_D, dt)
        # 转向控制的输出限制为1.0（CARLA中的最大转向值）
        self.output_limit = 1.0
        # 添加调试输出
        self.debug = True
        
    def step(self, error):
        """更新转向控制量，应用非线性映射以提高小误差的响应"""
        # 使用基本PID计算控制量
        control = super().step(error)
        
        if self.debug:
            print(f"PID控制器: 误差={error:.3f}, 原始输出={control:.3f}")
        
        # 应用非线性映射，使小误差有更大的响应
        if abs(control) < 0.1:
            # 小误差区域，大幅提高响应
            control = control * 2.0
        elif abs(control) > 0.5:
            # 大误差区域，保持较大响应
            control = 0.5 + 0.8 * (control - 0.5) if control > 0 else -0.5 + 0.8 * (control + 0.5)
        else:
            # 中等误差区域，适度增强
            control = control * 1.5
        
        # 确保最小响应
        if abs(error) > 0.05 and abs(control) < 0.1:
            control = 0.1 if error > 0 else -0.1
            if self.debug:
                print(f"应用最小响应: {control:.3f}")
        
        # 确保在限制范围内
        output = max(-self.output_limit, min(self.output_limit, control))
        
        if self.debug:
            print(f"PID控制器: 最终输出={output:.3f}")
            
        return output


class LongitudinalPIDController(PIDController):
    """纵向PID控制器，专门用于车辆速度控制"""
    
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.05):
        super().__init__(K_P, K_I, K_D, dt)
        # 油门和刹车的输出限制为1.0
        self.output_limit = 1.0
        
    def step(self, error):
        """更新速度控制量，区分油门和刹车"""
        # 使用基本PID计算控制量
        control = super().step(error)
        
        # 区分油门和刹车
        throttle = max(0.0, control)
        brake = max(0.0, -control)
        
        return throttle, brake