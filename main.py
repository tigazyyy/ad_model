#!/usr/bin/env python
import sys
import os
import glob
import traceback
import pygame
import carla

# 添加CARLA Python API路径
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# 使用绝对导入，不再使用相对导入
from vehicle_control import VehicleController
from config import Config

def main():
    """主函数"""
    pygame.init()
    pygame.display.set_caption("CARLA 自动驾驶系统")
    
    try:
        # 连接CARLA服务器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        print("已连接到CARLA服务器")
        
        # 获取世界
        world = client.get_world()
        
        # 启用同步模式
        settings = world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
            print("已启用同步模式")
        
        # 在初始化控制器之前，清除世界中的所有车辆
        for actor in world.get_actors():
            if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
                actor.destroy()
        
        # 创建配置对象
        config = Config()
        
        # 创建车辆控制器
        controller = VehicleController(client, world, config)
        
        # 运行控制器
        controller.run()
        
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    except Exception as e:
        print(f"程序异常: {e}")
        traceback.print_exc()
    finally:
        pygame.quit()
        print("程序已退出")

if __name__ == '__main__':
    main()