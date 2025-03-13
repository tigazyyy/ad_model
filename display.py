import pygame
import datetime
import math
import numpy as np
import os

class Display:
    def __init__(self):
        """初始化HUD"""
        # 设置字体
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(pygame.font.Font(mono, 20))
        
        # 存储信息
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def render(self, screen, status):
        """渲染HUD"""
        try:
            # 创建半透明的HUD背景
            hud_surface = pygame.Surface((300, 200))
            hud_surface.set_alpha(128)
            hud_surface.fill((0, 0, 0))
            
            # 准备文本
            text_lines = [
                f"速度: {status.get('speed', 0):.1f} km/h",
                f"模式: {status.get('control_mode', 'AUTO')}",
                f"转向: {status.get('steer', 0):.2f}",
                f"车道偏移: {status.get('lane_center', 0):.2f}m",
                f"车道置信度: {status.get('lane_confidence', 0):.2f}"
            ]
            
            # 添加交叉路口和轨迹信息
            if status.get('approaching_junction', False):
                text_lines.append("状态: 接近交叉路口")
            
            if status.get('in_junction', False):
                text_lines.append(f"交叉路口方向: {status.get('junction_direction', 'unknown')}")
            
            if status.get('following_trajectory', False):
                text_lines.append("正在跟随轨迹")
                
                # 如果有轨迹完成百分比信息
                if 'trajectory_progress' in status:
                    text_lines.append(f"轨迹完成: {status.get('trajectory_progress', 0):.0f}%")
            
            # 渲染文本
            y_offset = 10
            for line in text_lines:
                text_surface = self._font_mono.render(line, True, (255, 255, 255))
                hud_surface.blit(text_surface, (10, y_offset))
                y_offset += 25
            
            # 在屏幕左上角显示HUD
            screen.blit(hud_surface, (10, 10))
            
            # 如果正在跟随轨迹，显示轨迹进度条
            if status.get('following_trajectory', False) and 'trajectory_progress' in status:
                progress = status.get('trajectory_progress', 0) / 100.0
                
                # 创建进度条背景
                progress_bg = pygame.Surface((200, 20))
                progress_bg.fill((50, 50, 50))
                
                # 创建进度条
                progress_bar = pygame.Surface((int(200 * progress), 20))
                
                # 根据方向设置不同颜色
                if status.get('junction_direction') == 'left':
                    progress_bar.fill((0, 0, 255))  # 蓝色表示左转
                elif status.get('junction_direction') == 'right':
                    progress_bar.fill((255, 0, 0))  # 红色表示右转
                else:
                    progress_bar.fill((0, 255, 0))  # 绿色表示直行
                
                # 显示进度条
                screen.blit(progress_bg, (10, screen.get_height() - 30))
                screen.blit(progress_bar, (10, screen.get_height() - 30))
                
                # 显示进度文本
                progress_text = self._font_mono.render(f"轨迹进度: {int(progress * 100)}%", True, (255, 255, 255))
                screen.blit(progress_text, (220, screen.get_height() - 30))
            
        except Exception as e:
            print(f"HUD渲染错误: {e}")

class FadingText:
    def __init__(self, font):
        """初始化渐变文本"""
        self.font = font
        self.seconds_left = 0
        self.surface = None
        self.text = ''
        self.color = (255, 255, 255)

    def set_text(self, text, seconds=2.0, color=(255, 255, 255)):
        """设置文本"""
        self.surface = self.font.render(text, True, color)
        self.seconds_left = seconds
        self.text = text
        self.color = color

    def render(self, display):
        """渲染文本"""
        if self.seconds_left > 0:
            alpha = self.seconds_left * 255
            self.surface.set_alpha(alpha)
            display.blit(self.surface, (80, 10))
            self.seconds_left -= 0.05 