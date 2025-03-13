import subprocess
import json
import traceback
import os
import time
from pathlib import Path
import socket
import requests
import base64

class OllamaClient:
    """与Ollama大模型通信的客户端"""
    
    def __init__(self, config):
        """初始化客户端
        
        Args:
            config: 配置对象
        """
        # 修改这一行，使用正确的配置路径
        self.model_name = config.model["model_name"]
        self.timeout = config.model["timeout"]
        print(f"Ollama客户端已初始化，使用模型: {self.model_name}")
        
        # 添加缓存，避免频繁请求模型
        self.decision_cache = {}
        self.cache_timeout = 30  # 缓存有效期（秒）
        
        self.photo_dir = Path(__file__).parent / "photo"
    
    def ask(self, prompt, image_path=None):
        """询问大模型
        
        Args:
            prompt: 提示文本
            image_path: 可选的图像路径
        
        Returns:
            dict: 模型的回复，解析为字典
        """
        try:
            # 使用API方式调用Ollama
            response_text = self.call_ollama_api(prompt, image_path)
            
            # 尝试从响应中提取JSON
            try:
                # 查找JSON开始和结束的位置
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    print(f"无法从响应中提取JSON: {response_text}")
                    return None
            except json.JSONDecodeError:
                print(f"JSON解析错误: {response_text}")
                return None
            
        except Exception as e:
            print(f"询问模型时出错: {e}")
            traceback.print_exc()
            return None
    
    def ask_simple_question(self, status):
        """询问简单问题
        
        Args:
            status: 包含图像路径和问题的字典
        
        Returns:
            dict: 模型回答
        """
        try:
            image_path = status.get('image_path')
            question = status.get('question', '')
            
            if question == 'is_junction':
                prompt = """
分析这张图片，判断车辆是否在交叉路口或接近交叉路口。
请考虑以下特征：
1. 交通信号灯
2. 道路标线（如停止线、人行横道）
3. 道路宽度变化
4. 多个行驶方向选择

请以JSON格式回答：
{
  "is_junction": true/false,
  "reason": "判断理由"
}
"""
                response = self.ask(prompt, image_path)
                return response
            
            return None
        
        except Exception as e:
            print(f"询问简单问题异常: {e}")
            traceback.print_exc()
            return None
    
    def get_decision(self, status):
        """获取驾驶决策
        
        Args:
            status: 状态信息，包含车速、车道状态等
            
        Returns:
            dict: 包含throttle, steering, brake和reason的字典
        """
        try:
            # 检查是否接近交叉路口
            if 'lane_status' in status and 'approaching_junction' in status['lane_status']:
                return self.get_junction_decision(status)
            
            # 构建提示词
            prompt = self._build_prompt(status)
            
            # 获取图像路径
            image_path = status.get('image_path')
            
            # 调用模型
            response = self.ask(prompt, image_path)
            
            # 解析响应
            if response and isinstance(response, dict):
                # 确保包含所有必要字段
                if 'throttle' not in response:
                    response['throttle'] = 0.3
                if 'steering' not in response:
                    response['steering'] = 0.0
                if 'brake' not in response:
                    response['brake'] = 0.0
                if 'reason' not in response:
                    response['reason'] = '模型未提供原因'
                
                return response
            
            # 如果无法获取有效响应，返回默认决策
            return {
                'throttle': 0.3,
                'steering': 0.0,
                'brake': 0.0,
                'reason': '无法获取有效的模型响应'
            }
            
        except Exception as e:
            print(f"获取决策异常: {e}")
            traceback.print_exc()
            
            # 返回安全的默认决策
            return {
                'throttle': 0.3,
                'steering': 0.0,
                'brake': 0.0,
                'reason': f'获取决策异常: {e}'
            }
    
    def get_junction_decision(self, status):
        """获取交叉路口决策
        
        Args:
            status: 包含图像路径和可能方向的字典
            
        Returns:
            dict: 包含direction字段的字典，表示选择的方向
        """
        try:
            # 获取图像路径
            image_path = status.get('image_path')
            if not image_path or not os.path.exists(image_path):
                print("未找到交叉路口图像")
                return {'direction': '直行', 'reason': '未找到交叉路口图像，默认直行'}
            
            print(f"使用图像: {image_path}")
            
            # 获取可能的方向
            directions = status.get('directions', [{'type': '直行', 'confidence': 1.0}])
            
            # 构建可选方向的描述
            direction_desc = ""
            for i, direction in enumerate(directions):
                direction_desc += f"{i+1}. {direction['type']}\n"
            
            # 构建交叉路口提示词
            prompt = f"""
分析这张交叉路口图像，判断应该选择哪个方向行驶。
可选方向:
{direction_desc}

请考虑以下因素：
1. 道路标线和箭头
2. 交通信号灯状态
3. 道路布局和交通规则
4. 车辆当前位置和朝向

请以JSON格式回答，必须包含direction字段：
{{
  "direction": "左转/右转/直行",
  "reason": "判断理由",
  "confidence": 0.8  // 置信度，0-1之间
}}

direction字段必须是以下值之一：左转, 右转, 直行
"""
            
            # 调用大模型获取决策
            print("调用大模型分析交叉路口图像...")
            response = self.ask(prompt, image_path)
            
            print(f"大模型原始响应: {response}")
            
            # 解析响应
            if response and isinstance(response, dict):
                # 确保包含direction字段
                if 'direction' not in response:
                    # 尝试从reason中提取方向
                    if 'reason' in response:
                        reason = response['reason'].lower()
                        if '左' in reason or 'left' in reason:
                            response['direction'] = '左转'
                        elif '右' in reason or 'right' in reason:
                            response['direction'] = '右转'
                        else:
                            response['direction'] = '直行'
                    else:
                        response['direction'] = '直行'
                
                # 确保direction是有效值
                valid_directions = {'左转', '右转', '直行', 'left', 'right', 'straight'}
                if 'direction' in response:
                    direction = response['direction']
                    
                    # 标准化方向值
                    if direction.lower() in ['left', '左', '左转']:
                        response['direction'] = '左转'
                    elif direction.lower() in ['right', '右', '右转']:
                        response['direction'] = '右转'
                    elif direction.lower() in ['straight', '直', '直行']:
                        response['direction'] = '直行'
                    else:
                        print(f"无效的方向值: {direction}，修正为直行")
                        response['direction'] = '直行'
                
                # 添加控制参数
                if response['direction'] == '左转':
                    response['throttle'] = 0.3
                    response['steering'] = -0.5
                    response['brake'] = 0.0
                elif response['direction'] == '右转':
                    response['throttle'] = 0.3
                    response['steering'] = 0.5
                    response['brake'] = 0.0
                else:  # 直行
                    response['throttle'] = 0.5
                    response['steering'] = 0.0
                    response['brake'] = 0.0
                
                return response
            
            # 如果无法获取有效响应，返回默认决策
            print("无法获取有效响应，返回默认决策")
            return {
                'direction': '直行',
                'reason': '无法获取有效的模型响应，默认直行',
                'throttle': 0.5,
                'steering': 0.0,
                'brake': 0.0
            }
            
        except Exception as e:
            print(f"获取交叉路口决策异常: {e}")
            traceback.print_exc()
            
            # 返回默认决策
            return {
                'direction': '直行',
                'reason': f'获取决策异常: {e}',
                'throttle': 0.5,
                'steering': 0.0,
                'brake': 0.0
            }
    
    def _build_prompt(self, status):
        """构建提示词
        
        Args:
            status: 状态信息
            
        Returns:
            str: 提示词
        """
        return f"""
速度: {status.get('speed', 0.0):.1f} km/h
方向: {status.get('curve_direction', 'straight')}
弯度: {status.get('curve_strength', 0.0):.3f}
车道: {status.get('lane_status', '')}
车辆: {status.get('nearby_vehicles', '')}
信号: {status.get('traffic_light_state', '')}

根据以上驾驶状态和图像，生成一个JSON格式的驾驶指令，包含throttle(油门0-1)、steering(转向-1到1)、brake(刹车0-1)和reason(原因)字段。
"""
    
    def call_ollama_api(self, prompt, image_path=None):
        """调用Ollama API
        
        Args:
            prompt: 提示文本
            image_path: 可选的图像路径
        
        Returns:
            str: 模型的回复文本
        """
        try:
            # 构建API请求
            url = "http://localhost:11434/api/generate"
            
            # 准备请求数据
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1024
                }
            }
            
            # 如果有图像，添加图像数据
            if image_path and os.path.exists(image_path):
                print(f"添加图像到请求: {image_path}")
                with open(image_path, "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                    data["images"] = [img_base64]
            
            # 发送请求
            print(f"发送请求到Ollama API，模型: {self.model_name}")
            print(f"提示词: {prompt[:100]}...")  # 只打印前100个字符
            
            # 设置超时时间
            timeout = self.timeout
            
            # 重试机制
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = requests.post(url, json=data, timeout=timeout)
                    response.raise_for_status()  # 如果响应状态码不是200，抛出异常
                    
                    # 解析响应
                    response_data = response.json()
                    response_text = response_data.get("response", "")
                    
                    print(f"收到响应，长度: {len(response_text)}")
                    return response_text
                    
                except requests.exceptions.Timeout:
                    retry_count += 1
                    print(f"请求超时，重试 ({retry_count}/{max_retries})...")
                    timeout += 5  # 每次重试增加超时时间
                    
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    print(f"请求异常: {e}，重试 ({retry_count}/{max_retries})...")
                    time.sleep(1)  # 等待1秒后重试
            
            # 如果所有重试都失败
            print("所有重试都失败，返回默认响应")
            return '{"error": "无法连接到模型服务器"}'
            
        except Exception as e:
            print(f"调用Ollama API异常: {e}")
            traceback.print_exc()
            return '{"error": "调用模型API异常"}'

    def get_latest_photo(self):
        """获取最新的交叉路口照片"""
        try:
            photos = list(self.photo_dir.glob("junction_*.jpg"))
            if not photos:
                return None
            # 按文件修改时间排序，获取最新的照片
            latest_photo = max(photos, key=lambda x: x.stat().st_mtime)
            return str(latest_photo)
        except Exception as e:
            print(f"获取最新照片失败: {e}")
            return None

    def get_model_decision(self, timeout=2.0):
        """获取模型决策，带超时处理"""
        try:
            # 设置超时时间
            socket.setdefaulttimeout(timeout)
            
            # 如果缓存有效，直接返回缓存的决策
            if hasattr(self, '_cached_decision') and time.time() - self._last_decision_time < 30:
                return self._cached_decision
            
            # 获取最新的图片
            image_path = self.get_latest_photo()
            if not image_path:
                print("未找到交叉路口图像")
                return {'direction': 'straight', 'reason': '未找到交叉路口图像，默认直行'}
            
            print(f"使用图像: {image_path}")
            
            # 构建交叉路口提示词
            prompt = """
分析这张图片，判断在这个交叉路口应该往哪个方向行驶（左转、右转或直行）。
请考虑以下因素：
1. 道路标线和箭头
2. 交通信号灯状态
3. 道路布局

请以JSON格式回答，必须包含direction字段：
{
  "direction": "left/right/straight",
  "reason": "判断理由"
}

direction字段只能是以下三个值之一：left, right, straight
"""
            
            # 直接调用ask方法获取决策
            print("调用大模型分析交叉路口图像...")
            direction_response = self.ask(prompt, image_path)
            
            print(f"大模型原始响应: {direction_response}")
            
            if direction_response and isinstance(direction_response, dict):
                # 确保包含direction字段
                if 'direction' not in direction_response:
                    # 尝试从reason中提取方向
                    if 'reason' in direction_response:
                        reason = direction_response['reason'].lower()
                        if 'left' in reason:
                            direction_response['direction'] = 'left'
                        elif 'right' in reason:
                            direction_response['direction'] = 'right'
                        else:
                            direction_response['direction'] = 'straight'
                    else:
                        direction_response['direction'] = 'straight'
                
                # 确保direction是小写的
                if 'direction' in direction_response:
                    direction_response['direction'] = direction_response['direction'].lower()
                    
                    # 验证direction值是否有效
                    if direction_response['direction'] not in ['left', 'right', 'straight']:
                        print(f"无效的方向值: {direction_response['direction']}，修正为straight")
                        direction_response['direction'] = 'straight'
                
                # 缓存决策
                self._cached_decision = direction_response
                self._last_decision_time = time.time()
                return direction_response
            
            # 如果无法获取有效响应，返回默认直行
            default_response = {'direction': 'straight', 'reason': '无法获取有效响应，默认直行'}
            self._cached_decision = default_response
            self._last_decision_time = time.time()
            return default_response
            
        except socket.timeout:
            print("模型决策超时，使用默认决策")
            return {'direction': 'straight', 'reason': '模型响应超时'}
        except Exception as e:
            print(f"获取模型决策出错: {e}")
            traceback.print_exc()
            return {'direction': 'straight', 'reason': f'获取决策出错: {e}'}