# model_server.py
import socket
import json
import threading
import time
import traceback
import subprocess
import os
import requests
from PIL import Image
import io
import base64

class ModelServer:
    def __init__(self):
        # 初始化模型
        self.model_name = "car"  # Ollama模型名称
        
        # 简化prompt模板，减少token长度
        self.prompt_template = """
速度:{speed:.1f}km/h
方向:{curve_direction}
弯度:{curve_strength:.3f}
车道:{lane_status}
车辆:{nearby_vehicles}
信号:{traffic_light_state}

生成驾驶指令:
{
"""

    def call_ollama_api(self, prompt, image_path=None):
        """调用Ollama API"""
        try:
            # 准备请求数据
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # 如果有图像，添加到请求中
            if image_path and os.path.exists(image_path):
                # 读取图像并转换为base64
                with open(image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
                
                # 添加图像到请求
                data["images"] = [image_data]
            
            # 明确指定本地回环地址，避免通过VPN
            api_url = "http://127.0.0.1:11434/api/generate"
            print(f"发送请求到: {api_url}")
            
            # 发送请求到Ollama API，添加超时参数
            response = requests.post(api_url, json=data, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", "")
        
        except Exception as e:
            print(f"调用Ollama API时出错: {e}")
            traceback.print_exc()
            return f"调用模型出错: {e}"

    def generate_response(self, state):
        """生成规范的响应"""
        try:
            # 从状态中提取信息
            speed = state.get('speed', 0.0)
            curve_direction = state.get('curve_direction', 'straight')
            curve_strength = state.get('curve_strength', 0.0)
            lane_status = state.get('lane_status', '')
            nearby_vehicles = state.get('nearby_vehicles', '')
            traffic_light_state = state.get('traffic_light_state', '')
            image_path = state.get('image_path', '')
            
            print("\n=== 接收到的状态信息 ===")
            print(f"速度: {speed:.1f} km/h")
            print(f"道路状况: {lane_status}")
            print(f"弯道方向: {curve_direction}")
            print(f"弯道强度: {curve_strength:.3f}")
            print(f"附近车辆: {nearby_vehicles}")
            print(f"交通信号: {traffic_light_state}")
            print(f"图像路径: {image_path}")
            print("=====================\n")
            
            # 构建提示词
            prompt = f"""
速度:{speed:.1f}km/h
方向:{curve_direction}
弯度:{curve_strength:.3f}
车道:{lane_status}
车辆:{nearby_vehicles}
信号:{traffic_light_state}

根据以上驾驶状态和图像，生成一个JSON格式的驾驶指令，包含throttle(油门0-1)、steering(转向-1到1)、brake(刹车0-1)和reason(原因)字段。
"""
            
            # 调用Ollama模型
            model_output = self.call_ollama_api(prompt, image_path)
            print(f"模型原始输出: {model_output}")
            
            # 尝试从模型输出中提取JSON
            try:
                # 查找JSON开始和结束的位置
                start_idx = model_output.find('{')
                end_idx = model_output.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = model_output[start_idx:end_idx]
                    response = json.loads(json_str)
                    
                    # 确保包含所有必要字段
                    if 'throttle' not in response:
                        response['throttle'] = 0.3
                    if 'steering' not in response:
                        response['steering'] = 0.0
                    if 'brake' not in response:
                        response['brake'] = 0.0
                    if 'reason' not in response:
                        response['reason'] = '模型未提供原因'
                else:
                    # 如果没有找到JSON，使用默认响应
                    response = {
                        'throttle': 0.3,
                        'steering': 0.0,
                        'brake': 0.0,
                        'reason': f'无法从模型输出中提取JSON: {model_output}'
                    }
            except Exception as e:
                print(f"解析模型输出时出错: {e}")
                # 使用基于规则的备用逻辑
                if 'approaching_junction' in lane_status:
                    # 交叉路口决策逻辑
                    response = {
                        'throttle': 0.3,
                        'steering': 0.0,
                        'brake': 0.0,
                        'reason': '检测到交叉路口，保持直行并降低速度'
                    }
                    
                    # 如果速度过快，增加刹车
                    if speed > 15.0:
                        response['throttle'] = 0.0
                        response['brake'] = 0.3
                        response['reason'] += '，当前速度过快，需要减速'
                else:
                    # 普通道路决策逻辑
                    response = {
                        'throttle': 0.3,
                        'steering': 0.0,
                        'brake': 0.0,
                        'reason': '正常行驶'
                    }
            
            print("=== 模型决策 ===")
            print(f"油门: {response['throttle']:.3f}")
            print(f"转向: {response['steering']:.3f}")
            print(f"刹车: {response['brake']:.3f}")
            print(f"原因: {response['reason']}")
            print("===============\n")
            
            return response
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            traceback.print_exc()
            return {
                'throttle': 0.3,
                'steering': 0.0,
                'brake': 0.0,
                'reason': f'生成决策出错: {e}'
            }

def handle_client(client_socket):
    """处理客户端连接"""
    try:
        # 接收数据
        data = b""
        while True:
            try:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                
                # 检查是否接收到完整的JSON
                try:
                    json.loads(data.decode('utf-8'))
                    break  # 如果成功解析，说明数据接收完毕
                except:
                    pass  # 继续接收
                    
            except socket.timeout:
                print("接收数据超时")
                break
        
        if data:
            # 解析请求
            try:
                request = json.loads(data.decode('utf-8'))
                print(f"收到请求: {request}")
                
                # 获取状态
                status = request.get('status', {})
                
                # 生成响应
                model_server = ModelServer()
                response = model_server.generate_response(status)
                
                # 使用json库序列化
                response_str = json.dumps(response) + "\n"
                client_socket.sendall(response_str.encode('utf-8'))
                print(f"已发送响应: {response}")
                
            except json.JSONDecodeError:
                print(f"无法解析JSON: {data.decode('utf-8')}")
                # 发送错误响应
                error_response = {
                    'throttle': 0.3,
                    'steering': 0.0,
                    'reason': '请求格式错误'
                }
                # 使用json库序列化错误响应
                error_str = json.dumps(error_response) + "\n"
                client_socket.sendall(error_str.encode('utf-8'))
    
    except Exception as e:
        print(f"处理客户端时出错: {e}")
        traceback.print_exc()
    
    finally:
        client_socket.close()

def start_server():
    """启动服务器"""
    host = '127.0.0.1'
    port = 51738
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    
    print(f"模型服务器启动，监听 {host}:{port}")
    
    try:
        while True:
            client_socket, address = server_socket.accept()
            print(f"接受来自 {address} 的连接")
            
            # 设置超时
            client_socket.settimeout(5.0)
            
            # 创建新线程处理客户端
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()
    
    except KeyboardInterrupt:
        print("服务器关闭")
    
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()