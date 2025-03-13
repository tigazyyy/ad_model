#!/usr/bin/env python
"""
测试车道线检测器
这个脚本用于测试SimpleOpenCVLaneDetector的效果
"""

import cv2
import numpy as np
import os
import sys
import glob
import argparse
import time
from lane_detector import SimpleOpenCVLaneDetector

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试车道线检测器')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--video', type=str, help='输入视频路径')
    parser.add_argument('--camera', type=int, default=-1, help='使用摄像头ID')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    args = parser.parse_args()
    
    # 创建车道线检测器
    detector = SimpleOpenCVLaneDetector(None, None, None)
    
    # 设置输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    if args.image:
        # 处理单张图像
        img = cv2.imread(args.image)
        if img is None:
            print(f"无法读取图像: {args.image}")
            return
        
        # 检测车道线
        lane_info, result = detector.detect_lanes(img)
        
        # 显示结果
        cv2.imshow('Lane Detection', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif args.video:
        # 处理视频
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"无法打开视频: {args.video}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 创建输出视频
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测车道线
            start_time = time.time()
            lane_info, result = detector.detect_lanes(frame)
            process_time = time.time() - start_time
            
            # 添加FPS信息
            fps_text = f"FPS: {1/process_time:.1f}"
            cv2.putText(result, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 写入输出视频
            out.write(result)
            
            # 显示结果
            cv2.imshow('Lane Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        
    elif args.camera >= 0:
        # 使用摄像头
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"无法打开摄像头: {args.camera}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        
        # 创建输出视频
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测车道线
            start_time = time.time()
            lane_info, result = detector.detect_lanes(frame)
            process_time = time.time() - start_time
            
            # 添加FPS信息
            fps_text = f"FPS: {1/process_time:.1f}"
            cv2.putText(result, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # 写入输出视频
            out.write(result)
            
            # 显示结果
            cv2.imshow('Lane Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    
    else:
        print("请指定输入源: --image, --video 或 --camera")
        return
    
    # 释放资源
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"处理完成，输出保存到: {args.output}")

if __name__ == "__main__":
    main() 