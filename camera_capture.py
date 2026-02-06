import json
import os
import time
import logging
from datetime import datetime
import cv2
import subprocess
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('camera_capture.log'),
        logging.StreamHandler()
    ]
)

# 读取配置文件
def load_config():
    try:
        with open('cameras.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        raise

# 创建存储目录
def ensure_store_dir(store_path):
    if not os.path.exists(store_path):
        try:
            os.makedirs(store_path)
            logging.info(f"创建存储目录: {store_path}")
        except Exception as e:
            logging.error(f"创建存储目录失败: {e}")
            raise

# 从RTSP获取图片（使用ffmpeg命令行）
def capture_frame(rtsp_url, protocol=None):
    try:
        logging.info(f"尝试打开RTSP流: {rtsp_url}, 协议: {protocol}")
        
        # 构建ffmpeg命令
        # 使用-t 1只获取1秒的视频
        # 使用-vframes 1只获取1帧
        # 使用-f mjpeg输出为mjpeg格式
        # 使用-来输出到标准输出
        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp' if protocol == 'tcp' else 'udp',
            '-i', rtsp_url,
            '-t', '1',
            '-vframes', '1',
            '-f', 'mjpeg',
            '-y',
            'pipe:1'
        ]
        
        # 执行命令并捕获输出
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10  # 设置10秒超时
        )
        
        if result.returncode != 0:
            logging.error(f"ffmpeg执行失败: {result.stderr.decode('utf-8', errors='ignore')}")
            return None
        
        # 将输出转换为numpy数组
        img_data = np.frombuffer(result.stdout, dtype=np.uint8)
        if len(img_data) == 0:
            logging.error(f"ffmpeg未返回图像数据")
            return None
        
        # 解码图像
        frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if frame is None:
            logging.error(f"无法解码图像数据")
            return None
        
        logging.info(f"成功从RTSP流获取帧: {rtsp_url}")
        return frame
    except subprocess.TimeoutExpired:
        logging.error(f"ffmpeg执行超时: {rtsp_url}")
        return None
    except Exception as e:
        logging.error(f"获取图片时发生错误: {e}")
        return None

# 生成文件名
def generate_filename(camera_name):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return f"camera_{camera_name}_{timestamp}.jpg"

# 保存图片
def save_frame(frame, store_path, filename):
    file_path = os.path.join(store_path, filename)
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        logging.warning(f"文件已存在，跳过保存: {file_path}")
        return False
    
    try:
        cv2.imwrite(file_path, frame)
        logging.info(f"图片已保存: {file_path}")
        return True
    except Exception as e:
        logging.error(f"保存图片失败: {e}")
        return False

# 处理单个摄像头
def process_camera(camera, store_path):
    name = camera.get('name')
    rtsp = camera.get('rtsp')
    protocol = camera.get('protocol')
    interval = camera.get('interval', 60)  # 默认60秒
    
    logging.info(f"开始处理摄像头: {name}, 间隔: {interval}秒, 协议: {protocol}")
    
    while True:
        frame = capture_frame(rtsp, protocol)
        if frame is not None:
            filename = generate_filename(name)
            save_frame(frame, store_path, filename)
        
        # 等待指定的间隔时间
        time.sleep(interval)

if __name__ == "__main__":
    try:
        # 加载配置
        config = load_config()
        store_path = config.get('store_path', 'test')
        cameras = config.get('cameras', [])
        
        # 创建存储目录
        ensure_store_dir(store_path)
        
        # 启动多个线程处理不同的摄像头
        import threading
        threads = []
        
        for camera in cameras:
            thread = threading.Thread(target=process_camera, args=(camera, store_path))
            thread.daemon = True
            threads.append(thread)
            thread.start()
        
        # 主线程保持运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
    except Exception as e:
        logging.error(f"程序运行出错: {e}")
