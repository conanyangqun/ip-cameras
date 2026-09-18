import json
import os
import time
import logging
import threading
import subprocess
from datetime import datetime

import cv2
import numpy as np

from human_detection import detect_humans, draw_boxes, DEFAULT_YOLO_MODEL
from dingtalk import DingTalkNotifier

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
    """
    读取cameras.json配置文件
    返回配置字典
    """
    try:
        with open('cameras.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        raise

# 创建存储目录
def ensure_store_dir(store_path):
    """
    确保存储目录可用
    不存在时创建
    """
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
def generate_filename(camera_name, suffix=None):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tag = f"_{suffix}" if suffix else ""
    return f"camera_{camera_name}_{timestamp}{tag}.jpg"

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
def process_camera(camera, store_path, notifier=None,
                   detection_method='hog', yolo_model=DEFAULT_YOLO_MODEL):
    """
    单个摄像头处理程序
    params:
        camera - 摄像头配置
        store_path - 存储路径
        notifier - 钉钉通知器
        detection_method - 检测方法: hog / yolo
        yolo_model - YOLO ONNX 模型文件路径（method=yolo 时使用）
    """
    name = camera.get('name')
    rtsp = camera.get('rtsp')
    protocol = camera.get('protocol')
    # 是否开启人形检测（未配置时默认开启）
    human_detection = camera.get('human_detection', True)
    # 每摄像头的报出阈值（未配置时使用检测方法的默认值）
    conf_threshold = camera.get('human_detection_threshold')
    # 检测周期：每隔多少秒抓一帧做检测（决定检测灵敏度的最小粒度）
    capture_cycle = max(camera.get('capture_cycle', 3), 1)
    timelapse_interval = camera.get('timelapse_interval', 60)
    motion_interval = camera.get('motion_interval', 5)

    logging.info(
        f"开始处理摄像头: {name}, 人形检测: {'开启' if human_detection else '关闭'}, "
        f"检测周期: {capture_cycle}秒, "
        f"延时摄影间隔: {timelapse_interval}秒, 人形间隔: {motion_interval}秒, "
        f"协议: {protocol}, 报出阈值: {conf_threshold if conf_threshold is not None else '默认'}"
    )

    last_save_time = 0

    while True:
        frame = capture_frame(rtsp, protocol) # 获取当前帧
        now = time.time()

        if frame is not None:
            boxes = []
            motion = False
            # 人形检测（关闭时跳过，仅按延时摄影间隔保存）
            if human_detection:
                try:
                    boxes = detect_humans(frame, method=detection_method, yolo_model=yolo_model,
                                          conf_threshold=conf_threshold)
                except Exception as e:
                    logging.error(f"人形检测出错: {e}")
                    boxes = []
                motion = len(boxes) > 0

            # 根据是否检测到人形选择保存间隔
            interval = motion_interval if motion else timelapse_interval
            if now - last_save_time >= interval:
                filename = generate_filename(name)
                if save_frame(frame, store_path, filename):
                    last_save_time = now
                    # 检测到人形时，额外保存带矩形框标注的图像
                    if motion:
                        annotated_filename = generate_filename(name, suffix='detected')
                        if save_frame(draw_boxes(frame, boxes), store_path, annotated_filename):
                            logging.info(f"标注图已保存: {annotated_filename}")

            if motion:
                # 检测到人形
                logging.info(f"摄像头 {name} 检测到人形，数量: {len(boxes)}")
                # 检测到人形时发送钉钉通知（内部带冷却时间和限流）
                if notifier:
                    try:
                        notifier.send_motion_alert(name, len(boxes))
                    except Exception as e:
                        logging.error(f"发送钉钉通知出错: {e}")

        # 等待下一个检测周期
        time.sleep(capture_cycle)


if __name__ == "__main__":
    try:
        # 加载配置
        config = load_config()
        store_path = config.get('store_path', 'test')
        cameras = config.get('cameras', [])

        # 创建存储目录
        ensure_store_dir(store_path)

        # 读取检测方法配置
        detection_method = config.get('detection_method', 'hog')
        if detection_method not in ('hog', 'yolo'):
            logging.error(f"无效的检测方法: {detection_method}，可选: hog / yolo")
            raise SystemExit(1)
        yolo_model = config.get('yolo_model', DEFAULT_YOLO_MODEL)
        if detection_method == 'yolo' and not os.path.exists(yolo_model):
            logging.error(f"检测方法为 yolo 但模型文件不存在: {yolo_model}，请先导出或下载")
            raise SystemExit(1)
        if detection_method == 'yolo':
            logging.info(f"检测方法: yolo, 模型: {yolo_model}")
        else:
            logging.info("检测方法: hog")

        # 初始化钉钉机器人通知器
        notifier = None
        dingtalk_cfg = config.get('dingtalk', {})
        if dingtalk_cfg.get('enabled'):
            # 启用dingtalk机器人
            webhook = dingtalk_cfg.get('webhook')
            if webhook:
                notifier = DingTalkNotifier(
                    webhook=webhook,
                    secret=dingtalk_cfg.get('secret'),
                    notify_cooldown=dingtalk_cfg.get('notify_cooldown', 300),
                    max_per_minute=dingtalk_cfg.get('max_per_minute', 20),
                )
                logging.info("钉钉机器人通知已启用")
            else:
                logging.warning("已启用钉钉通知但未配置 webhook，通知功能不可用")

        # 启动多个线程处理不同的摄像头
        threads = []

        for camera in cameras:
            thread = threading.Thread(target=process_camera,
                                      args=(camera, store_path, notifier,
                                            detection_method, yolo_model))
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
