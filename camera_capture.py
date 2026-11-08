import json
import os
import time
import logging
import threading
import subprocess
from datetime import datetime

import cv2
import numpy as np

from human_detection import (detect_humans, draw_boxes, draw_roi, normalize_roi,
                             as_polygon_list, DEFAULT_YOLO_MODEL,
                             ROI_MODE_FILTER, ROI_MODE_CROP)
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
                   detection_method='hog', yolo_model=DEFAULT_YOLO_MODEL, defaults=None):
    """
    单个摄像头处理程序
    params:
        camera - 摄像头配置
        store_path - 存储路径
        notifier - 钉钉通知器
        detection_method - 检测方法: hog / yolo
        yolo_model - YOLO ONNX 模型文件路径（method=yolo 时使用）
        defaults - 顶层默认配置（roi / roi_mode / roi_overlap / roi_debug），
                   摄像头配置中未指定的项从这里回退
    """
    defaults = defaults or {}
    name = camera.get('name')
    rtsp = camera.get('rtsp')
    protocol = camera.get('protocol')
    # 是否开启人形检测（未配置时默认开启）
    human_detection = camera.get('human_detection', True)
    # 每摄像头的报出阈值（未配置时使用检测方法的默认值）
    conf_threshold = camera.get('human_detection_threshold')
    # ROI 检测区域：归一化多边形（单个或多个取并集），未配置时整幅画面都参与检测
    roi = camera.get('roi', defaults.get('roi'))
    # ROI 开关：关闭时忽略 roi 配置，整幅画面都参与人形检测（标定好的坐标仍可保留）
    roi_enabled = camera.get('roi_enabled', defaults.get('roi_enabled', True))
    roi_mode = camera.get('roi_mode', defaults.get('roi_mode', ROI_MODE_FILTER))
    roi_overlap = camera.get('roi_overlap', defaults.get('roi_overlap'))
    # 是否在标注图上叠加 ROI 轮廓，便于事后核对标定是否贴合
    roi_debug = camera.get('roi_debug', defaults.get('roi_debug', False))
    # 检测周期：每隔多少秒抓一帧做检测（决定检测灵敏度的最小粒度）
    capture_cycle = max(camera.get('capture_cycle', 3), 1)
    timelapse_interval = camera.get('timelapse_interval', 60)
    motion_interval = camera.get('motion_interval', 5)
    if not roi_enabled:
        roi = None
        roi_desc = "已关闭(整幅画面参与检测)"
    elif roi:
        try:
            polygons = as_polygon_list(roi)
            roi_desc = (f"{len(polygons)}个多边形, 共{sum(len(p) for p in polygons)}个顶点, "
                        f"模式: {roi_mode}")
        except ValueError as e:
            roi = None
            roi_desc = f"配置无效({e}), 整幅画面参与检测"
    else:
        roi_desc = "未配置(整幅画面参与检测)"

    logging.info(
        f"开始处理摄像头: {name}, 人形检测: {'开启' if human_detection else '关闭'}, "
        f"检测周期: {capture_cycle}秒, "
        f"延时摄影间隔: {timelapse_interval}秒, 人形间隔: {motion_interval}秒, "
        f"协议: {protocol}, 报出阈值: {conf_threshold if conf_threshold is not None else '默认'}, "
        f"ROI: {roi_desc}"
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
                                          conf_threshold=conf_threshold, roi=roi,
                                          roi_mode=roi_mode, roi_overlap=roi_overlap)
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
                        annotated = draw_boxes(frame, boxes)
                        if roi_debug and roi:
                            # 叠加 ROI 轮廓，便于事后核对标定是否贴合
                            pixel_roi = normalize_roi(roi, frame.shape[1], frame.shape[0])
                            if pixel_roi:
                                annotated = draw_roi(annotated, pixel_roi)
                        if save_frame(annotated, store_path, annotated_filename):
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

        # ROI 默认配置（顶层配置对所有摄像头生效，摄像头内同名配置可覆盖）
        defaults = {
            'roi': config.get('roi'),
            'roi_enabled': config.get('roi_enabled', True),
            'roi_mode': config.get('roi_mode', ROI_MODE_FILTER),
            'roi_overlap': config.get('roi_overlap'),
            'roi_debug': config.get('roi_debug', False),
        }

        # 启动前校验 ROI 配置，避免运行中才发现配置错误
        for camera in cameras:
            camera_name = camera.get('name')
            camera_enabled = camera.get('roi_enabled', defaults['roi_enabled'])
            if not camera_enabled:
                logging.info(f"摄像头 {camera_name} 的 ROI 已关闭，整幅画面都参与人形检测")
                continue
            camera_mode = camera.get('roi_mode', defaults['roi_mode'])
            if camera_mode not in (ROI_MODE_FILTER, ROI_MODE_CROP):
                logging.error(f"摄像头 {camera_name} 的 roi_mode 无效: {camera_mode}，"
                              f"可选: {ROI_MODE_FILTER} / {ROI_MODE_CROP}")
                raise SystemExit(1)
            camera_roi = camera.get('roi', defaults['roi'])
            if camera_roi is None:
                logging.info(f"摄像头 {camera_name} 未配置 ROI，整幅画面都参与人形检测")
                continue
            try:
                # 传 1x1 尺寸只为借用其格式校验，不依赖真实图像分辨率
                polygons = normalize_roi(camera_roi, 1, 1)
            except ValueError as e:
                logging.error(f"摄像头 {camera_name} 的 roi 配置无效: {e}")
                raise SystemExit(1)
            logging.info(f"摄像头 {camera_name} 已启用检测区域 ROI: "
                         f"{len(polygons)}个多边形, 共{sum(len(p) for p in polygons)}个顶点, "
                         f"模式: {camera_mode}")

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
                                            detection_method, yolo_model, defaults))
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
