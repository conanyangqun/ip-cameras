import cv2
import numpy as np
import os
import argparse
import threading
import logging

logger = logging.getLogger(__name__)

# HOG 检测器加载较慢，全局缓存，避免重复初始化
_hog = None


def get_hog_detector():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog


# YOLO 检测方法
DEFAULT_YOLO_MODEL = 'yolo26n.onnx'
# YOLO 输入尺寸（越小越快，416 兼顾速度与精度）
YOLO_INPUT_SIZE = 640
# person 置信度阈值
YOLO_CONF_THRESHOLD = 0.4

# YOLO 网络加载较慢，全局缓存，避免重复初始化
_yolo_net = None
_yolo_model_path = None
# cv2.dnn.Net 非线程安全，多摄像头线程共享时必须串行化推理，否则结果会被污染
_yolo_infer_lock = threading.Lock()


def get_yolo_detector(model_path=DEFAULT_YOLO_MODEL):
    """加载并缓存 YOLO ONNX 模型"""
    global _yolo_net, _yolo_model_path
    if _yolo_net is None or _yolo_model_path != model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO 模型文件不存在: {model_path}，"
                f"请先导出或下载，例如: yolo export format=onnx imgsz={YOLO_INPUT_SIZE}"
            )
        _yolo_net = cv2.dnn.readNetFromONNX(model_path)
        _yolo_model_path = model_path
    return _yolo_net


def detect_humans_yolo(image, model_path=DEFAULT_YOLO_MODEL, conf_threshold=None):
    """
    用 YOLO nano ONNX 模型检测人形，返回检测框列表 [(x, y, w, h), ...]
    模型输出类别索引 0 为 person
    params:
        conf_threshold - person 置信度阈值，为 None 时使用默认值 YOLO_CONF_THRESHOLD
    """
    if image is None:
        return []

    net = get_yolo_detector(model_path)
    ih, iw = image.shape[:2]
    input_size = YOLO_INPUT_SIZE
    threshold = YOLO_CONF_THRESHOLD if conf_threshold is None else conf_threshold

    blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (input_size, input_size),
                                 swapRB=True, crop=False)
    with _yolo_infer_lock:
        net.setInput(blob)
        output = net.forward()  # (1, 4+80, num_anchors)
    preds = output[0].T     # (num_anchors, 4+80)

    # person 类别（索引 0）的置信度过滤
    person_scores = preds[:, 4]
    # 调试：打印置信度最高的前 10 个分数，用于确定 YOLO_CONF_THRESHOLD 合适取值
    if len(person_scores) > 0:
        top_scores = np.sort(person_scores)[::-1][:10]
        logger.debug("person_scores top10: %s", np.round(top_scores, 4))
    mask = person_scores > threshold
    if not np.any(mask):
        return []

    # cxcywh（输入尺寸）-> xywh（原图尺寸）
    b = preds[mask, :4]
    scale_x = iw / input_size
    scale_y = ih / input_size
    boxes = np.stack([
        (b[:, 0] - b[:, 2] / 2) * scale_x,
        (b[:, 1] - b[:, 3] / 2) * scale_y,
        b[:, 2] * scale_x,
        b[:, 3] * scale_y,
    ], axis=1)

    # 丢弃含 NaN/Inf 的无效检测框（坏帧可能导致模型输出异常值）
    boxes = boxes[np.all(np.isfinite(boxes), axis=1)]
    if len(boxes) == 0:
        return []

    # 将坐标裁剪到图像范围内，宽高至少为 1，确保后续绘制不会溢出
    boxes[:, 0] = np.clip(boxes[:, 0], 0, iw - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, ih - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 1, iw - boxes[:, 0])
    boxes[:, 3] = np.clip(boxes[:, 3], 1, ih - boxes[:, 1])

    # 非极大值抑制去除重叠框
    return non_max_suppression(np.rint(boxes).astype("int")).tolist()


def non_max_suppression(boxes, overlapThresh=0.4):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


# 在图像上绘制人形检测框
def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2, label='person'):
    """
    在图像副本上绘制检测框及标签，返回标注后的图像（不修改原图）
    params:
        image - 原始图像
        boxes - 检测框列表 [(x, y, w, h), ...]
        color - 框和标签颜色（BGR）
        thickness - 线宽
        label - 标签文本，为空时不绘制
    """
    annotated = image.copy()
    for (x, y, w, h) in boxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        if label:
            cv2.putText(annotated, label, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return annotated


# HOG 默认置信度阈值
HOG_CONF_THRESHOLD = 0.75


def detect_humans(image, method='hog', yolo_model=DEFAULT_YOLO_MODEL, conf_threshold=None):
    """
    检测图片中的人形，返回过滤后的检测框列表 [(x, y, w, h), ...]
    params:
        method - 检测方法: hog=HOG+SVM(默认), yolo=YOLO nano ONNX(误报更低)
        yolo_model - YOLO ONNX 模型文件路径（method=yolo 时使用）
        conf_threshold - 报出阈值: yolo 为 person 置信度阈值（默认 YOLO_CONF_THRESHOLD），
                         hog 为检测权重阈值（默认 HOG_CONF_THRESHOLD）
    """
    if image is None:
        return []

    if method == 'yolo':
        return detect_humans_yolo(image, model_path=yolo_model, conf_threshold=conf_threshold)

    hog = get_hog_detector()

    # 检测人形 - 调整参数平衡假阳性和漏检
    (rects, weights) = hog.detectMultiScale(
        image,
        winStride=(12, 12),  # 适中的步长
        padding=(20, 20),  # 适中的填充
        scale=1.1,  # 适中的缩放因子
        useMeanshiftGrouping=False  # 禁用均值漂移分组，使用非极大值抑制
    )

    # 过滤检测结果，平衡假阳性和漏检
    weight_threshold = HOG_CONF_THRESHOLD if conf_threshold is None else conf_threshold
    filtered_rects = []
    for (x, y, w, h), weight in zip(rects, weights):
        # 报出阈值（可通过配置按摄像头调整）
        if weight > weight_threshold:
            # 放宽宽高比范围
            aspect_ratio = h / w
            if aspect_ratio > 1.2 and aspect_ratio < 3.0:
                # 降低最小尺寸要求
                if h > 60 and w > 25:
                    # 降低面积要求
                    area = w * h
                    if area > 1500:  # 25*60=1500
                        # 降低边缘密度阈值
                        roi = image[y:y+h, x:x+w]
                        if roi.size > 0:
                            # 转换为灰度图
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            # 计算边缘
                            edges = cv2.Canny(gray_roi, 50, 150)
                            # 计算边缘密度
                            edge_density = cv2.countNonZero(edges) / (w * h)
                            # 降低边缘密度阈值
                            if edge_density > 0.03:
                                filtered_rects.append((x, y, w, h))

    # 应用非极大值抑制(NMS)进一步过滤重叠检测
    if len(filtered_rects) > 0:
        filtered_rects = non_max_suppression(np.array(filtered_rects))

    return filtered_rects


def detect_humans_in_folder(folder_path, method='hog', yolo_model=DEFAULT_YOLO_MODEL,
                            conf_threshold=None):
    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)

            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                logger.error("无法读取图片: %s", filename)
                continue

            filtered_rects = detect_humans(image, method=method, yolo_model=yolo_model,
                                           conf_threshold=conf_threshold)

            # 输出检测结果
            if len(filtered_rects) > 0:
                logger.info("图片 %s 中检测到人形，数量: %d", filename, len(filtered_rects))
                # 在图片上绘制检测框并保存
                annotated = draw_boxes(image, filtered_rects)
                output_path = os.path.join(folder_path, f"detected_{filename}")
                cv2.imwrite(output_path, annotated)
                logger.info("带有检测框的图片已保存为: detected_%s", filename)
            else:
                logger.info("图片 %s 中未检测到人形", filename)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='检测文件夹中图片是否包含人形')
    parser.add_argument('folder', type=str, help='包含图片的文件夹路径')
    parser.add_argument('--method', type=str, choices=['hog', 'yolo'], default='hog',
                        help='检测方法: hog=HOG+SVM(默认), yolo=YOLO nano ONNX(误报更低)')
    parser.add_argument('--model', type=str, default=DEFAULT_YOLO_MODEL,
                        help='YOLO ONNX 模型文件路径，默认为 %s（--method yolo 时使用）' % DEFAULT_YOLO_MODEL)
    parser.add_argument('--threshold', type=float, default=None,
                        help='报出阈值: yolo 为 person 置信度（默认 %.2f），hog 为检测权重（默认 %.2f）'
                             % (YOLO_CONF_THRESHOLD, HOG_CONF_THRESHOLD))
    parser.add_argument('--debug', action='store_true',
                        help='输出 DEBUG 级别日志（打印每张图片的置信度得分，用于调整 YOLO_CONF_THRESHOLD）')
    args = parser.parse_args()

    # 配置日志输出
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    )

    # 检查文件夹是否存在
    if not os.path.isdir(args.folder):
        logger.error("文件夹 %s 不存在", args.folder)
    else:
        # 开始检测
        detect_humans_in_folder(args.folder, method=args.method, yolo_model=args.model,
                                conf_threshold=args.threshold)
