import cv2
import numpy as np
import os
import json
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


# ===== ROI（检测区域）支持 =====
# ROI 使用归一化的多边形表示，x/y 为相对图像宽高的 0~1 浮点数，支持两种写法:
#   单个多边形: [[x1, y1], [x2, y2], ...]
#   多个多边形: [[[x1, y1], [x2, y2], ...], [[x1, y1], ...]]（并集，命中任意一个即算命中）
# 只有命中 ROI 的人形才算检出，ROI 之外一律忽略；未配置 ROI 时行为与原逻辑完全一致（全图检测）。
# 坐标使用归一化值而非像素，是为了在分辨率/码流变化后配置依然有效。
ROI_MODE_FILTER = 'filter'  # 全图检测后按 ROI 过滤（默认，检测质量最高）
ROI_MODE_CROP = 'crop'      # 仅裁剪 ROI 区域送检测（省算力，但 ROI 边缘的人形可能被截断）
ROI_MODES = (ROI_MODE_FILTER, ROI_MODE_CROP)


def _is_point(value):
    """判断是否为形如 [x, y] 的坐标点"""
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in value))


def as_polygon_list(roi):
    """
    将 ROI 配置统一为多边形列表 [[[x, y], ...], ...]，同时完成格式校验
    params:
        roi - None / 单个多边形 [[x, y], ...] / 多个多边形 [[[x, y], ...], ...]
    返回: 多边形列表（坐标为 float）；roi 为空时返回 []；格式非法时抛出 ValueError
    """
    if roi is None:
        return []
    if not isinstance(roi, (list, tuple)) or len(roi) == 0:
        raise ValueError("ROI 应为多边形顶点列表，当前: %r" % (roi,))

    # 第一个元素形如 [x, y] 时是单个多边形，否则视为多边形组
    polygons = [roi] if _is_point(roi[0]) else list(roi)

    result = []
    for index, polygon in enumerate(polygons):
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
            raise ValueError("ROI 第 %d 个多边形至少需要 3 个顶点，当前: %r" % (index + 1, polygon))
        points = []
        for point in polygon:
            if not _is_point(point):
                raise ValueError("ROI 顶点格式错误，应为 [x, y]: %r" % (point,))
            try:
                points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                raise ValueError("ROI 顶点坐标必须为数值: %r" % (point,))
        result.append(points)
    return result


def parse_roi_arg(text):
    """
    解析命令行传入的 ROI 文本为归一化多边形列表
    支持三种写法:
        JSON 单个多边形: [[0,0],[1,0],[1,0.72]]
        JSON 多个多边形: [[[0,0],[1,0],[1,0.72]], [[0,0],[0.3,0],[0.3,0.3]]]
        坐标对: "0,0 1,0 1,0.72"（多个多边形之间用 | 分隔）
    """
    if text is None:
        return None
    text = text.strip() if isinstance(text, str) else text
    if not text:
        return None

    if isinstance(text, str) and text.startswith('['):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError("ROI JSON 解析失败: %s" % e)
        return as_polygon_list(data)

    polygons = []
    for group in str(text).split('|'):
        points = []
        for pair in group.replace(';', ' ').split():
            parts = pair.split(',')
            if len(parts) != 2:
                raise ValueError("ROI 坐标对格式错误，应为 x,y: %r" % (pair,))
            points.append([float(parts[0]), float(parts[1])])
        polygons.append(points)
    return as_polygon_list(polygons)


def normalize_roi(roi, image_w, image_h):
    """
    将归一化的 ROI 转换为像素坐标的多边形列表
    params:
        roi - 归一化多边形配置（单个多边形或多边形组，见 as_polygon_list），坐标取值 0~1
        image_w / image_h - 当前图像宽高
    返回: 像素多边形列表 [[(x, y), ...], ...]；roi 为空时返回 None；格式非法时抛出 ValueError
    """
    polygons = as_polygon_list(roi)
    if not polygons or image_w <= 0 or image_h <= 0:
        return None

    result = []
    for polygon in polygons:
        points = []
        for x, y in polygon:
            # 越界坐标裁剪到图像范围内，避免绘制和判定异常
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            points.append((int(round(x * image_w)), int(round(y * image_h))))
        result.append(points)
    return result


def build_roi_mask(roi, image_w, image_h):
    """生成 ROI 二值掩码（多个多边形取并集），用于重叠比例判定（像素多边形列表输入）"""
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32) for polygon in roi], 1)
    return mask


def box_hits_roi(box, roi, roi_mask=None, overlap_threshold=None):
    """
    判断单个检测框是否命中 ROI
    params:
        box - 检测框 (x, y, w, h)
        roi - 像素坐标的多边形列表（normalize_roi 的输出）
        roi_mask - 可选，ROI 二值掩码（build_roi_mask 的输出），重叠比例判定时使用
        overlap_threshold - 可选，重叠比例阈值: 框与 ROI 的交集面积 / 框面积 >= 该值时也视为命中。
                            为 None（默认）时仅按框中心点是否落在 ROI 内判定。
    返回: bool
    """
    if not roi:
        return True

    x, y, w, h = (float(v) for v in box)
    center = (x + w / 2.0, y + h / 2.0)
    # 多个多边形取并集，命中任意一个即算命中
    for polygon in roi:
        if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), center, False) >= 0:
            return True

    # 中心点不在 ROI 内时，可选地按重叠比例兜底（用于 ROI 边缘只露出半个身子的人）
    if overlap_threshold is None or roi_mask is None or w <= 0 or h <= 0:
        return False

    mask_h, mask_w = roi_mask.shape[:2]
    x1 = int(max(0, np.floor(x)))
    y1 = int(max(0, np.floor(y)))
    x2 = int(min(mask_w, np.ceil(x + w)))
    y2 = int(min(mask_h, np.ceil(y + h)))
    if x2 <= x1 or y2 <= y1:
        return False

    roi_pixels = cv2.countNonZero(roi_mask[y1:y2, x1:x2])
    return roi_pixels / float((x2 - x1) * (y2 - y1)) >= overlap_threshold


def filter_boxes_by_roi(boxes, roi, overlap_threshold=None, image_shape=None):
    """
    过滤掉 ROI 之外的检测框
    params:
        boxes - 检测框列表 [(x, y, w, h), ...]
        roi - 像素坐标的多边形列表；为空时原样返回
        overlap_threshold - 可选，重叠比例阈值（见 box_hits_roi），默认不启用
        image_shape - 图像 (高, 宽)，提供时用于生成重叠比例判定所需的掩码
    返回: 命中 ROI 的检测框列表 [(x, y, w, h), ...]
    """
    if not roi or len(boxes) == 0:
        return list(boxes)

    roi_mask = None
    if overlap_threshold is not None and image_shape:
        roi_mask = build_roi_mask(roi, image_shape[1], image_shape[0])

    return [tuple(int(v) for v in box) for box in boxes
            if box_hits_roi(box, roi, roi_mask, overlap_threshold)]


def draw_roi(image, roi, color=(0, 255, 255), thickness=2, label='ROI'):
    """
    在图像副本上绘制 ROI 多边形轮廓（支持多个多边形），返回标注后的图像（不修改原图）
    params:
        image - 原始图像
        roi - 像素坐标的多边形列表（normalize_roi 的输出）
        color - 轮廓和标签颜色（BGR）
        thickness - 线宽
        label - 标签文本，为空时不绘制；多个多边形时只有第一个会标注
    """
    annotated = image.copy()
    if not roi:
        return annotated

    for index, polygon in enumerate(roi):
        points = np.array(polygon, dtype=np.int32)
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=thickness)
        if label and index == 0:
            px, py = points[0]
            cv2.putText(annotated, label, (int(px), max(int(py) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return annotated


# HOG 默认置信度阈值
HOG_CONF_THRESHOLD = 0.75


def _detect_humans_raw(image, method='hog', yolo_model=DEFAULT_YOLO_MODEL, conf_threshold=None):
    """
    原始检测逻辑（不含 ROI 过滤），一般不用直接调用，请使用 detect_humans
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


def detect_humans(image, method='hog', yolo_model=DEFAULT_YOLO_MODEL, conf_threshold=None,
                  roi=None, roi_mode=ROI_MODE_FILTER, roi_overlap=None):
    """
    检测图片中的人形，返回过滤后的检测框列表 [(x, y, w, h), ...]
    params:
        method - 检测方法: hog=HOG+SVM(默认), yolo=YOLO nano ONNX(误报更低)
        yolo_model - YOLO ONNX 模型文件路径（method=yolo 时使用）
        conf_threshold - 报出阈值: yolo 为 person 置信度阈值（默认 YOLO_CONF_THRESHOLD），
                         hog 为检测权重阈值（默认 HOG_CONF_THRESHOLD）
        roi - 归一化检测区域，坐标相对图像宽高取值 0~1：
              单个多边形 [[x, y], ...] 或多个多边形 [[[x, y], ...], ...]（并集）；
              为空时全图检测（默认，行为与原逻辑一致）
        roi_mode - ROI 生效方式: filter=全图检测后过滤(默认), crop=仅裁剪 ROI 区域检测
        roi_overlap - 可选重叠比例阈值，检测框与 ROI 的交集占比达到该值时也算命中
                      （默认 None，即仅按框中心点是否落在 ROI 内判定）
    """
    if image is None:
        return []

    image_h, image_w = image.shape[:2]
    pixel_roi = normalize_roi(roi, image_w, image_h)
    if not pixel_roi:
        return _detect_humans_raw(image, method, yolo_model, conf_threshold)

    if roi_mode == ROI_MODE_CROP:
        # 只把 ROI 的外接矩形（覆盖全部多边形）送检测，再把坐标偏移回原图
        all_points = np.vstack([np.array(polygon, dtype=np.int32) for polygon in pixel_roi])
        x, y, w, h = cv2.boundingRect(all_points)
        if w <= 0 or h <= 0:
            return []
        boxes = _detect_humans_raw(image[y:y + h, x:x + w], method, yolo_model, conf_threshold)
        boxes = [(bx + x, by + y, bw, bh) for (bx, by, bw, bh) in boxes]
        # 多边形 ROI（如 L 形）的凹口仍落在裁剪矩形内，需再过滤一次
        return filter_boxes_by_roi(boxes, pixel_roi, roi_overlap, (image_h, image_w))

    boxes = _detect_humans_raw(image, method, yolo_model, conf_threshold)
    return filter_boxes_by_roi(boxes, pixel_roi, roi_overlap, (image_h, image_w))


def detect_humans_in_folder(folder_path, method='hog', yolo_model=DEFAULT_YOLO_MODEL,
                            conf_threshold=None, roi=None, roi_mode=ROI_MODE_FILTER,
                            roi_overlap=None, roi_outline=False):
    """
    批量检测文件夹中的图片是否包含人形，命中的人形框会标注后另存为 detected_原文件名
    params:
        conf_threshold / roi / roi_mode / roi_overlap - 含义同 detect_humans
        roi_outline - 为 True 时在输出的标注图上叠加 ROI 轮廓，便于核对标定结果
    """
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
                                           conf_threshold=conf_threshold, roi=roi,
                                           roi_mode=roi_mode, roi_overlap=roi_overlap)

            # 输出检测结果
            if len(filtered_rects) > 0:
                logger.info("图片 %s 中检测到人形，数量: %d", filename, len(filtered_rects))
                # 在图片上绘制检测框并保存
                annotated = draw_boxes(image, filtered_rects)
                if roi_outline:
                    pixel_roi = normalize_roi(roi, image.shape[1], image.shape[0])
                    if pixel_roi:
                        annotated = draw_roi(annotated, pixel_roi)
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
    parser.add_argument('--roi', type=str, default=None,
                        help='检测区域（归一化多边形），如 "0,0 1,0 1,0.72 0,0.72" 或 JSON 格式；'
                             '多个多边形（并集）用 | 分隔坐标对，或直接给 JSON 数组嵌套；'
                             '不指定则全图检测')
    parser.add_argument('--roi-mode', type=str, choices=list(ROI_MODES), default=ROI_MODE_FILTER,
                        help='ROI 生效方式: filter=全图检测后过滤(默认), crop=仅裁剪 ROI 区域检测')
    parser.add_argument('--roi-overlap', type=float, default=None,
                        help='框与 ROI 交集占比阈值，达到该值时也算命中（默认不启用，仅按框中心点判定）')
    parser.add_argument('--roi-outline', action='store_true',
                        help='在输出的标注图上叠加 ROI 轮廓，便于核对标定结果')
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
        try:
            roi = parse_roi_arg(args.roi)
        except ValueError as e:
            logger.error("ROI 参数无效: %s", e)
            raise SystemExit(1)
        if roi:
            logger.info("启用检测区域 ROI: %d个多边形, 共%d个顶点, 模式: %s",
                        len(roi), sum(len(polygon) for polygon in roi), args.roi_mode)
        # 开始检测
        detect_humans_in_folder(args.folder, method=args.method, yolo_model=args.model,
                                conf_threshold=args.threshold, roi=roi,
                                roi_mode=args.roi_mode, roi_overlap=args.roi_overlap,
                                roi_outline=args.roi_outline)
