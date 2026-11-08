"""
ROI 标定工具

在抓拍画面上用鼠标点选多边形检测区域，输出归一化坐标，可直接粘贴到 cameras.json，
也可以加 --write 自动写回配置。只有落在该区域内的检测框才会被判定为人形。

用法:
    # 从摄像头实时抓一帧后标定
    python select_roi.py --camera yuanzi

    # 用已有抓拍图标定
    python select_roi.py --image test/camera_yuanzi_2026-09-22-10-00-00.jpg

    # 标定并直接写回 cameras.json 中该摄像头的 roi 字段
    python select_roi.py --camera yuanzi --write

    # 每 10 秒自动重抓一帧刷新底图，用于确认误报物体位置是否稳定
    python select_roi.py --camera yuanzi --refresh 10

操作说明（窗口内）:
    鼠标左键      - 为当前多边形添加顶点
    鼠标右键 / u  - 撤销当前多边形的最后一个顶点
    回车 / 空格   - 确认当前多边形；当前没有未确认的顶点时结束标定并输出
    b             - 删除上一个已确认的多边形
    r             - 清空全部顶点重新点选
    c / ESC / q   - 取消退出

一个摄像头可以标定多个多边形，它们取并集：命中原有的任意一个区域都算有效检出区域。
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np

from camera_capture import capture_frame

# 显示窗口的最大尺寸，避免大分辨率画面超出屏幕导致无法操作
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 800

WINDOW_NAME = 'select_roi'


def load_config(config_path):
    """读取 cameras.json"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_camera(config, name):
    """按名称查找摄像头配置，找不到返回 None"""
    for camera in config.get('cameras', []):
        if camera.get('name') == name:
            return camera
    return None


def fetch_frame_from_camera(camera):
    """从摄像头实时抓取一帧画面，失败返回 None"""
    frame = capture_frame(camera.get('rtsp'), camera.get('protocol'))
    if frame is None:
        print(f"错误: 无法从摄像头 {camera.get('name')} 抓取画面，请检查 rtsp 地址与网络")
    return frame


def resize_for_display(image):
    """
    等比缩小到适合屏幕显示的尺寸
    返回: (显示用图像, 缩放比例)
    """
    height, width = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return image, 1.0
    return cv2.resize(image, (int(width * scale), int(height * scale))), scale


def select_roi_interactive(image, refresh_callback=None, refresh_interval=0):
    """
    在窗口中用鼠标点选多边形顶点，支持连续标定多个多边形
    params:
        image - 底图（抓拍画面）
        refresh_callback - 可选，返回新底图的回调，用于定时刷新画面
        refresh_interval - 刷新间隔（秒），需与 refresh_callback 配合使用
    返回: 归一化多边形列表 [[[x, y], ...], ...]；取消或没有有效多边形时返回 None
    """
    display, scale = resize_for_display(image)
    polygons = []  # 已确认的多边形（显示坐标）
    current = []   # 正在点选的多边形（显示坐标）
    cancelled = False
    last_refresh = time.time()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and current:
            current.pop()

    def draw_canvas():
        canvas = display.copy()
        # 已确认的多边形：闭合轮廓
        for polygon in polygons:
            cv2.polylines(canvas, [np.array(polygon, dtype=np.int32)], True, (0, 255, 255), 2)
        # 正在点选的多边形：顶点圆点 + 依次连线
        for i, point in enumerate(current):
            cv2.circle(canvas, point, 4, (0, 255, 255), -1)
            if i > 0:
                cv2.line(canvas, current[i - 1], point, (0, 255, 255), 2)
        if len(current) >= 3:
            # 用不同颜色画出待闭合的边，提示多边形形状
            cv2.line(canvas, current[-1], current[0], (0, 165, 255), 1)
        hint = ("左键加顶点 | 回车确认多边形(当前为空则完成) | u 撤销点 | b 删多边形 | "
                "r 清空 | c/ESC 取消 | 已确认 %d 个, 当前 %d 点"
                % (len(polygons), len(current)))
        cv2.putText(canvas, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return canvas

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            # 定时重抓底图：保留已点顶点（归一化位置不变），仅验证物体是否仍在同一位置
            if refresh_callback and refresh_interval > 0:
                if time.time() - last_refresh >= refresh_interval:
                    last_refresh = time.time()
                    new_frame = refresh_callback()
                    if new_frame is not None:
                        display, scale = resize_for_display(new_frame)

            cv2.imshow(WINDOW_NAME, draw_canvas())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10, 32):  # 回车 / 空格
                if len(current) >= 3:
                    # 确认当前多边形，接着可以继续点选下一个
                    polygons.append(list(current))
                    current.clear()
                elif current:
                    print("提示: 当前多边形至少需要 3 个顶点才能确认")
                elif polygons:
                    # 没有未确认的顶点，结束标定
                    break
                else:
                    print("提示: 至少需要确认一个多边形（3 个顶点以上）")
            elif key == ord('u') and current:
                current.pop()
            elif key == ord('b') and polygons:
                polygons.pop()
            elif key == ord('r'):
                polygons.clear()
                current.clear()
            elif key in (ord('c'), 27, ord('q')):  # c / ESC / q
                cancelled = True
                break
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        # 部分平台需要多轮 waitKey 才能关闭窗口
        cv2.waitKey(1)

    if cancelled or not polygons:
        return None

    # 显示坐标 -> 原图归一化坐标
    height, width = image.shape[:2]
    result = []
    for polygon in polygons:
        result.append([[round((x / scale) / width, 4), round((y / scale) / height, 4)]
                       for (x, y) in polygon])
    return result


def main():
    parser = argparse.ArgumentParser(description='ROI 标定工具：鼠标点选多边形检测区域并输出配置')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=str, help='已有抓拍图片路径')
    source.add_argument('--camera', type=str, help='摄像头名称，从配置读取 rtsp 后实时抓一帧')
    parser.add_argument('--config', type=str, default='cameras.json',
                        help='配置文件路径，默认为 cameras.json')
    parser.add_argument('--write', action='store_true',
                        help='将标定结果写回配置文件中该摄像头的 roi 字段（仅 --camera 模式可用）')
    parser.add_argument('--refresh', type=float, default=0,
                        help='--camera 模式下每隔多少秒重抓一帧刷新底图，0 表示不刷新')
    args = parser.parse_args()

    config = None
    camera = None

    if args.camera:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            print(f"错误: 找不到配置文件 {args.config}")
            return 1
        except Exception as e:
            print(f"错误: 读取配置文件失败: {e}")
            return 1

        camera = find_camera(config, args.camera)
        if camera is None:
            print(f"错误: 在 {args.config} 中找不到摄像头 {args.camera}")
            return 1
        print(f"正在从摄像头 {args.camera} 抓取画面...")
        image = fetch_frame_from_camera(camera)
        if image is None:
            return 1
    else:
        image = cv2.imread(args.image)
        if image is None:
            print(f"错误: 无法读取图片 {args.image}")
            return 1

    height, width = image.shape[:2]
    print(f"图像尺寸: {width}x{height}（ROI 输出为相对宽高的归一化坐标）")
    print("标定多个多边形: 回车确认当前多边形后可继续点选下一个，没有未确认顶点时回车结束")

    refresh_callback = None
    if args.camera and args.refresh > 0:
        refresh_callback = lambda: fetch_frame_from_camera(camera)

    roi = select_roi_interactive(image, refresh_callback=refresh_callback,
                                 refresh_interval=args.refresh)
    if roi is None:
        print("已取消，未生成 ROI")
        return 1

    print("=" * 64)
    print("ROI 标定结果（%d 个多边形，归一化坐标，可粘贴到 cameras.json 对应摄像头中）:" % len(roi))
    print('    "roi": %s,' % json.dumps(roi))
    print("=" * 64)

    if args.write:
        if camera is None:
            print("提示: --write 仅在 --camera 模式下可用（需要知道写回哪个摄像头）")
        else:
            camera['roi'] = roi
            with open(args.config, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                f.write('\n')
            print(f"已写入 {args.config} 中摄像头 {args.camera} 的 roi 字段")

    return 0


if __name__ == '__main__':
    sys.exit(main())
