# ip-cameras

从 IP 摄像头（RTSP 协议）定时抓拍画面，并结合人形检测实现智能保存与钉钉告警通知。

## 功能特性

- **多摄像头定时抓拍**：支持配置多个摄像头，每个摄像头独立线程采集，支持 TCP/UDP 传输协议
- **人形检测智能保存**：
  - 未检测到人形时，按 `timelapse_interval` 间隔保存延时摄影截图
  - 检测到人形时，按 `motion_interval` 间隔保存截图，并额外保存带矩形框标注的图像（文件名带 `_detected` 后缀）
- **钉钉机器人告警**：检测到人形时自动推送告警消息到钉钉群，支持加签验证、全局限流（每分钟最多 20 条，符合钉钉平台限制）和每摄像头告警冷却时间
- **ROI 检测区域**：可为每个摄像头配置多边形检测区域（归一化坐标），只有落在区域内的人形才算检出，用于屏蔽固定机位下墙角等区域的稳定误报
- **离线人形检测**：可对已抓拍的图片文件夹批量执行人形检测，并输出带标注框的图片

## 环境依赖

- Python 3.7+
- 系统需安装 [ffmpeg](https://ffmpeg.org/)（用于 RTSP 拉流）
- Python 依赖：

```bash
pip install -r requirements.txt
```

依赖包括：`opencv-python`、`numpy`、`requests`

## 配置说明

编辑 `cameras.json`：

```jsonc
{
    "store_path": "test",              // 截图保存目录
    "detection_method": "hog",         // 人形检测方法: hog=HOG+SVM(默认), yolo=YOLO ONNX(误报更低、更准)
    "yolo_model": "yolo26n.onnx",      // YOLO ONNX 模型文件路径（detection_method 为 yolo 时使用）
    "dingtalk": {                      // 钉钉机器人配置（可选）
        "enabled": false,              // 是否启用钉钉通知
        "webhook": "https://oapi.dingtalk.com/robot/send?access_token=xxxxxxxx",
        "secret": "SECxxxxxxxx",       // 加签密钥（机器人安全设置选择"加签"时必填）
        "notify_cooldown": 300,        // 同一摄像头告警冷却时间（秒），避免刷屏
        "max_per_minute": 20           // 每分钟最多发送条数（不超过钉钉平台 20 条限制）
    },
    "cameras": [
        {
            "name": "hutong",          // 摄像头名称（用于文件名和告警消息）
            "rtsp": "<real rtsp url>", // RTSP 流地址
            "protocol": "tcp",         // 传输协议: tcp / udp
            "capture_cycle": 3,        // 检测周期（秒），决定检测灵敏度的最小粒度，需小于 motion_interval
            "timelapse_interval": 60,  // 无人形时的保存间隔（秒）
            "motion_interval": 5,      // 检测到人形时的保存间隔（秒）
            "human_detection": true,   // 是否开启人形检测（默认开启）
            "human_detection_threshold": 0.4, // 每摄像头报出阈值（可选，默认 hog=0.75 / yolo=0.4）
            "roi_enabled": true,       // 是否启用 ROI 检测区域（可选，默认开启）
            "roi": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], // 人形检测区域: 归一化多边形，默认整幅画面；支持多个多边形（并集），见下方说明
            "roi_mode": "filter",      // ROI 生效方式: filter=全图检测后按 ROI 过滤(默认), crop=仅裁剪 ROI 区域送检测
            "roi_overlap": null,       // 框与 ROI 交集占比阈值（可选，默认 null 即仅按框中心点是否落在 ROI 内判定）
            "roi_debug": false         // 是否在 *_detected 标注图上叠加 ROI 轮廓，便于核对标定（可选）
        }
    ]
}
```

`roi` 相关配置既可写在单个摄像头内，也可写在顶层（与 `store_path` 同级）作为所有摄像头的默认值，摄像头内的同名配置优先。

- `roi_enabled` 为 `true`（默认）时按 `roi` 圈定检测区域；为 `false` 时忽略 `roi`，整幅画面都参与人形检测，已标定好的坐标会保留下来，随时可以再打开；
- `roi` 默认是整幅画面 `[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]`，等价于全图检测，标定后用实际区域替换即可；
- 完全未配置 `roi` 的摄像头同样是全图检测，行为与加入该功能之前完全一致。

`roi` 支持配置**多个多边形**：两层数组表示单个多边形，三层数组表示多个多边形，它们取并集（命中任意一个即算命中），用于画面中互不相连的多块区域。

```jsonc
// 单个多边形（区域本身可以是凹多边形，例如 L 形）
"roi": [[0.0, 0.0], [1.0, 0.0], [1.0, 0.72], [0.0, 0.72]]

// 两个多边形（并集）
"roi": [
    [[0.0, 0.0], [1.0, 0.0], [1.0, 0.72], [0.0, 0.72]],
    [[0.0, 0.78], [0.4, 0.78], [0.4, 1.0], [0.0, 1.0]]
]
```

## 使用方法

### 1. 定时抓拍 + 人形检测 + 钉钉告警

将 `cameras.json` 中的 `rtsp` 替换为真实摄像头地址，然后运行：

```bash
python camera_capture.py
```

程序将持续运行（Ctrl+C 退出），每个摄像头占用一个线程，抓拍结果保存到 `store_path` 目录，日志写入 `camera_capture.log`。

### 2. 测试钉钉机器人连通性

```bash
# 从 cameras.json 读取配置
python dingtalk.py

# 或通过命令行参数指定
python dingtalk.py --webhook "https://oapi.dingtalk.com/robot/send?access_token=xxx" --secret "SECxxx"
```

### 3. 对已抓拍图片批量人形检测

```bash
python human_detection.py test
```

对 `test` 目录下所有图片执行人形检测，检测结果打印到控制台，带标注框的图片保存为 `detected_原文件名`。
指定 `--roi` 后只有命中检测区域的人形才会被计数和标注，可用于回归验证 ROI 标定效果：

```bash
python human_detection.py test --roi "0,0 1,0 1,0.72 0,0.72" --roi-outline
```

### 4. 标定 ROI 检测区域

误报区域往往是固定机位下长期存在的一块区域（例如广角镜头画面左下角的静止物体），用 ROI 把它排除掉即可。ROI 用**归一化多边形顶点**描述（`x, y` 为相对图像宽高的 `0~1` 浮点数），分辨率或码流变化后配置依然有效。

标定靠鼠标点选，不需要手算坐标：

```bash
# 从摄像头实时抓一帧后点选
python select_roi.py --camera yuanzi

# 或直接用已有抓拍图
python select_roi.py --image test/camera_yuanzi_2026-09-22-10-00-00.jpg

# 标定完直接写回 cameras.json 中该摄像头的 roi 字段
python select_roi.py --camera yuanzi --write
```

窗口内操作：

| 操作 | 说明 |
|---|---|
| 鼠标左键 | 为当前多边形添加顶点 |
| 鼠标右键 / `u` | 撤销当前多边形的最后一个顶点 |
| 回车 / 空格 | 确认当前多边形（至少需要 3 个顶点）并开始下一个；当前没有未确认顶点时结束并输出 |
| `b` | 删除上一个已确认的多边形 |
| `r` | 清空全部顶点重新点选 |
| `c` / `ESC` / `q` | 取消退出 |

`cameras.json` 中已为每个摄像头预置了整幅画面的 `roi`，标定后替换即可。工具会把结果打印成可直接粘贴的配置片段（加 `--write` 则自动覆盖对应摄像头已有的 `roi`）：

```jsonc
"roi": [
    [[0.0, 0.0], [1.0, 0.72], [0.55, 0.72], [0.55, 0.0]],
    [[0.0, 0.8], [0.35, 0.8], [0.35, 1.0], [0.0, 1.0]]
],
```

一个摄像头可以标定多个多边形，用回车逐个确认：

- 区域带凹口（例如只排除左下角、保留右下角）→ 一个 L 形多边形就能表达；
- 画面中有互不相连的多块区域（例如左右两个墙角）→ 拆成多个多边形，取并集。

标定建议：

- 加 `--refresh 10` 让底图每 10 秒自动重抓一帧，确认误报物体长期停在相同位置，避免把偶尔路过的行人一并圈进禁区；
- 标定后先用 `python human_detection.py <样本目录> --roi "..." --roi-outline` 在历史图片上回归，**误报样本和真人样本都要测**，避免 ROI 收得太小导致漏检；
- 打开 `roi_debug` 后，实时链路保存的 `*_detected.jpg` 会叠加 ROI 轮廓，便于持续核对标定是否贴合。

标定工具需要图形界面（`cv2.imshow`），请在有桌面的环境运行；`--camera` 模式还需要系统已安装 ffmpeg 且能连通摄像头。

## 工作原理

1. `camera_capture.py` 通过 `ffmpeg` 命令行从 RTSP 流抓取单帧图像
2. 使用 OpenCV 内置 HOG+SVM 行人检测器（或 YOLO ONNX 模型）判断画面中是否有人形（含置信度、宽高比、尺寸、边缘密度过滤和非极大值抑制，降低误报）
3. 若配置了 `roi`，只保留落在该多边形区域内的人形：默认全图检测后过滤（`filter`），也可设为 `crop` 只把 ROI 区域送检测
4. 根据检测结果动态选择保存间隔，并触发钉钉机器人告警
5. 钉钉通知采用滑动窗口限流 + 每摄像头冷却双层保护，避免触发钉钉平台禁言（超限每分钟 20 条会被禁言 10 分钟）
