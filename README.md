# ip-cameras

从 IP 摄像头（RTSP 协议）定时抓拍画面，并结合人形检测实现智能保存与钉钉告警通知。

## 功能特性

- **多摄像头定时抓拍**：支持配置多个摄像头，每个摄像头独立线程采集，支持 TCP/UDP 传输协议
- **人形检测智能保存**：
  - 未检测到人形时，按 `timelapse_interval` 间隔保存延时摄影截图
  - 检测到人形时，按 `motion_interval` 间隔保存截图，并额外保存带矩形框标注的图像（文件名带 `_detected` 后缀）
- **钉钉机器人告警**：检测到人形时自动推送告警消息到钉钉群，支持加签验证、全局限流（每分钟最多 20 条，符合钉钉平台限制）和每摄像头告警冷却时间
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
            "motion_interval": 5       // 检测到人形时的保存间隔（秒）
        }
    ]
}
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

## 工作原理

1. `camera_capture.py` 通过 `ffmpeg` 命令行从 RTSP 流抓取单帧图像
2. 使用 OpenCV 内置 HOG+SVM 行人检测器判断画面中是否有人形（含置信度、宽高比、尺寸、边缘密度过滤和非极大值抑制，降低误报）
3. 根据检测结果动态选择保存间隔，并触发钉钉机器人告警
4. 钉钉通知采用滑动窗口限流 + 每摄像头冷却双层保护，避免触发钉钉平台禁言（超限每分钟 20 条会被禁言 10 分钟）
