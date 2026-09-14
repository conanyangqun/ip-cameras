import time
import hmac
import hashlib
import base64
import logging
import threading
import urllib.parse
from collections import deque

import requests

logger = logging.getLogger(__name__)


class DingTalkNotifier:
    """
    钉钉机器人通知器。
    注意钉钉机器人限流规则：每个机器人每分钟最多发送 20 条消息，
    超出限流后会被平台禁言 10 分钟。因此本类做了两层保护：
    1. 全局滑动窗口限流（默认每分钟最多 20 条，不可突破平台上限）；
    2. 每个摄像头的告警冷却时间（notify_cooldown），避免同一摄像头频繁刷屏。
    """

    # 钉钉平台硬限制：每分钟最多 20 条
    PLATFORM_MAX_PER_MINUTE = 20

    def __init__(self, webhook, secret=None, notify_cooldown=300,
                 max_per_minute=PLATFORM_MAX_PER_MINUTE):
        self.webhook = webhook
        self.secret = secret
        self.notify_cooldown = max(notify_cooldown, 0)
        self.max_per_minute = min(max_per_minute, self.PLATFORM_MAX_PER_MINUTE)
        self._lock = threading.Lock()
        self._send_history = deque()  # 最近 60 秒内的发送时间戳
        self._last_notify = {}  # 每个摄像头最近一次成功通知的时间戳

    def _build_url(self):
        """
        构建带加签的 webhook 地址（安全设置为自定义密钥时需要）
        """
        if not self.secret:
            return self.webhook
        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode('utf-8'),
            string_to_sign.encode('utf-8'),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return f"{self.webhook}&timestamp={timestamp}&sign={sign}"

    def _wait_for_slot(self):
        """
        滑动窗口限流：确保不超过每分钟 max_per_minute 条
        """
        with self._lock:
            now = time.time()
            while self._send_history and now - self._send_history[0] >= 60:
                self._send_history.popleft()
            if len(self._send_history) < self.max_per_minute:
                self._send_history.append(now)
                return
            # 需要等待最早一条记录滑出 60 秒窗口
            wait = 60 - (now - self._send_history[0]) + 0.5

        logger.warning("钉钉机器人达到每分钟发送上限，等待 %.1f 秒", wait)
        time.sleep(wait)

        with self._lock:
            self._send_history.append(time.time())

    def send_text(self, content, at_mobiles=None, at_all=False):
        """
        发送文本消息
        params:
            content 文本内容,字符串
        """
        payload = {
            "msgtype": "text",
            "text": {"content": content},
            "at": {
                "atMobiles": at_mobiles or [],
                "isAtAll": at_all,
            },
        }
        return self._send(payload)

    def _send(self, payload):
        """
        发送消息,判断是否成功发送
        """
        try:
            self._wait_for_slot()
            resp = requests.post(self._build_url(), json=payload, timeout=10)
            data = resp.json()
            if data.get('errcode') == 0:
                logger.info("钉钉通知发送成功")
                return True
            logger.error("钉钉通知发送失败: %s", data)
            return False
        except Exception as e:
            logger.error("钉钉通知发送异常: %s", e)
            return False

    def send_motion_alert(self, camera_name, human_count):
        """
        发送人形检测告警，带每摄像头冷却时间，返回是否实际发送
        """
        now = time.time()
        with self._lock:
            last = self._last_notify.get(camera_name, 0)
            if now - last < self.notify_cooldown:
                # 未通过冷却期，不发送告警
                return False

        from datetime import datetime
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        content = f"【人形告警】摄像头 {camera_name} 于 {time_str} 检测到人形，数量: {human_count}"
        sent = self.send_text(content)

        if sent:
            with self._lock:
                self._last_notify[camera_name] = now
        return sent


def _run_test(webhook, secret=None, notify_cooldown=300):
    """测试钉钉机器人是否可以正常工作"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    notifier = DingTalkNotifier(
        webhook=webhook,
        secret=secret,
        notify_cooldown=notify_cooldown,
    )

    print("=" * 50)
    print("测试 1: 发送普通文本消息")
    print("=" * 50)
    ok = notifier.send_text("【测试消息】钉钉机器人连通性测试，收到此消息说明机器人工作正常。")
    print("结果: %s" % ("成功 ✅" if ok else "失败 ❌，请检查 webhook/secret 配置"))

    print("=" * 50)
    print("测试 2: 发送人形告警消息（模拟）")
    print("=" * 50)
    ok2 = notifier.send_motion_alert("test-camera", 1)
    print("结果: %s" % ("成功 ✅" if ok2 else "失败 ❌"))

    if ok and ok2:
        print("\n测试通过: 钉钉机器人工作正常，请到钉钉群确认已收到两条消息。")
    else:
        print("\n测试失败: 请检查以下常见问题:")
        print("  1. webhook 地址是否正确（须包含 access_token）")
        print("  2. 若机器人启用了加签，secret 是否正确（以 SEC 开头）")
        print("  3. 机器人安全设置是否与代码使用的验证方式一致")
        print("  4. 是否触发了钉钉限流（每分钟最多 20 条）")


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description='测试钉钉机器人是否可以正常工作')
    parser.add_argument('--webhook', type=str, default=None,
                        help='钉钉机器人 webhook 地址（不指定则从 cameras.json 读取）')
    parser.add_argument('--secret', type=str, default=None,
                        help='加签密钥（不指定则从 cameras.json 读取）')
    parser.add_argument('--config', type=str, default='cameras.json',
                        help='配置文件路径，默认为 cameras.json')
    args = parser.parse_args()

    webhook = args.webhook
    secret = args.secret

    # 未通过命令行指定时，从配置文件读取
    if not webhook:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            dingtalk_cfg = cfg.get('dingtalk', {})
            if not dingtalk_cfg.get('enabled'):
                print(f"配置文件 {args.config} 中 dingtalk.enabled 为 false，"
                      f"仍将按现有配置尝试发送测试消息...")
            webhook = dingtalk_cfg.get('webhook')
            secret = secret or dingtalk_cfg.get('secret')
        except FileNotFoundError:
            print(f"错误: 未指定 --webhook 且找不到配置文件 {args.config}")
            raise SystemExit(1)
        except Exception as e:
            print(f"错误: 读取配置文件失败: {e}")
            raise SystemExit(1)

    if not webhook or 'access_token=' not in webhook:
        print("错误: webhook 无效，须为包含 access_token 的钉钉机器人地址")
        raise SystemExit(1)

    _run_test(webhook, secret)
