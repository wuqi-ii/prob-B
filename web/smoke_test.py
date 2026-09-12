"""Minimal real-simulator connection test. Use exercise mode only."""

from __future__ import annotations

import argparse
import json
from urllib.parse import urlsplit

from simulator_client import Position, SimulatorClient, SimulatorError, probe


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe connection test for the B-problem simulator")
    parser.add_argument("--robot-id", help="team number currently logged into the simulator")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--measure", action="store_true", help="also make one measurement at (0,0), channel 1")
    parser.add_argument("--confirm-exercise", action="store_true", help="confirm that an exercise, not formal, test is active")
    args = parser.parse_args()

    parts = urlsplit(args.base_url)
    if not probe(parts.hostname or "127.0.0.1", parts.port or 2026):
        print("连接未开放：请启动模拟机中的演练测试，并等待 5 秒倒计时结束。")
        return 2
    print("端口已开放。")
    if not args.robot_id:
        print("仅完成无动作探测；要联调请传入 --robot-id 当前登录队号。")
        return 0
    if not args.confirm_exercise:
        print("已停止：真实动作只允许在演练测试中运行，请添加 --confirm-exercise。")
        return 3

    client = SimulatorClient(args.robot_id, base_url=args.base_url)
    entered = False
    try:
        entered_result = client.enter()
        entered = True
        print("ENTER:", json.dumps(entered_result, ensure_ascii=False))
        if args.measure:
            measured = client.measure(Position(0, 0), 1)
            print("MEASURE:", json.dumps(measured, ensure_ascii=False))
        exited = client.exit()
        entered = False
        print("EXIT:", json.dumps(exited, ensure_ascii=False))
        return 0
    except SimulatorError as exc:
        print(f"联调失败 [{type(exc).__name__}]: {exc}")
        if entered:
            print("提示：已成功 enter 但未确认 exit，请在模拟机界面检查本次演练状态。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
