"""真实模拟器后端：把 HTTP 客户端包装成与离线模拟器完全一致的接口。

这样策略代码不需要区分自己在和谁对话；切换到真实演练只需换一个后端对象。

安全护栏
--------
1. 真实运行时间看门狗：接近 ``max_real_s``（默认留出充分余量）立刻停止并 /exit，
   避免撞上"程序运行超时"或 25 分钟窗口而被记为异常结束；
2. 请求数上限，防止异常循环在短时间内打出海量请求（附件 1 对日志体积有要求）；
3. 虚拟时间自记账：响应体不返回虚拟时间，按附录规则自行累加，与实际扣时应一致。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

from .config import (
    CLEAR_HIT_COST_S,
    CLEAR_MISS_COST_S,
    DOG_SPEED_MPS,
    MEASURE_COST_S,
    SWITCH_COST_S,
    StrategyConfig,
)


class HttpBackend:
    """DogBackend 的真实实现（鸭子类型，不继承以便无网时也能 import）。"""

    def __init__(self, cfg: StrategyConfig, log_path: Optional[str] = None) -> None:
        from .client import Position, SimulatorClient  # 局部导入：离线模式无需依赖

        self.cfg = cfg
        self._Position = Position
        self.client = SimulatorClient(
            robot_id=cfg.robot_id,
            base_url=cfg.base_url,
            timeout_s=cfg.request_timeout_s,
            log_path=log_path,
        )
        self._x = 0.0
        self._y = 0.0
        self._channel = 1
        self._virtual_time = 0.0
        self._server_virtual = 0.0
        self.max_drift = 0.0
        self._requests = 0
        self._entered = False
        self.real_start = 0.0
        self.max_real_s = 900.0            # 15 分钟，给收尾留 5 分钟
        self.max_virtual_s = 6 * 3600.0    # 虚拟时间软预算
        self.clear_attempts = 0
        self.clear_hits = 0
        self.trace: list = []

    # ---- 状态 ----
    @property
    def virtual_time_s(self) -> float:
        # 模拟器回报的 virtual_time_s 是权威值；本地记账只作兜底与校验。
        # 取较大者，预算判断偏保守。
        return max(self._virtual_time, self._server_virtual)

    @property
    def request_count(self) -> int:
        return self._requests

    @property
    def position(self) -> tuple[float, float]:
        return (self._x, self._y)

    @property
    def current_channel(self) -> int:
        return self._channel

    @property
    def real_elapsed_s(self) -> float:
        return time.time() - self.real_start if self.real_start else 0.0

    def budget_exhausted(self) -> bool:
        """任一预算见底都应立即收尾，优先保证/log 正常落地。"""
        return (
            self.real_elapsed_s >= self.max_real_s
            or self._virtual_time >= self.max_virtual_s
            or self._requests >= self.cfg.max_requests
        )

    # ---- 指令 ----
    def _move(self, x: float, y: float) -> float:
        cost = math.hypot(x - self._x, y - self._y) / DOG_SPEED_MPS
        self._virtual_time += cost
        self._x, self._y = float(x), float(y)
        return cost

    def _sync_virtual_time(self, res: Dict[str, Any], kind: str) -> None:
        """用模拟器回报的 virtual_time_s 校准本地记账。

        模拟器在每个成功响应里都会给出 virtual_time_s，这是权威值；
        本地按附录规则自行累加只用于在没有回报时兜底，以及校验规则理解是否正确。
        """
        server = res.get("virtual_time_s")
        if not isinstance(server, (int, float)):
            return
        self._server_virtual = float(server)
        drift = self._virtual_time - self._server_virtual
        if abs(drift) > abs(self.max_drift):
            self.max_drift = drift
            self.trace.append({"kind": "virtual_time_drift", "at": kind,
                               "local_s": round(self._virtual_time, 3),
                               "server_s": round(self._server_virtual, 3),
                               "drift_s": round(drift, 3)})

    def enter(self) -> Dict[str, Any]:
        self.real_start = time.time()
        res = self.client.enter()
        self._entered = True
        self._requests += 1
        remaining = float(res.get("remaining_real_duration_s", 1200.0))
        # 给自己留出收尾时间：取配置上限与模拟器返回的较小者，再打九折
        self.max_real_s = min(self.max_real_s, remaining * 0.9)
        self.trace.append({"kind": "enter", "response": res})
        return res

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        if not self._entered:
            raise RuntimeError("必须先 /enter")
        move_cost = self._move(x, y)
        switch_cost = SWITCH_COST_S if channel != self._channel else 0.0
        self._channel = channel
        res = self.client.measure(self._Position(x, y), channel)
        detect_cost = MEASURE_COST_S
        self._virtual_time += switch_cost + detect_cost
        self._sync_virtual_time(res, "measure")
        self._requests += 1
        out = dict(res)
        out.update({"move_cost_s": move_cost, "switch_cost_s": switch_cost,
                    "detect_cost_s": detect_cost})
        self.trace.append({"kind": "measure", "x": x, "y": y, "channel": channel,
                           "virtual_time_s": round(self._virtual_time, 3),
                           "measure_result": res.get("measure_result"),
                           "svd_deg": res.get("svd_deg")})
        return out

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        if not self._entered:
            raise RuntimeError("必须先 /enter")
        move_cost = self._move(x, y)
        res = self.client.clear(self._Position(x, y), channel)
        hit = res.get("clear_result") == "success"
        cost = CLEAR_HIT_COST_S if hit else CLEAR_MISS_COST_S
        self._virtual_time += cost
        self._sync_virtual_time(res, "clear")
        self._requests += 1
        self.clear_attempts += 1
        self.clear_hits += int(hit)
        out = dict(res)
        out.update({"move_cost_s": move_cost, "clear_cost_s": cost})
        # 注意：/clear 不改变测向机频道状态
        self.trace.append({"kind": "clear", "x": x, "y": y, "channel": channel,
                           "virtual_time_s": round(self._virtual_time, 3),
                           "clear_result": res.get("clear_result")})
        return out

    def exit(self) -> Dict[str, Any]:
        res = self.client.exit()
        self._requests += 1
        self.trace.append({"kind": "exit", "virtual_time_s": round(self._virtual_time, 3),
                           "exit_reason": res.get("exit_reason")})
        return res
