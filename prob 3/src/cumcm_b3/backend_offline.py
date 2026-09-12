"""离线自验模拟器：在本地复现赛题附录 1、2 的全部规则。

存在意义：演练测试每次都要在模拟器界面手工启动（5 秒倒计时 + 25 分钟窗口），
不可能用它来做算法调试。于是把赛题规则原样实现成本地模拟器，
让策略先在"零风险、可批量、可重复"的环境里闭环验证，再上真实演练。

与真实模拟器严格对齐的行为约定（均出自附件 1《模拟器使用说明》）：
1. 一次 /measure 的虚拟耗时 = 移动耗时 + 频道切换耗时(变化时 1 s) + 检测 5 s；
2. 一次 /clear 的虚拟耗时 = 移动耗时 + 光学定位 3 s (+ 命中再 2 s)；
   **/clear 不会改变测向机当前频道**，因此不会引入切换耗时；
3. 检测结果三态：no_signal / near(<=5 m) / direction(>5 m 且在有效半径内)；
4. 示向度 = 真实方位角 + 误差，且"同一地点重复检测不改变误差"，
   所以用 (位置, 频道) 的确定性散列实现"当地电磁环境固定"这一性质；
5. 清除成功当且仅当指定频道的未清除源落在 20 m 内；同一源只能被清除一次。
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    ARENA_RADIUS_M,
    BEARING_ERROR_DEG,
    CHANNEL_COUNT,
    CLEAR_HIT_COST_S,
    CLEAR_MISS_COST_S,
    CLEAR_RADIUS_M,
    DOG_SPEED_MPS,
    MEASURE_COST_S,
    NEAR_RADIUS_M,
    PROGRAM_TIME_LIMIT_S,
    RECEIVER_MAX_M,
    RECEIVER_MIN_M,
    SOURCE_COUNT_MAX,
    SOURCE_COUNT_MIN,
    SWITCH_COST_S,
    TEST_WINDOW_S,
    VIRTUAL_TIME_LIMIT_S,
)


@dataclass
class Source:
    """一个干扰源。问题 3 中全部为全向源。

    非冻结：清除操作需要就地改写 cleared 标记。
    """

    channel: int
    x: float
    y: float
    receiver_radius_m: float
    cleared: bool = False

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class WorldCases:
    """一次测试案例的全部真值——只有"上帝视角"才看得到，策略无权访问。"""

    sources: List[Source]
    seed: int

    @property
    def total(self) -> int:
        return len(self.sources)

    def count_cleared(self) -> int:
        return sum(1 for s in self.sources if s.cleared)


def _deterministic_error(x: float, y: float, channel: int) -> float:
    """把 (检测点, 频道) 映射为 [-1, 1] 内的固定误差。

    sha1 保证同一地点重复检测得到完全相同的误差，这与"电磁环境固定"一致；
    不同频道使用不同盐值，避免所有频道共享同一偏差这种不真实的简化。
    """
    key = f"{x:.3f}|{y:.3f}|{channel}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    value = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF  # [0, 1]
    return (2.0 * value - 1.0) * BEARING_ERROR_DEG


def sample_uniform_in_disk(rng: random.Random, radius: float) -> Tuple[float, float]:
    """面积均匀抽样：r = R*sqrt(U)，否则点会向圆心堆积。"""
    r = radius * math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return (r * math.cos(theta), r * math.sin(theta))


def generate_case(seed: int, n_sources: Optional[int] = None) -> WorldCases:
    """随机生成一个测试案例。

    n_sources 为空时在 [10, 16] 内随机取值，符合赛题"个数未知"的设定。
    """
    rng = random.Random(seed)
    n = n_sources if n_sources is not None else rng.randint(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX)
    channels = rng.sample(range(1, CHANNEL_COUNT + 1), n)
    sources: List[Source] = []
    for ch in channels:
        x, y = sample_uniform_in_disk(rng, ARENA_RADIUS_M)
        radius = rng.uniform(RECEIVER_MIN_M, RECEIVER_MAX_M)
        sources.append(Source(channel=ch, x=x, y=y, receiver_radius_m=radius))
    return WorldCases(sources=sources, seed=seed)


class DogBackend:
    """机器狗接口的最小抽象：真实客户端与离线模拟器都实现它。"""

    def enter(self) -> Dict[str, Any]:  # pragma: no cover - 接口定义
        raise NotImplementedError

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        raise NotImplementedError

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        raise NotImplementedError

    def exit(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def virtual_time_s(self) -> float:
        raise NotImplementedError

    @property
    def request_count(self) -> int:
        raise NotImplementedError


class OfflineSimulator(DogBackend):
    """本地模拟器。对外行为、字段名、耗时规则与真实模拟器一致。"""

    def __init__(self, case: WorldCases) -> None:
        self.case = case
        self._x = 0.0
        self._y = 0.0
        self._channel = 1
        self._entered = False
        self._exited = False
        self._virtual_time = 0.0
        self._requests = 0
        self.trace: List[Dict[str, Any]] = []
        self.clear_attempts = 0
        self.clear_hits = 0

    # ---- 状态访问 ----
    @property
    def virtual_time_s(self) -> float:
        return self._virtual_time

    @property
    def request_count(self) -> int:
        return self._requests

    @property
    def position(self) -> Tuple[float, float]:
        return (self._x, self._y)

    @property
    def current_channel(self) -> int:
        return self._channel

    # ---- 内部工具 ----
    def _move(self, x: float, y: float) -> float:
        distance = math.hypot(x - self._x, y - self._y)
        cost = distance / DOG_SPEED_MPS
        self._virtual_time += cost
        self._x, self._y = float(x), float(y)
        return cost

    def _record(self, kind: str, payload: Dict[str, Any]) -> None:
        self.trace.append({"virtual_time_s": round(self._virtual_time, 3), "kind": kind, **payload})

    def _require_active(self) -> None:
        if not self._entered or self._exited:
            raise RuntimeError("必须先调用 /enter 且尚未 /exit")

    # ---- 四大指令 ----
    def enter(self) -> Dict[str, Any]:
        if self._entered:
            raise RuntimeError("重复 /enter")
        self._entered = True
        self._requests += 1
        self._x = self._y = 0.0
        self._channel = 1
        self._record("enter", {})
        return {
            "accepted": True,
            "max_virtual_duration_s": VIRTUAL_TIME_LIMIT_S,
            "max_real_duration_s": PROGRAM_TIME_LIMIT_S,
            "remaining_real_duration_s": PROGRAM_TIME_LIMIT_S,
            "test_window_s": TEST_WINDOW_S,
        }

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        self._require_active()
        if not isinstance(channel, int) or not 1 <= channel <= CHANNEL_COUNT:
            raise ValueError("channel 必须是 1..20 的整数")
        move_cost = self._move(x, y)
        switch_cost = SWITCH_COST_S if channel != self._channel else 0.0
        self._channel = channel
        self._virtual_time += switch_cost + MEASURE_COST_S
        self._requests += 1

        source = self._source_of(channel)
        result: Dict[str, Any] = {
            "accepted": True,
            "measure_result": "no_signal",
            "move_cost_s": round(move_cost, 6),
            "switch_cost_s": round(switch_cost, 6),
        }
        if source is not None and not source.cleared:
            distance = math.hypot(source.x - x, source.y - y)
            if distance <= NEAR_RADIUS_M:
                result["measure_result"] = "near"
                result["distance_m"] = round(distance, 3)
            elif distance <= source.receiver_radius_m:
                true_bearing = math.degrees(math.atan2(source.y - y, source.x - x)) % 360.0
                noisy = (true_bearing + _deterministic_error(x, y, channel)) % 360.0
                result["measure_result"] = "direction"
                result["svd_deg"] = round(noisy, 6)
                result["distance_m"] = round(distance, 3)
        self._record("measure", {"x": round(x, 3), "y": round(y, 3), "channel": channel, **{
            k: v for k, v in result.items() if k not in ("accepted",)}})
        return result

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        self._require_active()
        if not isinstance(channel, int) or not 1 <= channel <= CHANNEL_COUNT:
            raise ValueError("channel 必须是 1..20 的整数")
        move_cost = self._move(x, y)
        self._requests += 1
        source = self._source_of(channel)
        hit = (
            source is not None
            and not source.cleared
            and math.hypot(source.x - x, source.y - y) <= CLEAR_RADIUS_M
        )
        cost = CLEAR_HIT_COST_S if hit else CLEAR_MISS_COST_S
        self._virtual_time += cost
        if hit:
            source.cleared = True
        self.clear_attempts += 1
        self.clear_hits += int(hit)
        result = {
            "accepted": True,
            "clear_result": "success" if hit else "no_target_in_range",
            "move_cost_s": round(move_cost, 6),
        }
        # 注意：/clear 不改变测向机频道状态（附件 1 第 3 节示例）
        self._record("clear", {"x": round(x, 3), "y": round(y, 3), "channel": channel,
                               "result": result["clear_result"]})
        return result

    def exit(self) -> Dict[str, Any]:
        self._require_active()
        self._exited = True
        self._requests += 1
        self._record("exit", {"virtual_time_s": round(self._virtual_time, 3)})
        return {"accepted": True, "exit_reason": "user_exit"}

    # ---- 辅助 ----
    def _source_of(self, channel: int) -> Optional[Source]:
        for s in self.case.sources:
            if s.channel == channel:
                return s
        return None

    def remaining(self) -> Sequence[Source]:
        """仍未清除的源（仅测试脚本使用，策略不得访问）。"""
        return [s for s in self.case.sources if not s.cleared]
