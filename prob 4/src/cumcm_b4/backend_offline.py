"""离线自验模拟器：在本地复现赛题附录 1、2 的全部规则（问题 4：含定向源）。

与问题 3 的离线模拟器相比，新增定向干扰源：

- 全向源：有效覆盖角度 360°，距离 <= 有效接收半径即测到；
- 定向源：有效覆盖角度为定向方向两侧各 90°（含边界），只有检测点同时满足
  「距离 <= 有效接收半径」且「位于覆盖楔内」才测到，否则 no_signal；
- ``/clear`` 只看距离（<=20 m），与覆盖角无关——定向源也能被楔外的点清除。

其余行为（移动/切换/检测计时、固定地点误差、串行、幂等）与问题 3 完全一致。
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    ARENA_RADIUS_M,
    BEARING_ERROR_DEG,
    CHANNEL_COUNT,
    CLEAR_HIT_COST_S,
    CLEAR_MISS_COST_S,
    CLEAR_RADIUS_M,
    DIRECTIONAL_HALF_ANGLE_DEG,
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

# 热循环内建引用提升（与 geometry.py 6.4 同源加固）。
_FLOAT = float


@dataclass
class Source:
    """一个干扰源。问题 4 中可能是全向源或定向源。

    非冻结：清除操作需要就地改写 cleared 标记。
    """

    channel: int
    x: float
    y: float
    receiver_radius_m: float
    directional: bool = False
    direction_deg: float = 0.0      # 定向方向（x 轴正向逆时针，[0,360)）
    cleared: bool = False

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def covers(self, px: float, py: float) -> bool:
        """检测点 (px,py) 是否位于该源的信号覆盖范围内（距离+角度）。"""
        dist = math.hypot(px - self.x, py - self.y)
        if dist > self.receiver_radius_m:
            return False
        if not self.directional:
            return True
        if dist < 1e-9:
            return True
        vx, vy = px - self.x, py - self.y
        rad = math.radians(self.direction_deg)
        cosang = (vx * math.cos(rad) + vy * math.sin(rad)) / dist
        return math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) <= DIRECTIONAL_HALF_ANGLE_DEG + 1e-9


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

    def directional_count(self) -> int:
        return sum(1 for s in self.sources if s.directional)


def _deterministic_error(x: float, y: float, channel: int) -> float:
    """把 (检测点, 频道) 映射为 [-1, 1] 内的固定误差。"""
    key = f"{x:.3f}|{y:.3f}|{channel}".encode("utf-8")
    digest = hashlib.sha1(key).digest()
    value = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return (2.0 * value - 1.0) * BEARING_ERROR_DEG


def sample_uniform_in_disk(rng: random.Random, radius: float) -> Tuple[float, float]:
    """面积均匀抽样。"""
    r = radius * math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return (r * math.cos(theta), r * math.sin(theta))


def generate_case(seed: int, n_sources: Optional[int] = None,
                  directional_prob: float = 0.5) -> WorldCases:
    """随机生成一个测试案例。

    n_sources 为空时在 [10, 16] 内随机取值。每个源独立以 probability
    directional_prob 成为定向源（方向均匀取 [0,360)）。
    """
    rng = random.Random(seed)
    n = n_sources if n_sources is not None else rng.randint(SOURCE_COUNT_MIN, SOURCE_COUNT_MAX)
    channels = rng.sample(range(1, CHANNEL_COUNT + 1), n)
    sources: List[Source] = []
    for ch in channels:
        x, y = sample_uniform_in_disk(rng, ARENA_RADIUS_M)
        radius = rng.uniform(RECEIVER_MIN_M, RECEIVER_MAX_M)
        directional = rng.random() < directional_prob
        direction_deg = rng.uniform(0.0, 360.0) if directional else 0.0
        sources.append(Source(channel=ch, x=x, y=y, receiver_radius_m=radius,
                              directional=directional, direction_deg=direction_deg))
    return WorldCases(sources=sources, seed=seed)


class DogBackend:
    """机器狗接口的最小抽象。"""

    def enter(self) -> Dict[str, Any]:  # pragma: no cover
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

    def _move(self, x: float, y: float) -> float:
        distance = math.hypot(x - self._x, y - self._y)
        cost = distance / DOG_SPEED_MPS
        self._virtual_time += cost
        self._x, self._y = _FLOAT(x), _FLOAT(y)
        return cost

    def _record(self, kind: str, payload: Dict[str, Any]) -> None:
        self.trace.append({"virtual_time_s": round(self._virtual_time, 3), "kind": kind, **payload})

    def _require_active(self) -> None:
        if not self._entered or self._exited:
            raise RuntimeError("必须先调用 /enter 且尚未 /exit")

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
            if distance <= NEAR_RADIUS_M and source.covers(x, y):
                result["measure_result"] = "near"
                result["distance_m"] = round(distance, 3)
            elif distance <= source.receiver_radius_m and source.covers(x, y):
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
        hit = False
        if source is not None and not source.cleared:
            # /clear 只看距离，与覆盖角无关
            if math.hypot(source.x - x, source.y - y) <= CLEAR_RADIUS_M:
                hit = True
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
        self._record("clear", {"x": round(x, 3), "y": round(y, 3), "channel": channel,
                               "result": result["clear_result"]})
        return result

    def exit(self) -> Dict[str, Any]:
        self._require_active()
        self._exited = True
        self._requests += 1
        self._record("exit", {"virtual_time_s": round(self._virtual_time, 3)})
        return {"accepted": True, "exit_reason": "user_exit"}

    def _source_of(self, channel: int) -> Optional[Source]:
        for s in self.case.sources:
            if s.channel == channel:
                return s
        return None

    def remaining(self) -> Sequence[Source]:
        return [s for s in self.case.sources if not s.cleared]
