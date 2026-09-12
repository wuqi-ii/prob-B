"""第二检测点的候选生成——直接沿用问题 2 的成果。

问题 2 的结论（见 `../prob 2/res.md`）：在保留三项硬保证的前提下，
第二检测点应落在候选区域

    a >= 0,
    a^2 + b^2 <= R^2,
    (a - D)^2 + (|b| + h)^2 <= R^2,
    |b| >= t*a,
    |b| >= t*(D - a) + h

其中局部坐标以第一检测点为原点、a 轴沿第一次示向度方向，
    R = 1000 m   保守可接收距离（接收半径下界）
    D = 1500 m   源可能出现的最大距离（接收半径上界）
    eps = 1°     示向度误差半宽
    h = D*tan(eps) = 26.18 m    由误差张开量决定的纵向净空
    t = tan(30° + eps) = tan31° = 0.6009   交会角下界对应的斜率

在该区域内以「移动时间 + 定位表现的均值与 CVaR」为代价优选，
默认案例得到 (a, b) = (764, 526) m，行程 927.6 m；该坐标通过
``StrategyConfig.second_station_local_a_m/b_m`` 配置，可用于多组结果对照。
由于候选区域关于 a 轴对称，(764, +526) 与 (764, -526) 等价——
这正是"两个候选点"，第三问按**路径代价**在二者中取其一。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .config import StrategyConfig

Point = Tuple[float, float]


@dataclass(frozen=True)
class CandidatePair:
    """一对镜像候选点，以及生成它们时的中间量（便于写论文与自查）。"""

    left: Point          # 全局坐标：示向度左侧
    right: Point         # 全局坐标：示向度右侧
    first_station: Point
    bearing_deg: float
    local_optimum: Tuple[float, float]
    separation_m: float     # 第一站到候选点的距离


def _feasible(a: float, b: float, reach: float, far: float, h: float, t: float) -> bool:
    if a < 0:
        return False
    if math.hypot(a, b) > reach + 1e-9:
        return False
    if math.hypot(a - far, abs(b) + h) > reach + 1e-9:
        return False
    if abs(b) < t * a - 1e-9:
        return False
    if abs(b) < t * (far - a) + h - 1e-9:
        return False
    return True


def _local_optimal_point(cfg: StrategyConfig) -> Tuple[float, float]:
    """返回配置指定的候选区域代表点，并再次执行就地约束校验。"""
    reach = cfg.second_station_reach_m
    far = cfg.second_station_far_m
    h = cfg.second_station_clearance_h
    t = math.tan(math.radians(cfg.second_station_min_angle_deg))
    point = (cfg.second_station_local_a_m, cfg.second_station_local_b_m)
    if not _feasible(point[0], point[1], reach, far, h, t):
        raise ValueError(
            "配置的第二检测点不在候选区域内："
            f"(a, b)=({point[0]:g}, {point[1]:g})"
        )
    return point


def local_to_global(local: Point, station: Point, bearing_deg: float) -> Point:
    """局部坐标 -> 全局坐标：先按方位角旋转，再平移到第一检测点。"""
    a, b = local
    rad = math.radians(bearing_deg)
    x = a * math.cos(rad) - b * math.sin(rad)
    y = a * math.sin(rad) + b * math.cos(rad)
    return (station[0] + x, station[1] + y)


def candidate_pair(station: Point, bearing_deg: float, cfg: StrategyConfig) -> CandidatePair:
    """给出某频道第二检测点的两个镜像候选点。"""
    a, b = _local_optimal_point(cfg)
    left = local_to_global((a, b), station, bearing_deg)
    right = local_to_global((a, -b), station, bearing_deg)
    return CandidatePair(
        left=left,
        right=right,
        first_station=station,
        bearing_deg=bearing_deg,
        local_optimum=(a, b),
        separation_m=math.hypot(a, b),
    )


def feasible_lobe_points(station: Point, bearing_deg: float, cfg: StrategyConfig,
                         limit: int = 40) -> List[Point]:
    """候选区域的采样点（用于把候选点约束回地图内部时的备选）。"""
    reach = cfg.second_station_reach_m
    far = cfg.second_station_far_m
    h = cfg.second_station_clearance_h
    t = math.tan(math.radians(cfg.second_station_min_angle_deg))
    out: List[Point] = []
    a = 0.0
    while a <= reach and len(out) < limit:
        b = max(t * a, t * (far - a) + h)
        if _feasible(a, b, reach, far, h, t):
            out.append(local_to_global((a, b), station, bearing_deg))
            out.append(local_to_global((a, -b), station, bearing_deg))
        a += cfg.grid_step_m * 5
    return out


def clamp_to_arena(point: Point, arena_radius: float) -> Point:
    """机器狗允许走出目标区域，但把候选点拉回区域边缘通常更省路程。"""
    x, y = point
    r = math.hypot(x, y)
    if r <= arena_radius:
        return point
    scale = arena_radius / r
    return (x * scale, y * scale)
