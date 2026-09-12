"""七点覆盖普查：布局、覆盖性论证与数值校核。

为什么必须做"覆盖普查"
----------------------
干扰源个数未知（10~16）、分布在半径 1800 m 的圆域内，机器狗无法读取真值。
要让"所有干扰源都被清除"成为一个**可证明的结论**，第一步必须保证
每个可能存在源的位置，至少被某个必访扫描点的有效接收半径覆盖。

由于单个源的有效接收半径只在 [1000, 1500] m 之间、具体值未知，
覆盖设计只能按最坏情况 1000 m 来做——这是保守且唯一安全的选择。

几何结论（六边形环 + 中心点）
----------------------------
设环点距原点 d，区域半径 R=1800，则区域内任一点到最近扫描点的最大距离
必出现在"区域边界上、相邻两环点角平分线方向"，沿该方向距离函数对 r 是凸的，
因此只需比较两个端点 r=1000（中心点覆盖到此为止）与 r=1800。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .config import (
    ARENA_RADIUS_M,
    RECEIVER_MIN_M,
    StrategyConfig,
)


Point = Tuple[float, float]


def scan_points(cfg: StrategyConfig) -> List[Point]:
    """七个必访扫描点：原点 + 正 n 边形环点（默认六边形，d=1150 m）。"""
    points: List[Point] = [(0.0, 0.0)]
    n = cfg.scan_ring_count
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        points.append(
            (cfg.scan_ring_radius_m * math.cos(ang), cfg.scan_ring_radius_m * math.sin(ang))
        )
    return points


def worst_case_distance(cfg: StrategyConfig) -> Dict[str, float]:
    """返回最坏覆盖距离及其出现位置（解析结论）。"""
    d = cfg.scan_ring_radius_m
    n = cfg.scan_ring_count
    phi = 0.5 * (2.0 * math.pi / n)          # 相邻环点夹角的一半

    def dist_at(r: float) -> float:
        return math.sqrt(r * r + d * d - 2.0 * r * d * math.cos(phi))

    inner = dist_at(RECEIVER_MIN_M)
    outer = dist_at(ARENA_RADIUS_M)
    worst = max(inner, outer)
    worst_r = ARENA_RADIUS_M if outer >= inner else RECEIVER_MIN_M
    return {
        "ring_radius_m": d,
        "ring_count": float(n),
        "bisector_angle_deg": math.degrees(phi),
        "distance_at_r1000_m": inner,
        "distance_at_r1800_m": outer,
        "worst_distance_m": worst,
        "worst_at_radius_m": worst_r,
        "guaranteed_reception_m": RECEIVER_MIN_M,
        "margin_m": RECEIVER_MIN_M - worst,
        "covers": worst <= RECEIVER_MIN_M,
    }


def grid_verify(cfg: StrategyConfig, step_m: float = 10.0) -> Dict[str, float]:
    """独立数值校核：对区域做细网格扫描，求到最近扫描点的最大距离。

    与解析结论互为对照——两者应给出几乎相同的最大距离。
    """
    pts = scan_points(cfg)
    worst = 0.0
    arg = (0.0, 0.0)
    r = 0.0
    while r <= ARENA_RADIUS_M + 1e-9:
        # 每个半径上按弧长布点，避免极角均匀导致内圈过密、外圈过疏
        circumference = 2.0 * math.pi * max(r, 1.0)
        count = max(8, int(circumference / step_m))
        for k in range(count):
            th = 2.0 * math.pi * k / count
            x, y = r * math.cos(th), r * math.sin(th)
            m = min(math.hypot(x - px, y - py) for px, py in pts)
            if m > worst:
                worst, arg = m, (x, y)
        r += step_m
    return {
        "grid_worst_distance_m": worst,
        "worst_point_x": arg[0],
        "worst_point_y": arg[1],
        "margin_m": RECEIVER_MIN_M - worst,
        "covers": worst <= RECEIVER_MIN_M,
    }


def minimum_points_argument(arena_radius: float = ARENA_RADIUS_M,
                            reception: float = RECEIVER_MIN_M) -> Dict[str, float]:
    """说明"六个环形点不带中心点会把原点漏掉"，从而论证七点是下界之一。

    纯环点配置下原点距任一环点为 d，要让原点被覆盖必须 d <= 1000；
    而边界最差点的距离又会随之上升，两者不可兼得。
    """
    best = None
    d = 900.0
    while d <= 1790.0:
        phi = math.pi / 6.0
        outer = math.sqrt(arena_radius ** 2 + d ** 2 - 2 * arena_radius * d * math.cos(phi))
        worst = max(d, outer)          # 原点 + 边界中点
        if best is None or worst < best[1]:
            best = (d, worst)
        d += 5.0
    d, worst = best  # type: ignore[misc]
    return {
        "ring_only_best_radius_m": d,
        "ring_only_worst_distance_m": worst,
        "covers": worst <= reception,
        "shortfall_m": worst - reception,
    }
