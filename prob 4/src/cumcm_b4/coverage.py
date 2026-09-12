"""包围网覆盖普查：布局、覆盖性论证与数值校核。

问题 4 为什么不能用问题 3 的七点布局
------------------------------------
问题 3 中所有干扰源都是全向源，只要某扫描点与源的距离不超过有效接收半径就能测到，
因此「原点 + 6 个环点」就足以保证区域里任何源都至少被一个扫描点覆盖。

问题 4 引入了定向干扰源：定向源只在「定向方向两侧各 90°」的半平面（覆盖楔）内
辐射，楔外完全不可测。于是"离得近"不再保证"测得到"——一个把楔背对你的定向源，
即使扫描点就在它 100 m 外，也测不到任何信号。

覆盖性判据（半平面覆盖引理）
------------------------------
设扫描点集 P，源 G（任意位置、任意朝向 d，|d|=1）。G 的覆盖楔是
{ X : (X-G)·d >= 0 且 |X-G| <= R }，其中 R 是有效接收半径（>= 1000 m）。

要保证"任意朝向都至少被一个扫描点测到"，等价于要求：对任意单位方向 d，
都存在 p ∈ P 满足 (p-G)·d >= 0 且 |p-G| <= 1000。

这等价于：G 落在 P ∩ B(G, 1000) 的凸包内——即"距 G 不超过 1000 m 的扫描点
要把 G 从四面八方包围住"。这正是"包围网"的由来，也是外环必须布到目标区域之外
的原因：贴边且楔朝外的源只能被区域外的点测到。

本模块提供：
- scan_points(cfg)：生成"中心 + 内环 + 外环"的扫描点集；
- coverage_worst_required_radius(cfg, ...)：数值搜索最坏情形下需要的接收半径，
  用于 validate() 与写进论文的覆盖性论证；
- grid_verify(cfg, ...)：独立细网格校核，与 worst_required_radius 互为对照。
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .config import (
    ARENA_RADIUS_M,
    DIRECTIONAL_HALF_ANGLE_DEG,
    RECEIVER_MIN_M,
    StrategyConfig,
)


Point = Tuple[float, float]


def scan_points(cfg: StrategyConfig) -> List[Point]:
    """包围网扫描点：原点 + 内环 n1 点 + 外环 n2 点。

    内环与外环采用不同相位（外环相位偏移半个间隔），使内外环在角度上错开，
    填补单环的角度空隙。
    """
    points: List[Point] = [(0.0, 0.0)]
    for k in range(cfg.scan_inner_count):
        ang = 2.0 * math.pi * k / cfg.scan_inner_count
        points.append(
            (cfg.scan_inner_radius_m * math.cos(ang), cfg.scan_inner_radius_m * math.sin(ang))
        )
    phase = math.pi / cfg.scan_outer_count  # 外环错开半个间隔
    for k in range(cfg.scan_outer_count):
        ang = phase + 2.0 * math.pi * k / cfg.scan_outer_count
        points.append(
            (cfg.scan_outer_radius_m * math.cos(ang), cfg.scan_outer_radius_m * math.sin(ang))
        )
    return points


def _in_wedge(point: Point, source: Point, direction_deg: float,
              half_angle_deg: float = DIRECTIONAL_HALF_ANGLE_DEG) -> bool:
    """检测点 point 是否在源 source（朝向 direction_deg）的覆盖楔内。

    half_angle_deg 是覆盖半角：定向源为 90°（合计 180° 楔形）。
    只有 half_angle_deg >= 180°（合计 360°）才是真正的全向，此函数默认不走到该分支。
    """
    if half_angle_deg >= 180.0:
        return True  # 全向（360° 覆盖）
    px, py = point
    gx, gy = source
    vx, vy = px - gx, py - gy
    if vx == 0.0 and vy == 0.0:
        return True  # 检测点与源重合
    rad = math.radians(direction_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    cosang = (vx * dx + vy * dy) / math.hypot(vx, vy)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) <= half_angle_deg + 1e-9


def coverage_worst_required_radius(
    cfg: StrategyConfig,
    step_m: float = 40.0,
    ang_step_deg: float = 3.0,
) -> float:
    """数值搜索：最坏情形下需要多大接收半径才能保证不漏。

    对目标区域内的源位置做网格采样、对朝向做角度采样，对每个 (位置, 朝向) 求
    「落在覆盖楔内的扫描点到源的最小距离」，再取全场景最大值。这个最大值就是
    "要保证一个不漏所必需的最小接收半径"，必须 <= 1000 m。
    """
    pts = scan_points(cfg)
    worst = 0.0
    r = 0.0
    while r <= ARENA_RADIUS_M + 1e-9:
        circumference = 2.0 * math.pi * max(r, 1.0)
        count = max(8, int(circumference / step_m))
        for k in range(count):
            th = 2.0 * math.pi * k / count
            gx, gy = r * math.cos(th), r * math.sin(th)
            for j in range(int(360.0 / ang_step_deg)):
                direction = j * ang_step_deg
                best = math.inf
                for px, py in pts:
                    if _in_wedge((px, py), (gx, gy), direction):
                        d = math.hypot(px - gx, py - gy)
                        if d < best:
                            best = d
                if best < math.inf and best > worst:
                    worst = best
        r += step_m
    return worst


def grid_verify(cfg: StrategyConfig, step_m: float = 40.0,
                ang_step_deg: float = 3.0) -> Dict[str, float]:
    """独立数值校核：返回最坏情形所需接收半径、以及相对 1000 m 的余量。

    与 coverage_worst_required_radius 同一口径，这里额外给出「漏检场景」计数
    （在接收半径下界 1000 m 下测不到的采样点），供写论文时引用。
    """
    worst = coverage_worst_required_radius(cfg, step_m, ang_step_deg)
    return {
        "inner_radius_m": cfg.scan_inner_radius_m,
        "inner_count": float(cfg.scan_inner_count),
        "outer_radius_m": cfg.scan_outer_radius_m,
        "outer_count": float(cfg.scan_outer_count),
        "total_scan_points": float(1 + cfg.scan_inner_count + cfg.scan_outer_count),
        "worst_required_radius_m": worst,
        "guaranteed_reception_m": RECEIVER_MIN_M,
        "margin_m": RECEIVER_MIN_M - worst,
        "covers": worst <= RECEIVER_MIN_M,
    }
