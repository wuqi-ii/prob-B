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

# 热循环里对 math.* 的属性查找既慢又脆弱（曾观测到解释器层面 math.pi 被异常
# 解析为 list_iterator 的罕见异常，导致 coverage_worst_required_radius 崩溃）。
# 在导入期一次性取出，等价且更快。
_PI = math.pi
_TAU = 2.0 * math.pi
_COS = math.cos
_SIN = math.sin
_HYPOT = math.hypot
_ATAN2 = math.atan2
_RADIANS = math.radians
_DEGREES = math.degrees
_ACOS = math.acos
_INF = math.inf


def scan_points(cfg: StrategyConfig) -> List[Point]:
    """包围网扫描点：原点 + 内环 n1 点 + 外环 n2 点。

    内环与外环采用不同相位（外环相位偏移半个间隔），使内外环在角度上错开，
    填补单环的角度空隙。
    """
    points: List[Point] = [(0.0, 0.0)]
    for k in range(cfg.scan_inner_count):
        ang = _TAU * k / cfg.scan_inner_count
        points.append(
            (cfg.scan_inner_radius_m * _COS(ang), cfg.scan_inner_radius_m * _SIN(ang))
        )
    phase = _PI / cfg.scan_outer_count  # 外环错开半个间隔
    for k in range(cfg.scan_outer_count):
        ang = phase + _TAU * k / cfg.scan_outer_count
        points.append(
            (cfg.scan_outer_radius_m * _COS(ang), cfg.scan_outer_radius_m * _SIN(ang))
        )
    points.extend(getattr(cfg, "scan_extra_points", ()) or ())
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
    rad = _RADIANS(direction_deg)
    dx, dy = _COS(rad), _SIN(rad)
    cosang = (vx * dx + vy * dy) / _HYPOT(vx, vy)
    return _DEGREES(_ACOS(max(-1.0, min(1.0, cosang)))) <= half_angle_deg + 1e-9


def coverage_worst_required_radius(
    cfg: StrategyConfig,
    step_m: float = 40.0,
    ang_step_deg: float = 3.0,
) -> float:
    """位置网格搜索：最坏情形下需要多大接收半径才能保证不漏。

    对每个采样源位置，不再离散枚举朝向。扫描点按距离逐个加入；当这些点相对
    源的极角最大空隙不超过 180° 时，源恰好落入局部扫描点凸包，因此所有可能
    朝向的半平面内都至少有一个点。首次满足该条件的距离就是该位置所需半径。

    ``ang_step_deg`` 仅为兼容旧调用保留，当前算法对朝向做精确判定。位置仍采用
    网格搜索，因此最终表述应保留“位置细网格校核”，不能冒充连续域严格证明。
    """
    pts = scan_points(cfg)
    worst = 0.0
    r = 0.0
    while r <= ARENA_RADIUS_M + 1e-9:
        circumference = _TAU * max(r, 1.0)
        count = max(8, int(circumference / step_m))
        for k in range(count):
            th = _TAU * k / count
            gx, gy = r * _COS(th), r * _SIN(th)
            required = required_radius_at_position((gx, gy), pts)
            if required > worst:
                worst = required
        r += step_m
    return worst


def required_radius_at_position(source: Point, points: List[Point]) -> float:
    """对固定源位置，精确求覆盖任意朝向所需的最小接收半径。

    半平面覆盖等价于源位于接收半径内扫描点的凸包中。把扫描点看成以源为原点
    的向量后，这又等价于其极角不存在大于 180° 的循环空隙。按距离从近到远
    加点，第一次满足该角度条件时的距离即为答案。
    """
    gx, gy = source
    polar = []
    for px, py in points:
        dx, dy = px - gx, py - gy
        distance = _HYPOT(dx, dy)
        if distance <= 1e-12:
            return 0.0
        polar.append((distance, _ATAN2(dy, dx) % _TAU))
    polar.sort(key=lambda item: item[0])

    active_angles: List[float] = []
    index = 0
    while index < len(polar):
        distance = polar[index][0]
        while index < len(polar) and abs(polar[index][0] - distance) <= 1e-9:
            active_angles.append(polar[index][1])
            index += 1
        if len(active_angles) < 2:
            continue
        angles = sorted(active_angles)
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(angles[0] + _TAU - angles[-1])
        if max(gaps) <= _PI + 1e-12:
            return distance
    return _INF


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
