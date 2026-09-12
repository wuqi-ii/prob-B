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

from .config import ARENA_RADIUS_M, BEARING_ERROR_DEG, NEAR_RADIUS_M, StrategyConfig

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
    effective_far_m: float  # 本次使用的源最远可能距离 R
    boundary_clipped: bool  # 1500 m 示向扇形是否被场地边界截断


# ``prob 2new`` 的参数化快速试算节点。R=1500 使用旧方案的高精度
# (764,526)，其余节点用于分段线性近似；同一候选瓣的联合变量
# (R,a,b) 约束为凸约束，因此可行节点之间的线性插值仍保持可行。
_PARAMETRIC_KNOTS = (
    (0.0, 0.0, 0.0),
    (500.0, 297.0, 180.25818570826812),
    (750.0, 445.0, 267.3829754672644),
    (1000.0, 593.75, 356.76099254761397),
    (1250.0, 668.0, 437.0),
    (1500.0, 764.0, 526.0),
)


def _feasible(a: float, b: float, reach: float, far: float, h: float, t: float) -> bool:
    if a < 0:
        return False
    if math.hypot(a, b) > reach + 1e-9:
        return False
    if math.hypot(a - far, abs(b) + h) > reach + 1e-9:
        return False
    if abs(b) < t * a - 1e-9:
        return False
    return abs(b) >= t * (far - a) + h - 1e-9


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


def _ray_exit_distance(station: Point, angle_rad: float,
                       arena_radius: float = ARENA_RADIUS_M) -> float:
    """从 station 沿给定射线到场地圆的最远正向交点距离。"""
    ux, uy = math.cos(angle_rad), math.sin(angle_rad)
    dot = station[0] * ux + station[1] * uy
    disc = dot * dot + arena_radius * arena_radius - math.hypot(*station) ** 2
    if disc < 0.0:
        return 0.0
    return max(0.0, -dot + math.sqrt(max(0.0, disc)))


def effective_far_distance(station: Point, bearing_deg: float,
                           cfg: StrategyConfig) -> Tuple[float, bool]:
    """返回示向误差扇形受场地裁剪后的安全径向上界 R。

    取整个误差角区间内圆边界距离的最大值，不能只看中心示向线；
    否则会把仍可能位于扇形另一侧的源错误排除。
    """
    physical_far = cfg.second_station_far_m
    eps = math.radians(BEARING_ERROR_DEG + cfg.bearing_rounding_slack_deg)
    center = math.radians(bearing_deg)
    lo, hi = center - eps, center + eps
    angles = [lo, hi]
    # dot(station,u) 在与 station 反向时最小，此时正向出界距离最大。
    if math.hypot(*station) > 1e-12:
        opposite = math.atan2(station[1], station[0]) + math.pi
        for k in range(-2, 3):
            angle = opposite + 2.0 * math.pi * k
            if lo <= angle <= hi:
                angles.append(angle)
    exits = [_ray_exit_distance(station, angle) for angle in angles]
    clipped = min(exits) < physical_far - 1e-7
    far = min(physical_far, max(exits))
    # direction 而非 near 意味着源不在 5 m 内；该下限也避免退化坐标。
    return max(NEAR_RADIUS_M + 1e-6, far), clipped


def parametric_local_point(far: float, cfg: StrategyConfig) -> Point:
    """用 ``prob 2new`` 的分段函数返回 P*(R) 的 +b 相对坐标。"""
    far = min(max(0.0, far), cfg.second_station_far_m)
    for left, right in zip(_PARAMETRIC_KNOTS, _PARAMETRIC_KNOTS[1:]):
        if far <= right[0] + 1e-12:
            ratio = (far - left[0]) / (right[0] - left[0])
            a = left[1] + ratio * (right[1] - left[1])
            b = left[2] + ratio * (right[2] - left[2])
            h = far * math.tan(math.radians(BEARING_ERROR_DEG))
            t = math.tan(math.radians(cfg.second_station_min_angle_deg))
            if not _feasible(a, b, cfg.second_station_reach_m, far, h, t):
                raise ValueError(f"参数化第二检测点未通过硬约束：R={far:g}, (a,b)=({a:g},{b:g})")
            return (a, b)
    return _local_optimal_point(cfg)


def local_to_global(local: Point, station: Point, bearing_deg: float) -> Point:
    """局部坐标 -> 全局坐标：先按方位角旋转，再平移到第一检测点。"""
    a, b = local
    rad = math.radians(bearing_deg)
    x = a * math.cos(rad) - b * math.sin(rad)
    y = a * math.sin(rad) + b * math.cos(rad)
    return (station[0] + x, station[1] + y)


def global_to_local(point: Point, station: Point, bearing_deg: float) -> Point:
    """全局坐标 -> 以第一检测点和首次示向方向定义的局部坐标。"""
    dx, dy = point[0] - station[0], point[1] - station[1]
    rad = math.radians(bearing_deg)
    return (dx * math.cos(rad) + dy * math.sin(rad),
            -dx * math.sin(rad) + dy * math.cos(rad))


def certifies_reusable_station(point: Point, station: Point, bearing_deg: float,
                               cfg: StrategyConfig) -> Tuple[bool, float]:
    """判断已有停靠点能否安全兼任第二检测点。

    判定直接复用问题2候选区域的充分条件，因此同时保证：对首次示向
    扇形与场地边界共同限定的所有可能源，第二站最坏接收距离不超过
    1000 m，且两站交会角不小于配置下界。返回值中的 float 是本次
    边界修正后的最远径向距离。
    """
    far, _ = effective_far_distance(station, bearing_deg, cfg)
    a, b = global_to_local(point, station, bearing_deg)
    h = far * math.tan(math.radians(
        BEARING_ERROR_DEG + cfg.bearing_rounding_slack_deg))
    t = math.tan(math.radians(cfg.second_station_min_angle_deg))
    return _feasible(a, b, cfg.second_station_reach_m, far, h, t), far


def candidate_pair(station: Point, bearing_deg: float, cfg: StrategyConfig) -> CandidatePair:
    """给出某频道第二检测点的两个镜像候选点。"""
    far, clipped = effective_far_distance(station, bearing_deg, cfg)
    if cfg.dynamic_second_station and far < cfg.second_station_far_m - 1e-7:
        a, b = parametric_local_point(far, cfg)
    else:
        a, b = _local_optimal_point(cfg)
        far = cfg.second_station_far_m
    left = local_to_global((a, b), station, bearing_deg)
    right = local_to_global((a, -b), station, bearing_deg)
    return CandidatePair(
        left=left,
        right=right,
        first_station=station,
        bearing_deg=bearing_deg,
        local_optimum=(a, b),
        separation_m=math.hypot(a, b),
        effective_far_m=far,
        boundary_clipped=clipped,
    )


def fixed_candidate_pair(station: Point, bearing_deg: float,
                         cfg: StrategyConfig) -> CandidatePair:
    """显式返回原 1500 m 方案，供动态点的全路线兜底比较。"""
    a, b = _local_optimal_point(cfg)
    return CandidatePair(
        left=local_to_global((a, b), station, bearing_deg),
        right=local_to_global((a, -b), station, bearing_deg),
        first_station=station,
        bearing_deg=bearing_deg,
        local_optimum=(a, b),
        separation_m=math.hypot(a, b),
        effective_far_m=cfg.second_station_far_m,
        boundary_clipped=False,
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
