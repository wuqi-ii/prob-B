"""交割定位（交会定位）几何内核：凸集表示与半平面裁剪。

坐标系：目标区域为原点在 (0,0)、半径 1800 m 的圆域，x 轴正东，y 轴正北。
角度用「度」，方位角定义为 x 轴正向逆时针旋转到目标向量的角度，范围 [0, 360)。

设计原则（与问题 1、2 的实现保持一致）：
1. 源的可行域始终用**凸多边形**表示；
2. 圆域用**外接**半平面多边形逼近（只会放大，不会丢源），这是保守方向；
3. 示向度约束用「前向楔形」而不是两条无向直线，避免把背后的区域误判为可行。

本模块只依赖标准库，便于在任意 Python 3.9+ 环境运行。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

Point = Tuple[float, float]
Polygon = List[Point]

DIST_TOL = 1e-9

# 热循环中的 math.* 属性查找提升为模块级常量（与 coverage.py 同样的加固）。
# 历史观测到极罕见的「math 属性在热循环中被篡改」型 TypeError
# （'float' object is not callable / 'list_iterator' and 'float'），
# 见 RESULTS.md 6.4；提升后崩溃点不复存在，且省去重复属性查找。
_DIST = math.dist
_SQRT = math.sqrt
_SIN = math.sin
_COS = math.cos
_RADIANS = math.radians
_HYPOT = math.hypot
_PI = math.pi
# 热循环中对内建 float 的引用同样提升为模块级常量，规避与 6.4 同源的
# 「名字在热循环中被篡改成 float」型 'float' object is not callable 崩溃。
_FLOAT = float


# --------------------------------------------------------------------------
# 凸多边形基本操作
# --------------------------------------------------------------------------

def clean_polygon(vertices: Sequence[Point], tol: float = DIST_TOL) -> Polygon:
    """去掉相邻重复点，保持顶点顺序。"""
    out: Polygon = []
    for p in vertices:
        p = (_FLOAT(p[0]), _FLOAT(p[1]))
        if not out or _DIST(p, out[-1]) > tol:
            out.append(p)
    if len(out) > 1 and _DIST(out[0], out[-1]) <= tol:
        out.pop()
    return out


def signed_area(poly: Sequence[Point]) -> float:
    """多边形有向面积，逆时针为正。"""
    if len(poly) < 3:
        return 0.0
    total = 0.0
    n = len(poly)
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def convex_hull(points: Sequence[Point], tol: float = 1e-9) -> Polygon:
    """Andrew 单调链求凸包，返回逆时针有序顶点。

    旋转卡壳要求输入是**有序**的凸多边形；散点必须先过这里。
    """
    ordered = sorted(set((_FLOAT(p[0]), _FLOAT(p[1])) for p in points))
    ps: Polygon = []
    for p in ordered:
        if not ps or _DIST(p, ps[-1]) > tol:
            ps.append(p)
    if len(ps) <= 2:
        return ps

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: Polygon = []
    for p in ps:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: Polygon = []
    for p in reversed(ps):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def ensure_ccw(poly: Sequence[Point]) -> Polygon:
    """统一为逆时针顺序；零面积输入原样返回。"""
    if len(poly) < 3 or signed_area(poly) >= 0:
        return list(poly)
    return list(reversed(poly))


def clip_halfplane(
    poly: Sequence[Point],
    nx: float,
    ny: float,
    rhs: float,
    tol: float = 1e-9,
) -> Polygon:
    """用半平面 nx*x + ny*y <= rhs 裁剪凸多边形（Sutherland-Hodgman）。"""
    if not poly:
        return []
    result: Polygon = []
    a = poly[-1]
    da = nx * a[0] + ny * a[1] - rhs
    for b in poly:
        db = nx * b[0] + ny * b[1] - rhs
        in_a, in_b = da <= tol, db <= tol
        if in_a != in_b:
            den = da - db
            if abs(den) > 1e-15:
                ratio = min(1.0, max(0.0, da / den))
                result.append(
                    (a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1]))
                )
        if in_b:
            result.append((_FLOAT(b[0]), _FLOAT(b[1])))
        a, da = b, db
    return clean_polygon(result)


# --------------------------------------------------------------------------
# 约束构造器
# --------------------------------------------------------------------------

def _bounding_box(half: float) -> Polygon:
    """以原点为中心的正方形，作为裁剪起点。"""
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


def disk_outer_polygon(radius: float, sides: int = 64) -> Polygon:
    """圆域的外接多边形：由 sides 条支撑半平面截出，保证包含整个圆盘。

    用外接而非内接，是因为任何被丢掉的区域都可能藏着真实干扰源。
    额外代价是半径放大 sec(pi/sides) 倍；sides=64 时约 1.0012，可忽略。
    """
    if radius <= 0 or sides < 8:
        raise ValueError("radius 必须为正，sides 至少为 8")
    poly = _bounding_box(2.0 * radius)
    for k in range(sides):
        angle = 2.0 * _PI * k / sides
        nx, ny = _COS(angle), _SIN(angle)
        poly = clip_halfplane(poly, nx, ny, radius)
        if len(poly) < 3:
            break
    return ensure_ccw(poly)


def clip_wedge(
    poly: Sequence[Point],
    station: Point,
    bearing_deg: float,
    half_angle_deg: float,
) -> Polygon:
    """把示向度约束作为**前向楔形**施加：源必须落在站点的该扇形内。

    两条约束分别是：
      (a) 源在 bearing - half_angle 射线的逆时针侧；
      (b) 源在 bearing + half_angle 射线的顺时针侧。
    """
    if not 0 < half_angle_deg < 90:
        raise ValueError("half_angle_deg 必须位于 (0, 90)")
    px, py = station
    lo = _RADIANS(bearing_deg - half_angle_deg)
    hi = _RADIANS(bearing_deg + half_angle_deg)
    # (a) cross(d_lo, X-P) >= 0  <=>  sin(lo)*X - cos(lo)*Y <= sin(lo)*px - cos(lo)*py
    nx, ny = _SIN(lo), -_COS(lo)
    out = clip_halfplane(poly, nx, ny, nx * px + ny * py)
    if not out:
        return out
    # (b) cross(d_hi, X-P) <= 0  <=> -sin(hi)*X + cos(hi)*Y <= -sin(hi)*px + cos(hi)*py
    nx, ny = -_SIN(hi), _COS(hi)
    return clip_halfplane(out, nx, ny, nx * px + ny * py)


def clip_max_radius(poly: Sequence[Point], station: Point, radius: float, sides: int = 32) -> Polygon:
    """追加「到站点距离不超过 radius」的约束（外接多边形逼近）。

    用于刻画有效接收半径：某频道**没有信号**意味着源在该站点的接收半径之外。

    .. note::
       本函数目前**未被 tracker 接线**（``locate`` 的 ``max_range_m`` 参数默认不传）。
       离线 A/B（60 案例 ×2，种子 9000~9059）实测：对 ``direction`` 观测施加
       ``max_range_m = 1500``（接收半径上界，只收紧不丢源）平均只省 **0.31%** 虚拟时间
       （4016.6 s → 4004.1 s），请求数几乎不变——因为 ±1° 楔形约束本身已经把可行域
       压得很紧，1500 m 的球面约束很少真的切掉区域。收益小于噪声，故保持不接线。
       若将来要接，改 ``ChannelTrack.polygon`` 传入 ``RECEIVER_MAX_M`` 即可。
    """
    out = ensure_ccw(poly)
    px, py = station
    for k in range(sides):
        angle = 2.0 * _PI * k / sides
        nx, ny = _COS(angle), _SIN(angle)
        out = clip_halfplane(out, nx, ny, radius + nx * px + ny * py)
        if len(out) < 3:
            return []
    return out


# --------------------------------------------------------------------------
# 度量
# --------------------------------------------------------------------------

def centroid(poly: Sequence[Point]) -> Point:
    """多边形质心（面积加权）；退化为线段时退化为中点。"""
    pts = list(poly)
    if not pts:
        raise ValueError("空多边形没有质心")
    if len(pts) <= 2:
        x = sum(p[0] for p in pts) / len(pts)
        y = sum(p[1] for p in pts) / len(pts)
        return (x, y)
    pts = ensure_ccw(pts)
    area2 = 0.0
    cx = cy = 0.0
    n = len(pts)
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-12:
        # 面积退化：退回顶点平均值
        return (
            sum(p[0] for p in pts) / n,
            sum(p[1] for p in pts) / n,
        )
    return (cx / (3.0 * area2), cy / (3.0 * area2))


def diameter_bruteforce(poly: Sequence[Point]) -> Tuple[float, Tuple[Point, Point]]:
    """直径的二次精确解，用作旋转卡壳的对照实现。"""
    if len(poly) < 2:
        return 0.0, ((poly[0], poly[0]) if poly else ((0.0, 0.0), (0.0, 0.0)))
    best, pair = 0.0, (poly[0], poly[0])
    for i, a in enumerate(poly):
        for b in poly[i + 1:]:
            d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d > best:
                best, pair = d, (a, b)
    return _SQRT(best), pair


def diameter(poly: Sequence[Point]) -> Tuple[float, Tuple[Point, Point]]:
    """旋转卡壳求凸多边形直径；小多边形或退化情形退回二次算法。"""
    pts = ensure_ccw(list(poly))
    n = len(pts)
    if n < 10 or abs(signed_area(pts)) <= 1e-6:
        return diameter_bruteforce(pts)
    best, pair, j = 0.0, (pts[0], pts[0]), 1

    def area(i: int, k: int) -> float:
        a, b, q = pts[i], pts[(i + 1) % n], pts[k % n]
        return abs((b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]))

    for i in range(n):
        moved = 0
        while moved < n and area(i, j + 1) > area(i, j) + 1e-9:
            j = (j + 1) % n
            moved += 1
        for k in (j, (j + 1) % n):
            for ii in (i, (i + 1) % n):
                a, b = pts[ii], pts[k]
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                if d > best:
                    best, pair = d, (a, b)
    return _SQRT(best), pair


def max_distance_from(point: Point, poly: Sequence[Point]) -> float:
    """点到多边形各顶点的最大距离——这是「距真源最坏情况」的安全上界。"""
    if not poly:
        return _FLOAT("inf")
    return max(_DIST(point, p) for p in poly)


def minimum_enclosing_circle(points: Sequence[Point]) -> Tuple[Point, float]:
    """最小覆盖圆（Welzl 算法），返回 (圆心, 半径)。

    凸多边形的极值点都在顶点上，因此对顶点集求最小覆盖圆即对整个可行域求解。
    该圆心是「到集合内所有点的最大距离」最小的点（minimax 最优），比质心
    （期望平方误差最优）更适合作为「确保一次清除命中」的站位：它把最坏情况
    距离压到最小，且半径 <= 质心到最远顶点的距离，等号只在对称形状成立。
    """
    pts = clean_polygon(points)
    if not pts:
        raise ValueError("空点集没有最小覆盖圆")

    def trivial(r: List[Point]) -> Tuple[Point, float]:
        if not r:
            return (0.0, 0.0), 0.0
        if len(r) == 1:
            return r[0], 0.0
        if len(r) == 2:
            a, b = r
            cx = (a[0] + b[0]) / 2.0
            cy = (a[1] + b[1]) / 2.0
            return (cx, cy), _DIST(a, b) / 2.0
        return _mec_from_3(r[0], r[1], r[2])

    def welzl(p: List[Point], r: List[Point]) -> Tuple[Point, float]:
        if not p or len(r) == 3:
            return trivial(r)
        q = p[-1]
        c, rad = welzl(p[:-1], r)
        if _DIST(q, c) <= rad + DIST_TOL:
            return c, rad
        return welzl(p[:-1], r + [q])

    return welzl(list(pts), [])


def _mec_from_3(a: Point, b: Point, c: Point) -> Tuple[Point, float]:
    """三点最小覆盖圆：钝角/直角取最长边为直径，锐角取外接圆。"""
    ab2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    bc2 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    ca2 = (c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2
    sides = sorted([(ab2, a, b), (bc2, b, c), (ca2, c, a)], key=lambda t: -t[0])
    longest, p1, p2 = sides[0]
    # 钝角或直角：外接圆圆心会跑到三角形外，最小覆盖圆应取最长边为直径
    if longest >= sides[1][0] + sides[2][0] - 1e-12:
        cx = (p1[0] + p2[0]) / 2.0
        cy = (p1[1] + p2[1]) / 2.0
        return (cx, cy), _SQRT(longest) / 2.0
    ax, ay = a
    bx, by = b
    cx_, cy_ = c
    d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-12:
        # 三点共线退化
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0), _SQRT(longest) / 2.0
    ux = ((ax * ax + ay * ay) * (by - cy_)
          + (bx * bx + by * by) * (cy_ - ay)
          + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx_ - bx)
          + (bx * bx + by * by) * (ax - cx_)
          + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
    return (ux, uy), _DIST((ux, uy), a)


def _cross(o: Point, a: Point, b: Point) -> float:
    """(a-o) × (b-o)。CCW 多边形下结果 >= 0 表示 b 在 o->a 的左侧（内侧）。"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def point_to_polygon_distance(point: Point, poly: Sequence[Point]) -> float:
    """点到凸多边形集合的最短距离；点在多边形内时返回0。

    这里同时检查边的内部投影，不能用“到最近顶点的距离”代替。
    """
    vertices = ensure_ccw(poly)
    if not vertices:
        return _FLOAT("inf")
    if len(vertices) == 1:
        return _DIST(point, vertices[0])

    def segment_distance(a: Point, b: Point) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        denom = dx * dx + dy * dy
        if denom <= 1e-18:
            return _DIST(point, a)
        t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom
        t = max(0.0, min(1.0, t))
        q = (a[0] + t * dx, a[1] + t * dy)
        return _DIST(point, q)

    if len(vertices) >= 3 and all(
        _cross(vertices[i], vertices[(i + 1) % len(vertices)], point) >= -DIST_TOL
        for i in range(len(vertices))
    ):
        return 0.0
    return min(
        segment_distance(vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    )


def polygon_area(poly: Sequence[Point]) -> float:
    return abs(signed_area(poly))


def point_along(start: Point, target: Point, distance: float) -> Point:
    """从 start 指向 target 的方向上前进 distance 米后的点。"""
    dx, dy = target[0] - start[0], target[1] - start[1]
    length = _HYPOT(dx, dy)
    if length <= 1e-12 or distance <= 0:
        return (_FLOAT(start[0]), _FLOAT(start[1]))
    ratio = min(1.0, distance / length)
    return (start[0] + dx * ratio, start[1] + dy * ratio)


def locate(
    observations: Iterable[Tuple[Point, float]],
    error_deg: float,
    arena_radius: float,
    max_range_m: float | None = None,
) -> Polygon:
    """把一组 (观测点, 示向度) 观测累积成交叉定位多边形。

    observations 中的每个元素都会贡献一个前向楔形约束；
    若给出 max_range_m，则额外施加「源在各观测点接收半径内」的约束。
    """
    poly = disk_outer_polygon(arena_radius)
    for station, bearing in observations:
        poly = clip_wedge(poly, station, bearing, error_deg)
        # 半平面交可能合法地退化为线段或单点；这些集合仍包含有效定位
        # 信息，diameter()/centroid() 也都支持，不能误判为空集。
        if not poly:
            return []
        if max_range_m is not None:
            poly = clip_max_radius(poly, station, max_range_m)
            if not poly:
                return []
    return ensure_ccw(poly)
