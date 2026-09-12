"""Fixed feasible region: keep the user's reception and 30-degree constraints."""
import math
import numpy as np
from .geometry import clip_disk_outer, global_to_local, bearing_interval


def triangle(config):
    return [(0.0, 0.0), (config.max_reception_m, -config.h), (config.max_reception_m, config.h)]


def source_polygon(config):
    poly = clip_disk_outer(triangle(config), (0.0, 0.0), config.max_reception_m, config.circle_sides)
    if config.use_map_boundary:
        center = global_to_local(config.map_center_m, config)
        poly = clip_disk_outer(poly, center, config.map_radius_m, config.circle_sides)
    if len(poly) < 3:
        raise ValueError("No two-dimensional source region; check first station/bearing and map.")
    return poly


def margins(point, config):
    """All returned slacks are metres; feasible iff every slack >=0."""
    x, y = map(float, point)
    y = abs(y)
    return np.array([
        x,
        config.guaranteed_reception_m - math.hypot(x, y),
        config.guaranteed_reception_m - math.hypot(x - config.max_reception_m, y + config.h),
        y - config.t * x,
        y - config.t * (config.max_reception_m - x) - config.h,
    ])


def feasible_mask(points, config, tol=1e-8):
    p = np.asarray(points, dtype=float)
    x, y = p[..., 0], np.abs(p[..., 1])
    return ((x >= -tol)
            & (np.hypot(x, y) <= config.guaranteed_reception_m + tol)
            & (np.hypot(x - config.max_reception_m, y + config.h) <= config.guaranteed_reception_m + tol)
            & (y >= config.t * x - tol)
            & (y >= config.t * (config.max_reception_m - x) + config.h - tol))


def lobe_bounds(x, config):
    """Lower and upper |y| bounds at x, for an exact analytic region boundary."""
    x = np.asarray(x, dtype=float)
    low = np.maximum(config.t * x, config.t * (config.max_reception_m - x) + config.h)
    R = config.guaranteed_reception_m
    r1, r2 = R * R - x * x, R * R - (x - config.max_reception_m) ** 2
    high = np.minimum(np.sqrt(np.maximum(0, r1)), np.sqrt(np.maximum(0, r2)) - config.h)
    high = np.where((r1 >= 0) & (r2 >= 0) & (x >= 0), high, np.nan)
    return low, high


def uniform_diameter_bound(config):
    w1, w2 = 2 * config.h, 2 * config.guaranteed_reception_m * math.sin(config.eps)
    a = config.alpha
    return math.sqrt(w1 * w1 + w2 * w2 + 2 * w1 * w2 * math.cos(a)) / math.sin(a)


def _is_feasible_x(x, config):
    low, high = lobe_bounds(x, config)
    return bool(np.isfinite(low)) and bool(np.isfinite(high)) and bool(high >= low)


def exact_tips(config, iterations=90):
    """The two corner points where the lower and upper boundary meet.

    They are where two constraints intersect, so they are found by bisecting the
    feasibility of x instead of sampling x on a grid.
    """
    R = config.guaranteed_reception_m
    x = np.linspace(0.0, R, 4001)
    low, high = lobe_bounds(x, config)
    ok = np.isfinite(low) & np.isfinite(high) & (high >= low)
    if not ok.any():
        return np.empty((0, 2))
    feasible = x[ok]
    tips = []
    for inside, outside in ((feasible[0], feasible[0] - (x[1] - x[0])),
                            (feasible[-1], feasible[-1] + (x[1] - x[0]))):
        a, b = outside, inside  # a infeasible, b feasible
        for _ in range(iterations):
            mid = 0.5 * (a + b)
            if _is_feasible_x(mid, config):
                b = mid
            else:
                a = mid
        value = float(np.mean(lobe_bounds(b, config)))
        tips.append((b, value))
        tips.append((b, -value))
    return np.asarray(tips)


def boundary_candidates(config, spacing=5.0, resolution=4001):
    """Points on the boundary of the fixed candidate region, tips included.

    A monotone distance/accuracy trade-off can place the optimum exactly on a
    constraint (the boundary case does: the optimum sits on the far crossing
    angle and the far reception circle), where a coarse interior grid cannot
    reach it. Sampling the analytic lower/upper bounds covers every boundary
    piece, and the extreme feasible x values are the tips where two constraints
    meet; the tips themselves come from a bisection on feasibility.
    """
    R = config.guaranteed_reception_m
    x = np.linspace(0.0, R, resolution)
    low, high = lobe_bounds(x, config)
    ok = np.isfinite(low) & np.isfinite(high) & (high >= low)
    if not ok.any():
        return np.empty((0, 2))
    xs, lo, hi = x[ok], low[ok], high[ok]
    # Thin along x so that consecutive samples stay at least `spacing` apart.
    keep = np.zeros(len(xs), dtype=bool)
    keep[0] = True
    last = xs[0]
    for i in range(1, len(xs)):
        if xs[i] - last >= spacing:
            keep[i] = True
            last = xs[i]
    keep[-1] = True
    xs, lo, hi = xs[keep], lo[keep], hi[keep]
    blocks = []
    for side in (1.0, -1.0):
        blocks.append(np.column_stack([xs, side * lo]))
        blocks.append(np.column_stack([xs, side * hi]))
    blocks.append(exact_tips(config))
    points = np.vstack(blocks)
    points = points[feasible_mask(points, config, tol=1e-6)]
    return deduplicate(points)


def deduplicate(points, tol=1e-6):
    points = np.asarray(points, dtype=float)
    if not len(points):
        return points
    _, unique = np.unique(np.round(points, int(-math.log10(tol))), axis=0, return_index=True)
    return points[np.sort(unique)]


def min_measured_acute_deg(point, poly, config):
    lo, hi = bearing_interval(poly, point)
    lo, hi = lo - config.eps, hi + config.eps
    # Acute angle is distance to the nearest multiple of pi.
    if math.ceil(lo / math.pi) <= math.floor(hi / math.pi):
        return 0.0
    return math.degrees(min(math.acos(min(1.0, abs(math.cos(v)))) for v in (lo, hi)))


def physical_checks(point, poly, config):
    named = dict(zip(["forward", "near_vertex_reception", "far_vertex_reception", "angle_near", "angle_far"], margins(point, config).tolist()))
    tri = np.asarray(triangle(config))
    source = np.asarray(poly)
    max_tri = float(np.linalg.norm(tri - point, axis=1).max())
    # The triangle lies in |y|<=h; this lower bound also excludes the <=5m blind zone.
    min_separation_lower = abs(float(point[1])) - config.h
    angle = min_measured_acute_deg(point, triangle(config), config)
    return {
        "slacks_m": named,
        "min_slack_m": min(named.values()),
        "max_distance_to_triangle_m": max_tri,
        "max_distance_to_source_outer_polygon_m": float(np.linalg.norm(source - point, axis=1).max()),
        "min_distance_lower_bound_m": min_separation_lower,
        "min_measured_acute_angle_deg": angle,
        "uniform_diameter_upper_bound_m": uniform_diameter_bound(config),
        "reception_guaranteed": max_tri <= config.guaranteed_reception_m + 1e-5,
        "angle_guaranteed": angle >= config.min_crossing_deg - 1e-6,
        "strong_signal_blind_zone_excluded": min_separation_lower > 5.0,
        "candidate_feasible": bool(feasible_mask(np.asarray(point), config, tol=1e-5)),
    }
