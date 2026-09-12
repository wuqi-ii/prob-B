"""Scenario metrics: mean diameter, tail (CVaR) diameter, and worst-case audit.

For every scenario the localisation update uses ONLY the fabricated reading
theta = arg(G - P) + e. The scenario's true source position never enters the
clipping step, so the reported diameters are the ones the robot could actually
compute before moving on.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

from .geometry import clip_wedge, diameter


@dataclass
class PointMetrics:
    point_local_m: tuple
    distance_m: float
    mean_diameter_m: float
    cvar_diameter_m: float
    max_diameter_m: float
    min_diameter_m: float
    composite_m: float
    empty_updates: int

    def to_dict(self):
        data = asdict(self)
        data["point_local_m"] = list(self.point_local_m)
        return data


def cvar(values, weights, alpha):
    """Weighted upper-tail CVaR; the boundary scenario is split by tail mass.

    The number of tail scenarios need not be an integer: once the accumulated
    tail mass would exceed 1-alpha, only the remaining fraction of the boundary
    scenario's weight is taken. Never round the tail size up or down silently.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    w = w / total
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    order = np.argsort(-v, kind="stable")
    v, w = v[order], w[order]
    target = 1.0 - alpha
    used, accumulated = 0.0, 0.0
    for value, weight in zip(v, w):
        take = min(weight, target - accumulated)
        if take <= 0:
            break
        accumulated += take
        used += take * value
        if accumulated >= target - 1e-15:
            break
    if accumulated <= 0:
        raise ValueError("Empty tail: increase the number of scenarios.")
    return float(used / accumulated)


def var_alpha(values, weights, alpha):
    """Weighted quantile; the boundary value where the upper tail starts."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    order = np.argsort(-v, kind="stable")
    v, w = v[order], w[order]
    target = 1.0 - alpha
    accumulated = 0.0
    for value, weight in zip(v, w):
        accumulated += weight
        if accumulated >= target - 1e-15:
            return float(value)
    return float(v[-1])


class PointEvaluator:
    """Cached per-point scenario evaluation on a fixed source polygon."""

    def __init__(self, poly, scenarios, config):
        self.poly = list(poly)
        self.scenarios = scenarios
        self.config = config
        self.cache = {}
        self.calls = 0

    def diameters(self, point):
        readings = self.scenarios.readings(point)
        out = np.empty(len(readings), dtype=float)
        empty = 0
        for i, theta in enumerate(readings):
            clipped = clip_wedge(self.poly, point, float(theta), self.config.eps)
            if len(clipped) < 3:
                empty += 1
                out[i] = 0.0
            else:
                out[i] = diameter(clipped)[0]
        return out, empty

    def evaluate(self, point):
        # The cache key is rounded (nearby refinements must reuse work) but the
        # reported coordinates keep full precision, so distance-to-station and
        # feasibility stay faithful to the point that was actually evaluated.
        key = (round(float(point[0]), 6), round(float(point[1]), 6))
        if key not in self.cache:
            self.calls += 1
            values, empty = self.diameters(point)
            weights = self.scenarios.weights
            mean = float(np.sum(weights * values))
            tail = cvar(values, weights, self.config.cvar_alpha)
            self.cache[key] = PointMetrics(
                point_local_m=(float(point[0]), float(point[1])),
                distance_m=math.hypot(float(point[0]), float(point[1])),
                mean_diameter_m=mean,
                cvar_diameter_m=tail,
                max_diameter_m=float(values.max()),
                min_diameter_m=float(values.min()),
                composite_m=(1 - self.config.tail_weight) * mean + self.config.tail_weight * tail,
                empty_updates=empty,
            )
        return self.cache[key]

    def __call__(self, point):
        return self.evaluate(point)


def evaluate_points(points, evaluator):
    """Evaluate an (m,2) array; returns a list of PointMetrics."""
    return [evaluator(point) for point in np.asarray(points, dtype=float)]
