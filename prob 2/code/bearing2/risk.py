"""Worst second-bearing outcome, with an auditable angular interval enclosure.

For a convex K outside P, possible true bearings form one interval [a,b].
Possible readings fill [a-eps,b+eps]. This removes the need for a 2D source
grid times an error grid. No uniform distribution assumption is introduced.
"""
from dataclasses import dataclass, asdict
import heapq
import math
import numpy as np
from .geometry import bearing_interval, clip_wedge, diameter


@dataclass
class RiskResult:
    lower_m: float
    upper_m: float
    worst_bearing_rad: float
    polygon: list
    diameter_pair: object
    evaluations: int
    splits: int
    converged: bool

    def to_dict(self):
        result = asdict(self)
        result["gap_m"] = self.upper_m - self.lower_m
        result["worst_bearing_deg_local"] = math.degrees(self.worst_bearing_rad)
        return result


def measured_interval(point, poly, eps):
    lo, hi = bearing_interval(poly, point)
    return lo - eps, hi + eps


def bearing_profile(point, poly, eps, count):
    lo, hi = measured_interval(point, poly, eps)
    readings = np.linspace(lo, hi, count)
    ds = np.array([diameter(clip_wedge(poly, point, float(a), eps))[0] for a in readings])
    return readings, ds


def sampled_risk(point, poly, eps, count):
    readings, ds = bearing_profile(point, poly, eps, count)
    index = int(np.argmax(ds))
    return float(ds[index]), float(readings[index])


def certify_risk(point, poly, config, tolerance=None, max_splits=None):
    """Bracket the maximum over ALL continuous readings, for the outer polygon.

On [a,b], W(P,theta,eps) is contained in W(P,(a+b)/2,eps+(b-a)/2).
The diameter of K intersected with that wider wedge bounds all diameters
on the interval. A max-heap refines only intervals that can still win.
This is a floating-point numerical certificate, not exact-arithmetic proof.
"""
    tol = config.certificate_tolerance_m if tolerance is None else tolerance
    limit = config.certificate_max_splits if max_splits is None else max_splits
    lo, hi = measured_interval(point, poly, config.eps)
    best = -1.0
    best_theta = (lo + hi) / 2
    evaluations = 0
    cache = {}

    def sample(theta):
        nonlocal best, best_theta, evaluations
        if theta not in cache:
            cache[theta] = diameter(clip_wedge(poly, point, theta, config.eps))[0]
            evaluations += 1
        value = cache[theta]
        if value > best:
            best, best_theta = value, theta
        return value

    def upper(a, b):
        nonlocal evaluations
        mid = (a + b) / 2
        sample(mid)
        enlarged = config.eps + (b - a) / 2
        if enlarged >= math.pi / 2:
            value = diameter(poly)[0]
        else:
            value = diameter(clip_wedge(poly, point, mid, enlarged))[0]
        evaluations += 1
        return value

    sample(lo)
    sample(hi)
    heap = [(-upper(lo, hi), lo, hi)]
    splits = 0
    while heap and -heap[0][0] > best + tol and splits < limit:
        negub, a, b = heapq.heappop(heap)
        mid = (a + b) / 2
        if b - a < 1e-14:
            heapq.heappush(heap, (negub, a, b))
            break
        for left, right in ((a, mid), (mid, b)):
            ub = upper(left, right)
            if ub > best:
                heapq.heappush(heap, (-ub, left, right))
        splits += 1
    ub = max(best, -heap[0][0] if heap else best)
    witness = clip_wedge(poly, point, best_theta, config.eps)
    _, pair = diameter(witness)
    return RiskResult(best, ub, best_theta, witness, pair, evaluations, splits, ub - best <= tol + 1e-8)
