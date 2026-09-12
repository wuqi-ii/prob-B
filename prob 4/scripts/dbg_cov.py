"""临时调试：定位 coverage_worst_required_radius 崩溃的确切网格点。"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cumcm_b4 import coverage  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402

cfg = StrategyConfig(
    scan_inner_count=9, scan_inner_radius_m=980.0,
    scan_outer_count=13, scan_outer_radius_m=1854.0,
)
pts = coverage.scan_points(cfg)
print("python", sys.version.split()[0], "| module math is", coverage.math)
print("pts", len(pts))

step = 1.0
worst = 0.0
r = 0.0
bad = 0
while r <= 1800.0 + 1e-9:
    circumference = 2.0 * math.pi * max(r, 1.0)
    count = max(8, int(circumference / step))
    for k in range(count):
        th = 2.0 * math.pi * k / count
        gx, gy = r * math.cos(th), r * math.sin(th)
        try:
            required = coverage.required_radius_at_position((gx, gy), pts)
        except Exception:
            bad += 1
            if bad <= 3:
                print("FAIL at", (round(gx, 3), round(gy, 3)), "r=", r)
                print("   coverage.math =", coverage.math)
                traceback.print_exc()
            required = math.inf
        if required > worst:
            worst = required
    r += step
print("worst =", worst, "| bad =", bad)
