"""Screen concentric surrounding-net layouts for fewer/shorter scan routes.

The coarse grid is only a filter. Any retained layout must subsequently pass the
normal 40 m / 3 degree validation and the finer 20 m / 1 degree audit before use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.coverage import coverage_worst_required_radius, scan_points  # noqa: E402


def nearest_neighbor_length(points: list[tuple[float, float]]) -> float:
    current = 0
    remaining = set(range(1, len(points)))
    length = 0.0
    while remaining:
        nxt = min(remaining, key=lambda idx: math.dist(points[current], points[idx]))
        length += math.dist(points[current], points[nxt])
        current = nxt
        remaining.remove(nxt)
    return length


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-m", type=float, default=100.0)
    parser.add_argument("--angle-step", type=float, default=10.0)
    parser.add_argument("--out", default="outputs/layout_search/coarse.json")
    args = parser.parse_args()

    base = StrategyConfig()
    rows = []
    for inner_count in (6, 7, 8, 9):
        for outer_count in (16, 17, 18):
            for inner_radius in range(700, 1001, 50):
                for outer_radius in (1805.0, 1820.0, 1840.0):
                    cfg = replace(
                        base,
                        scan_inner_count=inner_count,
                        scan_outer_count=outer_count,
                        scan_inner_radius_m=float(inner_radius),
                        scan_outer_radius_m=outer_radius,
                    )
                    worst = coverage_worst_required_radius(
                        cfg,
                        step_m=args.step_m,
                        ang_step_deg=args.angle_step,
                    )
                    rows.append(
                        {
                            "points": 1 + inner_count + outer_count,
                            "inner_count": inner_count,
                            "outer_count": outer_count,
                            "inner_radius_m": float(inner_radius),
                            "outer_radius_m": outer_radius,
                            "coarse_worst_m": worst,
                            "coarse_margin_m": 1000.0 - worst,
                            "scan_route_m": nearest_neighbor_length(scan_points(cfg)),
                        }
                    )
    rows.sort(key=lambda row: (row["points"], row["scan_route_m"]))
    feasible = [row for row in rows if row["coarse_worst_m"] <= 980.0]
    print("Coarse candidates with at least 20 m sampled margin:")
    for row in feasible[:30]:
        print(json.dumps(row, ensure_ascii=False))

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} candidates to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
