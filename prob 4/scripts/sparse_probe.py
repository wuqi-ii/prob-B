"""搜索「按更大接收半径设计」的稀疏包围网，用于实测减点的漏检代价。

背景：当前 23 点网按最坏接收半径 RECEIVER_MIN_M=1000 m 设计，保证任意位置的源
都至少被一个扫描点测到。但真机观测到不少源在 1300 m 外仍被测到，说明按更大的
设计半径（如 1200 m）布网可以显著减少点数、缩短巡回。代价是：接收半径小于该
设计值的源会落在盲区而漏检。

本脚本：枚举布局，找出满足 worst <= target 且点数最少/巡回最短的候选，
供 ab_test 直接评估全清率。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.coverage import coverage_worst_required_radius, scan_points  # noqa: E402


def tour_len(pts: list[tuple[float, float]]) -> float:
    """开放路径 2-opt（与项目布局搜索脚本同口径）。"""
    n = len(pts)

    def length(order: list[int]) -> float:
        return sum(math.dist(pts[order[i]], pts[order[i + 1]]) for i in range(len(order) - 1))

    order = list(range(n))
    best = length(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                cur = length(cand)
                if cur < best - 1e-9:
                    best, order, improved = cur, cand, True
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1200.0, help="允许的最坏所需半径")
    ap.add_argument("--step", type=float, default=30.0)
    ap.add_argument("--ang", type=float, default=6.0)
    args = ap.parse_args()

    results = []
    for ri in (900.0, 1050.0, 1200.0, 1350.0):
        for ni in (4, 5, 6, 7, 8):
            for ro in (1850.0, 1880.0):
                for no in (8, 9, 10, 11, 12, 13, 14):
                    cfg = StrategyConfig(
                        scan_inner_radius_m=ri,
                        scan_inner_count=ni,
                        scan_outer_radius_m=ro,
                        scan_outer_count=no,
                    )
                    try:
                        worst = coverage_worst_required_radius(cfg, step_m=args.step,
                                                               ang_step_deg=args.ang)
                    except Exception:  # noqa: BLE001
                        continue
                    if not math.isfinite(worst) or worst > args.target:
                        continue
                    pts = scan_points(cfg)
                    results.append((len(pts), tour_len(pts), worst, ri, ni, ro, no))

    results.sort(key=lambda r: (r[0], r[1]))
    print(f"满足 worst <= {args.target:.0f} 的布局（按点数、巡回排序）：")
    print(f"{'点数':>4}{'巡回m':>9}{'worst':>9}   布局")
    seen = set()
    for npts, tl, worst, ri, ni, ro, no in results[:14]:
        if npts in seen and list(seen)[-1] != npts:
            continue
        seen.add(npts)
        print(f"{npts:>4}{tl:>9.0f}{worst:>9.1f}   内环{ni}×{ri:.0f} + 外环{no}×{ro:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
