"""布局搜索 v2：在满足覆盖（最坏所需接收半径 <= 1000 m 且留余量）的前提下，
最小化包围网巡回路线长度（= 移动时间），从而给出比"内环8×980 + 外环16×1840"更省时的
等效布局。

与旧 search_coverage_layout.py 的区别：
1. 目标函数从"最少点数"改成"最小巡回长度"——点少不等于跑得短（环上点越多，
   总弧长越接近 2πr，所以点数与路长方向相反）；
2. 对每个候选布局先做粗网格覆盖筛选，再对优胜者做 5 m 细网格复核；
3. 巡回长度用 最近邻 + 2-opt + Or-opt 求近似最优开放路线（从原点出发）。

用法：
    python scripts/search_layout_v2.py --coarse-step 60 --topk 40
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import ARENA_RADIUS_M, RECEIVER_MIN_M, StrategyConfig  # noqa: E402
from cumcm_b4.coverage import coverage_worst_required_radius, scan_points  # noqa: E402


# --------------------------------------------------------------------------
# 巡回长度
# --------------------------------------------------------------------------

def tour_length(points, start=(0.0, 0.0)) -> float:
    """最近邻开放路线长度（从 start 出发，访问全部点，不要求回到起点）。"""
    remaining = list(points)
    cur = start
    total = 0.0
    while remaining:
        nxt = min(remaining, key=lambda p: (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2)
        total += math.dist(cur, nxt)
        cur = nxt
        remaining.remove(nxt)
    return total


def _path_length(path, start):
    total = math.dist(start, path[0])
    for a, b in zip(path, path[1:]):
        total += math.dist(a, b)
    return total


def two_opt(points, start=(0.0, 0.0), max_pass: int = 60) -> float:
    """对开放路线做 2-opt + Or-opt（单点搬家、双点搬家）局部搜索。"""
    path = list(points)
    best = _path_length(path, start)
    improved = True
    passes = 0
    while improved and passes < max_pass:
        improved = False
        passes += 1
        n = len(path)
        # 2-opt：反转 path[i..j]
        for i in range(n - 1):
            for j in range(i + 1, n):
                new = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
                cand = _path_length(new, start)
                if cand < best - 1e-9:
                    path, best, improved = new, cand, True
        # Or-opt：把长度 1..3 的片段搬到别处
        for seg_len in (1, 2, 3):
            for i in range(0, n - seg_len + 1):
                seg = path[i:i + seg_len]
                rest = path[:i] + path[i + seg_len:]
                for k in range(len(rest) + 1):
                    if k == i:
                        continue
                    new = rest[:k] + seg + rest[k:]
                    cand = _path_length(new, start)
                    if cand < best - 1e-9:
                        path, best, improved = new, cand, True
    return best


# --------------------------------------------------------------------------
# 覆盖筛选
# --------------------------------------------------------------------------

def coverage_worst(n1, r1, n2, r2, step_m):
    cfg = StrategyConfig(
        scan_inner_count=n1, scan_inner_radius_m=r1,
        scan_outer_count=n2, scan_outer_radius_m=r2,
    )
    pts = scan_points(cfg)
    return coverage_worst_required_radius(cfg, step_m=step_m, ang_step_deg=3.0), pts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-step", type=float, default=60.0)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--min-margin", type=float, default=5.0,
                    help="粗筛阶段要求的最小覆盖余量(m)")
    args = ap.parse_args()

    inner_counts = (5, 6, 7, 8, 9)
    inner_radii = tuple(float(r) for r in range(700, 1001, 25))
    outer_counts = (12, 14, 16, 18, 20)
    outer_radii = (1836.0, 1850.0, 1870.0, 1890.0)

    base = StrategyConfig()
    base_pts = scan_points(base)
    base_tour = two_opt(base_pts)
    base_worst = coverage_worst_required_radius(base, step_m=40.0)

    print(f"当前布局: n1={base.scan_inner_count} r1={base.scan_inner_radius_m} "
          f"n2={base.scan_outer_count} r2={base.scan_outer_radius_m}  "
          f"点数={len(base_pts)}  巡回={base_tour:.1f} m  "
          f"最坏所需接收半径={base_worst:.1f} m  余量={RECEIVER_MIN_M-base_worst:.1f} m")
    print()

    rows = []
    total = len(inner_counts) * len(inner_radii) * len(outer_counts) * len(outer_radii)
    done = 0
    for n1, r1, n2, r2 in itertools.product(inner_counts, inner_radii, outer_counts, outer_radii):
        done += 1
        if done % 200 == 0:
            print(f"  ... {done}/{total}", file=sys.stderr)
        # 外环必须真的能"出圈"：cos(半间隔)*r2 > 1800 才有外侧覆盖点
        half_gap = math.pi / n2
        if r2 * math.cos(half_gap) <= ARENA_RADIUS_M:
            continue
        worst, pts = coverage_worst(n1, r1, n2, r2, args.coarse_step)
        if RECEIVER_MIN_M - worst < args.min_margin:
            continue
        tour = two_opt(pts)
        rows.append({
            "n1": n1, "r1": r1, "n2": n2, "r2": r2,
            "points": len(pts), "tour": tour, "worst": worst,
            "margin": RECEIVER_MIN_M - worst,
        })

    rows.sort(key=lambda x: x["tour"])
    print(f"粗筛通过 {len(rows)} 个布局（余量 >= {args.min_margin} m）。按巡回长度排序 Top{args.topk}：")
    print(f"{'n1':>3s} {'r1':>6s} {'n2':>3s} {'r2':>7s} {'pts':>4s} "
          f"{'tour_m':>9s} {'vs_cur':>8s} {'worst':>8s} {'margin':>8s} {'refine':>7s}")
    print("-" * 78)
    for row in rows[:args.topk]:
        refined = coverage_worst_required_radius(
            StrategyConfig(scan_inner_count=row["n1"], scan_inner_radius_m=row["r1"],
                           scan_outer_count=row["n2"], scan_outer_radius_m=row["r2"]),
            step_m=5.0,
        )
        row["worst_fine"] = refined
        ok = "OK" if refined <= RECEIVER_MIN_M else "FAIL"
        print(f"{row['n1']:3d} {row['r1']:6.0f} {row['n2']:3d} {row['r2']:7.0f} "
              f"{row['points']:4d} {row['tour']:9.1f} "
              f"{row['tour']/base_tour-1:8.2%} {row['worst']:8.1f} {row['margin']:8.1f} "
              f"{refined:7.1f} {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
