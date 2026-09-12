"""布局搜索 v3：以"估计总时间"为目标，严格校核覆盖余量。

目标函数（秒）：
    T_hat = tour_length / 5 + n_points * MEASURE_PER_POINT_S
其中 tour_length 是 25 点集合的最近邻+2-opt 开放路线长度（从原点出发），
n_points 是扫描点数（每个扫描点都要对未清除频道逐个普查，实测约 13.7 次检测，
每次 5 s 检测 + 1 s 切换）。这一项是 v2 忽略掉的——v2 只看巡回长度，
于是把"点多但路短"的布局排在了前面，实测恰恰相反。

覆盖校核采用三级网格 30/5/2 m：粗网格会低估最坏值（实测可达 40 m），
所以粗网格只用来筛掉明显不合格的，最终结论一律以细网格为准。

用法：
    python scripts/search_layout_v3.py --screen-step 30 --fine-step 2 --topk 60
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cumcm_b4.config import ARENA_RADIUS_M, RECEIVER_MIN_M, StrategyConfig  # noqa: E402
from cumcm_b4.coverage import coverage_worst_required_radius, scan_points  # noqa: E402
from search_layout_v2 import two_opt  # noqa: E402

MEASURE_PER_POINT_S = 82.2   # 13.7 次检测 × (5 s 检测 + 1 s 切换)


def build_cfg(n1, r1, n2, r2) -> StrategyConfig:
    return StrategyConfig(scan_inner_count=n1, scan_inner_radius_m=float(r1),
                          scan_outer_count=n2, scan_outer_radius_m=float(r2))


def worst_radius(cfg: StrategyConfig, step: float) -> float:
    return coverage_worst_required_radius(cfg, step_m=step)


def objective(tour_m: float, n_points: int) -> float:
    return tour_m / 5.0 + n_points * MEASURE_PER_POINT_S


def screen(args):
    n1, r1, n2, r2 = args
    if r2 * math.cos(math.pi / n2) <= ARENA_RADIUS_M:
        return None
    cfg = build_cfg(n1, r1, n2, r2)
    pts = scan_points(cfg)
    rough = worst_radius(cfg, 30.0)
    if rough > RECEIVER_MIN_M:
        return None
    tour = two_opt(pts)
    return {
        "n1": n1, "r1": r1, "n2": n2, "r2": r2, "pts": len(pts),
        "tour": tour, "rough": rough, "obj": objective(tour, len(pts)),
    }


def verify(args):
    row, step = args
    cfg = build_cfg(row["n1"], row["r1"], row["n2"], row["r2"])
    fine = worst_radius(cfg, float(step))
    return {**row, "fine": fine, "margin": RECEIVER_MIN_M - fine}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-step", type=float, default=30.0)
    ap.add_argument("--fine-step", type=float, default=2.0)
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--margin-floor", type=float, default=5.0,
                    help="细网格余量下限，低于此值一律淘汰")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    inner_counts = (6, 7, 8, 9)
    inner_radii = tuple(range(940, 1001, 10))
    outer_counts = (12, 13, 14, 15, 16)
    outer_radii = tuple(range(1836, 1873, 6))

    grid = list(itertools.product(inner_counts, inner_radii, outer_counts, outer_radii))
    print(f"枚举 {len(grid)} 个布局，粗筛 step={args.screen_step:.0f} m ...")

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, row in enumerate(pool.map(screen, grid, chunksize=8)):
            if row is not None:
                rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(grid)}", file=sys.stderr)

    base = StrategyConfig()
    base_pts = scan_points(base)
    base_tour = two_opt(base_pts)
    base_obj = objective(base_tour, len(base_pts))
    print(f"粗筛通过 {len(rows)} 个。当前布局目标值 T_hat={base_obj:.1f} s "
          f"(tour={base_tour:.0f} m, pts={len(base_pts)})")

    rows.sort(key=lambda r: r["obj"])
    top = rows[:args.topk]
    print(f"对目标值最好的 {len(top)} 个做 step={args.fine_step:.0f} m 细网格校核 ...")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        verified = list(pool.map(verify, [(r, args.fine_step) for r in top], chunksize=2))

    ok = [r for r in verified if r["margin"] >= args.margin_floor]
    ok.sort(key=lambda r: r["obj"])
    print(f"细网格余量 >= {args.margin_floor:.0f} m 的布局共 {len(ok)} 个。"
          f"按 T_hat 排序：")
    print(f"{'n1':>3s} {'r1':>5s} {'n2':>3s} {'r2':>5s} {'pts':>4s} "
          f"{'tour_m':>8s} {'obj_s':>8s} {'vs_cur':>7s} {'rough':>7s} {'fine':>7s} {'margin':>7s}")
    print("-" * 84)
    for r in ok[:20]:
        print(f"{r['n1']:3d} {r['r1']:5d} {r['n2']:3d} {r['r2']:5d} {r['pts']:4d} "
              f"{r['tour']:8.1f} {r['obj']:8.1f} {r['obj']/base_obj-1:7.2%} "
              f"{r['rough']:7.1f} {r['fine']:7.1f} {r['margin']:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
