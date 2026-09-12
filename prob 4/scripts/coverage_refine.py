"""覆盖性"局部细化"校核：先粗网格定位最坏位置，再在其邻域做 1 m 级精搜。

动机：coverage_worst_required_radius 只做均匀网格采样，网格步长越大越可能
错过真正的最坏位置（相邻采样点间的函数斜率可达 ~1 m/m）。实测同一布局
step=60 给出 962 m、step=5 给出 1010 m，差异远超可接受的余量。
因此判断"是否覆盖"必须做局部细化，不能只信粗网格。

用法：
    python scripts/coverage_refine.py            # 校核当前默认布局
    python scripts/coverage_refine.py --config configs/baseline.json
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import ARENA_RADIUS_M, RECEIVER_MIN_M, StrategyConfig  # noqa: E402
from cumcm_b4.coverage import required_radius_at_position, scan_points  # noqa: E402


def worst_over_grid(points, step_m, radius_max=ARENA_RADIUS_M):
    """返回 (最坏所需半径, 最坏位置)。位置采样为同心圆环上的均匀点。"""
    worst = 0.0
    worst_pos = (0.0, 0.0)
    r = 0.0
    while r <= radius_max + 1e-9:
        count = max(8, int(2.0 * math.pi * max(r, 1.0) / step_m))
        for k in range(count):
            th = 2.0 * math.pi * k / count
            gx, gy = r * math.cos(th), r * math.sin(th)
            req = required_radius_at_position((gx, gy), points)
            if req > worst:
                worst, worst_pos = req, (gx, gy)
        r += step_m
    return worst, worst_pos


def refine_around(points, center, box_m, fine_step):
    """在 center 的方形邻域内做 fine_step 精搜，返回 (最坏值, 位置)。"""
    worst = 0.0
    worst_pos = center
    n = int(box_m / fine_step)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            gx = center[0] + i * fine_step
            gy = center[1] + j * fine_step
            if math.hypot(gx, gy) > ARENA_RADIUS_M + 1e-9:
                continue
            req = required_radius_at_position((gx, gy), points)
            if req > worst:
                worst, worst_pos = req, (gx, gy)
    return worst, worst_pos


def analyse(cfg: StrategyConfig, coarse: float = 40.0, fine: float = 1.0,
            box: float = 80.0) -> dict:
    pts = scan_points(cfg)
    c_worst, c_pos = worst_over_grid(pts, coarse)
    f_worst, f_pos = refine_around(pts, c_pos, box, fine)
    return {
        "points": len(pts),
        "coarse_step": coarse, "coarse_worst": c_worst, "coarse_pos": c_pos,
        "fine_step": fine, "fine_worst": f_worst, "fine_pos": f_pos,
        "margin": RECEIVER_MIN_M - f_worst,
        "covers": f_worst <= RECEIVER_MIN_M + 1e-9,
        "underestimate": f_worst - c_worst,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--coarse", type=float, default=40.0)
    ap.add_argument("--fine", type=float, default=1.0)
    ap.add_argument("--box", type=float, default=80.0)
    args = ap.parse_args()

    cfg = StrategyConfig.from_file(args.config) if args.config else StrategyConfig()
    r = analyse(cfg, args.coarse, args.fine, args.box)
    print(f"布局: 中心 + 内环{cfg.scan_inner_count}x{cfg.scan_inner_radius_m:.0f} "
          f"+ 外环{cfg.scan_outer_count}x{cfg.scan_outer_radius_m:.0f}  点数={r['points']}")
    print(f"粗网格(step={r['coarse_step']:.0f} m) 最坏所需半径 = {r['coarse_worst']:.3f} m "
          f"@ ({r['coarse_pos'][0]:.1f}, {r['coarse_pos'][1]:.1f})")
    print(f"局部细化(step={r['fine_step']:.1f} m, 邻域±{args.box:.0f} m) = "
          f"{r['fine_worst']:.3f} m @ ({r['fine_pos'][0]:.1f}, {r['fine_pos'][1]:.1f})")
    print(f"粗网格低估量 = {r['underestimate']:.3f} m")
    print(f"相对 1000 m 余量 = {r['margin']:.3f} m   -> {'覆盖成立' if r['covers'] else '覆盖失败'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
