"""在稀疏包围网上做「盲区补点」，目标：用最少的额外点把 worst 压回安全线。

背景：均匀双环（内环 n1×r1 + 外环 n2×r2）受整数点数约束，减到 19 点后
worst 从 983 涨到 1124，超出 RECEIVER_MIN_M=1000 的安全线，实测会漏检。
但盲区通常只集中在少数几个扇区，未必需要一个整环来填——本脚本用贪心在
盲区补自由点，尝试用 1~2 个点换回全覆盖。

用法：
    python scripts/patch_cover.py                 # 默认从 19 点稀疏网出发
    python scripts/patch_cover.py --inner-r 900 --inner-n 7 --outer-r 1880 --outer-n 11
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.coverage import (  # noqa: E402
    ARENA_RADIUS_M,
    required_radius_at_position,
    scan_points,
)

SAFE = 1000.0


def grid(step: float) -> list[tuple[float, float]]:
    """区域内均匀采样网格（含圆心）。"""
    pts = [(0.0, 0.0)]
    r = step
    while r <= ARENA_RADIUS_M + 1e-9:
        count = max(8, int(2 * math.pi * max(r, 1.0) / step))
        for k in range(count):
            th = 2 * math.pi * k / count
            pts.append((r * math.cos(th), r * math.sin(th)))
        r += step
    return pts


def worst_of(points, samples) -> tuple[float, tuple[float, float]]:
    worst, at = 0.0, (0.0, 0.0)
    for g in samples:
        v = required_radius_at_position(g, points)
        if v > worst:
            worst, at = v, g
    return worst, at


def tour_len(pts: list[tuple[float, float]]) -> float:
    n = len(pts)

    def length(order):
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
    ap.add_argument("--inner-r", type=float, default=900.0)
    ap.add_argument("--inner-n", type=int, default=7)
    ap.add_argument("--outer-r", type=float, default=1880.0)
    ap.add_argument("--outer-n", type=int, default=11)
    ap.add_argument("--base-step", type=float, default=60.0, help="补点搜索用粗网格")
    ap.add_argument("--check-step", type=float, default=8.0, help="最终校核用细网格")
    ap.add_argument("--max-add", type=int, default=3)
    args = ap.parse_args()

    cfg = StrategyConfig(scan_inner_radius_m=args.inner_r, scan_inner_count=args.inner_n,
                         scan_outer_radius_m=args.outer_r, scan_outer_count=args.outer_n)
    pts = scan_points(cfg)
    coarse = grid(args.base_step)
    w0, at0 = worst_of(pts, coarse)
    print(f"基础布局 {len(pts)} 点（内环{args.inner_n}×{args.inner_r:.0f} "
          f"+ 外环{args.outer_n}×{args.outer_r:.0f}）")
    print(f"  粗网格 worst = {w0:.1f}   最坏位置 r={math.hypot(*at0):.0f} "
          f"θ={math.degrees(math.atan2(at0[1], at0[0])) % 360:.0f}°")
    print(f"  巡回基线 {tour_len(pts):.0f} m")

    # 盲区诊断：worst 超标的采样点分布
    bad = [(g, required_radius_at_position(g, pts)) for g in coarse]
    bad = [(g, v) for g, v in bad if v > SAFE]
    if not bad:
        print("  粗网格下已满足 worst <= 1000，无需补点。")
        return 0
    rr = sorted(math.hypot(*g) for g, _ in bad)
    th = sorted(math.degrees(math.atan2(g[1], g[0])) % 360 for g, _ in bad)
    print(f"  盲区采样点 {len(bad)} 个：半径 {rr[0]:.0f}~{rr[-1]:.0f} m，"
          f"中位 {rr[len(rr) // 2]:.0f} m")
    print(f"    角度分布（度）: {[round(a) for a in th[:12]]}"
          f"{' ...' if len(th) > 12 else ''}")

    # 贪心补点：候选限制在盲区附近的网格
    cand = grid(50.0)
    cur = list(pts)
    added = []
    for step_i in range(args.max_add):
        w_cur, _ = worst_of(cur, coarse)
        if w_cur <= SAFE:
            break
        best = None
        for c in cand:
            if any(math.dist(c, p) < 100.0 for p in cur):
                continue  # 离现有点太近，无意义
            w, _ = worst_of(cur + [c], coarse)
            if best is None or w < best[0]:
                best = (w, c)
        if best is None:
            print("  无可用候选点。")
            break
        w_new, c = best
        cur.append(c)
        added.append(c)
        print(f"  补点{step_i + 1}: ({c[0]:.0f}, {c[1]:.0f})  r={math.hypot(*c):.0f} m "
              f"θ={math.degrees(math.atan2(c[1], c[0])) % 360:.0f}°  "
              f"worst {w_cur:.1f} -> {w_new:.1f}")
        if w_new <= SAFE:
            break

    fine = grid(args.check_step)
    wf, atf = worst_of(cur, fine)
    print(f"\n补 {len(added)} 点后共 {len(cur)} 点：")
    print(f"  细网格(step={args.check_step:.0f}) worst = {wf:.1f}  "
          f"{'✅ 达标' if wf <= SAFE else '❌ 仍超安全线'}"
          f"   最坏位置 r={math.hypot(*atf):.0f}")
    print(f"  巡回 {tour_len(cur):.0f} m（基线 {tour_len(pts):.0f} m）")
    print("  补点坐标: " + ", ".join(f"({x:.0f},{y:.0f})" for x, y in added))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
