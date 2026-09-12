"""自由布局优化：不锁死均匀双环，直接搜「worst <= 1000」的最少点布局。

背景：均匀双环受整数点数约束。内环 7 + 外环 11 = 19 点时 worst=1124，
超出 RECEIVER_MIN_M=1000 的安全线，实测 600 例会漏 1 例。脚本 patch_cover.py
已证明「在均匀网上补自由点」无效——盲区是整圈的，不是局部扇区。

但均匀性本身是一条人为约束：外环点不必等角分布，内环半径也不必统一。
本脚本把每个扫描点当作自由变量（极坐标），直接最小化 worst，看能否用
19 个（甚至更少）点达到 worst <= 1000——那就既能省下点数，又不赌全清率。

方法：多起点随机重启 + 坐标扰动（带退火的贪心）。worst 是对位置的 max，
非光滑，因此只保证找到可行解，不保证全局最优。
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.coverage import ARENA_RADIUS_M, required_radius_at_position  # noqa: E402

SAFE = 1000.0


def grid(step: float) -> list[tuple[float, float]]:
    pts = [(0.0, 0.0)]
    r = step
    while r <= ARENA_RADIUS_M + 1e-9:
        count = max(8, int(2 * math.pi * max(r, 1.0) / step))
        for k in range(count):
            th = 2 * math.pi * k / count
            pts.append((r * math.cos(th), r * math.sin(th)))
        r += step
    return pts


def worst_of(points, samples) -> float:
    w = 0.0
    for g in samples:
        v = required_radius_at_position(g, points)
        if v > w:
            w = v
    return w


def tour_len(pts) -> float:
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


def to_xy(genes):
    return [(0.0, 0.0)] + [(r * math.cos(t), r * math.sin(t)) for r, t in genes]


def anneal(genes, samples, rng, budget_s, r_range, t_step0):
    cur = worst_of(to_xy(genes), samples)
    best, best_g = cur, list(genes)
    t_end = time.time() + budget_s
    step = 0
    while time.time() < t_end:
        step += 1
        temp = max(0.02, 1.0 - step / 4000.0)
        i = rng.randrange(len(genes))
        r, t = genes[i]
        if rng.random() < 0.5:
            nr = min(max(r + rng.gauss(0, 120.0), r_range[0]), r_range[1])
            cand = list(genes)
            cand[i] = (nr, t)
        else:
            dth = rng.gauss(0, t_step0 * temp)
            cand = list(genes)
            cand[i] = (r, t + dth)
        w = worst_of(to_xy(cand), samples)
        if w < cur or rng.random() < math.exp(-(w - cur) / max(temp * 60.0, 1e-6)):
            genes, cur = cand, w
            if w < best:
                best, best_g = w, list(cand)
        if best <= SAFE:
            break
    return best, best_g, step


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=18, help="除原点外的点数（总点数 = n+1）")
    ap.add_argument("--restarts", type=int, default=12)
    ap.add_argument("--budget", type=float, default=25.0, help="每次重启的秒数")
    ap.add_argument("--step", type=float, default=70.0, help="优化用粗网格")
    ap.add_argument("--check-step", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    samples = grid(args.step)
    print(f"目标：{args.n + 1} 点（含原点），worst <= {SAFE:.0f}"
          f"   粗网格采样 {len(samples)} 点")
    rng = random.Random(args.seed)
    overall = None
    for k in range(args.restarts):
        # 多起点：均匀双环 / 随机散布 / 抖动后的均匀环
        if k == 0:
            n_in = args.n // 2
            genes = []
            for i in range(args.n):
                r = 980.0 if i < n_in else 1880.0
                m = n_in if i < n_in else args.n - n_in
                genes.append((r, 2 * math.pi * (i - (n_in if i >= n_in else 0)) / m
                              + (math.pi / m if i >= n_in else 0.0)))
        else:
            genes = [(rng.uniform(300.0, 1900.0), rng.uniform(0, 2 * math.pi))
                     for _ in range(args.n)]
        w, g, steps = anneal(genes, samples, rng, args.budget, (200.0, 1980.0), 0.6)
        tag = "✅" if w <= SAFE else "  "
        print(f"  重启{k + 1:>2}: worst(粗) = {w:8.1f} {tag}  ({steps} 步)")
        if overall is None or w < overall[0]:
            overall = (w, g)
        if w <= SAFE:
            break

    w, g = overall
    pts = to_xy(g)
    fine = grid(args.check_step)
    wf = worst_of(pts, fine)
    print(f"\n最优解：{len(pts)} 点")
    print(f"  细网格(step={args.check_step:.0f}) worst = {wf:.1f}  "
          f"{'✅ 达标' if wf <= SAFE else '❌ 仍超安全线'}")
    print(f"  巡回 {tour_len(pts):.0f} m")
    print("  点坐标（极坐标 r,θ°）: " + ", ".join(
        f"({math.hypot(x, y):.0f},{math.degrees(math.atan2(y, x)) % 360:.0f})" for x, y in pts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
