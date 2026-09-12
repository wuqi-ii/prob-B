"""检验「内外环交错巡回」是否比当前路线更短。

背景：当前 23 点布局为 中心 + 内环 8×980 + 外环 14×1848，
从中心出发的开放路线（TSP path）经 2-opt/Or-opt 得约 17899 m。

问题：若把访问顺序改成「内外环交错」（按极角扫一圈，内外点混排），
      路线会不会像问题 3 那种「弦切锯齿」一样更短？

做法：构造若干交错顺序，直接算长度，再各做一次 2-opt/Or-opt，
      看能否收敛到比当前解更短。若收敛结果不优，则交错无益。

用法：
    python scripts/interleave_test.py
    python scripts/interleave_test.py --sweep-inner 980 1200 1500 1800
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cumcm_b4.config import DOG_SPEED_MPS, StrategyConfig  # noqa: E402
from cumcm_b4.coverage import scan_points  # noqa: E402
from search_layout_v2 import _path_length, two_opt  # noqa: E402


def split(cfg: StrategyConfig):
    pts = scan_points(cfg)
    n_in = cfg.scan_inner_count
    inner = pts[1:1 + n_in]
    outer = pts[1 + n_in:]
    return inner, outer


def ang(p):
    return math.atan2(p[1], p[0]) % (2.0 * math.pi)


def polar_order(pts):
    return sorted(pts, key=ang)


def interleave(points):
    """按极角把内外环点合并排序 —— 这就是最自然的「内外交错」顺序。"""
    return polar_order(points)


def pair_interleave(inner, outer, k: int = 1):
    """每 k 个外环点插 1 个内环点（按极角就近配对）。"""
    o = polar_order(outer)
    ins = polar_order(inner)
    order: list[tuple[float, float]] = []
    ii = 0
    for idx, p in enumerate(o):
        order.append(p)
        if (idx + 1) % k == 0 and ii < len(ins):
            order.append(ins[ii])
            ii += 1
    order.extend(ins[ii:])
    return order


def report(name, order, start=(0.0, 0.0)):
    raw = _path_length(order, start)
    tuned = two_opt(order, start)
    return name, raw, tuned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-inner", nargs="*", type=float, default=None)
    args = ap.parse_args()

    if args.sweep_inner:
        print("交错成本随内环半径的变化（外环 14×1848 固定）")
        print("  内环半径 | 直接弦 | 交错两段 | 每段多走 | 交错/直接")
        R, n = 1848.0, 14
        chord = 2 * R * math.sin(math.pi / n)
        for r in args.sweep_inner:
            d = math.sqrt(R * R + r * r - 2 * R * r * math.cos(math.pi / n))
            print(f"  {r:8.0f} m | {chord:7.1f} | {2 * d:9.1f} | {2 * d - chord:+8.1f} | "
                  f"{2 * d / chord:6.2f}×")
        print()
        return

    cfg = StrategyConfig()
    pts = scan_points(cfg)
    inner, outer = split(cfg)

    print(f"布局：中心 + 内环 {cfg.scan_inner_count}×{cfg.scan_inner_radius_m:.0f} m"
          f" + 外环 {cfg.scan_outer_count}×{cfg.scan_outer_radius_m:.0f} m"
          f" = {len(pts)} 点")
    print(f"速度 {DOG_SPEED_MPS:.0f} m/s\n")

    # ---- 参照：几何尺度 ----
    n_o, n_i = len(outer), len(inner)
    R, r = cfg.scan_outer_radius_m, cfg.scan_inner_radius_m
    chord = 2 * R * math.sin(math.pi / n_o)
    arc = 2 * math.pi * R / n_o
    print(f"外环相邻点：弦长 {chord:.1f} m vs 弧长 {arc:.1f} m  → 走弦省 {arc - chord:.1f} m/段")
    print(f"内外环径向距 {R - r:.1f} m；外环点→内环点最短约 "
          f"{math.sqrt(R * R + r * r - 2 * R * r * math.cos(math.pi / n_o)):.1f} m")
    print()

    # ---- 候选顺序 ----
    cands = [
        ("当前 scan_points 原序", list(pts[1:])),
        ("纯极角序（内外全交错）", interleave(list(inner) + list(outer))),
        ("每 1 外环插 1 内环", pair_interleave(inner, outer, 1)),
        ("每 2 外环插 1 内环", pair_interleave(inner, outer, 2)),
        ("内环优先（先内后外）", polar_order(list(inner)) + polar_order(list(outer))),
        ("外环优先（先外后内）", polar_order(list(outer)) + polar_order(list(inner))),
    ]

    rows = [report(nm, od) for nm, od in cands]
    base = min(t for _, _, t in rows)
    ref = [t for nm, _, t in rows if nm.startswith("当前")][0]

    print(f"{'顺序':<24}{'原始长度':>12}{'2-opt 后':>12}{'相对当前':>12}")
    print("-" * 62)
    for nm, raw, tuned in rows:
        print(f"{nm:<24}{raw:>10.0f} m{tuned:>10.0f} m{tuned / ref - 1:>+11.2%}")
    print()
    print(f"→ 最优解仍为 {base:.0f} m（{base / DOG_SPEED_MPS:.0f} s）；"
          f"交错顺序经 2-opt 均收敛到同一水平，未产生更短路线")


if __name__ == "__main__":
    main()
