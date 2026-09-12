"""搜索23点安全网中的19点高发现率前缀。

从22个非原点扫描点中选择4个作为安全补全点；其余19点构成第一阶段。
仅当第一阶段已经发现16个频道时才能合法早停，否则继续访问4个补全点。
搜索目标依次为：16源案例全部发现率、平均发现比例、前缀开放路线长度。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import generate_case  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.coverage import scan_points  # noqa: E402


def nearest_route(indices, points):
    order = [0]
    remaining = set(indices) - {0}
    while remaining:
        nxt = min(remaining, key=lambda j: math.dist(points[order[-1]], points[j]))
        order.append(nxt)
        remaining.remove(nxt)
    return order


def route_length(order, points):
    return sum(math.dist(points[a], points[b]) for a, b in zip(order, order[1:]))


def detection_masks(cases, points):
    rows = []
    full_mask = (1 << len(points)) - 1
    uncovered = 0
    for case in cases:
        masks = []
        for source in case.sources:
            mask = 0
            for i, (x, y) in enumerate(points):
                if source.covers(x, y):
                    mask |= 1 << i
            if not mask & full_mask:
                uncovered += 1
            masks.append(mask)
        rows.append(masks)
    if uncovered:
        raise RuntimeError(f"完整23点仍有 {uncovered} 个离线源未覆盖")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--start-seed", type=int, default=30000)
    ap.add_argument("--out", default="outputs/progressive_prefix_search.json")
    args = ap.parse_args()

    points = scan_points(StrategyConfig())
    cases = [generate_case(seed, n_sources=16, directional_prob=0.5)
             for seed in range(args.start_seed, args.start_seed + args.cases)]
    masks = detection_masks(cases, points)
    all_indices = set(range(len(points)))
    candidates = []
    for supplement in itertools.combinations(range(1, len(points)), 4):
        core = sorted(all_indices - set(supplement))
        core_mask = sum(1 << i for i in core)
        complete = 0
        found_sum = 0
        for source_masks in masks:
            found = sum(bool(mask & core_mask) for mask in source_masks)
            found_sum += found
            complete += found == 16
        core_order = nearest_route(core, points)
        # 补全阶段从前缀终点继续走最近邻。
        tail = []
        remaining = set(supplement)
        cur = core_order[-1]
        while remaining:
            nxt = min(remaining, key=lambda j: math.dist(points[cur], points[j]))
            tail.append(nxt)
            remaining.remove(nxt)
            cur = nxt
        full_order = core_order + tail
        candidates.append({
            "supplement": list(supplement),
            "core": core,
            "core_order": core_order,
            "full_order": full_order,
            "early_complete": complete,
            "early_rate": complete / args.cases,
            "mean_found": found_sum / args.cases,
            "core_route_m": route_length(core_order, points),
            "full_route_m": route_length(full_order, points),
        })

    for row in candidates:
        # 源数10~16近似等概率，只有16源局能够早停；路线代理目标按1/7早停、6/7补全。
        row["expected_route_proxy_m"] = (
            row["early_rate"] * row["core_route_m"] / 7.0
            + (1.0 - row["early_rate"] / 7.0) * row["full_route_m"]
        )

    candidates.sort(key=lambda r: (-r["early_complete"], -r["mean_found"],
                                   r["core_route_m"], r["full_route_m"]))
    payload = {
        "cases": args.cases,
        "start_seed": args.start_seed,
        "point_count": len(points),
        "points": points,
        "top": candidates[:50],
        "top_expected_route": sorted(
            candidates, key=lambda r: (r["expected_route_proxy_m"], -r["early_rate"])
        )[:50],
        "top_with_full_route_le_18500": sorted(
            (r for r in candidates if r["full_route_m"] <= 18500.0),
            key=lambda r: (-r["early_complete"], -r["mean_found"], r["full_route_m"])
        )[:50],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in candidates[:15]:
        print(f"supp={row['supplement']} early={row['early_complete']}/{args.cases} "
              f"mean_found={row['mean_found']:.3f} "
              f"route={row['core_route_m']:.1f}/{row['full_route_m']:.1f} m")
    print("--- best expected-route proxy ---")
    for row in payload["top_expected_route"][:10]:
        print(f"supp={row['supplement']} early={row['early_complete']}/{args.cases} "
              f"route={row['core_route_m']:.1f}/{row['full_route_m']:.1f} "
              f"proxy={row['expected_route_proxy_m']:.1f} m")
    print("--- best early rate with full route <= 18500 m ---")
    for row in payload["top_with_full_route_le_18500"][:10]:
        print(f"supp={row['supplement']} early={row['early_complete']}/{args.cases} "
              f"route={row['core_route_m']:.1f}/{row['full_route_m']:.1f} m")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
