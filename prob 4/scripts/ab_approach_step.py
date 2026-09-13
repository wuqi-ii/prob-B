"""逼近步长 A/B：减小逼近步长，避免大幅越过源，从而缩短二分区间、减少二分次数。

瓶颈诊断（diagnose_per_target）：最难源 700~957s 几乎全来自 bisect（最多触顶 20 次）。
根源：approach step = 0.85 * dist(cur, anchor)，上限 400m，当 anchor 偏（定向源可行域瘦长）
时一步大幅越过源，二分区间 [最后有信号点, 越界点] 过长，0.85 衰减要 20 次才收敛。

本脚本扫 approach_step_ratio 与 approach_max_step_m，看是否能压低 max 单源时间与总时间。

用法：
    python scripts/ab_approach_step.py --cases 100 --start-seed 14000
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


def run(seed: int, cfg: StrategyConfig, directional_prob: float) -> dict:
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    st = DogStrategy(sim, cfg).run()
    bisect = sum(1 for e in st.events if e.get("event") == "bisect")
    # 单源最大耗时（排除扫描摊派，只看定位清除时间）
    per_src = list(st.per_target_time_s.values())
    return {
        "time": sim.virtual_time_s,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "bisect": bisect,
        "max_per_src": max(per_src) if per_src else 0.0,
        "req": st.requests,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    base = StrategyConfig()
    seeds = list(range(args.start_seed, args.start_seed + args.cases))
    variants: list[tuple[str, StrategyConfig]] = [("base", base)]
    for ratio in (0.65, 0.75):
        variants.append((f"ratio{ratio}", replace(base, approach_step_ratio=ratio)))
    for mx in (200.0, 300.0):
        variants.append((f"max{mx:.0f}", replace(base, approach_max_step_m=mx)))
    # 组合：小步长 + 小上限
    variants.append(("ratio0.7_max250", replace(base, approach_step_ratio=0.7, approach_max_step_m=250.0)))

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.directional_prob}")
    results = {}
    for name, cfg in variants:
        rows = [run(s, cfg, args.directional_prob) for s in seeds]
        results[name] = rows
        print(
            f"{name:16s}: mean {statistics.fmean(r['time'] for r in rows):7.1f}s | "
            f"perfect {sum(r['cleared']==r['total'] for r in rows):3d}/{len(rows)} | "
            f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
            f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f} | "
            f"max_src {statistics.fmean(r['max_per_src'] for r in rows):6.1f}s | "
            f"req {statistics.fmean(r['req'] for r in rows):6.1f}"
        )

    base_rows = results["base"]
    base_mean = statistics.fmean(r["time"] for r in base_rows)
    print("\n配对差 vs base:")
    for name, _ in variants[1:]:
        rows = results[name]
        d = [r["time"] - b["time"] for r, b in zip(rows, base_rows)]
        dmax = [r["max_per_src"] - b["max_per_src"] for r, b in zip(rows, base_rows)]
        se = statistics.pstdev(d) / math.sqrt(len(d))
        failed = [s for s, r in zip(seeds, rows) if r["cleared"] != r["total"]]
        print(
            f"{name:16s}: time {statistics.fmean(d):+8.1f}s ({statistics.fmean(d)/base_mean:+.2%}, se {se:.1f}) | "
            f"max_src {statistics.fmean(dmax):+7.1f}s | failed {len(failed)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
