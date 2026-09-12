"""参数敏感性扫描（问题 4）：找出真正影响总时间与清除率的旋钮。

用法
----
    python scripts/sweep.py --cases 30 --start-seed 300
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.metrics import aggregate  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


def evaluate(cfg: StrategyConfig, seeds: list[int], directional_prob: float) -> dict:
    results = []
    for seed in seeds:
        case = generate_case(seed, directional_prob=directional_prob)
        sim = OfflineSimulator(case)
        stats = DogStrategy(sim, cfg).run()
        cleared = sim.case.count_cleared()
        results.append({
            "seed": seed,
            "total": case.total,
            "cleared": cleared,
            "ratio": cleared / case.total,
            "time_s": sim.virtual_time_s,
            "requests": sim.request_count,
        })
    ratios = [r["ratio"] for r in results]
    times = [r["time_s"] for r in results]
    return {
        "cases": len(results),
        "ratio_mean": statistics.fmean(ratios),
        "perfect": sum(1 for r in ratios if r >= 1.0 - 1e-12),
        "time_mean_s": statistics.fmean(times),
        "time_median_s": statistics.median(times),
        "time_max_s": max(times),
        "requests_mean": statistics.fmean([r["requests"] for r in results]),
        "raw": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=300)
    parser.add_argument("--directional-prob", type=float, default=0.5)
    parser.add_argument("--outdir", type=str, default="outputs/sweep")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.cases))
    base = StrategyConfig()

    # 问题 4 的关键旋钮：包围网密度（外环点数）、环形兜底、逼近步长。
    grid = {
        "scan_outer_count": [18, 24],
        "ring_clear_enabled": [True, False],
        "approach_step_ratio": [0.65, 0.85],
    }
    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))

    rows = []
    print(f"共 {len(combos)} 组配置 × {len(seeds)} 个案例（定向概率 {args.directional_prob}）")
    for combo in combos:
        cfg = replace(base, **dict(zip(keys, combo)))
        cfg.validate()
        t0 = time.time()
        stat = evaluate(cfg, seeds, args.directional_prob)
        label = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
        rows.append({"params": dict(zip(keys, combo)), **{k: v for k, v in stat.items() if k != "raw"}})
        print(f"{label:<60s} 比例={stat['ratio_mean']:.3f} "
              f"满分={stat['perfect']}/{stat['cases']} "
              f"均值时间={stat['time_mean_s']:.0f}s "
              f"({time.time()-t0:.1f}s 墙钟)")

    rows.sort(key=lambda r: r["time_mean_s"])
    print("\n=== 按平均总时间排序 ===")
    for r in rows:
        print(f"  {r['time_mean_s']:8.0f}s  满分 {r['perfect']:3d}/{r['cases']}  "
              f"请求 {r['requests_mean']:6.1f}  {r['params']}")

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sweep.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {outdir / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
