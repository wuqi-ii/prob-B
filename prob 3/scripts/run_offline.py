"""离线批量演练：在本地模拟器上验证策略，不消耗真实测试机会。

用法
----
    python scripts/run_offline.py --cases 20 --start-seed 1
    python scripts/run_offline.py --cases 1 --start-seed 7 --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b3.config import StrategyConfig  # noqa: E402
from cumcm_b3.metrics import (  # noqa: E402
    aggregate, build_result, format_table, write_json,
)
from cumcm_b3.experimental_strategy import make_strategy as DogStrategy  # noqa: E402


def run_one(seed: int, cfg: StrategyConfig, verbose: bool = False,
            n_sources: int | None = None):
    case = generate_case(seed, n_sources=n_sources)
    sim = OfflineSimulator(case)
    strat = DogStrategy(sim, cfg, verbose=verbose)
    stats = strat.run()
    result = build_result(f"offline-{seed}", seed, case.total, stats)
    detail = {
        "result": result.as_row(),
        "truth": [
            {"channel": s.channel, "x": round(s.x, 2), "y": round(s.y, 2),
             "radius": round(s.receiver_radius_m, 1), "cleared": s.cleared}
            for s in case.sources
        ],
        "events": stats.events,
    }
    return result, detail, case


def main() -> int:
    parser = argparse.ArgumentParser(description="离线演练测试")
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--second-a", type=float, default=None,
                        help="第二检测点局部坐标 a（米）")
    parser.add_argument("--second-b", type=float, default=None,
                        help="第二检测点局部横向距离 |b|（米）")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-sources", type=int, default=0,
                        help="固定干扰源个数；0 表示按赛题随机取 10~16")
    parser.add_argument("--outdir", type=str, default="outputs/offline")
    args = parser.parse_args()

    cfg = StrategyConfig.from_file(args.config) if args.config else StrategyConfig()
    overrides = {}
    if args.second_a is not None:
        overrides["second_station_local_a_m"] = args.second_a
    if args.second_b is not None:
        overrides["second_station_local_b_m"] = args.second_b
    if overrides:
        cfg = replace(cfg, **overrides)
    cfg.validate()

    results, details = [], []
    started = time.time()
    for i in range(args.cases):
        seed = args.start_seed + i
        n_src = args.n_sources or None
        result, detail, case = run_one(seed, cfg, verbose=args.verbose, n_sources=n_src)
        results.append(result)
        details.append(detail)
        print(f"[{i+1:3d}/{args.cases}] seed={seed:<6d} 总数={case.total:2d} "
              f"清除={result.cleared_count:2d} 比例={result.cleared_ratio:5.3f} "
              f"总时间={result.total_time_s:8.1f}s 均值={result.mean_clear_time_s:7.1f}s "
              f"请求={result.requests:4d} 停止={result.stop_reason}")

    agg = aggregate(results)
    print("\n=== 汇总 ===")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    outdir = ROOT / args.outdir
    write_json(outdir / "summary.json", {"effective_config": cfg.to_dict(),
                                         "aggregate": agg,
                                         "results": [r.as_row() for r in results]})
    write_json(outdir / "details.json", details)
    (outdir / "table.md").write_text(format_table(results), encoding="utf-8")
    print(f"\n结果已写入 {outdir}")
    print(f"墙钟耗时 {time.time() - started:.1f} s")
    return 0 if agg.get("perfect_cases") == args.cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
