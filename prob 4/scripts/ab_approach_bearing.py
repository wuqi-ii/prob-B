"""逼近方向 A/B：朝 anchor（历史） vs 沿 bearing（避免定向源 overshoot）。

统计总时间 / miss / bisect 次数 / 全清率 的配对差。

用法：
    python scripts/ab_approach_bearing.py --cases 400 --start-seed 14000
    python scripts/ab_approach_bearing.py --cases 300 --start-seed 20000 --directional-prob 1.0
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


def run(seed, cfg, directional_prob):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    st = DogStrategy(sim, cfg).run()
    bisect = sum(1 for e in st.events if e.get("event") == "bisect")
    approach = sum(1 for e in st.events if e.get("event") == "approach")
    return {
        "time": sim.virtual_time_s,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "bisect": bisect,
        "approach": approach,
    }


def summarize(rows):
    n = len(rows)
    return (
        f"mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
        f"perfect {sum(1 for r in rows if r['cleared'] == r['total']):4d}/{n} | "
        f"miss {statistics.fmean(r['miss'] for r in rows):6.2f} | "
        f"bisect {statistics.fmean(r['bisect'] for r in rows):6.2f} | "
        f"approach {statistics.fmean(r['approach'] for r in rows):6.2f} | "
        f"req {statistics.fmean(r['req'] for r in rows):6.1f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    base = StrategyConfig()
    var = replace(base, approach_along_bearing=True)
    seeds = list(range(args.start_seed, args.start_seed + args.cases))

    base_rows = [run(s, base, args.directional_prob) for s in seeds]
    var_rows = [run(s, var, args.directional_prob) for s in seeds]

    print(f"cases={args.cases}  start_seed={args.start_seed}  "
          f"directional_prob={args.directional_prob}")
    print(f"  anchor  : {summarize(base_rows)}")
    print(f"  bearing : {summarize(var_rows)}")

    dt = [v["time"] - b["time"] for b, v in zip(base_rows, var_rows)]
    dm = [v["miss"] - b["miss"] for b, v in zip(base_rows, var_rows)]
    db = [v["bisect"] - b["bisect"] for b, v in zip(base_rows, var_rows)]
    n = len(dt)
    print(f"  paired dtime   : {statistics.fmean(dt):+8.1f}s "
          f"(se {statistics.pstdev(dt) / math.sqrt(n):.1f}) "
          f"rel {statistics.fmean(dt) / statistics.fmean(r['time'] for r in base_rows):+.2%}")
    print(f"  paired dmiss   : {statistics.fmean(dm):+8.2f}")
    print(f"  paired dbisect : {statistics.fmean(db):+8.2f}")
    fewer = sum(1 for d in db if d < 0)
    more = sum(1 for d in db if d > 0)
    print(f"  bisect 减少 {fewer} 局 / 增加 {more} 局 / 持平 {n - fewer - more} 局")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
