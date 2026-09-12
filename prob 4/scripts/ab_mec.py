"""MEC 圆心 vs 质心 清除站位的配对 A/B。

清除站位从「质心 + 质心到最远顶点距离」换成「可行域顶点最小覆盖圆圆心 + 半径」，
看 miss 数、总时间、全清率是否改善。miss 是时间浪费主因（每次 miss 触发 7 点环形兜底）。

用法：
    python scripts/ab_mec.py --cases 400 --start-seed 14000
    python scripts/ab_mec.py --cases 300 --start-seed 20000 --directional-prob 1.0
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
    return {
        "time": sim.virtual_time_s,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
    }


def summarize(rows):
    n = len(rows)
    return (
        f"mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
        f"perfect {sum(1 for r in rows if r['cleared'] == r['total']):4d}/{n} | "
        f"miss {statistics.fmean(r['miss'] for r in rows):6.2f} "
        f"(max {max(r['miss'] for r in rows)}) | "
        f"req {statistics.fmean(r['req'] for r in rows):6.1f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    base = StrategyConfig()
    mec = replace(base, use_mec_clear=True)
    seeds = list(range(args.start_seed, args.start_seed + args.cases))

    base_rows = [run(s, base, args.directional_prob) for s in seeds]
    mec_rows = [run(s, mec, args.directional_prob) for s in seeds]

    print(f"cases={args.cases}  start_seed={args.start_seed}  "
          f"directional_prob={args.directional_prob}")
    print(f"  centroid : {summarize(base_rows)}")
    print(f"  mec      : {summarize(mec_rows)}")

    dt = [m["time"] - b["time"] for b, m in zip(base_rows, mec_rows)]
    dm = [m["miss"] - b["miss"] for b, m in zip(base_rows, mec_rows)]
    dr = [m["req"] - b["req"] for b, m in zip(base_rows, mec_rows)]
    n = len(dt)
    print(f"  paired dtime : {statistics.fmean(dt):+8.1f}s "
          f"(se {statistics.pstdev(dt) / math.sqrt(n):.1f}) "
          f"rel {statistics.fmean(dt) / statistics.fmean(r['time'] for r in base_rows):+.2%}")
    print(f"  paired dmiss : {statistics.fmean(dm):+8.2f}")
    print(f"  paired dreq  : {statistics.fmean(dr):+8.1f}")
    # 有多少局 miss 变少了 / 变多了
    fewer = sum(1 for d in dm if d < 0)
    more = sum(1 for d in dm if d > 0)
    same = sum(1 for d in dm if d == 0)
    print(f"  miss 减少 {fewer} 局 / 增加 {more} 局 / 持平 {same} 局")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
