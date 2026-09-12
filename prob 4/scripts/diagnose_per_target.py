"""每源定位清除耗时诊断：定向 vs 全向，找出「每源耗时大」的根因。

统计每个源从发现到清除的过程中：approach / bisect / ring / clear 尝试 / miss 次数，
按源类型（定向/全向）分组，看时间到底花在逼近、二分、还是环形兜底上。

用法：
    python scripts/diagnose_per_target.py --cases 60 --start-seed 14000
"""

from __future__ import annotations

import argparse
import collections
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    cfg = StrategyConfig()
    # 按源类型聚合：dir -> 每个源的 {time, approach, bisect, ring, clear_try, miss}
    agg = {True: [], False: []}
    per_source_rows = []

    for seed in range(args.start_seed, args.start_seed + args.cases):
        sim = OfflineSimulator(generate_case(seed, directional_prob=args.directional_prob))
        st = DogStrategy(sim, cfg).run()
        src_by_ch = {s.channel: s for s in sim.case.sources}
        # 每源的计数
        cnt = collections.defaultdict(lambda: collections.Counter())
        for e in st.events:
            ch = e.get("channel")
            if ch is None:
                continue
            if e["event"] in ("approach", "bisect", "ring", "clear", "clear_failed"):
                cnt[ch][e["event"]] += 1
        for ch, t in st.per_target_time_s.items():
            s = src_by_ch.get(ch)
            if s is None:
                continue
            row = {
                "time": t,
                "approach": cnt[ch]["approach"],
                "bisect": cnt[ch]["bisect"],
                "ring": cnt[ch]["ring"],
                "clear_try": cnt[ch]["clear"],
                "miss": cnt[ch]["clear_failed"],
            }
            per_source_rows.append((s.directional, row))
            agg[s.directional].append(row)

    def stat(rows, key):
        vals = [r[key] for r in rows]
        return f"avg {statistics.fmean(vals):6.2f}"

    print(f"cases={args.cases}  directional_prob={args.directional_prob}")
    for is_dir, label in ((False, "全向源"), (True, "定向源")):
        rows = agg[is_dir]
        n = len(rows)
        if not n:
            continue
        print(f"\n[{label}] n={n}")
        print(f"  定位清除时间   : {stat(rows, 'time')} s")
        print(f"  approach 次数  : {stat(rows, 'approach')}")
        print(f"  bisect 次数    : {stat(rows, 'bisect')}")
        print(f"  ring 次数      : {stat(rows, 'ring')}")
        print(f"  clear 尝试     : {stat(rows, 'clear_try')}")
        print(f"  miss 次数      : {stat(rows, 'miss')}")
        # 有 miss 的源占比
        missy = sum(1 for r in rows if r["miss"] > 0)
        print(f"  有 miss 的源占比: {missy}/{n} = {missy/n*100:.0f}%")

    # 最难源 top 8（按时间）
    print("\n最难源 top8（按定位清除时间）:")
    per_source_rows.sort(key=lambda x: -x[1]["time"])
    for is_dir, r in per_source_rows[:8]:
        tag = "定向" if is_dir else "全向"
        print(f"  {tag} time={r['time']:6.1f}s  approach={r['approach']}  "
              f"bisect={r['bisect']}  ring={r['ring']}  try={r['clear_try']}  "
              f"miss={r['miss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
