"""越界恢复的非对称二分 A/B。

机器狗触发越界时位于 hi（无信号端），标准中点法第一步要回走半个括号。
本实验令探测点 = lo + fraction*(hi-lo)，fraction>0.5 时更靠近机器狗，
始终保留“lo 有信号、hi 无信号”的正确括号，因此只改变搜索顺序与路程。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


class FractionalBisectStrategy(DogStrategy):
    fraction = 0.5

    def __init__(self, *args, fraction: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.fraction = fraction

    def _resolve_overshoot(self, ch, lo_pt, hi_pt):
        lo, hi = lo_pt, hi_pt
        for _ in range(40):
            if math.dist(lo, hi) <= self.cfg.bisect_tolerance_m:
                break
            f = self.fraction
            point = (lo[0] + f * (hi[0] - lo[0]),
                     lo[1] + f * (hi[1] - lo[1]))
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            self._log("bisect", channel=ch,
                      at=[round(point[0], 1), round(point[1], 1)], result=kind)
            if kind == "near":
                return self._try_clear(point, ch)
            if kind == "direction":
                lo = point
            else:
                hi = point
        center = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
        return self._ring_clear(center, ch)


def run(seed, cfg, directional_prob, fraction):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    st = FractionalBisectStrategy(sim, cfg, fraction=fraction).run()
    return {
        "time": sim.virtual_time_s,
        "per_source": sim.virtual_time_s / sim.case.total,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "bisect": sum(e.get("event") == "bisect" for e in st.events),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=20000)
    ap.add_argument("--directional-prob", type=float, default=1.0)
    ap.add_argument("--fractions", default="0.5,0.7,0.8,0.9,0.92,0.94,0.96",
                    help="逗号分隔；长实验可分进程运行以规避解释器偶发故障")
    args = ap.parse_args()
    cfg = StrategyConfig()
    seeds = range(args.start_seed, args.start_seed + args.cases)
    fractions = tuple(float(x) for x in args.fractions.split(","))
    results = {}
    for fraction in fractions:
        rows = [run(seed, cfg, args.directional_prob, fraction) for seed in seeds]
        results[fraction] = rows
        ps = [r["per_source"] for r in rows]
        print(
            f"f={fraction:.2f}: mean={statistics.fmean(r['time'] for r in rows):.1f}s "
            f"per-source={min(ps):.1f}/{statistics.fmean(ps):.1f}/{max(ps):.1f}s "
            f"perfect={sum(r['cleared'] == r['total'] for r in rows)}/{len(rows)} "
            f"miss={statistics.fmean(r['miss'] for r in rows):.2f} "
            f"bisect={statistics.fmean(r['bisect'] for r in rows):.2f} "
            f"req={statistics.fmean(r['req'] for r in rows):.1f}"
        )
    if 0.5 not in results:
        return 0
    base = results[0.5]
    base_mean = statistics.fmean(r["time"] for r in base)
    for fraction in fractions[1:]:
        diffs = [r["time"] - b["time"] for r, b in zip(results[fraction], base)]
        print(
            f"f={fraction:.2f}: paired={statistics.fmean(diffs):+.1f}s "
            f"({statistics.fmean(diffs) / base_mean:+.2%}, "
            f"se={statistics.pstdev(diffs) / math.sqrt(len(diffs)):.1f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
