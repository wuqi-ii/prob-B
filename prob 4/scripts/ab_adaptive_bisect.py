"""自适应二分 f A/B：最难源 bisect 触顶 20 次的根因是 f=0.85 收敛慢。

二分 [lo(有信号), hi(无信号)]，mid = lo + f*(hi-lo)。f=0.85 时若 mid 仍 no_signal
（大幅越过源、源在 lo 附近），区间仅缩到 0.85，20 次才从 ~400m 缩到 ~15m（触顶）。

自适应：no_signal 时改用更小的 f（快速退回 lo），direction 时保持大 f（源在 hi 侧）。

用法：
    python scripts/ab_adaptive_bisect.py --cases 100 --start-seed 14000
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
from cumcm_b4.config import StrategyConfig, CLEAR_RADIUS_M  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


class AdaptiveBisectStrategy(DogStrategy):
    """no_signal 时改用 backoff_f，其余保持 bisect_fraction。"""

    def __init__(self, *args, backoff_f: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.backoff_f = backoff_f

    def _resolve_overshoot(self, ch, lo_pt, hi_pt):
        lo = lo_pt
        hi = hi_pt
        f = self.cfg.bisect_fraction
        for _ in range(20):
            if math.dist(lo, hi) <= self.cfg.bisect_tolerance_m:
                break
            mid = (lo[0] + f * (hi[0] - lo[0]), lo[1] + f * (hi[1] - lo[1]))
            payload = self.backend.measure(mid[0], mid[1], ch)
            kind = self.book.apply_measure(ch, mid, payload)
            self._log("bisect", channel=ch, at=[round(mid[0], 1), round(mid[1], 1)], result=kind)
            if kind == "near":
                return self._try_clear(mid, ch)
            if kind == "direction":
                lo = mid
                f = self.cfg.bisect_fraction  # 源在 hi 侧，保持大步
            else:
                hi = mid
                f = self.backoff_f  # no_signal：源在 lo 侧，改用小 f 快速退回
        center = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
        return self._ring_clear(center, ch)


def run(seed: int, cfg: StrategyConfig, directional_prob: float, backoff: float | None):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    if backoff is None:
        st = DogStrategy(sim, cfg).run()
    else:
        st = AdaptiveBisectStrategy(sim, cfg, backoff_f=backoff).run()
    bisect = sum(1 for e in st.events if e.get("event") == "bisect")
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

    cfg = StrategyConfig()
    seeds = list(range(args.start_seed, args.start_seed + args.cases))
    variants = [("base", None), ("adapt0.5", 0.5), ("adapt0.35", 0.35)]

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.directional_prob}")
    results = {}
    for name, backoff in variants:
        rows = [run(s, cfg, args.directional_prob, backoff) for s in seeds]
        results[name] = rows
        print(
            f"{name:12s}: mean {statistics.fmean(r['time'] for r in rows):7.1f}s | "
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
        dbis = [r["bisect"] - b["bisect"] for r, b in zip(rows, base_rows)]
        se = statistics.pstdev(d) / math.sqrt(len(d))
        failed = [s for s, r in zip(seeds, rows) if r["cleared"] != r["total"]]
        print(
            f"{name:12s}: time {statistics.fmean(d):+8.1f}s ({statistics.fmean(d)/base_mean:+.2%}, se {se:.1f}) | "
            f"max_src {statistics.fmean(dmax):+7.1f}s | bisect {statistics.fmean(dbis):+5.2f} | failed {len(failed)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
