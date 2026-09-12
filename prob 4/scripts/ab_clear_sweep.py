"""越界后沿括号线段做单向光学扫线的 A/B。

/clear 未命中只耗时3秒且不受定向覆盖角限制。机器狗从无信号端 hi 向有信号端
lo 单向移动，每隔 spacing 米尝试清除；若整段未命中，再回退到安全的括号搜索。
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


class ClearSweepStrategy(DogStrategy):
    def __init__(self, *args, spacing_m: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacing_m = spacing_m

    def _resolve_overshoot(self, ch, lo_pt, hi_pt):
        length = math.dist(lo_pt, hi_pt)
        if length <= 1e-9:
            return super()._resolve_overshoot(ch, lo_pt, hi_pt)
        count = max(1, math.ceil(length / self.spacing_m))
        # 从当前所在的 hi 单向向 lo 扫；端点间距不超过 spacing。
        for k in range(count + 1):
            t = k / count
            point = (hi_pt[0] + t * (lo_pt[0] - hi_pt[0]),
                     hi_pt[1] + t * (lo_pt[1] - hi_pt[1]))
            self._log("clear_sweep", channel=ch,
                      at=[round(point[0], 1), round(point[1], 1)])
            if self._try_clear(point, ch):
                return True
        self._log("clear_sweep_fallback", channel=ch, length=round(length, 1))
        return super()._resolve_overshoot(ch, lo_pt, hi_pt)


def run(seed, cfg, directional_prob, spacing):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    if spacing is None:
        st = DogStrategy(sim, cfg).run()
    else:
        st = ClearSweepStrategy(sim, cfg, spacing_m=spacing).run()
    return {
        "time": sim.virtual_time_s,
        "per_source": sim.virtual_time_s / sim.case.total,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "fallback": sum(e.get("event") == "clear_sweep_fallback" for e in st.events),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=20000)
    ap.add_argument("--directional-prob", type=float, default=1.0)
    args = ap.parse_args()
    cfg = StrategyConfig()
    variants = [("baseline", None), ("s25", 25.0), ("s30", 30.0),
                ("s35", 35.0), ("s40", 40.0)]
    results = {}
    for name, spacing in variants:
        rows = [run(seed, cfg, args.directional_prob, spacing)
                for seed in range(args.start_seed, args.start_seed + args.cases)]
        results[name] = rows
        ps = [r["per_source"] for r in rows]
        print(
            f"{name:8s}: mean={statistics.fmean(r['time'] for r in rows):.1f}s "
            f"per-source={min(ps):.1f}/{statistics.fmean(ps):.1f}/{max(ps):.1f}s "
            f"perfect={sum(r['cleared'] == r['total'] for r in rows)}/{len(rows)} "
            f"miss={statistics.fmean(r['miss'] for r in rows):.2f} "
            f"req={statistics.fmean(r['req'] for r in rows):.1f} "
            f"fallback={sum(r['fallback'] for r in rows)}"
        )
    base = results["baseline"]
    base_mean = statistics.fmean(r["time"] for r in base)
    for name, _ in variants[1:]:
        diffs = [r["time"] - b["time"] for r, b in zip(results[name], base)]
        print(f"{name:8s}: paired={statistics.fmean(diffs):+.1f}s "
              f"({statistics.fmean(diffs)/base_mean:+.2%}, "
              f"se={statistics.pstdev(diffs)/math.sqrt(len(diffs)):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
