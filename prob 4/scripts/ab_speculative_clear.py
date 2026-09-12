"""逼近途中先尝试低成本光学清除，再测量的安全 A/B。

清除未命中只增加3秒且机器狗仍停在原点位；随后照常测量、逼近和二分，因而失败
不会丢失定位信息。若提前命中，则省去后续测量与移动。threshold 控制只在 MEC
最坏半径不超过多少米时尝试。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4 import geometry  # noqa: E402
from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import CLEAR_RADIUS_M, StrategyConfig  # noqa: E402
from cumcm_b4.second_station import approach_point  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


class SpeculativeClearStrategy(DogStrategy):
    def __init__(self, *args, threshold_m: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold_m = threshold_m

    def _approach_and_clear(self, ch):
        cfg = self.cfg
        track = self.book[ch]
        if track.near_points:
            return self._try_clear(track.near_points[-1], ch)
        for _ in range(cfg.approach_max_iterations):
            anchor, worst = self._clear_anchor(ch)
            if anchor is None:
                return False
            if worst <= CLEAR_RADIUS_M - cfg.clear_safety_margin_m:
                if self._try_clear(anchor, ch):
                    return True
                payload = self.backend.measure(anchor[0], anchor[1], ch)
                kind = self.book.apply_measure(ch, anchor, payload)
                if kind == "near":
                    return self._try_clear(anchor, ch)
                if kind == "direction":
                    continue
                return self._ring_clear(anchor, ch)

            cur = self._pos()
            d = math.dist(cur, anchor)
            step = max(cfg.approach_min_step_m,
                       min(cfg.approach_step_ratio * d, cfg.approach_max_step_m))
            if cfg.approach_along_bearing and track.observations:
                target = approach_point(cur, track.observations[-1][1], step)
            else:
                target = geometry.point_along(cur, anchor, step)

            if worst <= self.threshold_m:
                self._log("speculative_clear", channel=ch, worst=round(worst, 1))
                if self._try_clear(target, ch):
                    return True

            payload = self.backend.measure(target[0], target[1], ch)
            kind = self.book.apply_measure(ch, target, payload)
            self._log("approach", channel=ch,
                      at=[round(target[0], 1), round(target[1], 1)],
                      worst=round(worst, 1) if math.isfinite(worst) else -1,
                      result=kind)
            if kind == "near":
                return self._try_clear(target, ch)
            if kind == "no_signal":
                lo = track.observations[-1][0] if track.observations else cur
                if self._resolve_overshoot(ch, lo, target):
                    return True
                return self._ring_clear(anchor, ch)
        return False


def run(seed, cfg, directional_prob, threshold):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    if threshold is None:
        st = DogStrategy(sim, cfg).run()
    else:
        st = SpeculativeClearStrategy(sim, cfg, threshold_m=threshold).run()
    return {"time": sim.virtual_time_s, "per_source": sim.virtual_time_s/sim.case.total,
            "cleared": st.cleared_count, "total": sim.case.total,
            "miss": st.clear_miss, "req": st.requests,
            "tries": sum(e.get("event") == "speculative_clear" for e in st.events)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()
    cfg = StrategyConfig()
    variants = [("baseline", None), ("t30", 30.), ("t50", 50.),
                ("t80", 80.), ("t120", 120.)]
    results = {}
    for name, threshold in variants:
        rows = [run(s, cfg, args.directional_prob, threshold)
                for s in range(args.start_seed, args.start_seed + args.cases)]
        results[name] = rows
        ps = [r["per_source"] for r in rows]
        print(f"{name:8s}: mean={statistics.fmean(r['time'] for r in rows):.1f}s "
              f"per-source={min(ps):.1f}/{statistics.fmean(ps):.1f}/{max(ps):.1f}s "
              f"perfect={sum(r['cleared']==r['total'] for r in rows)}/{len(rows)} "
              f"miss={statistics.fmean(r['miss'] for r in rows):.2f} "
              f"tries={statistics.fmean(r['tries'] for r in rows):.2f}")
    base = results["baseline"]
    bm = statistics.fmean(r["time"] for r in base)
    for name, _ in variants[1:]:
        d = [r["time"]-b["time"] for r,b in zip(results[name],base)]
        print(f"{name:8s}: paired={statistics.fmean(d):+.1f}s "
              f"({statistics.fmean(d)/bm:+.2%}, se={statistics.pstdev(d)/math.sqrt(len(d)):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
