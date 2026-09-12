"""首次逼近横向偏置（lateral）在当前配置下的细扫 A/B。

首次逼近点从「沿示向度」改成「与示向度成 phi 度、前进 L 米」，增大两次观测的交会角，
改善定向源定位（当前瓶颈）。统计时间/miss/bisect/全清。

用法：
    python scripts/ab_lateral2.py --cases 200 --start-seed 14000
    python scripts/ab_lateral2.py --cases 200 --start-seed 20000 --directional-prob 1.0
"""

from __future__ import annotations

import argparse
import contextlib
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4 import strategy as strategy_module  # noqa: E402
from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


@contextlib.contextmanager
def lateral_verify(phi_deg, length_m):
    original = strategy_module.approach_point

    def patched(station, bearing_deg, step_m=None, cfg=None):
        length = length_m if length_m is not None else (step_m if step_m else 300.0)
        rad = math.radians(bearing_deg + phi_deg)
        return (station[0] + math.cos(rad) * length, station[1] + math.sin(rad) * length)

    strategy_module.approach_point = patched
    try:
        yield
    finally:
        strategy_module.approach_point = original


def run(seed, cfg, directional_prob):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    st = DogStrategy(sim, cfg).run()
    return {
        "time": sim.virtual_time_s,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "bisect": sum(1 for e in st.events if e.get("event") == "bisect"),
    }


def summarize(rows):
    n = len(rows)
    return (f"mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
            f"perfect {sum(1 for r in rows if r['cleared'] == r['total']):4d}/{n} | "
            f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
            f"bisect {statistics.fmean(r['bisect'] for r in rows):6.2f} | "
            f"req {statistics.fmean(r['req'] for r in rows):6.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    cfg = StrategyConfig()
    seeds = list(range(args.start_seed, args.start_seed + args.cases))

    variants = [
        ("phi0_L300", lateral_verify(0.0, 300.0)),
        ("phi15_L300", lateral_verify(15.0, 300.0)),
        ("phi20_L400", lateral_verify(20.0, 400.0)),
        ("phi30_L300", lateral_verify(30.0, 300.0)),
        ("phi30_L500", lateral_verify(30.0, 500.0)),
        ("phi20_L500", lateral_verify(20.0, 500.0)),
    ]

    print(f"cases={args.cases}  start_seed={args.start_seed}  "
          f"directional_prob={args.directional_prob}")
    results = {}
    for name, patch in variants:
        with patch:
            results[name] = [run(s, cfg, args.directional_prob) for s in seeds]

    base = results["phi0_L300"]
    for name, _ in variants:
        rows = results[name]
        print(f"  {name:12s}: {summarize(rows)}")
        if name != "phi0_L300":
            dt = [r["time"] - b["time"] for b, r in zip(base, rows)]
            db = [r["bisect"] - b["bisect"] for b, r in zip(base, rows)]
            n = len(dt)
            print(f"               paired dtime {statistics.fmean(dt):+8.1f}s "
                  f"(se {statistics.pstdev(dt) / math.sqrt(n):.1f}) "
                  f"rel {statistics.fmean(dt) / statistics.fmean(r['time'] for r in base):+.2%} | "
                  f"dbisect {statistics.fmean(db):+5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
