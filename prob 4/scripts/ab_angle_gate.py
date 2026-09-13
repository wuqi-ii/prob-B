"""angle_gate_bisect A/B：越界后按「lo→hi 方向 vs bearing」夹角门控跳过无效二分。

配对 baseline vs angle_gate（阈值 15°/20°/30°）。重点看：
- 时间是否下降、bisect 次数是否减少；
- 全清率是否保持 100%（跳过二分是否导致漏检）。
"""
import sys, math, statistics, argparse
from dataclasses import replace
sys.path.insert(0, "src")

from cumcm_b4.backend_offline import OfflineSimulator, generate_case
from cumcm_b4.config import StrategyConfig
from cumcm_b4.strategy import DogStrategy


def run(seed, cfg, directional_prob):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    st = DogStrategy(sim, cfg).run()
    per = list(st.per_target_time_s.values())
    return {
        "time": sim.virtual_time_s,
        "per_source": sim.virtual_time_s / sim.case.total,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "bisect": sum(e.get("event") == "bisect" for e in st.events),
        "max_src": max(per) if per else 0.0,
    }


def summary(rows):
    n = len(rows)
    ps = [r["per_source"] for r in rows]
    mx = [r["max_src"] for r in rows]
    perfect = sum(r["cleared"] == r["total"] for r in rows)
    return (f"mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
            f"per-src min/mean/max {min(ps):6.1f}/{statistics.fmean(ps):6.1f}/{max(ps):6.1f}s | "
            f"max_src(mean) {statistics.fmean(mx):6.1f}s | "
            f"perfect {perfect:4d}/{n} | "
            f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
            f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--dir-prob", type=float, default=0.5)
    args = ap.parse_args()

    base_cfg = StrategyConfig()
    seeds = range(args.start_seed, args.start_seed + args.cases)
    variants = [("baseline", base_cfg)]
    for th in (15.0, 20.0, 30.0):
        variants.append((f"gate{int(th)}",
                         replace(base_cfg, angle_gate_bisect=True,
                                 angle_gate_threshold_deg=th)))

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.dir_prob}")
    results = {}
    for name, cfg in variants:
        rows = [run(s, cfg, args.dir_prob) for s in seeds]
        results[name] = rows
        print(f"{name:12s}: {summary(rows)}")

    base = results["baseline"]
    bmean = statistics.fmean(r["time"] for r in base)
    for name, _ in variants[1:]:
        rows = results[name]
        delta = [r["time"] - b["time"] for r, b in zip(rows, base)]
        se = statistics.pstdev(delta) / math.sqrt(len(delta))
        print(f"{name:12s}: paired {statistics.fmean(delta):+8.1f}s "
              f"({statistics.fmean(delta)/bmean:+.2%}, se {se:.1f}) | "
              f"bisect {statistics.fmean(r['bisect']-b['bisect'] for r,b in zip(rows,base)):+.1f}")


if __name__ == "__main__":
    main()
