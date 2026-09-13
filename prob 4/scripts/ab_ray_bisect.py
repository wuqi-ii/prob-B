"""ray_bisect A/B：越界后二分方向从 lo→hi 连线改为沿 bearing 射线。

配对 baseline（ray_bisect=False）vs ray_bisect=True。重点看：
- max_src（单源长尾，807s 那类）是否被压缩；
- mean / per-source 是否有收益且全清率保持 100%。
用 CPython 3.12.14 跑（规避 3.11 长循环偶发内存污染崩溃）。
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
        "dir": sim.case.directional_count(),
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
            f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f} | "
            f"dir {statistics.fmean(r['dir'] for r in rows):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--dir-prob", type=float, default=0.5)
    args = ap.parse_args()

    base_cfg = StrategyConfig()
    seeds = range(args.start_seed, args.start_seed + args.cases)
    variants = [("baseline", base_cfg),
                ("ray_bisect", replace(base_cfg, ray_bisect=True))]

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.dir_prob}")
    results = {}
    for name, cfg in variants:
        rows = [run(s, cfg, args.dir_prob) for s in seeds]
        results[name] = rows
        print(f"{name:12s}: {summary(rows)}")

    base = results["baseline"]
    bmean = statistics.fmean(r["time"] for r in base)
    rows = results["ray_bisect"]
    delta = [r["time"] - b["time"] for r, b in zip(rows, base)]
    se = statistics.pstdev(delta) / math.sqrt(len(delta))
    print(f"ray_bisect paired: time {statistics.fmean(delta):+8.1f}s "
          f"({statistics.fmean(delta)/bmean:+.2%}, se {se:.1f})")
    dmax = [r["max_src"] - b["max_src"] for r, b in zip(rows, base)]
    print(f"ray_bisect paired: max_src {statistics.fmean(dmax):+8.1f}s "
          f"(baseline {statistics.fmean(r['max_src'] for r in base):.1f}s -> "
          f"ray {statistics.fmean(r['max_src'] for r in rows):.1f}s)")
    dbis = [r["bisect"] - b["bisect"] for r, b in zip(rows, base)]
    print(f"ray_bisect paired: bisect {statistics.fmean(dbis):+6.2f} 次/局")


if __name__ == "__main__":
    main()
