"""ring_clear 重定位补救 A/B：验证 ch17 类「二分横向出楔带偏定位」的补救收益。

背景（drill24 ch17）：4 远观测定位准（MEC 圆心距源 0.1m，半径 32.8m），但 32.8m>30m
判据 → 逼近越界 → 二分横向出楔收敛到距源 80m 处 → ring_clear 7 次全 miss → 冻结 →
终局重试才用全部观测命中。relocate 在 ring 全 miss 后立即用全部观测重定位补救。
"""
import sys, math, statistics, argparse
from dataclasses import replace
sys.path.insert(0, "src")

from cumcm_b4.backend_offline import OfflineSimulator, generate_case
from cumcm_b4.config import StrategyConfig
from cumcm_b4.strategy import DogStrategy


def run(seed, cfg, dp):
    sim = OfflineSimulator(generate_case(seed, directional_prob=dp))
    st = DogStrategy(sim, cfg).run()
    per = list(st.per_target_time_s.values())
    return {
        "time": sim.virtual_time_s,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "bisect": sum(e.get("event") == "bisect" for e in st.events),
        "reloc": sum(e.get("event") == "ring_relocate" for e in st.events),
        "max_src": max(per) if per else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--dir-prob", type=float, default=0.5)
    args = ap.parse_args()

    base = StrategyConfig()  # relocate=True（新默认）
    off = replace(base, ring_clear_relocate=False)
    seeds = range(args.start_seed, args.start_seed + args.cases)

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.dir_prob}")
    results = {}
    for name, cfg in [("reloc_off", off), ("reloc_on", base)]:
        rows = [run(s, cfg, args.dir_prob) for s in seeds]
        results[name] = rows
        perfect = sum(r["cleared"] == r["total"] for r in rows)
        print(f"{name:10s}: mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
              f"max_src {statistics.fmean(r['max_src'] for r in rows):6.1f}s | "
              f"perfect {perfect:4d}/{args.cases} | "
              f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
              f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f} | "
              f"reloc {statistics.fmean(r['reloc'] for r in rows):5.2f}")

    on = results["reloc_on"]
    of = results["reloc_off"]
    bmean = statistics.fmean(r["time"] for r in of)
    delta = [a["time"] - b["time"] for a, b in zip(on, of)]
    se = statistics.pstdev(delta) / math.sqrt(len(delta))
    dperfect = sum(r["cleared"] != r["total"] for r in on) - sum(r["cleared"] != r["total"] for r in of)
    print(f"\nreloc_on vs off: paired {statistics.fmean(delta):+8.1f}s "
          f"({statistics.fmean(delta)/bmean:+.2%}, se {se:.1f}) | "
          f"漏检Δ {dperfect:+d} | "
          f"missΔ {statistics.fmean(a['miss']-b['miss'] for a,b in zip(on,of)):+.2f} | "
          f"reloc触发 {statistics.fmean(r['reloc'] for r in on):.2f}/局")


if __name__ == "__main__":
    main()
