"""扫描 task_bias_clear_m：清除时机对「总时间 / 测量次数 / 移动」的联合影响。

背景：清除时机越早（bias 正值），已定位源越早被清除，后续扫描点对该频道的
冗余测量越少（测量省时间）；但清除会打断扫描路径（移动增）。这是一个明确的
tradeoff，之前只测过负值（延后清除），正值（提前清除）未系统测过。
"""
import sys, math, statistics, argparse
sys.path.insert(0, "src")

from dataclasses import replace
from cumcm_b4.backend_offline import OfflineSimulator, generate_case
from cumcm_b4.strategy import DogStrategy
from cumcm_b4.config import StrategyConfig


def run(seed, cfg):
    sim = OfflineSimulator(generate_case(seed, directional_prob=0.5))
    st = DogStrategy(sim, cfg).run()
    # 从 events 统计扫描点测量次数（scan_done 事件里的 channels 计数）
    scan_meas = 0
    for e in st.events:
        if isinstance(e, dict) and e.get("event") == "scan_done":
            scan_meas += e.get("channels", 0)
    return dict(
        time=sim.virtual_time_s,
        cleared=st.cleared_count,
        total=sim.case.total,
        miss=st.clear_miss,
        requests=st.requests,
        scan_meas=scan_meas,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=120)
    ap.add_argument("--start-seed", type=int, default=14000)
    args = ap.parse_args()

    base = StrategyConfig()
    seeds = list(range(args.start_seed, args.start_seed + args.cases))

    variants = [-300, -150, -100, -50, 0, 50, 100, 150, 200, 300]
    base_results = {s: run(s, base) for s in seeds}
    base_time = statistics.fmean(r["time"] for r in base_results.values())

    print(f"{'bias':>6} {'time':>8} {'Δvs-100':>8} {'perfect':>9} {'miss':>6} {'req':>5} {'scan_meas':>10}")
    for b in variants:
        cfg = replace(base, task_bias_clear_m=float(b))
        results = [run(s, cfg) for s in seeds]
        t = statistics.fmean(r["time"] for r in results)
        perfect = sum(1 for r in results if r["cleared"] == r["total"])
        miss = statistics.fmean(r["miss"] for r in results)
        req = statistics.fmean(r["requests"] for r in results)
        sm = statistics.fmean(r["scan_meas"] for r in results)
        paired = [results[i]["time"] - base_results[s]["time"] for i, s in enumerate(seeds)]
        dp = statistics.fmean(paired)
        print(f"{b:>6} {t:>8.1f} {dp:>+8.1f} {perfect:>4}/{len(results):<4} {miss:>6.2f} {req:>5.1f} {sm:>10.1f}")

    print(f"\nbaseline(-100) time = {base_time:.1f} s")


if __name__ == "__main__":
    main()
