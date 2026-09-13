"""margin 安全边界精确锁定：ch17 案例（MEC 半径 32.8m 卡在 30m 判据）证明 -10 不够激进。

之前 700 例只验证了 -10 零漏检。这里用更大样本验证 -13/-15/-17 的零漏检稳健性，
以及与 angle_gate 阈值的组合（gate5 能拦截 ch17 那种 8.5° 横向出楔）。
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
        "max_src": max(per) if per else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=500)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--dir-prob", type=float, default=0.5)
    args = ap.parse_args()

    base = StrategyConfig()  # 当前 margin=-10, gate=15
    seeds = range(args.start_seed, args.start_seed + args.cases)

    variants = [
        ("m-10_g15", base),
        ("m-13_g15", replace(base, clear_safety_margin_m=-13.0)),
        ("m-15_g15", replace(base, clear_safety_margin_m=-15.0)),
        ("m-17_g15", replace(base, clear_safety_margin_m=-17.0)),
        ("m-15_g10", replace(base, clear_safety_margin_m=-15.0, angle_gate_threshold_deg=10.0)),
        ("m-15_g5", replace(base, clear_safety_margin_m=-15.0, angle_gate_threshold_deg=5.0)),
    ]

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.dir_prob}")
    results = {}
    for name, cfg in variants:
        rows = [run(s, cfg, args.dir_prob) for s in seeds]
        results[name] = rows
        perfect = sum(r["cleared"] == r["total"] for r in rows)
        print(f"{name:10s}: mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
              f"max_src {statistics.fmean(r['max_src'] for r in rows):6.1f}s | "
              f"perfect {perfect:4d}/{args.cases} | "
              f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
              f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f}")

    base_rows = results["m-10_g15"]
    bmean = statistics.fmean(r["time"] for r in base_rows)
    print("\n--- 配对差异（vs m-10_g15）---")
    for name, _ in variants[1:]:
        rows = results[name]
        delta = [r["time"] - b["time"] for r, b in zip(rows, base_rows)]
        se = statistics.pstdev(delta) / math.sqrt(len(delta))
        # 漏检数对比（全清率下降即致命）
        dperfect = sum(r["cleared"] != r["total"] for r in rows) - \
                   sum(r["cleared"] != r["total"] for r in base_rows)
        print(f"{name:10s}: paired {statistics.fmean(delta):+8.1f}s "
              f"({statistics.fmean(delta)/bmean:+.2%}, se {se:.1f}) | "
              f"漏检Δ {dperfect:+d} | missΔ {statistics.fmean(r['miss']-b['miss'] for r,b in zip(rows,base_rows)):+.2f}")


if __name__ == "__main__":
    main()
