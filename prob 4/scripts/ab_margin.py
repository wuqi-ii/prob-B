"""clear_safety_margin + angle_gate 阈值 A/B：攻 ch5 这类「定位准但被过度保守送去逼近」的长尾。

背景（drill23 ch5）：4 个 direction 观测 MEC 圆心距源仅 11.2m、MEC 半径 19.2m，
但判据 `mec_radius <= CLEAR_RADIUS(20) - margin(2) = 18m` 不满足 → 被送去逼近+二分，
二分又因越界夹角 12.8°<15° 未触发 angle_gate，白做 7 次，绕 ~400s。

两个独立变量：
  clear_safety_margin_m: 2 -> 0（更激进直接 clear）甚至负值（赌源在圆心附近）
  angle_gate_threshold_deg: 15 -> 10 / 5（更激进跳过横向越界二分）
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

    variants = [("margin2_gate15", base_cfg)]  # 当前默认（基线）
    for m in (0.0, -5.0, -10.0, -15.0, -20.0):
        variants.append((f"margin{int(m)}", replace(base_cfg, clear_safety_margin_m=m)))
    for th in (10.0, 5.0):
        variants.append((f"gate{int(th)}",
                         replace(base_cfg, angle_gate_threshold_deg=th)))
    # 组合：margin-10 + gate5（两个最优方向叠加）
    variants.append(("margin-10_gate5",
                     replace(base_cfg, clear_safety_margin_m=-10.0,
                             angle_gate_threshold_deg=5.0)))

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.dir_prob}")
    results = {}
    for name, cfg in variants:
        rows = [run(s, cfg, args.dir_prob) for s in seeds]
        results[name] = rows
        print(f"{name:16s}: {summary(rows)}")

    base = results["margin2_gate15"]
    bmean = statistics.fmean(r["time"] for r in base)
    print("\n--- 配对差异（vs 当前默认 margin2_gate15）---")
    for name, _ in variants[1:]:
        rows = results[name]
        delta = [r["time"] - b["time"] for r, b in zip(rows, base)]
        se = statistics.pstdev(delta) / math.sqrt(len(delta))
        print(f"{name:16s}: paired {statistics.fmean(delta):+8.1f}s "
              f"({statistics.fmean(delta)/bmean:+.2%}, se {se:.1f}) | "
              f"miss {statistics.fmean(r['miss']-b['miss'] for r,b in zip(rows,base)):+.2f} | "
              f"bisect {statistics.fmean(r['bisect']-b['bisect'] for r,b in zip(rows,base)):+.1f}")


if __name__ == "__main__":
    main()
