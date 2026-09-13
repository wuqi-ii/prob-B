"""调度 bias 联合扫描：验证「提前清除/提前 verify」在 margin=-10 新框架下是否更优。

背景（drill24 profile）：move 占 72.7%（位移 26891m）。scan 的 TSP 移动是覆盖下界
动不了，可优化的是 clear 折返（~1300s）+ locate 折返（~490s）+ 对已定位源的反复测量。

根因假设：task_bias_clear_m=-100、task_bias_verify_m=-700 把清除/验证延后到扫描快
结束，导致 (a) 单观测频道在后续每个扫描点被反复 measure（5s+1s switch/次）；
(b) 扫描结束后从最后扫描点折返回各源集中清除（大位移）。提前清除/verify 应能同时
降低 measure 次数与折返位移，代价是可能打断扫描路径——需要数据权衡。

扫描变量：
  task_bias_clear_m  : -100(当前) / 0 / +100 / +200
  task_bias_verify_m : -700(当前) / -300 / -100 / 0
输出额外含 move_m（总位移）、measure 次数，用于精确定位节省来源。
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
    # 从 trace 累计位移与 measure 次数
    move_m = 0.0
    n_measure = 0
    n_clear = 0
    px = py = 0.0
    for rec in sim.trace:
        kind = rec.get("kind")
        if kind == "measure":
            n_measure += 1
            move_m += math.hypot(rec["x"] - px, rec["y"] - py)
            px, py = rec["x"], rec["y"]
        elif kind == "clear":
            n_clear += 1
            move_m += math.hypot(rec["x"] - px, rec["y"] - py)
            px, py = rec["x"], rec["y"]
    return {
        "time": sim.virtual_time_s,
        "per_source": sim.virtual_time_s / sim.case.total,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "bisect": sum(e.get("event") == "bisect" for e in st.events),
        "max_src": max(per) if per else 0.0,
        "move_m": move_m,
        "n_measure": n_measure,
        "n_clear": n_clear,
    }


def summary(rows):
    n = len(rows)
    ps = [r["per_source"] for r in rows]
    mx = [r["max_src"] for r in rows]
    perfect = sum(r["cleared"] == r["total"] for r in rows)
    return (f"mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
            f"per-src {statistics.fmean(ps):6.1f}s | "
            f"max_src {statistics.fmean(mx):6.1f}s | "
            f"perfect {perfect:4d}/{n} | "
            f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
            f"move {statistics.fmean(r['move_m'] for r in rows):7.0f}m | "
            f"meas {statistics.fmean(r['n_measure'] for r in rows):5.1f} | "
            f"clear {statistics.fmean(r['n_clear'] for r in rows):4.1f} | "
            f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--dir-prob", type=float, default=0.5)
    args = ap.parse_args()

    base_cfg = StrategyConfig()
    seeds = range(args.start_seed, args.start_seed + args.cases)

    variants = [("c-100_v-700", base_cfg)]  # 当前默认（基线）
    # 提前清除（bias_clear 调正/调零）
    for bc in (0.0, 100.0, 200.0):
        variants.append((f"c{int(bc)}_v-700", replace(base_cfg, task_bias_clear_m=bc)))
    # 提前 verify（bias_verify 调正/调零）
    for bv in (-300.0, -100.0, 0.0):
        variants.append((f"c-100_v{int(bv)}", replace(base_cfg, task_bias_verify_m=bv)))
    # 组合：两者都提前
    variants.append(("c0_v-100", replace(base_cfg, task_bias_clear_m=0.0, task_bias_verify_m=-100.0)))
    variants.append(("c100_v0", replace(base_cfg, task_bias_clear_m=100.0, task_bias_verify_m=0.0)))

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.dir_prob}")
    results = {}
    for name, cfg in variants:
        rows = [run(s, cfg, args.dir_prob) for s in seeds]
        results[name] = rows
        print(f"{name:14s}: {summary(rows)}")

    base = results["c-100_v-700"]
    bmean = statistics.fmean(r["time"] for r in base)
    print("\n--- 配对差异（vs 当前默认 c-100_v-700）---")
    for name, _ in variants[1:]:
        rows = results[name]
        delta = [r["time"] - b["time"] for r, b in zip(rows, base)]
        se = statistics.pstdev(delta) / math.sqrt(len(delta))
        print(f"{name:14s}: paired {statistics.fmean(delta):+8.1f}s "
              f"({statistics.fmean(delta)/bmean:+.2%}, se {se:.1f}) | "
              f"miss {statistics.fmean(r['miss']-b['miss'] for r,b in zip(rows,base)):+.2f} | "
              f"move {statistics.fmean(r['move_m']-b['move_m'] for r,b in zip(rows,base)):+7.0f}m | "
              f"meas {statistics.fmean(r['n_measure']-b['n_measure'] for r,b in zip(rows,base)):+.1f}")


if __name__ == "__main__":
    main()
