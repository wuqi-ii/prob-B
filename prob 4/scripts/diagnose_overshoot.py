"""诊断越界类型：径向越过源 vs 横向擦出楔。

用真值（离线模拟器可访问）计算每次越界（_resolve_overshoot 入口）时源到
lo→hi 线段的最短距离。距离小（<=20m，清除半径）说明二分能夹到源（径向越界）；
距离大说明这条线段根本不经过源附近，二分 20 次基本白费（横向擦出楔边缘）。
"""
import argparse, statistics, sys, math, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case
from cumcm_b4.config import StrategyConfig
from cumcm_b4.strategy import DogStrategy


def point_seg_dist(p, a, b):
    px, py = p; ax, ay = a; bx, by = b
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / L2
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * vx, ay + t * vy
    return math.hypot(px - qx, py - qy)


class Probe(DogStrategy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.overshoots = []

    def _resolve_overshoot(self, ch, lo_pt, hi_pt):
        src = next((s for s in self.backend.case.sources if s.channel == ch), None)
        seg_dist = point_seg_dist((src.x, src.y), lo_pt, hi_pt) if src else -1.0
        # lo→hi 方向 vs 最后 bearing 的夹角（bearing 从 lo 指向源）
        ang = None
        bearing = None
        for pt, b in reversed(self.book[ch].observations):
            if math.dist(pt, lo_pt) < 1e-6:
                bearing = b
                break
        if bearing is not None:
            lo_hi_deg = math.degrees(math.atan2(hi_pt[1] - lo_pt[1], hi_pt[0] - lo_pt[0])) % 360.0
            d = (lo_hi_deg - bearing) % 360.0
            ang = min(d, 360.0 - d)
        n_bisect_before = sum(1 for e in self.stats.events if e.get("event") == "bisect")
        ok = super()._resolve_overshoot(ch, lo_pt, hi_pt)
        n_bisect_after = sum(1 for e in self.stats.events if e.get("event") == "bisect")
        self.overshoots.append({
            "seg_dist": seg_dist,
            "ok": ok,
            "directional": bool(src.directional) if src else False,
            "bisect_iter": n_bisect_after - n_bisect_before,
            "angle": ang,
        })
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    cfg = StrategyConfig()
    all_rows = []
    for seed in range(args.start_seed, args.start_seed + args.cases):
        sim = OfflineSimulator(generate_case(seed, directional_prob=args.directional_prob))
        probe = Probe(sim, cfg)
        probe.run()
        all_rows.extend(probe.overshoots)

    n = len(all_rows)
    print(f"cases={args.cases} 越界总次数={n}")
    if not n:
        return

    # 按 seg_dist 分桶
    buckets = [(0, 20, "<=20m 夹得到源"), (20, 50, "20-50m"),
               (50, 150, "50-150m"), (150, 1e9, ">150m 纯白费")]
    for lo, hi, label in buckets:
        rows = [r for r in all_rows if lo <= r["seg_dist"] < hi]
        if not rows:
            print(f"  {label:16s}: 0 次")
            continue
        ok = sum(r["ok"] for r in rows)
        bis = statistics.fmean(r["bisect_iter"] for r in rows)
        print(f"  {label:16s}: {len(rows):4d} 次 ({len(rows)/n*100:4.0f}%) | "
              f"二分成功 {ok:3d}/{len(rows)} ({ok/len(rows)*100:3.0f}%) | "
              f"平均二分迭代 {bis:5.1f}")

    # 定向 vs 全向的 seg_dist
    for is_dir, label in ((False, "全向源越界"), (True, "定向源越界")):
        rows = [r for r in all_rows if r["directional"] == is_dir]
        if not rows:
            print(f"  {label}: 0 次")
            continue
        sd = [r["seg_dist"] for r in rows]
        ok = sum(r["ok"] for r in rows)
        print(f"  {label}: {len(rows)} 次 | seg_dist 中位 {statistics.median(sd):.1f}m "
              f"平均 {statistics.fmean(sd):.1f}m | 二分成功 {ok}/{len(rows)}")

    # 关键：seg_dist > 20m 的越界里，二分"成功"的其实是 ring_clear 兜底命中，还是真找到源？
    far = [r for r in all_rows if r["seg_dist"] > 20]
    if far:
        print(f"\n关键结论：{len(far)}/{n} ({len(far)/n*100:.0f}%) 的越界线段不经过源附近"
              f"（seg_dist>20m），这些二分的测量点几乎全部落在无信号区。")
        print(f"  其中二分平均迭代 {statistics.fmean(r['bisect_iter'] for r in far):.1f} 次"
              f"（约等于触顶 20 次），是纯白费。")

    # 判断信号：lo→hi 方向 vs 最后 bearing 夹角，能否区分 seg_dist 大小
    print("\n=== 判断信号：|lo→hi 方向 - bearing| 夹角 vs seg_dist ===")
    ang_buckets = [(0, 15, "0-15°"), (15, 30, "15-30°"), (30, 60, "30-60°"),
                   (60, 181, "60-180°")]
    for lo, hi, label in ang_buckets:
        rows = [r for r in all_rows if r["angle"] is not None and lo <= r["angle"] < hi]
        if not rows:
            print(f"  {label:10s}: 0 次")
            continue
        sd = [r["seg_dist"] for r in rows]
        ok = sum(r["ok"] for r in rows)
        n_small = sum(1 for d in sd if d <= 20)
        print(f"  {label:10s}: {len(rows):4d} 次 | seg_dist 中位 {statistics.median(sd):5.1f}m "
              f"| seg_dist<=20m 占比 {n_small}/{len(rows)} ({n_small/len(rows)*100:3.0f}%) "
              f"| 二分成功 {ok}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
