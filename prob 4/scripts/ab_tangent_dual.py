"""切线双站定位 A/B v2：在 _approach_and_clear 内，定位不准时用「切线走位 + 交点定位
+ 扩大环形兜底」替代「径向逼近 + 二分」。

v1 只替换单观测 verify 路径，几乎不触发（单观测频道 0.37 次/局）。v2 把切线双站
放进所有 clear 任务的逼近入口，覆盖全部定位不准的源。
"""
import sys, math, statistics, argparse
from dataclasses import dataclass
sys.path.insert(0, "src")

from cumcm_b4.backend_offline import OfflineSimulator, generate_case
from cumcm_b4.config import StrategyConfig
from cumcm_b4.strategy import DogStrategy

Point = tuple[float, float]


def _intersect(p1: Point, b1: float, p2: Point, b2: float):
    x1, y1 = p1; x2, y2 = p2
    a1 = math.radians(b1); a2 = math.radians(b2)
    d1 = (math.cos(a1), math.sin(a1)); d2 = (math.cos(a2), math.sin(a2))
    dx = x2 - x1; dy = y2 - y1
    det = d1[0] * (-d2[1]) - (-d2[0]) * d1[1]
    if abs(det) < 1e-9:
        return None
    t1 = (dx * (-d2[1]) - dy * (-d2[0])) / det
    return (x1 + t1 * d1[0], y1 + t1 * d1[1])


@dataclass
class TangentSpec:
    lateral_m: float = 160.0
    ring_radius_m: float = 40.0
    min_worst_m: float = 20.0   # worst 超过此值才触发切线双站


class TangentDualStrategy(DogStrategy):
    def __init__(self, *args, spec: TangentSpec, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = spec

    def _ring_clear_radiused(self, center: Point, ch: int, radius: float) -> bool:
        if self._try_clear(center, ch):
            return True
        n = self.cfg.ring_clear_count
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            p = (center[0] + radius * math.cos(ang),
                 center[1] + radius * math.sin(ang))
            if self._try_clear(p, ch):
                return True
        return False

    def _approach_and_clear(self, ch: int) -> bool:
        track = self.book[ch]
        if track.near_points:
            return self._try_clear(track.near_points[-1], ch)

        # 切线双站：定位不准（worst 大）且有 bearing 观测时，先做切线走位 + 交点定位
        if track.observations:
            _, worst = self._clear_anchor(ch)
            if worst > self.spec.min_worst_m:
                last_pt, last_bearing = track.observations[-1]
                L = self.spec.lateral_m
                rad = math.radians(last_bearing)
                normal = (-math.sin(rad), math.cos(rad))
                p2a = (last_pt[0] + L * normal[0], last_pt[1] + L * normal[1])
                p2b = (last_pt[0] - L * normal[0], last_pt[1] - L * normal[1])
                # 两个切线方向都试（从近到远），测到 direction 就交点定位
                order = sorted((p2a, p2b), key=lambda p: math.dist(self._pos(), p))
                for pt in order:
                    payload = self.backend.measure(pt[0], pt[1], ch)
                    kind = self.book.apply_measure(ch, pt, payload)
                    self._log("tangent", channel=ch,
                              at=[round(pt[0], 1), round(pt[1], 1)], result=kind)
                    if kind == "near":
                        return self._try_clear(pt, ch)
                    if kind == "direction":
                        loc = _intersect(last_pt, last_bearing, pt, payload.get("svd_deg"))
                        if loc is not None:
                            if self._ring_clear_radiused(loc, ch, self.spec.ring_radius_m):
                                return True
                            break  # 交点定位失败，退回默认逼近
                # no_signal（两方向都出楔）或交点失败：退回默认
        return super()._approach_and_clear(ch)


def run(seed, cfg, directional_prob, spec):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    st = (DogStrategy(sim, cfg).run() if spec is None
          else TangentDualStrategy(sim, cfg, spec=spec).run())
    return {
        "time": sim.virtual_time_s,
        "per_source": sim.virtual_time_s / sim.case.total,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "bisect": sum(e.get("event") == "bisect" for e in st.events),
        "tangent": sum(e.get("event") == "tangent" for e in st.events),
    }


def summary(rows):
    ps = [r["per_source"] for r in rows]
    return (f"mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
            f"per-src min/mean/max {min(ps):6.1f}/{statistics.fmean(ps):6.1f}/{max(ps):6.1f}s | "
            f"perfect {sum(r['cleared']==r['total'] for r in rows):4d}/{len(rows)} | "
            f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
            f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f} | "
            f"req {statistics.fmean(r['req'] for r in rows):6.1f} | "
            f"tangent {statistics.fmean(r['tangent'] for r in rows):5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    cfg = StrategyConfig()
    seeds = range(args.start_seed, args.start_seed + args.cases)
    variants = [("baseline", None)]
    for L in (120.0, 160.0, 200.0):
        for rr in (30.0, 40.0, 50.0):
            variants.append((f"L{int(L)}_r{int(rr)}", TangentSpec(L, rr)))

    print(f"cases={args.cases} start_seed={args.start_seed} dir_prob={args.directional_prob}")
    results = {}
    for name, spec in variants:
        rows = [run(s, cfg, args.directional_prob, spec) for s in seeds]
        results[name] = rows
        print(f"{name:12s}: {summary(rows)}")

    base = results["baseline"]
    bmean = statistics.fmean(r["time"] for r in base)
    for name, _ in variants[1:]:
        rows = results[name]
        delta = [r["time"] - b["time"] for r, b in zip(rows, base)]
        se = statistics.pstdev(delta) / math.sqrt(len(delta))
        print(f"{name:12s}: paired {statistics.fmean(delta):+8.1f}s "
              f"({statistics.fmean(delta)/bmean:+.2%}, se {se:.1f})")


if __name__ == "__main__":
    main()
