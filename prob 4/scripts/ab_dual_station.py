"""主动双站定位 A/B：用横向基线增大两次示向观测的交会角。

基线的第二站位于首次观测射线前方 300 m。实验策略将第二站横向移动 L 米；
若该侧落到定向源覆盖楔外，可回到中心，或先尝试另一侧再回中心。所有变体只在
本脚本内以子类实现，不修改默认策略。

用法：
    python scripts/ab_dual_station.py --cases 100 --start-seed 20000 --directional-prob 1
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.second_station import approach_point  # noqa: E402
from cumcm_b4.strategy import DogStrategy, Task  # noqa: E402


Point = tuple[float, float]


@dataclass(frozen=True)
class DualSpec:
    lateral_m: float
    try_opposite: bool
    probe_before_clear: bool = True
    min_uncertainty_m: float = 80.0


class DualStationStrategy(DogStrategy):
    """把单示向度频道的第二站改为离射线横向 L 米的主动测站。"""

    def __init__(self, *args, dual_spec: DualSpec, **kwargs):
        super().__init__(*args, **kwargs)
        self.dual_spec = dual_spec
        self._dual_probed: set[int] = set()

    @staticmethod
    def _side_points(first: Point, bearing_deg: float, step_m: float,
                     lateral_m: float) -> tuple[Point, Point, Point]:
        center = approach_point(first, bearing_deg, step_m)
        rad = math.radians(bearing_deg)
        normal = (-math.sin(rad), math.cos(rad))
        plus = (center[0] + lateral_m * normal[0],
                center[1] + lateral_m * normal[1])
        minus = (center[0] - lateral_m * normal[0],
                 center[1] - lateral_m * normal[1])
        return center, plus, minus

    def select_task(self):
        task = super().select_task()
        if task is None or task.kind != "verify" or task.channel is None:
            return task
        track = self.book[task.channel]
        if len(track.observations) != 1:
            return task
        first, bearing = track.observations[0]
        _, plus, minus = self._side_points(
            first, bearing, self.cfg.first_approach_step_m,
            self.dual_spec.lateral_m,
        )
        # 先走离机器狗当前位置更近的一侧，减少主动造基线的额外路程。
        chosen = min((plus, minus), key=lambda p: math.dist(self._pos(), p))
        return Task("verify", chosen, task.channel, note="dual-lateral")

    def _measure_verify_point(self, ch: int, point: Point, label: str) -> str:
        payload = self.backend.measure(point[0], point[1], ch)
        kind = self.book.apply_measure(ch, point, payload)
        self._log(label, channel=ch,
                  at=[round(point[0], 1), round(point[1], 1)], result=kind)
        return kind

    def _finish_from_measure(self, ch: int, point: Point, kind: str,
                             first: Point, center: Point) -> bool:
        if kind == "direction":
            return self._approach_and_clear(ch)
        if kind == "near":
            return self._try_clear(point, ch)
        # 中心点 no_signal 才可沿首次射线使用原有二分；横向点 no_signal 不满足单调性。
        return self._resolve_overshoot(ch, first, center)

    def _do_verify(self, task: Task) -> None:
        assert task.channel is not None
        ch = task.channel
        track = self.book[ch]
        first, bearing = track.observations[0]
        center, plus, minus = self._side_points(
            first, bearing, self.cfg.first_approach_step_m,
            self.dual_spec.lateral_m,
        )
        first_side = task.point
        other_side = minus if math.dist(first_side, plus) < 1e-6 else plus

        t0 = self.backend.virtual_time_s
        self._clear_attempts[ch] = self._clear_attempts.get(ch, 0) + 1

        kind = self._measure_verify_point(ch, first_side, "dual_verify")
        if kind in ("direction", "near"):
            success = self._finish_from_measure(ch, first_side, kind, first, center)
        else:
            success = False
            if self.dual_spec.try_opposite:
                kind = self._measure_verify_point(ch, other_side, "dual_opposite")
                if kind in ("direction", "near"):
                    success = self._finish_from_measure(ch, other_side, kind, first, center)
            if not success:
                kind = self._measure_verify_point(ch, center, "dual_center_fallback")
                success = self._finish_from_measure(ch, center, kind, first, center)

        self._finalize(ch, success, t0)
        self._opportunistic_pass(self._pos(), exclude=ch)

    def _do_clear(self, task: Task) -> None:
        """高不确定度目标在常规逼近前，主动制造一条横向观测基线。"""
        assert task.channel is not None
        ch = task.channel
        track = self.book[ch]
        _, worst = self._clear_anchor(ch)
        should_probe = (
            self.dual_spec.probe_before_clear
            and ch not in self._dual_probed
            and not track.near_points
            and bool(track.observations)
            and worst > self.dual_spec.min_uncertainty_m
        )
        if not should_probe:
            return super()._do_clear(task)

        self._dual_probed.add(ch)
        first, bearing = track.observations[-1]
        center, plus, minus = self._side_points(
            first, bearing, self.cfg.first_approach_step_m,
            self.dual_spec.lateral_m,
        )
        first_side = min((plus, minus), key=lambda p: math.dist(self._pos(), p))
        other_side = minus if math.dist(first_side, plus) < 1e-6 else plus

        t0 = self.backend.virtual_time_s
        self._clear_attempts[ch] = self._clear_attempts.get(ch, 0) + 1
        kind = self._measure_verify_point(ch, first_side, "dual_preclear")
        success = False
        if kind == "near":
            success = self._try_clear(first_side, ch)
        elif kind == "direction":
            success = self._approach_and_clear(ch)
        else:
            if self.dual_spec.try_opposite:
                kind = self._measure_verify_point(ch, other_side, "dual_preclear_opposite")
                if kind == "near":
                    success = self._try_clear(other_side, ch)
                elif kind == "direction":
                    success = self._approach_and_clear(ch)
            if not success:
                kind = self._measure_verify_point(ch, center, "dual_preclear_center")
                if kind == "near":
                    success = self._try_clear(center, ch)
                elif kind == "direction":
                    success = self._approach_and_clear(ch)
                else:
                    success = self._resolve_overshoot(ch, first, center)

        self._finalize(ch, success, t0)
        self._opportunistic_pass(self._pos(), exclude=ch)


def run(seed: int, cfg: StrategyConfig, directional_prob: float,
        spec: DualSpec | None) -> dict:
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    if spec is None:
        st = DogStrategy(sim, cfg).run()
    else:
        st = DualStationStrategy(sim, cfg, dual_spec=spec).run()
    return {
        "time": sim.virtual_time_s,
        "per_source": sim.virtual_time_s / sim.case.total,
        "cleared": st.cleared_count,
        "total": sim.case.total,
        "miss": st.clear_miss,
        "req": st.requests,
        "bisect": sum(e.get("event") == "bisect" for e in st.events),
        "dual_no_signal": sum(
            e.get("event") in ("dual_verify", "dual_preclear")
            and e.get("result") == "no_signal"
            for e in st.events
        ),
        "dual_probes": sum(
            e.get("event") in ("dual_verify", "dual_preclear") for e in st.events
        ),
    }


def summary(rows: list[dict]) -> str:
    ps = [r["per_source"] for r in rows]
    return (
        f"mean {statistics.fmean(r['time'] for r in rows):8.1f}s | "
        f"per-source min/mean/max {min(ps):6.1f}/{statistics.fmean(ps):6.1f}/{max(ps):6.1f}s | "
        f"perfect {sum(r['cleared'] == r['total'] for r in rows):4d}/{len(rows)} | "
        f"miss {statistics.fmean(r['miss'] for r in rows):5.2f} | "
        f"bisect {statistics.fmean(r['bisect'] for r in rows):5.2f} | "
        f"req {statistics.fmean(r['req'] for r in rows):6.1f} | "
        f"dual {statistics.fmean(r['dual_probes'] for r in rows):5.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--start-seed", type=int, default=20000)
    ap.add_argument("--directional-prob", type=float, default=1.0)
    args = ap.parse_args()

    cfg = StrategyConfig()
    seeds = range(args.start_seed, args.start_seed + args.cases)
    variants: list[tuple[str, DualSpec | None]] = [("baseline", None)]
    for lateral in (50.0, 100.0, 200.0):
        for threshold in (20.0, 40.0, 80.0):
            variants.append(
                (f"L{int(lateral)}_u{int(threshold)}",
                 DualSpec(lateral, False, min_uncertainty_m=threshold))
            )

    print(f"cases={args.cases} start_seed={args.start_seed} directional_prob={args.directional_prob}")
    results: dict[str, list[dict]] = {}
    for name, spec in variants:
        rows = [run(seed, cfg, args.directional_prob, spec) for seed in seeds]
        results[name] = rows
        print(f"{name:14s}: {summary(rows)}")

    baseline = results["baseline"]
    base_mean = statistics.fmean(r["time"] for r in baseline)
    for name, _ in variants[1:]:
        rows = results[name]
        delta = [r["time"] - b["time"] for r, b in zip(rows, baseline)]
        n = len(delta)
        se = statistics.pstdev(delta) / math.sqrt(n)
        print(
            f"{name:14s}: paired {statistics.fmean(delta):+8.1f}s "
            f"({statistics.fmean(delta) / base_mean:+.2%}, se {se:.1f}), "
            f"lateral no-signal {sum(r['dual_no_signal'] for r in rows)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
