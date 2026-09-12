"""精细时间画像：把虚拟时间按「策略阶段」和「动作类型」双重分解。

用途：回答"时间到底花在哪"，从而判断下一步该优化什么。

阶段划分（按策略代码路径，而非结果类型）：
  scan         包围网扫描点上的逐频道普查
  verify       单示向度频道的"同向推进"补第二观测点
  approach     质心逼近循环中的推进检测
  bisect       越过源之后的二分恢复
  ring         光学环形兜底（/clear 七点）
  clear        单次 /clear 调用（含中心点）
  postverify   清除成功后的就地复测
  opportunistic 顺路捎带检测

用法：
    python scripts/profile_time.py --cases 200 --start-seed 7000
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import (  # noqa: E402
    CLEAR_HIT_COST_S,
    CLEAR_MISS_COST_S,
    MEASURE_COST_S,
    StrategyConfig,
)
from cumcm_b4.strategy import DogStrategy  # noqa: E402

PHASES = (
    "scan",
    "verify",
    "approach",
    "bisect",
    "ring",
    "clear",
    "postverify",
    "opportunistic",
)


class ProfilingStrategy(DogStrategy):
    """在策略各代码路径入口打上阶段标签。"""

    def __init__(self, backend, cfg, verbose: bool = False):
        super().__init__(backend, cfg, verbose)
        self._phase = "scan"

    # ---- 打点 ----
    def _visit_scan_point(self, idx: int) -> None:
        self._phase = "scan"
        return super()._visit_scan_point(idx)

    def _do_verify(self, task) -> None:
        self._phase = "verify"
        return super()._do_verify(task)

    def _do_clear(self, task) -> None:
        self._phase = "clear"
        return super()._do_clear(task)

    def _approach_and_clear(self, ch: int) -> bool:
        self._phase = "approach"
        return super()._approach_and_clear(ch)

    def _resolve_overshoot(self, ch: int, lo, hi) -> bool:
        prev, self._phase = self._phase, "bisect"
        try:
            return super()._resolve_overshoot(ch, lo, hi)
        finally:
            self._phase = prev

    def _ring_clear(self, center, ch: int) -> bool:
        prev, self._phase = self._phase, "ring"
        try:
            return super()._ring_clear(center, ch)
        finally:
            self._phase = prev

    def _try_clear(self, p, ch: int) -> bool:
        prev, self._phase = self._phase, "clear"
        try:
            return super()._try_clear(p, ch)
        finally:
            self._phase = prev

    def _finalize(self, ch: int, success: bool, t0: float) -> None:
        prev, self._phase = self._phase, "postverify"
        try:
            return super()._finalize(ch, success, t0)
        finally:
            self._phase = prev

    def _opportunistic_pass(self, point, exclude=None) -> None:
        prev, self._phase = self._phase, "opportunistic"
        try:
            return super()._opportunistic_pass(point, exclude)
        finally:
            self._phase = prev


class PhaseTrackingBackend:
    """包一层后端，把每次动作按当前阶段的移动/检测/切换/清除成本记账。"""

    def __init__(self, sim: OfflineSimulator):
        self._sim = sim
        self.strategy = None  # 由外部注入，用于读 _phase
        self.buckets = {
            p: {"calls": 0, "measures": 0, "clears": 0, "move_m": 0.0,
                "move_s": 0.0, "measure_s": 0.0, "switch_s": 0.0,
                "clear_s": 0.0, "hits": 0}
            for p in PHASES
        }
        self.scan_redundant_measures = 0
        self.scan_measures = 0

    def _phase(self) -> str:
        return getattr(self.strategy, "_phase", "scan")

    def measure(self, x, y, channel):
        phase = self._phase()
        before_time = self._sim.virtual_time_s
        before_pos = self._sim.position
        res = self._sim.measure(x, y, channel)
        b = self.buckets[phase]
        b["calls"] += 1
        b["measures"] += 1
        d = math.hypot(x - before_pos[0], y - before_pos[1])
        b["move_m"] += d
        b["move_s"] += d / 5.0
        b["measure_s"] += MEASURE_COST_S
        b["switch_s"] += 1.0 if x == before_pos[0] and y == before_pos[1] and False else 0.0
        # 切换代价 = 总耗时 - 移动 - 检测
        b["switch_s"] += (self._sim.virtual_time_s - before_time) - d / 5.0 - MEASURE_COST_S
        if phase == "scan":
            self.scan_measures += 1
            track = self.strategy.book[channel] if self.strategy else None
            # 该频道在本次扫描前已经 known 且已有 >=2 个观测 => 本次普查大概率冗余
            if track is not None and self._known_before.get(channel):
                self.scan_redundant_measures += 1
        return res

    def clear(self, x, y, channel):
        phase = self._phase()
        before_time = self._sim.virtual_time_s
        before_pos = self._sim.position
        res = self._sim.clear(x, y, channel)
        b = self.buckets[phase]
        b["calls"] += 1
        b["clears"] += 1
        d = math.hypot(x - before_pos[0], y - before_pos[1])
        b["move_m"] += d
        b["move_s"] += d / 5.0
        hit = res.get("clear_result") == "success"
        b["hits"] += int(hit)
        b["clear_s"] += CLEAR_HIT_COST_S if hit else CLEAR_MISS_COST_S
        return res

    def __getattr__(self, name):
        return getattr(self._sim, name)

    # 供 measure 内部判断"扫描前该频道是否已 known"
    _known_before: dict = {}


def run_case(seed: int, cfg: StrategyConfig, directional_prob: float = 0.5):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    backend = PhaseTrackingBackend(sim)
    strategy = ProfilingStrategy(backend, cfg)
    backend.strategy = strategy
    # 在每个扫描点开始前记录"已 known 且已有 >=2 观测"的频道快照
    original_visit = strategy._visit_scan_point

    def visit(idx):
        backend._known_before = {
            ch: (t.known and len(t.observations) >= 2)
            for ch, t in strategy.book.tracks.items()
            if not t.cleared
        }
        return original_visit(idx)

    strategy._visit_scan_point = visit  # type: ignore[method-assign]

    stats = strategy.run()
    return {
        "seed": seed,
        "total_time_s": stats.virtual_time_s,
        "cleared": stats.cleared_count,
        "total": sim.case.total,
        "requests": stats.requests,
        "buckets": backend.buckets,
        "scan_measures": backend.scan_measures,
        "scan_redundant": backend.scan_redundant_measures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--start-seed", type=int, default=7000)
    parser.add_argument("--directional-prob", type=float, default=0.5)
    parser.add_argument("--set", action="append", default=[],
                        help="覆写配置项，形如 --set task_bias_verify_m=-700")
    args = parser.parse_args()

    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key.strip()] = float(value)
    cfg = replace(StrategyConfig(), **overrides)
    rows = [
        run_case(seed, cfg, args.directional_prob)
        for seed in range(args.start_seed, args.start_seed + args.cases)
    ]

    perfect = sum(1 for r in rows if r["cleared"] == r["total"])
    mean_total = statistics.fmean(r["total_time_s"] for r in rows)
    print(f"cases={len(rows)}  perfect={perfect}/{len(rows)}  mean_total={mean_total:.1f} s")
    print(f"mean_requests={statistics.fmean(r['requests'] for r in rows):.1f}")
    print()
    header = (f"{'phase':14s} {'calls':>7s} {'meas':>6s} {'move_m':>9s} {'move_s':>9s} "
              f"{'meas_s':>8s} {'switch_s':>9s} {'clear_s':>8s} {'phase_s':>9s} {'share':>7s}")
    print(header)
    print("-" * len(header))
    agg = {p: {k: 0.0 for k in ("calls", "measures", "clears", "move_m", "move_s",
                                "measure_s", "switch_s", "clear_s", "hits")}
           for p in PHASES}
    for r in rows:
        for p in PHASES:
            for k, v in r["buckets"][p].items():
                agg[p][k] += v
    n = len(rows)
    for p in PHASES:
        a = agg[p]
        phase_s = a["move_s"] + a["measure_s"] + a["switch_s"] + a["clear_s"]
        print(f"{p:14s} {a['calls']/n:7.1f} {a['measures']/n:6.1f} {a['move_m']/n:9.1f} "
              f"{a['move_s']/n:9.1f} {a['measure_s']/n:8.1f} {a['switch_s']/n:9.1f} "
              f"{a['clear_s']/n:8.1f} {phase_s/n:9.1f} {phase_s/n/mean_total:7.2%}")
    tot = sum(agg[p]["move_s"] + agg[p]["measure_s"] + agg[p]["switch_s"] + agg[p]["clear_s"]
              for p in PHASES) / n
    print("-" * len(header))
    print(f"{'TOTAL':14s} {'':>7s} {'':>6s} "
          f"{sum(agg[p]['move_m'] for p in PHASES)/n:9.1f} "
          f"{sum(agg[p]['move_s'] for p in PHASES)/n:9.1f} "
          f"{sum(agg[p]['measure_s'] for p in PHASES)/n:8.1f} "
          f"{sum(agg[p]['switch_s'] for p in PHASES)/n:9.1f} "
          f"{sum(agg[p]['clear_s'] for p in PHASES)/n:8.1f} {tot:9.1f}")
    print()
    scan_measures = statistics.fmean(r["scan_measures"] for r in rows)
    redundant = statistics.fmean(r["scan_redundant"] for r in rows)
    print(f"扫描点测量次数/案例  = {scan_measures:.1f}  (占全部检测 {scan_measures/statistics.fmean(r['requests'] for r in rows):.1%})")
    print(f"其中扫前已 known 且已有>=2观测 = {redundant:.1f}  ({redundant/scan_measures:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
