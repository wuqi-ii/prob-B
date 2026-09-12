"""19点高发现率前缀 + 4点安全补全的配对实验。

安全性：只有已经成功清除16个不同频道时才跳过补全点；题面规定总源数不超过16，
因此此时不存在未发现源。其余案例一律继续完成全部23点。
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa:E402
from cumcm_b4.config import SOURCE_COUNT_MAX, StrategyConfig  # noqa:E402
from cumcm_b4.strategy import DogStrategy  # noqa:E402


class ProgressivePrefixStrategy(DogStrategy):
    def __init__(self, *args, supplement, **kwargs):
        super().__init__(*args, **kwargs)
        self.supplement = frozenset(supplement)
        self.core = frozenset(range(len(self.scan_points))) - self.supplement

    def _early_complete(self):
        return self.book.count_cleared() >= SOURCE_COUNT_MAX

    def select_task(self):
        if self._early_complete():
            return None
        return super().select_task()

    def _priority(self, task, cur):
        if task.kind == "scan":
            idx = self.scan_points.index(task.point)
            core_remaining = any(not self.visited_scan[i] for i in self.core)
            if core_remaining and idx in self.supplement:
                return math.inf
        return super()._priority(task, cur)

    def _no_task_stop_reason(self):
        return "all_done" if self._early_complete() else super()._no_task_stop_reason()

    def status(self):
        out = super().status()
        if self._early_complete():
            out["all_clear"] = True
            out["early_complete_at_max_sources"] = True
        return out


class EarlyMaxStrategy(DogStrategy):
    """不约束扫描顺序；清除数达到题面上限16时安全早停。"""
    def _early_complete(self):
        return self.book.count_cleared() >= SOURCE_COUNT_MAX

    def select_task(self):
        return None if self._early_complete() else super().select_task()

    def _no_task_stop_reason(self):
        return "all_done" if self._early_complete() else super()._no_task_stop_reason()

    def status(self):
        out = super().status()
        if self._early_complete():
            out["all_clear"] = True
            out["early_complete_at_max_sources"] = True
        return out


class AdaptivePrefixStrategy(ProgressivePrefixStrategy):
    """仅在已发现频道很多、很可能是16源局时才强制完成高发现率核心。"""
    def __init__(self, *args, gate_count, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_count = gate_count

    def _discovered_count(self):
        return self.book.count_cleared() + len(self.book.known_channels())

    def _priority(self, task, cur):
        if task.kind == "scan" and self._discovered_count() >= self.gate_count:
            idx = self.scan_points.index(task.point)
            if any(not self.visited_scan[i] for i in self.core) and idx in self.supplement:
                return math.inf
        return DogStrategy._priority(self, task, cur)


def run(seed, cfg, directional_prob, supplement):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    if isinstance(supplement, tuple) and supplement and supplement[0] == "adaptive":
        st = AdaptivePrefixStrategy(
            sim, cfg, supplement=supplement[1], gate_count=supplement[2]
        ).run()
    elif supplement == "early_only":
        st = EarlyMaxStrategy(sim, cfg).run()
    elif supplement is None:
        st = DogStrategy(sim, cfg).run()
    else:
        st = ProgressivePrefixStrategy(sim, cfg, supplement=supplement).run()
    return {
        "seed": seed, "n": sim.case.total, "time": sim.virtual_time_s,
        "per_source": sim.virtual_time_s / sim.case.total,
        "cleared": st.cleared_count, "visited": st.scan_points_visited,
        "perfect": st.cleared_count == sim.case.total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--start-seed", type=int, default=14000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    ap.add_argument("--only", default="", help="逗号分隔的变体名；空表示全部")
    args = ap.parse_args()
    cfg = StrategyConfig()
    variants = [
        ("baseline", None),
        ("early_only", "early_only"),
        ("adaptive_even_g12", ("adaptive", (2, 4, 6, 8), 12)),
        ("adaptive_even_g14", ("adaptive", (2, 4, 6, 8), 14)),
        ("adaptive_12322_g12", ("adaptive", (1, 2, 3, 22), 12)),
        ("adaptive_12322_g14", ("adaptive", (1, 2, 3, 22), 14)),
        ("tail_17_20", (17, 18, 19, 20)),
        ("detect_123_22", (1, 2, 3, 22)),
        ("detect_128_20", (1, 2, 8, 20)),
    ]
    if args.only:
        wanted = set(args.only.split(",")) | {"baseline"}
        variants = [v for v in variants if v[0] in wanted]
    results = {}
    for name, supplement in variants:
        rows = [run(s, cfg, args.directional_prob, supplement)
                for s in range(args.start_seed, args.start_seed + args.cases)]
        results[name] = rows
        ps = [r["per_source"] for r in rows]
        n16 = [r for r in rows if r["n"] == 16]
        print(
            f"{name:15s} perfect={sum(r['perfect'] for r in rows)}/{len(rows)} "
            f"time={statistics.fmean(r['time'] for r in rows):.1f} "
            f"per-source={min(ps):.1f}/{statistics.fmean(ps):.1f}/{max(ps):.1f} "
            f"visited={statistics.fmean(r['visited'] for r in rows):.2f} "
            f"n16_early={sum(r['visited'] < 23 for r in n16)}/{len(n16)}",
            flush=True,
        )
    base = results["baseline"]
    bm = statistics.fmean(r["time"] for r in base)
    for name, _ in variants[1:]:
        diffs = [r["time"] - b["time"] for r, b in zip(results[name], base)]
        d16 = [r["time"] - b["time"] for r, b in zip(results[name], base) if b["n"] == 16]
        print(f"{name:15s} paired={statistics.fmean(diffs):+.1f}s "
              f"({statistics.fmean(diffs)/bm:+.2%}, se={statistics.pstdev(diffs)/math.sqrt(len(diffs)):.1f}) "
              f"n16_delta={statistics.fmean(d16):+.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
