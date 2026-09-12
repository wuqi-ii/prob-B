"""问题4第二阶段安全时间优化实验。

不减少23个覆盖扫描点，只改变任务优先级与可证明无信息量的检测：

1. dynamic clear bonus：提前清除一个频道，可省下后续扫描点对该频道的检测；
2. current-channel first：扫描一组频道时先测当前频道，减少一次切换；
3. certified skip：仅跳过已经可保证一次清除，或整个可行域距扫描点超过
   最大接收半径1500 m的已知频道。

用法：
    python scripts/optimize_time_v2.py --cases 300 --start-seed 14000
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import (  # noqa: E402
    CLEAR_RADIUS_M,
    DOG_SPEED_MPS,
    MEASURE_COST_S,
    RECEIVER_MAX_M,
    SWITCH_COST_S,
    StrategyConfig,
)
from cumcm_b4.strategy import DogStrategy  # noqa: E402


def _point_segment_distance(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return math.dist(point, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _point_in_convex(point, poly):
    if len(poly) < 3:
        return False
    sign = 0
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) <= 1e-9:
            continue
        now = 1 if cross > 0 else -1
        if sign and now != sign:
            return False
        sign = now
    return True


def _point_polygon_distance(point, poly):
    if not poly:
        return math.inf
    if len(poly) == 1:
        return math.dist(point, poly[0])
    if _point_in_convex(point, poly):
        return 0.0
    return min(
        _point_segment_distance(point, poly[i], poly[(i + 1) % len(poly)])
        for i in range(len(poly))
    )


class CurrentChannelFirst(DogStrategy):
    """同点批量检测时先复用当前频道；不改变检测集合。"""

    def _scan_channels(self):
        channels = [ch for ch in range(1, 21) if not self.book[ch].cleared]
        current = getattr(self.backend, "current_channel", None)
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        return channels

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = self._scan_channels()
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log(
            "scan_done",
            point=list(point),
            index=idx,
            channels=len(channels),
            found=[ch for ch in channels if self.book[ch].known],
        )


class DynamicClearBonus(DogStrategy):
    """按剩余扫描点数给清除任务奖励。

    一次后续检测的时间成本约为检测5 s + 切换1 s，换算成机器狗移动距离为
    6*5=30 m。factor=1 表示完整计入这项可避免成本。
    """

    bonus_factor = 1.0

    def __init__(self, backend, cfg, verbose=False, bonus_factor=None):
        super().__init__(backend, cfg, verbose)
        if bonus_factor is not None:
            self.bonus_factor = float(bonus_factor)

    def _priority(self, task, cur):
        value = super()._priority(task, cur)
        if task.kind == "clear":
            remaining = sum(not done for done in self.visited_scan)
            avoided_m = (MEASURE_COST_S + SWITCH_COST_S) * DOG_SPEED_MPS * remaining
            value -= self.bonus_factor * avoided_m
        return value


class SafeCombined(CurrentChannelFirst, DynamicClearBonus):
    """动态清除奖励 + 频道切换复用 + 严格可证明的无信息测量跳过。"""

    def _scan_channels_at(self, point):
        channels = []
        for ch in range(1, 21):
            track = self.book[ch]
            if track.cleared:
                continue
            blocked = self._blocked_at_obs.get(ch)
            needs_recovery = blocked is not None and len(track.observations) <= blocked
            if track.known and not needs_recovery:
                summary = track.geometry_summary(self.cfg)
                # 可行域完全位于清除半径内：该频道已无需更多示向度。
                if summary["centroid_max_error_m"] <= CLEAR_RADIUS_M - self.cfg.clear_safety_margin_m:
                    continue
                poly = track.polygon(self.cfg)
                # 真源必在poly内；若扫描点到整个poly的最短距离都大于1500 m，
                # 无论实际接收半径如何，本次一定是no_signal。
                if poly and _point_polygon_distance(point, poly) > RECEIVER_MAX_M:
                    continue
            channels.append(ch)
        current = getattr(self.backend, "current_channel", None)
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        return channels

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = self._scan_channels_at(point)
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log(
            "scan_done",
            point=list(point),
            index=idx,
            channels=len(channels),
            found=[ch for ch in channels if self.book[ch].known],
        )


def _run(seed, cls, factor=None, directional_prob=0.5):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    # 实验基线显式关闭已经合入主干的两个开关，保证历史A/B仍可复现。
    cfg = StrategyConfig(certified_scan_skip=False, scan_current_channel_first=False)
    if factor is None:
        strategy = cls(sim, cfg)
    else:
        strategy = cls(sim, cfg, bonus_factor=factor)
    stats = strategy.run()
    return sim.virtual_time_s, stats.cleared_count, sim.case.total, stats.requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--start-seed", type=int, default=14000)
    parser.add_argument("--directional-prob", type=float, default=0.5)
    parser.add_argument("--refine", action="store_true",
                        help="只运行基线与组合方案的细网格权重")
    parser.add_argument("--stress", action="store_true",
                        help="只运行基线与纯认证跳过方案")
    args = parser.parse_args()

    variants = [
        ("baseline", DogStrategy, None),
        ("channel_first", CurrentChannelFirst, None),
        ("clear_f025", DynamicClearBonus, 0.25),
        ("clear_f050", DynamicClearBonus, 0.50),
        ("clear_f075", DynamicClearBonus, 0.75),
        ("clear_f100", DynamicClearBonus, 1.00),
        ("combined_f000", SafeCombined, 0.00),
        ("combined_f010", SafeCombined, 0.10),
        ("combined_f020", SafeCombined, 0.20),
        ("combined_f025", SafeCombined, 0.25),
        ("combined_f030", SafeCombined, 0.30),
        ("combined_f050", SafeCombined, 0.50),
        ("combined_f075", SafeCombined, 0.75),
    ]
    if args.refine:
        keep = {"baseline", "channel_first", "combined_f000", "combined_f010",
                "combined_f020", "combined_f025", "combined_f030"}
        variants = [v for v in variants if v[0] in keep]
    if args.stress:
        keep = {"baseline", "combined_f000"}
        variants = [v for v in variants if v[0] in keep]
    results = {}
    for name, cls, factor in variants:
        results[name] = [
            _run(seed, cls, factor, args.directional_prob)
            for seed in range(args.start_seed, args.start_seed + args.cases)
        ]
        print(f"completed {name}", flush=True)

    base = results["baseline"]
    base_mean = statistics.fmean(x[0] for x in base)
    print(f"cases={args.cases} seed={args.start_seed} directional_prob={args.directional_prob}")
    print(f"{'variant':18s} {'perfect':>10s} {'mean_s':>10s} {'delta_s':>10s} {'se':>8s} {'rel':>9s} {'req':>8s}")
    for name, _, _ in variants:
        rows = results[name]
        diffs = [row[0] - ref[0] for row, ref in zip(rows, base)]
        delta = statistics.fmean(diffs)
        se = statistics.pstdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        perfect = sum(row[1] == row[2] for row in rows)
        failed = [args.start_seed + i for i, row in enumerate(rows) if row[1] != row[2]]
        print(
            f"{name:18s} {perfect:4d}/{len(rows):<5d} {statistics.fmean(x[0] for x in rows):10.1f}"
            f" {delta:10.1f} {se:8.1f} {delta / base_mean:8.2%} {statistics.fmean(x[3] for x in rows):8.1f}"
        )
        if failed:
            print(f"  failed seeds: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
