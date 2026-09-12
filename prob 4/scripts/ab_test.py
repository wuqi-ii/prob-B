"""通用配对 A/B 测试台：同种子对比若干策略变体，时间差可归因于策略本身。

用法：
    python scripts/ab_test.py --suite scan --cases 300 --start-seed 7000
"""

from __future__ import annotations

import argparse
import contextlib
import math
import statistics
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4 import strategy as strategy_module  # noqa: E402
from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import RECEIVER_MIN_M, StrategyConfig  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


@contextlib.contextmanager
def lateral_verify(phi_deg: float, length_m: float | None = None):
    """把「同向推进点」改成与示向度成 phi_deg 的方向前进 length_m 米。

    phi=0 即当前实现（沿示向度朝源推进）。适度横向偏置能大幅改善两次观测的
    交会角（沿射线推进时两观测近乎共线，测距信息极差），代价是定向源有可能
    落到覆盖楔外——此时策略会自动走二分恢复，不会漏源。
    """
    original = strategy_module.approach_point

    def patched(station, bearing_deg, step_m=None, cfg=None):
        length = length_m if length_m is not None else (step_m if step_m else 300.0)
        rad = math.radians(bearing_deg + phi_deg)
        return (station[0] + math.cos(rad) * length, station[1] + math.sin(rad) * length)

    strategy_module.approach_point = patched
    try:
        yield
    finally:
        strategy_module.approach_point = original


# --------------------------------------------------------------------------
# 变体
# --------------------------------------------------------------------------

class PlannedClear(DogStrategy):
    """扫描网走完后，把剩余清除目标按「最近邻 + 2-opt」重新排序，按计划顺序执行。

    延后策略（task_bias_verify_m 取负）把工作方式变成了"先走完网、再集中清除"。
    此时剩余目标集合基本固定，等于一个静态 TSP；而默认滚动最近邻只是贪心，
    在多目标情形下通常比 2-opt 差 10%~20%。这里在待清除频道集合发生变化时
    重排一次计划，计划内按顺序执行。
    """

    def __init__(self, backend, cfg, verbose: bool = False):
        super().__init__(backend, cfg, verbose)
        self._order: list[int] = []
        self._order_key = None
        self._passes = 0

    # ---- 计划 ----
    def _target_point(self, ch):
        track = self.book[ch]
        summary = track.geometry_summary(self.cfg)
        point = summary["centroid"]
        if point is None and track.near_points:
            point = track.near_points[-1]
        return point

    def _rebuild_order(self, cur) -> None:
        channels = [ch for ch in self.book.known_channels() if ch not in self._give_up]
        points = {}
        for ch in channels:
            p = self._target_point(ch)
            if p is not None:
                points[ch] = p
        remaining = list(points)
        # 最近邻
        order = []
        pos = cur
        while remaining:
            nxt = min(remaining, key=lambda ch: math.dist(pos, points[ch]))
            order.append(nxt)
            pos = points[nxt]
            remaining.remove(nxt)
        # 2-opt（开放路线，起点固定为 cur）
        def length(seq):
            total = math.dist(cur, points[seq[0]]) if seq else 0.0
            for a, b in zip(seq, seq[1:]):
                total += math.dist(points[a], points[b])
            return total
        best = length(order)
        improved = True
        passes = 0
        while improved and passes < 30:
            improved = False
            passes += 1
            n = len(order)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                    val = length(cand)
                    if val < best - 1e-9:
                        order, best, improved = cand, val, True
        self._order = order
        self._order_key = tuple(sorted(order))
        self._passes = passes

    def _priority(self, task, cur):
        if task.kind == "scan" or any(not v for v in self.visited_scan):
            return super()._priority(task, cur)
        key = tuple(sorted(ch for ch in self.book.known_channels()
                           if ch not in self._give_up))
        if key != self._order_key:
            self._rebuild_order(cur)
        if task.channel in self._order:
            return float(self._order.index(task.channel))
        return float(len(self._order)) + super()._priority(task, cur) / 1e6


class PlannedClearNN(PlannedClear):
    """对照：只做最近邻重排，不加 2-opt。"""

    def _rebuild_order(self, cur) -> None:
        channels = [ch for ch in self.book.known_channels() if ch not in self._give_up]
        points = {}
        for ch in channels:
            p = self._target_point(ch)
            if p is not None:
                points[ch] = p
        remaining = list(points)
        order = []
        pos = cur
        while remaining:
            nxt = min(remaining, key=lambda ch: math.dist(pos, points[ch]))
            order.append(nxt)
            pos = points[nxt]
            remaining.remove(nxt)
        self._order = order
        self._order_key = tuple(sorted(order))


class LeanScanAll(DogStrategy):
    """扫描点只测「尚未探到的频道」+「清除失败后需要新观测解冻的频道」。

    已探到但尚未清除的频道不再重复普查：它的第二观测由 verify（同向推进）
    任务专门提供，普查重复测是纯浪费（5 s 检测 + 1 s 切换/次）。
    """

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = []
        for ch in range(1, 21):
            track = self.book[ch]
            if track.cleared:
                continue
            blocked = self._blocked_at_obs.get(ch)
            needs_recovery = blocked is not None and len(track.observations) <= blocked
            if not track.known or needs_recovery:
                channels.append(ch)
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(point), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])


class LeanScanAllGiveUpAware(LeanScanAll):
    """在上一版基础上，彻底放弃的频道也不再普查。"""

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = []
        for ch in range(1, 21):
            track = self.book[ch]
            if track.cleared or ch in self._give_up:
                continue
            blocked = self._blocked_at_obs.get(ch)
            needs_recovery = blocked is not None and len(track.observations) <= blocked
            if not track.known or needs_recovery:
                channels.append(ch)
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(point), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])


class LeanScan2(DogStrategy):
    """扫描点跳过「已知且已有 >=2 个观测」的频道，其余照旧。"""

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = []
        for ch in range(1, 21):
            track = self.book[ch]
            if track.cleared:
                continue
            blocked = self._blocked_at_obs.get(ch)
            needs_recovery = blocked is not None and len(track.observations) <= blocked
            if not track.known or len(track.observations) < 2 or needs_recovery:
                channels.append(ch)
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(point), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])


class SmartScan(DogStrategy):
    """组合扫描规则：信息无损跳过 + 已充分定位的远距离频道跳过。

    跳过条件（任一成立即跳过该频道在本次扫描点的检测）：
      A. 可行域整体距扫描点 > 1000 m：真源必在接收半径外，本次必返 no_signal，信息无损；
      B. 已有 >=2 个观测 且 可行域整体距扫描点 > far_skip_m：该频道已能直接走去清除，
         不需要更多示向度（free_skip 阈值越大越保守；0 表示只要 >=2 观测就跳，
         等价于 lean2）。
    需要解冻（清除失败后等新观测）的频道、以及尚未探到的频道，永不跳过。
    """

    far_skip_m = 500.0

    def __init__(self, backend, cfg, verbose: bool = False, far_skip_m=None):
        super().__init__(backend, cfg, verbose)
        if far_skip_m is not None:
            self.far_skip_m = float(far_skip_m)
        self.skipped_hard = 0
        self.skipped_soft = 0

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = []
        for ch in range(1, 21):
            track = self.book[ch]
            if track.cleared:
                continue
            blocked = self._blocked_at_obs.get(ch)
            needs_recovery = blocked is not None and len(track.observations) <= blocked
            if track.known and not needs_recovery:
                poly = track.polygon(self.cfg)
                if poly:
                    dmin = min(math.dist(point, v) for v in poly)
                    if dmin > RECEIVER_MIN_M:
                        self.skipped_hard += 1
                        continue
                    if len(track.observations) >= 2 and dmin > self.far_skip_m:
                        self.skipped_soft += 1
                        continue
            channels.append(ch)
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(point), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])


class FarSkipScan(DogStrategy):
    """扫描点跳过「可行域整体超出接收半径」的已知频道检测（信息无损）。

    已知频道的可行域 P 是 ±1° 示向度楔形交出的凸多边形，且**必含真源**。
    若扫描点到 P 的最小距离 > 1000 m（接收半径下界），则真源一定在 1000 m 之外，
    该次检测无论定向与否都只能返回 no_signal——跳过它不损失任何信息，却省下
    5 s 检测 + 1 s 切换。未探到的频道不能跳（可行域未知）。
    要求解冻的频道也不能跳（需要新观测来解除冻结）。
    """

    def __init__(self, backend, cfg, verbose: bool = False):
        super().__init__(backend, cfg, verbose)
        self.skipped = 0

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = []
        for ch in range(1, 21):
            track = self.book[ch]
            if track.cleared:
                continue
            blocked = self._blocked_at_obs.get(ch)
            needs_recovery = blocked is not None and len(track.observations) <= blocked
            if track.known and not needs_recovery:
                poly = track.polygon(self.cfg)
                if poly:
                    dmin = min(math.dist(point, v) for v in poly)
                    if dmin > RECEIVER_MIN_M:
                        self.skipped += 1
                        continue
            channels.append(ch)
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(point), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])


class DetourCost(DogStrategy):
    """把 clear / verify 的调度代价改成「插入绕行代价」。

    默认滚动最近邻只看 d(当前点, 任务点)，忽略了"绕完还得回到扫描网"的代价，
    于是容易被远处的源拽偏。绕行代价 = d(cur,t) + d(t, 锚点) - d(cur, 锚点)，
    锚点取最近的一个未访问扫描点。绕行便宜才去做，等价于把 bias 从"拍脑袋常数"
    变成"按几何算出来的真实绕路损失"。
    """

    def _priority(self, task, cur):
        if task.kind == "scan":
            return math.dist(cur, task.point)
        unvisited = [p for i, p in enumerate(self.scan_points) if not self.visited_scan[i]]
        if not unvisited:
            return math.dist(cur, task.point)
        anchor = min(unvisited, key=lambda p: math.dist(cur, p))
        return (math.dist(cur, task.point)
                + math.dist(task.point, anchor)
                - math.dist(cur, anchor))


class DetourCostBiased(DetourCost):
    """绕行代价 + 等效距离优惠（可正可负）。"""

    def __init__(self, backend, cfg, verbose=False, bias_scale=1.0):
        super().__init__(backend, cfg, verbose)
        self._bias_scale = bias_scale

    def _priority(self, task, cur):
        base = super()._priority(task, cur)
        if task.kind == "clear":
            base -= self.cfg.task_bias_clear_m * self._bias_scale
        elif task.kind == "verify":
            base -= self.cfg.task_bias_verify_m * self._bias_scale
        return base


class RampBias(DogStrategy):
    """延后权重随「剩余扫描点比例」衰减：网没走完就延后，快走完就立刻去做。

    常数 bias 的缺点是全程一个尺度：早期延后不够、后期又延后过头。
    这里让有效 bias = bias * (剩余未访问扫描点数 / 总扫描点数)，早期接近满额延后，
    网走完后自动归零，回到纯最近邻。
    """

    def _priority(self, task, cur):
        n_total = len(self.scan_points)
        remaining = sum(1 for v in self.visited_scan if not v)
        scale = remaining / n_total if n_total else 0.0
        d = math.dist(cur, task.point)
        if task.kind == "clear":
            d -= self.cfg.task_bias_clear_m * scale
        elif task.kind == "verify":
            d -= self.cfg.task_bias_verify_m * scale
        return d


def layout(n1, r1, n2, r2):
    """按覆盖搜索给出的布局构造配置（其余参数沿用当前默认）。"""
    return StrategyConfig(scan_inner_count=n1, scan_inner_radius_m=float(r1),
                          scan_outer_count=n2, scan_outer_radius_m=float(r2))


P22 = layout(8, 990, 13, 1854)
P23 = layout(8, 980, 14, 1848)
P23B = layout(8, 975, 14, 1850)
P23C = layout(9, 980, 13, 1854)
P23D = layout(8, 990, 14, 1848)


class OrderAwareScan(DogStrategy):
    """扫描点检测顺序微调：把"上一次停留的频道"排到最后测，省 1 s 切换。"""

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = [ch for ch in range(1, 21) if not self.book[ch].cleared]
        # 下一任务大概率要测的频道：优先把它排在最后测
        if channels:
            cur_ch = getattr(self.backend, "current_channel", None)
            if cur_ch in channels and len(channels) > 1:
                channels = [c for c in channels if c != cur_ch] + [cur_ch]
        for ch in channels:
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(point), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])


# --------------------------------------------------------------------------
# 运行
# --------------------------------------------------------------------------

def run_case(seed, cfg, cls, directional_prob=0.5):
    sim = OfflineSimulator(generate_case(seed, directional_prob=directional_prob))
    stats = cls(sim, cfg).run()
    return {
        "seed": seed,
        "time_s": sim.virtual_time_s,
        "cleared": stats.cleared_count,
        "total": sim.case.total,
        "requests": stats.requests,
    }


def summarize(rows):
    n = len(rows)
    return {
        "mean": statistics.fmean(r["time_s"] for r in rows),
        "perfect": sum(1 for r in rows if r["cleared"] == r["total"]),
        "n": n,
        "failed": [r["seed"] for r in rows if r["cleared"] != r["total"]],
        "req": statistics.fmean(r["requests"] for r in rows),
    }


def paired_delta(base_rows, var_rows):
    """逐案例配对差（消除案例难度差异）。"""
    diffs = [v["time_s"] - b["time_s"] for b, v in zip(base_rows, var_rows)]
    mean_diff = statistics.fmean(diffs)
    sd = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / math.sqrt(len(diffs)) if diffs else 0.0
    return mean_diff, se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="scan")
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--start-seed", type=int, default=7000)
    ap.add_argument("--directional-prob", type=float, default=0.5)
    args = ap.parse_args()

    base_cfg = StrategyConfig()
    suites = {
        "scan": {
            "baseline": (base_cfg, DogStrategy),
            "lean_all": (base_cfg, LeanScanAll),
            "lean_all_giveup": (base_cfg, LeanScanAllGiveUpAware),
            "order_aware": (base_cfg, OrderAwareScan),
        },
        "verify_bias": {
            "v0": (replace(base_cfg, task_bias_verify_m=0.0), DogStrategy),
            **{
                f"v{int(v)}": (replace(base_cfg, task_bias_verify_m=float(v)), DogStrategy)
                for v in (-100, -200, -300, -400, -500, -700, -900)
            },
        },
        "clear_bias": {
            "c0": (base_cfg, DogStrategy),
            **{
                f"c{int(v)}": (replace(base_cfg, task_bias_clear_m=float(v)), DogStrategy)
                for v in (-300, -200, -150, -100, -50, 150, 300, 500)
            },
        },
        "approach": {
            "r085_s400": (base_cfg, DogStrategy),
            **{
                f"r{int(r*100)}_s{s}": (
                    replace(base_cfg, approach_step_ratio=r, approach_max_step_m=float(s)),
                    DogStrategy,
                )
                for r, s in ((0.90, 400), (0.80, 400), (0.85, 300), (0.85, 600),
                             (0.90, 600), (0.95, 400), (0.90, 800), (0.85, 800))
            },
        },
        "verify_fine": {
            "v-300": (base_cfg, DogStrategy),
            **{
                f"v{v}": (replace(base_cfg, task_bias_verify_m=float(v)), DogStrategy)
                for v in (-500, -600, -700, -800, -1000, -1200, -1500, -2000)
            },
        },
        "combo": {
            "base_v300_s400": (base_cfg, DogStrategy),
            "v700": (replace(base_cfg, task_bias_verify_m=-700.0), DogStrategy),
            "v700_s300": (
                replace(base_cfg, task_bias_verify_m=-700.0, approach_max_step_m=300.0),
                DogStrategy,
            ),
            "v500_s300": (
                replace(base_cfg, task_bias_verify_m=-500.0, approach_max_step_m=300.0),
                DogStrategy,
            ),
            "v900_s300": (
                replace(base_cfg, task_bias_verify_m=-900.0, approach_max_step_m=300.0),
                DogStrategy,
            ),
        },
        "adopt": {
            "cur25_v300": (base_cfg, DogStrategy),
            "cur25_v700": (replace(base_cfg, task_bias_verify_m=-700.0), DogStrategy),
            "p22_v300": (P22, DogStrategy),
            "p22_v700": (replace(P22, task_bias_verify_m=-700.0), DogStrategy),
            "p22_v700_lean2": (replace(P22, task_bias_verify_m=-700.0), LeanScan2),
            "p23_v700": (replace(P23, task_bias_verify_m=-700.0), DogStrategy),
        },
        "layout2": {
            "cur_25": (base_cfg, DogStrategy),
            "p22_8_990_13_1854": (P22, DogStrategy),
            "p22_8_990_13_1860": (layout(8, 990, 13, 1860), DogStrategy),
            "p23_8_980_14_1848": (P23, DogStrategy),
            "p23_8_975_14_1850": (P23B, DogStrategy),
            "p23_9_980_13_1854": (P23C, DogStrategy),
            "p23_8_990_14_1848": (P23D, DogStrategy),
        },
        "final": {
            "cur_default": (base_cfg, DogStrategy),
            "p23_v700": (P23, DogStrategy),
            "p23_v700_lean2": (P23, LeanScan2),
            **{
                f"p23_v700_smart{f}": (P23, partial(SmartScan, far_skip_m=float(f)))
                for f in (0, 300, 600)
            },
            "p23_v700_lean2_opp": (
                replace(P23, opportunistic_radius_m=1000.0,
                        opportunistic_max_per_stop=6,
                        opportunistic_min_diameter_m=25.0),
                LeanScan2,
            ),
            "p23_v700_lean2_opp_bt20": (
                replace(P23, opportunistic_radius_m=1000.0,
                        opportunistic_max_per_stop=6,
                        opportunistic_min_diameter_m=25.0,
                        bisect_tolerance_m=20.0),
                LeanScan2,
            ),
            "p25_v700_lean2": (
                replace(base_cfg, task_bias_verify_m=-700.0), LeanScan2),
        },
        "smart": {
            "v700": (replace(base_cfg, task_bias_verify_m=-700.0), DogStrategy),
            "v700_lean2": (replace(base_cfg, task_bias_verify_m=-700.0), LeanScan2),
            **{
                f"v700_smart_f{m}": (
                    replace(base_cfg, task_bias_verify_m=-700.0),
                    partial(SmartScan, far_skip_m=float(m)),
                )
                for m in (0, 200, 400, 600, 800, 1000, 1e9)
            },
            "v700_farskip": (replace(base_cfg, task_bias_verify_m=-700.0), FarSkipScan),
        },
        "bisect": {
            "v700": (replace(base_cfg, task_bias_verify_m=-700.0), DogStrategy),
            **{
                f"v700_bt{m}": (
                    replace(base_cfg, task_bias_verify_m=-700.0,
                            bisect_tolerance_m=float(m)),
                    DogStrategy,
                )
                for m in (16, 20, 24, 28)
            },
        },
        "farskip": {
            "v700": (replace(base_cfg, task_bias_verify_m=-700.0), DogStrategy),
            "v700_farskip": (replace(base_cfg, task_bias_verify_m=-700.0), FarSkipScan),
            "v300_farskip": (base_cfg, FarSkipScan),
            "v700_farskip_s150": (
                replace(base_cfg, task_bias_verify_m=-700.0, approach_max_step_m=150.0),
                FarSkipScan,
            ),
        },
        "fine2": {
            "v300_s400": (base_cfg, DogStrategy),
            **{
                f"v300_s{s}": (replace(base_cfg, approach_max_step_m=float(s)), DogStrategy)
                for s in (150, 100, 80, 50)
            },
            "v700_s400": (replace(base_cfg, task_bias_verify_m=-700.0), DogStrategy),
            **{
                f"v700_s{s}": (
                    replace(base_cfg, task_bias_verify_m=-700.0,
                            approach_max_step_m=float(s)),
                    DogStrategy,
                )
                for s in (300, 250, 200, 150, 100)
            },
        },
        "layout": {
            "cur_25": (base_cfg, DogStrategy),
            "p23_8_975_14_1850": (
                replace(base_cfg, scan_inner_count=8, scan_inner_radius_m=975.0,
                        scan_outer_count=14, scan_outer_radius_m=1850.0), DogStrategy),
            "p26_9_950_16_1836": (
                replace(base_cfg, scan_inner_count=9, scan_inner_radius_m=950.0,
                        scan_outer_count=16, scan_outer_radius_m=1836.0), DogStrategy),
            "p26_7_975_18_1836": (
                replace(base_cfg, scan_inner_count=7, scan_inner_radius_m=975.0,
                        scan_outer_count=18, scan_outer_radius_m=1836.0), DogStrategy),
            "p27_8_950_18_1836": (
                replace(base_cfg, scan_inner_count=8, scan_inner_radius_m=950.0,
                        scan_outer_count=18, scan_outer_radius_m=1836.0), DogStrategy),
            "p28_9_925_18_1836": (
                replace(base_cfg, scan_inner_count=9, scan_inner_radius_m=925.0,
                        scan_outer_count=18, scan_outer_radius_m=1836.0), DogStrategy),
            "p23_9_925_16_1836": (
                replace(base_cfg, scan_inner_count=9, scan_inner_radius_m=925.0,
                        scan_outer_count=16, scan_outer_radius_m=1836.0), DogStrategy),
        },
        "planned": {
            "v700": (replace(base_cfg, task_bias_verify_m=-700.0), DogStrategy),
            "v700_planned_nn": (
                replace(base_cfg, task_bias_verify_m=-700.0), PlannedClearNN),
            "v700_planned_2opt": (
                replace(base_cfg, task_bias_verify_m=-700.0), PlannedClear),
            "v300_planned_2opt": (
                replace(base_cfg, task_bias_verify_m=-300.0), PlannedClear),
            "v700_lean2": (replace(base_cfg, task_bias_verify_m=-700.0), LeanScan2),
            "v700_leanall": (replace(base_cfg, task_bias_verify_m=-700.0), LeanScanAll),
        },
        "detour": {
            "baseline_v300": (base_cfg, DogStrategy),
            "detour_plain": (base_cfg, DetourCost),
            "detour_b0": (base_cfg, partial(DetourCostBiased, bias_scale=0.0)),
            "detour_b300": (base_cfg, partial(DetourCostBiased, bias_scale=1.0)),
            "detour_b700": (
                replace(base_cfg, task_bias_verify_m=-700.0),
                partial(DetourCostBiased, bias_scale=1.0),
            ),
            "ramp300": (replace(base_cfg, task_bias_verify_m=-300.0), RampBias),
            "ramp700": (replace(base_cfg, task_bias_verify_m=-700.0), RampBias),
            "ramp1400": (replace(base_cfg, task_bias_verify_m=-1400.0), RampBias),
        },
        "approach_fine": {            "r085_s400": (base_cfg, DogStrategy),
            **{
                f"r{int(r*100)}_s{s}": (
                    replace(base_cfg, approach_step_ratio=r, approach_max_step_m=float(s)),
                    DogStrategy,
                )
                for r, s in ((0.85, 300), (0.85, 250), (0.85, 200), (0.85, 150),
                             (0.75, 300), (0.70, 300), (0.80, 300), (0.90, 300))
            },
        },
        "sparse": {
            "p23_cur": (base_cfg, DogStrategy),
            "p19_w1098": (
                replace(base_cfg, scan_inner_radius_m=900.0, scan_inner_count=7,
                        scan_outer_radius_m=1880.0, scan_outer_count=11),
                DogStrategy,
            ),
            "p19_w1231": (
                replace(base_cfg, scan_inner_radius_m=900.0, scan_inner_count=4,
                        scan_outer_radius_m=1850.0, scan_outer_count=14),
                DogStrategy,
            ),
            "p17_w1244": (
                replace(base_cfg, scan_inner_radius_m=900.0, scan_inner_count=4,
                        scan_outer_radius_m=1880.0, scan_outer_count=12),
                DogStrategy,
            ),
        },
        "bisect_probe": {
            "b12": (base_cfg, DogStrategy),
            **{
                f"b{b}": (
                    replace(base_cfg, bisect_tolerance_m=float(b)),
                    DogStrategy,
                )
                for b in (20, 25, 30, 36, 44)
            },
        },
        "clear_tune": {
            "m2_b12": (base_cfg, DogStrategy),
            **{
                f"m{m}": (
                    replace(base_cfg, clear_safety_margin_m=float(m)),
                    DogStrategy,
                )
                for m in (5, 8, 12, 15, 18)
            },
            **{
                f"b{b}": (
                    replace(base_cfg, bisect_tolerance_m=float(b)),
                    DogStrategy,
                )
                for b in (6, 16, 20, 25)
            },
        },
        "lateral_v700": {
            "phi0_L300": (base_cfg, DogStrategy, lateral_verify(0.0, 300.0)),
            **{
                f"phi{p}_L{L}": (base_cfg, DogStrategy, lateral_verify(float(p), float(L)))
                for p, L in ((15, 300), (30, 300), (45, 300),
                             (15, 500), (30, 500), (45, 500),
                             (30, 400), (30, 700), (20, 500), (40, 500))
            },
        },
        "approach_micro": {
            "s400": (base_cfg, DogStrategy),
            **{
                f"r{int(r*100)}_s{s}": (
                    replace(base_cfg, approach_step_ratio=r, approach_max_step_m=float(s)),
                    DogStrategy,
                )
                for r, s in ((0.85, 300), (0.85, 200), (0.85, 150), (0.85, 120),
                             (0.85, 100), (0.85, 75), (0.85, 50),
                             (0.70, 150), (0.60, 150), (0.95, 150))
            },
        },
        "first_step": {
            "f300": (base_cfg, DogStrategy),
            **{
                f"f{v}": (replace(base_cfg, first_approach_step_m=float(v)), DogStrategy)
                for v in (150, 450, 600, 800, 1000, 1200)
            },
        },
        "lateral": {
            "phi0_L300": (base_cfg, DogStrategy, lateral_verify(0.0, 300.0)),
            **{
                f"phi{p}_L300": (base_cfg, DogStrategy, lateral_verify(float(p), 300.0))
                for p in (15, 30, 45, 60, 90)
            },
            **{
                f"phi{p}_L{L}": (base_cfg, DogStrategy, lateral_verify(float(p), float(L)))
                for p, L in ((30, 500), (45, 500), (45, 800))
            },
        },
        "opportunistic": {
            "opp_base": (base_cfg, DogStrategy),
            **{
                f"r{int(r)}_k{k}_d{int(d)}": (
                    replace(base_cfg, opportunistic_radius_m=float(r),
                            opportunistic_max_per_stop=k,
                            opportunistic_min_diameter_m=float(d)),
                    DogStrategy,
                )
                for r, k, d in ((800, 3, 40), (1000, 3, 40), (1000, 6, 25),
                                (1200, 6, 25), (1500, 6, 20), (1000, 6, 60))
            },
        },
    }
    configs = suites[args.suite]
    seeds = list(range(args.start_seed, args.start_seed + args.cases))

    results = {}
    for name, spec in configs.items():
        cfg, cls = spec[0], spec[1]
        patch = spec[2] if len(spec) > 2 else None
        with (patch if patch is not None else contextlib.nullcontext()):
            results[name] = [run_case(s, cfg, cls, args.directional_prob) for s in seeds]

    base_name = next(iter(configs))
    base = results[base_name]
    bs = summarize(base)
    print(f"suite={args.suite}  cases={bs['n']}  directional_prob={args.directional_prob}")
    print(f"{'variant':22s} {'perfect':>9s} {'mean_s':>9s} {'paired_dt':>10s} {'±se':>7s} "
          f"{'rel':>8s} {'req':>7s}")
    print("-" * 80)
    for name, rows in results.items():
        s = summarize(rows)
        if name == base_name:
            print(f"{name:22s} {s['perfect']:4d}/{s['n']:<4d} {s['mean']:9.1f} "
                  f"{'--':>10s} {'--':>7s} {'--':>8s} {s['req']:7.1f}")
        else:
            md, se = paired_delta(base, rows)
            rel = md / bs["mean"]
            print(f"{name:22s} {s['perfect']:4d}/{s['n']:<4d} {s['mean']:9.1f} "
                  f"{md:+10.1f} {se:7.1f} {rel:+8.2%} {s['req']:7.1f}")
            if s["failed"]:
                print(f"    failed seeds: {s['failed'][:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
