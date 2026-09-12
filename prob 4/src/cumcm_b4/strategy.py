"""问题四策略主引擎：包围网覆盖普查 + 滚动最近邻 + 定向源感知的逼近清除。

策略一句话
----------
维护三类"任务点"——**必访扫描点**（27 点包围网，保证任意朝向的定向源不漏检）、
**待清除点**（已观测到信号的频道，逼近到 20 m 内清除）、**同向推进点**（只有一次
示向度的频道，沿示向度朝源推进补第二观测）；每一步按路径代价最小挑一个执行，
直到"包围网走完 + 所有已知频道清除"。

与问题 3 的三处关键区别
------------------------
1. **扫描点从七点升级为包围网**：定向源只在 180° 覆盖楔内可测，因此必须用
   「中心 + 内环 + 外环（区域外）」把每个可能的源从四面八方包围住（见 coverage.py）。
2. **单示向度频道走"同向推进"而非垂直第二检测点**：定向源的覆盖楔外不可测，问题 2
   的垂直候选点可能落在楔外；沿示向度朝源推进则始终在楔内（见 second_station.py）。
3. **信号消失 ≠ 源消失**：逼近途中测到 ``no_signal``，对定向源而言通常意味着
   越过了源（走出覆盖楔），而不是源不存在。此时在「最后一个有信号点」与「无信号点」
   之间二分搜索，把源夹出来。光学清除（/clear）只与距离（≤20 m）有关、与覆盖楔无关，
   故即使站在楔外也能清除；收敛后用「中心 + 六点环形」兜底。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import geometry
from .config import CLEAR_RADIUS_M, StrategyConfig
from .coverage import scan_points
from .second_station import approach_point
from .tracker import TrackBook

Point = Tuple[float, float]


# --------------------------------------------------------------------------
# 任务定义
# --------------------------------------------------------------------------

@dataclass
class Task:
    kind: str                       # 'scan' | 'verify' | 'clear'
    point: Point
    channel: Optional[int] = None
    note: str = ""


@dataclass
class RunStats:
    """一次运行的统计与诊断信息。"""

    cleared_count: int = 0
    virtual_time_s: float = 0.0
    requests: int = 0
    clear_attempts: int = 0
    clear_success: int = 0
    clear_miss: int = 0
    scan_points_visited: int = 0
    per_target_time_s: Dict[int, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    channel_records: Dict[int, Dict[str, Any]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 策略主体
# --------------------------------------------------------------------------

class DogStrategy:
    """机器狗的"大脑"。只依赖 DogBackend 抽象，真实/离线模拟器通用。"""

    def __init__(self, backend, cfg: StrategyConfig, verbose: bool = False) -> None:
        self.backend = backend
        self.cfg = cfg
        self.verbose = verbose
        self.scan_points = scan_points(cfg)
        self.visited_scan: List[bool] = [False] * len(self.scan_points)
        self.book = TrackBook(list(range(1, 21)))
        self.stats = RunStats()
        self._clear_attempts: Dict[int, int] = {}   # 每频道已尝试清除次数
        self._give_up: set = set()                  # 超过尝试上限后放弃的频道
        self._blocked_at_obs: Dict[int, int] = {}   # 清除失败后"冻结"时的观测数，等新观测再重试
        self._real_start = 0.0

    # ---------------- 小工具 ----------------
    def _log(self, event: str, **payload: Any) -> None:
        record = {"virtual_time_s": round(self.backend.virtual_time_s, 3),
                  "event": event, **payload}
        self.stats.events.append(record)
        if self.verbose:
            print(f"[{record['virtual_time_s']:9.1f}s] {event} {payload}")

    def _pos(self) -> Point:
        return getattr(self.backend, "position", (0.0, 0.0))

    # ---------------- 主流程 ----------------
    def run(self) -> RunStats:
        self._real_start = time.time()
        info = self.backend.enter()
        self._log("enter", **{k: v for k, v in info.items() if k != "accepted"})

        self._visit_scan_point(0)   # 原点全频道普查

        guard = 0
        max_actions = 12000
        while guard < max_actions:
            guard += 1
            if self._budget_exhausted(info):
                self.stats.stop_reason = "budget_exhausted"
                break
            task = self.select_task()
            if task is None:
                self.stats.stop_reason = self._no_task_stop_reason()
                break
            self.execute(task)

        self.backend.exit()
        self.stats.cleared_count = self.book.count_cleared()
        self.stats.virtual_time_s = self.backend.virtual_time_s
        self.stats.requests = self.backend.request_count
        self.stats.scan_points_visited = sum(self.visited_scan)
        for ch, tr in self.book.tracks.items():
            if tr.known:
                self.stats.channel_records[ch] = {
                    "observations": len(tr.observations),
                    "cleared": tr.cleared,
                }
        if not self.stats.stop_reason:
            self.stats.stop_reason = "action_limit"
        self.stats.real_elapsed_s = time.time() - self._real_start  # type: ignore[attr-defined]
        return self.stats

    def _no_task_stop_reason(self) -> str:
        return "all_done" if self.status()["all_clear"] else "unresolved_targets"

    def _budget_exhausted(self, info: Dict[str, Any]) -> bool:
        cfg = self.cfg
        guard = getattr(self.backend, "budget_exhausted", None)
        if callable(guard) and guard():
            return True
        if self.backend.virtual_time_s >= cfg.virtual_time_budget_s:
            return True
        if self.backend.request_count >= cfg.max_requests:
            return True
        return False

    # ---------------- 任务生成与选择 ----------------
    def select_task(self) -> Optional[Task]:
        cfg = self.cfg
        cur = self._pos()
        candidates: List[Task] = []

        # (a) 必访扫描点
        for idx, p in enumerate(self.scan_points):
            if not self.visited_scan[idx]:
                candidates.append(Task("scan", p, None, note=f"S{idx}"))

        # (b) 已确认存在的频道
        for ch in self.book.known_channels():
            if ch in self._give_up:
                continue
            track = self.book[ch]
            # 清除失败后被"冻结"：除非又拿到了新的示向度观测，否则不再尝试清除，
            # 把调度机会让给剩余扫描点（它们会从更好的角度补观测）。
            if ch in self._blocked_at_obs and len(track.observations) <= self._blocked_at_obs[ch]:
                continue
            if track.near_points:
                # 距离 <=5 m 且在覆盖楔内：直接光学清除
                candidates.append(Task("clear", track.near_points[-1], ch, note="near"))
                continue
            if len(track.observations) == 1:
                # 只有一次示向度：沿示向度朝源推进补第二观测（保持在覆盖楔内）
                first_pt, first_bearing = track.observations[0]
                p = approach_point(first_pt, first_bearing, cfg.first_approach_step_m)
                candidates.append(Task("verify", p, ch, note="first-approach"))
                continue
            summary = track.geometry_summary(cfg)
            centroid = summary["centroid"]
            if centroid is not None:
                candidates.append(
                    Task("clear", centroid, ch,
                         note=f"u={summary['centroid_max_error_m']:.1f}")
                )

        if not candidates:
            return None

        def cost(t: Task) -> float:
            d = math.dist(cur, t.point)
            if t.kind == "clear":
                d -= cfg.task_bias_clear_m
            elif t.kind == "verify":
                d -= cfg.task_bias_verify_m
            return d

        return min(candidates, key=cost)

    # ---------------- 任务执行 ----------------
    def execute(self, task: Task) -> None:
        if task.kind == "scan":
            idx = self.scan_points.index(task.point)
            self._visit_scan_point(idx)
        elif task.kind == "verify":
            self._do_verify(task)
        else:  # clear
            self._do_clear(task)

    def _visit_scan_point(self, idx: int) -> None:
        p = self.scan_points[idx]
        channels = [ch for ch in range(1, 21) if not self.book[ch].cleared]
        for ch in channels:
            payload = self.backend.measure(p[0], p[1], ch)
            kind = self.book.apply_measure(ch, p, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(p), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(p), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])

    # ---------------- 单次示向度的同向推进 ----------------
    def _do_verify(self, task: Task) -> None:
        """只有一次示向度的频道：到同向推进点补第二观测，并就地完成逼近清除。

        关键：拿到第二观测后**不交还给调度器**，而是立即在同一任务内继续逼近。
        否则机器狗会离开示向线去做别的任务，等回来"清除"时已远离源、朝不可靠的
        质心走，二分搜索的线段不再经过源（定向源楔形边缘案例的常见失败模式）。
        """
        assert task.channel is not None
        ch = task.channel
        p = task.point
        t0 = self.backend.virtual_time_s
        self._clear_attempts[ch] = self._clear_attempts.get(ch, 0) + 1
        payload = self.backend.measure(p[0], p[1], ch)
        kind = self.book.apply_measure(ch, p, payload)
        self._log("verify", channel=ch,
                  at=[round(p[0], 1), round(p[1], 1)], result=kind)
        if kind == "direction":
            # 第二观测到手，狗就停在示向线上，立即继续逼近
            success = self._approach_and_clear(ch)
            self._finalize(ch, success, t0)
            self._opportunistic_pass(self._pos(), exclude=ch)
            return
        if kind == "near":
            success = self._try_clear(p, ch)
            self._finalize(ch, success, t0)
            self._opportunistic_pass(self._pos(), exclude=ch)
            return
        # no_signal：源在首个观测点与推进点之间（越过了），二分搜索夹出后清除
        first_pt = self.book[ch].observations[0][0]
        success = self._resolve_overshoot(ch, first_pt, p)
        self._finalize(ch, success, t0)
        self._opportunistic_pass(self._pos(), exclude=ch)

    # ---------------- 清除主路径 ----------------
    def _do_clear(self, task: Task) -> None:
        assert task.channel is not None
        ch = task.channel
        t0 = self.backend.virtual_time_s
        self._clear_attempts[ch] = self._clear_attempts.get(ch, 0) + 1
        success = self._approach_and_clear(ch)
        self._finalize(ch, success, t0)
        self._opportunistic_pass(self._pos(), exclude=ch)

    def _finalize(self, ch: int, success: bool, t0: float) -> None:
        """清除后的复测确认、记账与放弃判定。"""
        cfg = self.cfg
        if success and cfg.post_clear_verify:
            p = self._pos()
            payload = self.backend.measure(p[0], p[1], ch)
            kind = self.book.apply_measure(ch, p, payload)
            self._log("post_clear_check", channel=ch,
                      at=[round(p[0], 1), round(p[1], 1)], result=kind)
            if kind != "no_signal":
                self._log("clear_unconfirmed", channel=ch, result=kind)
                success = False
        if success:
            self.book[ch].cleared = True
            cost = self.backend.virtual_time_s - t0
            self.stats.per_target_time_s[ch] = cost
            self.stats.clear_success += 1
            self._log("cleared", channel=ch, cost_s=round(cost, 1),
                      attempts=self._clear_attempts[ch])
        else:
            self._log("clear_failed", channel=ch, attempts=self._clear_attempts[ch])
            # 冻结该频道：等新的示向度观测（来自后续扫描点/顺捎复测）再重试，
            # 避免用同一批（可能带偏的）观测反复重试直到耗光次数。
            self._blocked_at_obs[ch] = len(self.book[ch].observations)
            if self._clear_attempts[ch] >= self.cfg.max_clear_attempts_per_channel:
                self._give_up.add(ch)
                self._log("give_up", channel=ch,
                          attempts=self._clear_attempts[ch])

    # ---------------- 逼近与清除 ----------------
    def _try_clear(self, p: Point, ch: int) -> bool:
        res = self.backend.clear(p[0], p[1], ch)
        self.stats.clear_attempts += 1
        if res.get("clear_result") == "success":
            return True
        self.stats.clear_miss += 1
        return False

    def _ring_clear(self, center: Point, ch: int) -> bool:
        """光学环形兜底清除：中心 + 六点环形，共 7 次 /clear。

        /clear 只与距离有关、与定向源覆盖角无关，即便站在覆盖楔之外也能清。
        只要源在中心 15 m 附近，7 个点必有至少一个距源 <=20 m。
        """
        cfg = self.cfg
        if not cfg.ring_clear_enabled:
            return False
        if self._try_clear(center, ch):
            return True
        n = cfg.ring_clear_count
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            p = (center[0] + cfg.ring_clear_radius_m * math.cos(ang),
                 center[1] + cfg.ring_clear_radius_m * math.sin(ang))
            if self._try_clear(p, ch):
                return True
        return False

    def _approach_and_clear(self, ch: int) -> bool:
        """逼近循环（>=2 次示向度）：补观测 -> 质心 -> 再逼近，直到光学能命中。"""
        cfg = self.cfg
        track = self.book[ch]

        if track.near_points:
            return self._try_clear(track.near_points[-1], ch)

        for _ in range(cfg.approach_max_iterations):
            summary = track.geometry_summary(cfg)
            centroid = summary["centroid"]
            if centroid is None:
                return False
            worst = summary["centroid_max_error_m"]

            if worst <= CLEAR_RADIUS_M - cfg.clear_safety_margin_m:
                if self._try_clear(centroid, ch):
                    return True
                # 未命中：原地补一次观测修正，或进入环形兜底
                payload = self.backend.measure(centroid[0], centroid[1], ch)
                kind = self.book.apply_measure(ch, centroid, payload)
                if kind == "near":
                    return self._try_clear(centroid, ch)
                if kind == "direction":
                    continue
                return self._ring_clear(centroid, ch)

            cur = self._pos()
            d = math.dist(cur, centroid)
            step = max(cfg.approach_min_step_m,
                       min(cfg.approach_step_ratio * d, cfg.approach_max_step_m))
            target = geometry.point_along(cur, centroid, step)
            payload = self.backend.measure(target[0], target[1], ch)
            kind = self.book.apply_measure(ch, target, payload)
            self._log("approach", channel=ch,
                      at=[round(target[0], 1), round(target[1], 1)],
                      worst=round(worst, 1) if math.isfinite(worst) else -1,
                      result=kind)
            if kind == "near":
                return self._try_clear(target, ch)
            if kind == "no_signal":
                # 越过了源：在最后一个有信号点与当前位置之间二分搜索
                lo = track.observations[-1][0] if track.observations else cur
                if self._resolve_overshoot(ch, lo, target):
                    return True
                return self._ring_clear(centroid, ch)
        return False

    def _resolve_overshoot(self, ch: int, lo_pt: Point, hi_pt: Point) -> bool:
        """在 [lo_pt(有信号), hi_pt(无信号)] 之间二分搜索源，收敛后清除。

        沿逼近射线前进时，源之前返回 direction/near，越过源之后返回 no_signal
        （走出覆盖楔），因此可按 measure 结果二分夹逼。±1° 误差在短距离内造成的
        横向偏移远小于 20 m 清除半径，收敛点附近清除即可命中。
        """
        lo = lo_pt
        hi = hi_pt
        for _ in range(20):
            if math.dist(lo, hi) <= 12.0:
                break
            mid = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
            payload = self.backend.measure(mid[0], mid[1], ch)
            kind = self.book.apply_measure(ch, mid, payload)
            self._log("bisect", channel=ch,
                      at=[round(mid[0], 1), round(mid[1], 1)], result=kind)
            if kind == "near":
                return self._try_clear(mid, ch)
            if kind == "direction":
                lo = mid
            else:
                hi = mid
        center = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
        return self._ring_clear(center, ch)

    # ---------------- 机会性复测 ----------------
    def _opportunistic_pass(self, point: Point, exclude: Optional[int] = None) -> None:
        cfg = self.cfg
        if not cfg.opportunistic_scan:
            return
        done = 0
        for ch in self.book.known_channels():
            if done >= cfg.opportunistic_max_per_stop:
                break
            if exclude is not None and ch == exclude:
                continue
            track = self.book[ch]
            summary = track.geometry_summary(cfg)
            centroid = summary["centroid"]
            if centroid is None:
                continue
            if summary["centroid_max_error_m"] <= cfg.opportunistic_min_diameter_m:
                continue
            if math.dist(point, centroid) > cfg.opportunistic_radius_m:
                continue
            payload = self.backend.measure(point[0], point[1], ch)
            kind = self.book.apply_measure(ch, point, payload)
            self._log("opportunistic", channel=ch, result=kind)
            done += 1

    # ---------------- 终止条件的可解释输出 ----------------
    def status(self) -> Dict[str, Any]:
        unvisited = [i for i, v in enumerate(self.visited_scan) if not v]
        known_not_cleared = [ch for ch in self.book.known_channels()]
        return {
            "unvisited_scan_points": unvisited,
            "known_not_cleared": known_not_cleared,
            "cleared": self.book.cleared_channels(),
            "all_clear": not unvisited and not known_not_cleared,
        }
