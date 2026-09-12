"""问题三策略主引擎：七点覆盖普查 + 滚动最近邻 + 机会性复测。

策略一句话
----------
维护三类"任务点"——**必访扫描点**（保证覆盖，不让任何一个源被漏掉）、
**待验证点**（给只有一次示向的频道补第二个观测）、**待清除点**（已交会出
可行域的频道，逼近到 20 m 内清除）；每一步先求剩余扫描点的最短覆盖主路线，
再把验证和清除任务按最小绕路增量插入，只执行首项后滚动重算，
执行过程中在停留点顺带把附近频道的示向度补测掉，直到"七点走完 + 全频道无信号"。

这样设计的原因
--------------
1. **先保证不漏，再谈省**：七点布局使区域内任一点到最近扫描点不超过 988.5 m，
   小于最坏情况接收半径 1000 m，所以任何源都会被至少一次普查发现——这是
   "全部清除"这条硬指标的几何前提，也是策略里唯一不可动摇的部分。
2. **移动是主要成本，检测很便宜**：一次检测 5~6 s，而 1 km 行程要 200 s。
   因此凡是到了某个点，就把那里能测的频道一次测完；顺捎检测几乎免费。
3. **定位精度随距离自动提升**：±1° 误差对应的横向偏移约为 0.0175×距离，
   逼近到 100 m 时只有 1.75 m，所以"边走边测"比一次性远距定位更可靠。
"""

from __future__ import annotations

import json
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import geometry
from .config import (
    ARENA_RADIUS_M,
    BEARING_ERROR_DEG,
    CLEAR_RADIUS_M,
    MEASURE_COST_S,
    StrategyConfig,
)
from .coverage import scan_points
from .second_station import candidate_pair
from .tracker import TrackBook

Point = Tuple[float, float]


# --------------------------------------------------------------------------
# 任务定义
# --------------------------------------------------------------------------

@dataclass
class Task:
    kind: str                       # 'scan' | 'verify' | 'clear' | 'direct_clear'
    point: Point
    channel: Optional[int] = None
    note: str = ""
    near_point: Optional[Point] = None   # 验证任务：先试的近端候选点（失败则回退到 point）

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.channel if self.channel is not None else id(self)}"

    @property
    def execution_point(self) -> Point:
        """本次动作实际先到达的位置；近端验证不能再按标准远点计价。"""
        return self.near_point if self.near_point is not None else self.point


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
        self._verified_task_keys: set = set()
        self._clear_attempts: Dict[int, int] = {}   # 每频道已尝试清除次数
        self._give_up: set = set()                  # 超过尝试上限后放弃的频道
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

        # 步骤 0：原点全频道普查（这也是用户的策略起点）
        self._visit_scan_point(0)

        # 步骤 1..n：滚动决策
        guard = 0
        max_actions = 5000
        while guard < max_actions:
            guard += 1
            if self._budget_exhausted(info):
                self.stats.stop_reason = "budget_exhausted"
                break
            task = self.select_task()
            if task is None:
                # 没有可调度任务不等于全部完成：例如某频道连续清除失败后
                # 被列为 unresolved，或定位几何发生异常。绝不能在这种情况
                # 下把结果误报成 all_done。
                self.stats.stop_reason = self._no_task_stop_reason()
                break
            self.execute(task)

        result = self.backend.exit()
        self._log("exit", reason=result.get("exit_reason"))

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
        """区分真正完成与调度器无法继续处理的异常状态。"""
        return "all_done" if self.status()["all_clear"] else "unresolved_targets"

    def _budget_exhausted(self, info: Dict[str, Any]) -> bool:
        """软预算保护：宁可提前收尾，也不要被判超时退出。"""
        cfg = self.cfg
        # 真实后端自带墙钟看门狗，优先听它的
        guard = getattr(self.backend, "budget_exhausted", None)
        if callable(guard) and guard():
            return True
        if self.backend.virtual_time_s >= cfg.virtual_time_budget_s:
            return True
        if self.backend.request_count >= cfg.max_requests:
            return True
        return False

    # ---------------- 任务生成与选择 ----------------
    def _scan_tasks(self) -> List[Task]:
        return [
            Task("scan", p, None, note=f"S{idx}")
            for idx, p in enumerate(self.scan_points)
            if not self.visited_scan[idx]
        ]

    @staticmethod
    def _path_length(start: Point, tasks: Sequence[Task]) -> float:
        total = 0.0
        cur = start
        for task in tasks:
            nxt = task.execution_point
            total += math.dist(cur, nxt)
            cur = nxt
        return total

    def _shortest_scan_spine(self, cur: Point, scans: Sequence[Task]) -> List[Task]:
        """精确求剩余必访扫描点的最短开放路径（最多 6! = 720 种）。"""
        if len(scans) <= 1:
            return list(scans)
        return list(min(itertools.permutations(scans),
                        key=lambda order: self._path_length(cur, order)))

    @staticmethod
    def _insertion_delta(cur: Point, route: Sequence[Task], task: Task,
                         pos: int) -> float:
        """把 task 插到开放路线 pos 位置增加的移动距离。"""
        p = task.execution_point
        prev = cur if pos == 0 else route[pos - 1].execution_point
        if pos == len(route):
            return math.dist(prev, p)
        nxt = route[pos].execution_point
        return math.dist(prev, p) + math.dist(p, nxt) - math.dist(prev, nxt)

    def _task_bonus(self, task: Task) -> float:
        """兼容已有任务优惠；直接清除仍保持强优先。单位均为等效米。"""
        if task.kind == "clear":
            return self.cfg.task_bias_clear_m
        if task.kind == "verify":
            return self.cfg.task_bias_verify_m
        if task.kind == "direct_clear":
            return self.cfg.task_bias_clear_m + 1000.0
        return 0.0

    def _best_side(self, cur: Point, scans: Sequence[Task], track) -> Task:
        """左右镜像验证点按对扫描主路线的最小绕路增量选择。"""
        cfg = self.cfg
        first_point, bearing = track.observations[0]
        pair = candidate_pair(first_point, bearing, cfg)
        spine = self._shortest_scan_spine(cur, scans)
        options: List[Task] = []
        for label, full in (("left", pair.left), ("right", pair.right)):
            near = None
            if 0.0 < cfg.verify_near_fraction < 1.0:
                f = cfg.verify_near_fraction
                near = (first_point[0] + (full[0] - first_point[0]) * f,
                        first_point[1] + (full[1] - first_point[1]) * f)
            options.append(Task("verify", full, track.channel,
                                note=f"pair:{label}", near_point=near))

        if cfg.scheduler_mode == "nearest":
            return min(options, key=lambda task: math.dist(cur, task.execution_point))

        def detour(task: Task) -> float:
            return min(self._insertion_delta(cur, spine, task, pos)
                       for pos in range(len(spine) + 1))

        return min(options, key=detour)

    def _service_tasks(self, cur: Point, scans: Sequence[Task]) -> List[Task]:
        cfg = self.cfg
        tasks: List[Task] = []
        for ch in self.book.known_channels():
            if ch in self._give_up:
                continue
            track = self.book[ch]
            if track.near_points:
                tasks.append(Task("direct_clear", track.near_points[-1], ch, note="near"))
                continue
            summary = track.geometry_summary(cfg)
            centroid = summary["centroid"]
            if len(track.observations) >= 2 and centroid is not None:
                tasks.append(Task("clear", centroid, ch,
                                  note=f"u={summary['centroid_max_error_m']:.1f}"))
            elif len(track.observations) == 1:
                tasks.append(self._best_side(cur, scans, track))
        return tasks

    def _route_insert_choice(self, cur: Point, scans: Sequence[Task],
                             services: Sequence[Task]) -> Optional[Task]:
        """以扫描最短路为骨架，按最小增量插入动态任务，并返回首个动作。"""
        route = self._shortest_scan_spine(cur, scans)
        pending = list(services)
        while pending:
            best = None
            for task in pending:
                for pos in range(len(route) + 1):
                    delta = self._insertion_delta(cur, route, task, pos)
                    adjusted = delta - self._task_bonus(task)
                    candidate = (adjusted, delta, task.key, pos, task)
                    if best is None or candidate[:4] < best[:4]:
                        best = candidate
            assert best is not None
            _, _, _, pos, task = best
            route.insert(pos, task)
            pending.remove(task)
        return route[0] if route else None

    def _nearest_choice(self, cur: Point, candidates: Sequence[Task]) -> Optional[Task]:
        """旧版最近邻，保留用于同案例 A/B 对照。"""
        if not candidates:
            return None
        return min(candidates,
                   key=lambda task: math.dist(cur, task.execution_point)
                   - self._task_bonus(task))

    def select_task(self) -> Optional[Task]:
        """按配置使用旧最近邻或“覆盖主路线 + 动态插入”调度。"""
        cfg = self.cfg
        cur = self._pos()
        scans = self._scan_tasks()
        services = self._service_tasks(cur, scans)
        if cfg.scheduler_mode == "nearest":
            return self._nearest_choice(cur, [*scans, *services])
        return self._route_insert_choice(cur, scans, services)

    # ---------------- 任务执行 ----------------
    def execute(self, task: Task) -> None:
        if task.kind == "scan":
            idx = self.scan_points.index(task.point)
            self._visit_scan_point(idx)
        elif task.kind == "verify":
            self._do_verify(task)
        elif task.kind in ("clear", "direct_clear"):
            self._do_clear(task)

    def _visit_scan_point(self, idx: int) -> None:
        """到达第 idx 个必访扫描点，把未确认频道逐个测完。"""
        p = self.scan_points[idx]
        channels = [
            ch for ch in range(1, 21)
            if not self.book[ch].cleared
        ]
        for ch in channels:
            payload = self.backend.measure(p[0], p[1], ch)
            kind = self.book.apply_measure(ch, p, payload)
            if kind != "no_signal":
                self._log("discover", channel=ch, at=list(p), kind=kind)
        self.visited_scan[idx] = True
        self._log("scan_done", point=list(p), index=idx, channels=len(channels),
                  found=[ch for ch in channels if self.book[ch].known])

    def _do_verify(self, task: Task) -> None:
        """给只有一次观测的频道补第二个观测点（来自问题 2 的候选结论）。

        若配置了 ``verify_near_fraction``，先在更近的候选点上试一次；
        返回 ``no_signal`` 说明源太远、该点不在接收范围内，立即回退到标准候选点。
        标准点满足"二次接收 + 交会角 ≥ 31°"的硬保证，正常情况下必能拿到信号。
        """
        assert task.channel is not None
        ch = task.channel

        if task.near_point is not None:
            p = task.near_point
            payload = self.backend.measure(p[0], p[1], ch)
            kind = self.book.apply_measure(ch, p, payload)
            self._log("verify_near", channel=ch, at=[round(p[0], 1), round(p[1], 1)],
                      result=kind)
            if kind != "no_signal":
                self._opportunistic_pass(p, exclude=ch)
                return
            # 近端收不到：回退到标准点（负信息已记入 track，不影响后续定位）

        p = task.point
        payload = self.backend.measure(p[0], p[1], ch)
        kind = self.book.apply_measure(ch, p, payload)
        self._log("verify", channel=ch, at=[round(p[0], 1), round(p[1], 1)],
                  result=kind)
        self._opportunistic_pass(p, exclude=ch)

    def _do_clear(self, task: Task) -> None:
        """逼近并清除一个已初步定位的频道，清除后**就地复测确认**。

        复测只多 6 s，却把"确保全部清除"从一句口号变成可验证的闭环：
        只有复测返回 ``no_signal`` 才登记为已清除，否则该频道留在候选集中重试。
        """
        assert task.channel is not None
        ch = task.channel
        track = self.book[ch]
        cfg = self.cfg
        t0 = self.backend.virtual_time_s
        self._clear_attempts[ch] = self._clear_attempts.get(ch, 0) + 1
        kind: Optional[str] = None

        success = self._approach_and_clear(ch)
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
            track.cleared = True
            cost = self.backend.virtual_time_s - t0
            self.stats.per_target_time_s[ch] = cost
            self.stats.clear_success += 1
            self._log("cleared", channel=ch, cost_s=round(cost, 1),
                      attempts=self._clear_attempts[ch])
        else:
            self._log("clear_failed", channel=ch, result=kind,
                      attempts=self._clear_attempts[ch])
            if self._clear_attempts[ch] >= cfg.max_clear_attempts_per_channel:
                # 兜底：异常情况下宁可记录并放弃，也不能陷进无限重试把程序时限耗光
                self._give_up.add(ch)
                self._log("give_up", channel=ch,
                          attempts=self._clear_attempts[ch])
        self._opportunistic_pass(self._pos(), exclude=ch)

    def _approach_and_clear(self, ch: int) -> bool:
        """逼近循环：补观测 -> 质心 -> 再逼近，直到光学能命中。"""
        cfg = self.cfg
        track = self.book[ch]

        # near 情形：直接站在 near 点清除即可
        if track.near_points:
            p = track.near_points[-1]
            res = self.backend.clear(p[0], p[1], ch)
            self.stats.clear_attempts += 1
            if res.get("clear_result") == "success":
                return True
            self.stats.clear_miss += 1
            return False

        for _ in range(cfg.approach_max_iterations):
            summary = track.geometry_summary(cfg)
            centroid = summary["centroid"]
            if centroid is None:
                return False
            worst = summary["centroid_max_error_m"]

            # 质心到真源最坏距离已进入光学半径：直接命中
            if worst <= CLEAR_RADIUS_M - cfg.clear_safety_margin_m:
                res = self.backend.clear(centroid[0], centroid[1], ch)
                self.stats.clear_attempts += 1
                if res.get("clear_result") == "success":
                    return True
                self.stats.clear_miss += 1
                # 未命中说明观测被误差带偏：在原位补一次观测再试
                payload = self.backend.measure(centroid[0], centroid[1], ch)
                kind = self.book.apply_measure(ch, centroid, payload)
                if kind == "near":
                    res = self.backend.clear(centroid[0], centroid[1], ch)
                    self.stats.clear_attempts += 1
                    if res.get("clear_result") == "success":
                        return True
                    self.stats.clear_miss += 1
                    return False
                if kind == "no_signal":
                    return False
                continue

            # 朝质心推进，边走边测
            cur = self._pos()
            d = math.dist(cur, centroid)
            step = max(cfg.approach_min_step_m, d * cfg.approach_step_ratio)
            target = geometry.point_along(cur, centroid, step)
            payload = self.backend.measure(target[0], target[1], ch)
            kind = self.book.apply_measure(ch, target, payload)
            self._log("approach", channel=ch,
                      at=[round(target[0], 1), round(target[1], 1)],
                      worst=round(worst, 1) if math.isfinite(worst) else -1,
                      result=kind)
            if kind == "near":
                res = self.backend.clear(target[0], target[1], ch)
                self.stats.clear_attempts += 1
                if res.get("clear_result") == "success":
                    return True
                self.stats.clear_miss += 1
                return False
            if kind == "no_signal":
                # 走到了接收范围之外：该频道可能已无源，或估计严重偏离
                return False
        return False

    # ---------------- 机会性复测 ----------------
    def _opportunistic_pass(self, point: Point, exclude: Optional[int] = None) -> None:
        """在已经停下的位置，把附近频道的示向度顺手补掉。

        停留成本已经付出，额外的检测只多 5~6 s，却能换来后续路线质量的提升。
        """
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
        """当前任务是否满足"全部清除"的判据，便于诊断。"""
        unvisited = [i for i, v in enumerate(self.visited_scan) if not v]
        known_not_cleared = [ch for ch in self.book.known_channels()]
        return {
            "unvisited_scan_points": unvisited,
            "known_not_cleared": known_not_cleared,
            "cleared": self.book.cleared_channels(),
            "all_clear": not unvisited and not known_not_cleared,
        }
