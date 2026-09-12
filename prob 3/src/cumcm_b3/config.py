"""问题三的全局参数与策略参数。

所有物理常量直接取自赛题附录，策略参数集中在 CONFIG 中并可由 configs/*.json 覆盖，
便于做敏感性分析。单位：米 / 秒 / 度。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List


# --------------------------------------------------------------------------
# 赛题给定的物理常量（附录 1、2）——不可调整
# --------------------------------------------------------------------------
ARENA_RADIUS_M = 1800.0          # 目标区域半径
CHANNEL_COUNT = 20               # 频道编号 1..20
SOURCE_COUNT_MIN = 10            # 干扰源总数下界
SOURCE_COUNT_MAX = 16            # 干扰源总数上界
RECEIVER_MIN_M = 1000.0          # 有效接收半径下界（最坏情况）
RECEIVER_MAX_M = 1500.0          # 有效接收半径上界
BEARING_ERROR_DEG = 1.0          # 示向度误差半宽
DOG_SPEED_MPS = 5.0              # 移动速度
SWITCH_COST_S = 1.0              # 任意两频道之间的切换耗时
MEASURE_COST_S = 5.0             # 单次检测耗时
OPTICAL_COST_S = 3.0             # 光学精确定位耗时
LASER_COST_S = 2.0               # 激光清除耗时
CLEAR_RADIUS_M = 20.0            # 光学可精确定位的距离
NEAR_RADIUS_M = 5.0              # 信号过强、无法获得示向度的距离
VIRTUAL_TIME_LIMIT_S = 100 * 3600.0   # 虚拟时间上限
PROGRAM_TIME_LIMIT_S = 20 * 60.0      # 程序运行时间上限
TEST_WINDOW_S = 25 * 60.0             # 测试窗口

# 成功率/耗时换算：清除命中耗时 3+2=5，未命中 3
CLEAR_HIT_COST_S = OPTICAL_COST_S + LASER_COST_S
CLEAR_MISS_COST_S = OPTICAL_COST_S


@dataclass(frozen=True)
class StrategyConfig:
    """策略可调参数。默认值对应文档中的基准配置。"""

    # ---- 覆盖普查 ----
    scan_ring_radius_m: float = 1150.0     # 六个环点到原点的距离
    scan_ring_count: int = 6               # 环点个数（加原点共 7 个扫描点）
    disk_sides: int = 64                   # 圆域外接多边形的边数
    # 允许"留空隙"的覆盖余量（米）。默认 0 = 严格按接收半径下界 1000 m 全覆盖；
    # 设为正数 N 表示允许最坏点间距放宽到 1000+N，用于 model2 实验（留空隙省巡回、冒漏检风险）。
    coverage_gap_tolerance_m: float = 0.0

    # ---- 第二检测点（沿用问题 2 的结论）----
    # 候选区域：a>=0, a^2+b^2<=R^2, (a-D)^2+(|b|+h)^2<=R^2, |b|>=t*a, |b|>=t*(D-a)+h
    second_station_far_m: float = RECEIVER_MAX_M        # 假定的源最远可能距离 D
    second_station_reach_m: float = RECEIVER_MIN_M      # 保守可接收距离 R
    second_station_clearance_h: float = 5.0 + 21.18     # 5 m 盲区 + 余量
    second_station_min_angle_deg: float = 31.0          # 交会角下界（30° + 1° 误差）
    # 问题2给出的局部候选坐标：a 沿首次示向方向，±b 为两侧镜像点。
    # 独立设为配置项，便于用多组问题2结果做演练对照。
    second_station_local_a_m: float = 764.0
    second_station_local_b_m: float = 526.0
    grid_step_m: float = 10.0                           # 候选区域网格步长
    verify_near_fraction: float = 0.4   # 验证点近端优先比例 (0,1)；0 表示关闭（直接去标准点）。离线细扫最优 0.4（−5.7%）

    # ---- 逼近与清除 ----
    approach_step_ratio: float = 0.85      # 每次逼近前进到剩余估计距离的比例
    approach_min_step_m: float = 25.0      # 最小逼近步长，防止原地抖动
    approach_max_iterations: int = 40      # 单个目标的逼近迭代上限
    clear_safety_margin_m: float = 2.0     # 判定「质心可直接清除」的余量
    refine_diameter_m: float = 60.0        # 定位区域直径超过此值时先补观测再逼近

    # ---- 清除后的闭环确认（"确保全部清除"的可验证性）----
    post_clear_verify: bool = True         # 清除成功后是否就地复测确认无信号
    max_clear_attempts_per_channel: int = 4  # 单频道清除尝试上限，防止异常时死循环

    # ---- 决策 ----
    scheduler_mode: str = "route_insert"  # route_insert=覆盖主路线+动态插入；nearest=旧版最近邻
    adaptive_search: bool = False  # 实验：已执行的逐频道检测覆盖替代固定扫描点
    joint_service: bool = False    # 实验：验证、预计逼近清除联合计价与执行
    shared_observations: str = "off"  # off / opportunistic / batch
    route_polish: bool = False  # 对当前任务路线做开放路径2-opt
    adaptive_verify: bool = False
    route_multistart: bool = False
    route_time_score: bool = False
    step_replan: bool = False
    safe_clear_point: bool = False
    scan_relocation: bool = False
    received_range_bound: bool = False
    recovery_verify: bool = False
    dynamic_second_station: bool = False  # 场地截短首次示向扇形时，用动态 R 生成第二检测点
    planned_stop_reuse: bool = False  # 已实际到达点满足硬约束时，原地兼任第二检测点
    bearing_rounding_slack_deg: float = 0.0  # numerical rounding, not physical error relaxation
    stop_at_source_upper_bound: bool = False
    stop_search_when_all_sources_known: bool = False  # 已发现16个不同频道后取消剩余覆盖搜索
    opportunistic_scan: bool = True        # 是否启用顺路/顺频道捎带检测
    route_replan_every_step: bool = True   # 每完成一个任务点后是否重算路径
    scan_all_channels_at_scan_points: bool = True  # 在必访扫描点是否扫全部未确认频道
    task_bias_clear_m: float = 0.0         # 清除任务的等效距离优惠（越大越优先）
    task_bias_verify_m: float = 0.0        # 验证任务的等效距离优惠
    opportunistic_radius_m: float = 600.0  # 顺捎检测的作用半径
    opportunistic_max_per_stop: int = 3    # 单次停留最多顺捎几个频道
    opportunistic_min_diameter_m: float = 40.0  # 区域已足够小就不必顺捎
    opportunistic_gain_threshold_m: float = 30.0  # 顺路补测的收益阈值（可行域缩小量，米）；低于此不做，避免低价值检测

    # ---- 运行 ----
    base_url: str = "http://127.0.0.1:2026"
    robot_id: str = "202623001124"
    request_timeout_s: float = 10.0
    max_requests: int = 4000              # 请求数硬上限，防止异常高频循环
    virtual_time_budget_s: float = float("inf")  # 虚拟时间软预算，超出则提前收尾

    def validate(self) -> None:
        """参数可行性检查：覆盖必须成立，几何参数必须在合理区间。"""
        if not 0 < self.scan_ring_radius_m <= ARENA_RADIUS_M:
            raise ValueError("scan_ring_radius_m 必须位于 (0, 1800]")
        if self.scan_ring_count < 3:
            raise ValueError("scan_ring_count 至少为 3")
        gap = coverage_worst_gap(self.scan_ring_radius_m, self.scan_ring_count)
        if gap > RECEIVER_MIN_M + self.coverage_gap_tolerance_m + 1e-9:
            raise ValueError(
                "扫描点不能按最坏接收半径覆盖目标区域："
                f"最坏距离 {gap:.3f} m > {RECEIVER_MIN_M:.3f} m "
                f"(覆盖余量 {self.coverage_gap_tolerance_m:.3f} m)"
            )
        if not 0 < self.approach_step_ratio < 1:
            raise ValueError("approach_step_ratio 必须位于 (0, 1)")
        if not 0 <= self.clear_safety_margin_m < CLEAR_RADIUS_M:
            raise ValueError("clear_safety_margin_m 必须在 [0, 20)")
        if not 0 < self.second_station_min_angle_deg < 90:
            raise ValueError("second_station_min_angle_deg 必须位于 (0, 90)")
        if not 0.0 <= self.verify_near_fraction < 1.0:
            raise ValueError("verify_near_fraction 必须位于 [0, 1)")
        if self.scheduler_mode not in {"nearest", "route_insert"}:
            raise ValueError("scheduler_mode 只能是 nearest 或 route_insert")
        if not isinstance(self.adaptive_search, bool) or not isinstance(self.joint_service, bool):
            raise ValueError("adaptive_search 和 joint_service 必须是布尔值")
        if self.shared_observations not in {'off', 'opportunistic', 'batch'}:
            raise ValueError('shared_observations 必须是 off/opportunistic/batch')
        if not isinstance(self.route_polish, bool):
            raise ValueError('route_polish 必须是布尔值')
        if not isinstance(self.adaptive_verify, bool) or not isinstance(self.route_multistart, bool):
            raise ValueError('adaptive_verify/route_multistart 必须是布尔值')
        for name in ('route_time_score', 'step_replan', 'safe_clear_point', 'scan_relocation', 'received_range_bound', 'recovery_verify', 'dynamic_second_station', 'planned_stop_reuse', 'stop_at_source_upper_bound', 'stop_search_when_all_sources_known'):
            if not isinstance(getattr(self,name),bool):
                raise ValueError(name+' 必须是布尔值')
        if self.scan_relocation and (self.adaptive_search or self.joint_service or self.shared_observations=='batch'):
            raise ValueError('scan_relocation必须与其他覆盖/批量实验分开使用')
        if self.scan_relocation and gap > RECEIVER_MIN_M+1e-9:
            raise ValueError('scan_relocation必须保留1000米覆盖')
        if not math.isfinite(self.bearing_rounding_slack_deg) or not 0<=self.bearing_rounding_slack_deg<=0.01:
            raise ValueError('bearing_rounding_slack_deg必须位于[0,0.01]')
        if (self.route_time_score or self.step_replan or self.safe_clear_point) and (self.joint_service or self.shared_observations=='batch'):
            raise ValueError('细化实验不与joint_service/batch组合')
        if (self.adaptive_verify or self.route_multistart) and self.joint_service:
            raise ValueError('前瞻实验不能与 joint_service 叠加')
        if (self.shared_observations != 'off' or self.route_polish) and self.joint_service:
            raise ValueError('共享观测实验不与未获收益的 joint_service 叠加')
        if (self.adaptive_search or self.joint_service) and gap > RECEIVER_MIN_M + 1e-9:
            raise ValueError("搜索实验必须保留1000米全覆盖的兜底布局")
        a = self.second_station_local_a_m
        b = self.second_station_local_b_m
        if not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < 0:
            raise ValueError("第二检测点局部坐标 a、b 必须是有限非负数")
        reach = self.second_station_reach_m
        far = self.second_station_far_m
        h = self.second_station_clearance_h
        t = math.tan(math.radians(self.second_station_min_angle_deg))
        feasible = (
            math.hypot(a, b) <= reach + 1e-6
            and math.hypot(a - far, b + h) <= reach + 1e-6
            and b >= t * a - 1e-6
            and b >= t * (far - a) + h - 1e-6
        )
        if not feasible:
            raise ValueError(
                "第二检测点局部坐标不满足当前候选区域硬约束："
                f"(a, b)=({a:g}, {b:g})"
            )

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]

    @classmethod
    def from_file(cls, path: str | Path) -> "StrategyConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 以下划线开头的键（如 "_comment"）视为说明文字，不参与覆盖也不报错
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        known = set(cls().__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"配置含未知字段: {sorted(unknown)}")
        return replace(cls(), **data)


def coverage_worst_gap(ring_radius: float, ring_count: int = 6) -> float:
    """区域内任一点到「原点 + ring_count 个环点」的最大距离。

    由对称性，最差点必出现在区域边界上、相邻两环点角平分线方向；
    沿该方向距离函数对半径是凸的，故只需比较两个端点 (r=1000 与 r=1800)。
    """
    phi = 0.5 * (2.0 * math.pi / ring_count)
    def dist_at(r: float) -> float:
        return math.sqrt(r * r + ring_radius ** 2 - 2.0 * r * ring_radius * math.cos(phi))
    return max(dist_at(RECEIVER_MIN_M), dist_at(ARENA_RADIUS_M))


def default_scan_points(cfg: StrategyConfig) -> List[tuple]:
    """七个必访扫描点：原点 + 正 n 边形环点。"""
    pts = [(0.0, 0.0)]
    n = cfg.scan_ring_count
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        pts.append((cfg.scan_ring_radius_m * math.cos(ang), cfg.scan_ring_radius_m * math.sin(ang)))
    return pts
