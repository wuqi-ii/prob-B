"""问题四的全局参数与策略参数。

与问题三的区别：目标区域内既有全向干扰源又有定向干扰源。定向干扰源只在
「定向方向两侧各 90°」的楔形覆盖范围内辐射信号，楔外不可测。因此：

1. 物理常量新增定向覆盖半角 DIRECTIONAL_HALF_ANGLE_DEG = 90；
2. 覆盖普查从「七点」升级为「包围网」（中心 + 内环 + 外环），外环布在目标区域
   之外，从而保证任意位置、任意朝向的定向源都至少被一个扫描点测到；
3. 定位阶段对「只有一次示向度」的频道不再去问题 2 的垂直候选点，而是沿示向度
   **朝源推进**（保持在覆盖楔内），并在信号消失时改用「光学环形清除」兜底
   （/clear 只与距离有关、与覆盖角无关）。

所有物理常量直接取自赛题附录，策略参数集中在 CONFIG 中并可由 configs/*.json 覆盖。
单位：米 / 秒 / 度。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict

# 注意：不在此处顶层导入 coverage（coverage 需要这里的物理常量，会形成循环导入）。
# validate() 内部按需惰性导入 coverage_worst_required_radius。


# --------------------------------------------------------------------------
# 赛题给定的物理常量（附录 1、2）——不可调整
# --------------------------------------------------------------------------
ARENA_RADIUS_M = 1800.0          # 目标区域半径
CHANNEL_COUNT = 20               # 频道编号 1..20
SOURCE_COUNT_MIN = 10            # 干扰源总数下界
SOURCE_COUNT_MAX = 16            # 干扰源总数上界
RECEIVER_MIN_M = 1000.0          # 有效接收半径下界（最坏情况）
RECEIVER_MAX_M = 1500.0          # 有效接收半径上界
DIRECTIONAL_HALF_ANGLE_DEG = 90.0  # 定向源覆盖半角（定向方向两侧各 90°）
BEARING_ERROR_DEG = 1.0          # 示向度误差半宽
DOG_SPEED_MPS = 5.0              # 移动速度
SWITCH_COST_S = 1.0              # 任意两频道之间的切换耗时
MEASURE_COST_S = 5.0             # 单次检测耗时
OPTICAL_COST_S = 3.0             # 光学精确定位耗时
LASER_COST_S = 2.0               # 激光清除耗时
CLEAR_RADIUS_M = 20.0            # 光学可精确定位的距离
NEAR_RADIUS_M = 5.0              # 信号过强、无法获得示向度的距离（且在覆盖楔内）
VIRTUAL_TIME_LIMIT_S = 100 * 3600.0   # 虚拟时间上限
PROGRAM_TIME_LIMIT_S = 20 * 60.0      # 程序运行时间上限
TEST_WINDOW_S = 25 * 60.0             # 测试窗口

# 成功率/耗时换算：清除命中耗时 3+2=5，未命中 3
CLEAR_HIT_COST_S = OPTICAL_COST_S + LASER_COST_S
CLEAR_MISS_COST_S = OPTICAL_COST_S


@dataclass(frozen=True)
class StrategyConfig:
    """策略可调参数。默认值对应优化后的 23 点包围网（中心 + 内环 8 + 外环 14）。"""

    # ---- 包围网覆盖普查（问题 4 的核心改动）----
    # 中心点恒为原点（机器狗出发点）。内环 + 外环共同构成"包围网"：
    # 对任意位置、任意朝向的源，都至少存在一个扫描点既在其 180° 覆盖楔内、
    # 又在 1000 m 有效接收半径内。外环半径必须大于目标区域半径 1800，
    # 否则边界处、楔朝外的定向源会漏检（内接多边形盖不住圆盘边缘）。
    scan_inner_radius_m: float = 980.0    # 内环到原点距离
    scan_inner_count: int = 8             # 内环点数
    scan_outer_radius_m: float = 1848.0   # 外环到原点距离（> 1800，目标区域外）
    scan_outer_count: int = 14            # 外环点数（与内环合计 23 点）
    # 盲区补点：均匀双环受整数点数约束，减点后会在少数扇区留下盲区。
    # 这些自由点用于精准填补盲区，避免为覆盖一个小扇区而补整个环。
    # 默认空：不改动现有 23 点网的行为。
    scan_extra_points: tuple = ()
    # 显式声明「本布局的 worst 超过 1000 m 安全线，我接受漏检风险」。
    # 用于激进的稀疏网（省约 5.8% 时间，代价是全清率约 99.83%）。
    # 默认 False：validate() 会拒绝任何覆盖不足的布局。
    allow_unsafe_coverage: bool = False
    disk_sides: int = 64                  # 圆域外接多边形的边数

    # ---- 同向推进点（问题 4 取代问题 2 的垂直第二检测点）----
    # 只有一次示向度的频道：沿示向度方向朝源推进 first_approach_step_m 米再测。
    # 推进方向与"检测点指向源"一致，因此始终保持在定向源的覆盖楔内。
    first_approach_step_m: float = 300.0  # 单示向度频道的首次推进步长

    # ---- 逼近与清除 ----
    approach_step_ratio: float = 0.85     # 每次逼近前进到剩余估计距离的比例
    approach_min_step_m: float = 25.0     # 最小逼近步长，防止原地抖动
    approach_max_step_m: float = 400.0    # 逼近步长上限，防止越过源（定向源楔外不可测）
    # True 时后续逼近也沿「最近一次示向度」方向前进（与首次逼近一致），保证整段
    # 逼近线都落在定向源 180° 覆盖楔内。实测（150 例配对）反更差 +2.18%：
    # 沿 bearing 不自适应、误差累积，而朝 anchor 每步重算可行域中心收敛更快。
    # 故默认 False，仅作历史记录。
    approach_along_bearing: bool = False
    approach_max_iterations: int = 60     # 单个目标的逼近迭代上限
    clear_safety_margin_m: float = 2.0    # 判定「质心可直接清除」的余量
    # True 时清除站位/判据改用可行域顶点的最小覆盖圆圆心与半径（minimax 最优），
    # 替代质心 + 质心到最远顶点距离。700 例配对验证零漏检、−0.62%~−0.64% 显著收益，
    # 故设为默认。对非中心对称的可行域把最坏距离压小约 10%，更早满足一次命中判据。
    use_mec_clear: bool = True
    bisect_tolerance_m: float = 12.0      # 越过源后二分恢复的收敛阈值
    # 越界时机器狗位于无信号端 hi。取 lo + f(hi-lo)，f>0.5 可减少首次回走距离；
    # 始终保留 lo有信号/hi无信号 的括号，故不改变收敛与安全性。300个混合案例
    # 配对验证 f=0.85 全清且平均省 107.8 s（-1.56%）。
    bisect_fraction: float = 0.85
    # True 时越过源（no_signal）后不做二分，直接以越界点为圆心环形兜底清除。
    # 实测（150 例配对）省 −0.83% 但漏检 31/150（miss 8.61→17.63）：二分的
    # "边测边找源"价值不可省。故默认 False，仅作历史记录。
    skip_bisect_on_overshoot: bool = False
    # 二分的「有信号/无信号」括号长度收敛到该阈值即停，取中点清除。
    # 因真源必落在括号内、且 /clear 半径 20 m，阈值放大到 2*(20-横向误差) 仍安全，
    # 但放大能省下最后一次往返。默认 12.0 与历史行为一致。

    # ---- 定向源的光学环形兜底（/clear 只看距离、不看覆盖角）----
    ring_clear_radius_m: float = 15.0     # 环形清除的半径
    ring_clear_count: int = 6             # 环形清除的点数（+ 中心共 7 个）
    ring_clear_enabled: bool = True       # 是否启用环形兜底清除

    # ---- 清除后的闭环确认（"确保全部清除"的可验证性）----
    # /clear 的 success 已是权威成功结果；成功后原地复测只会固定增加5~6秒/源，
    # 不提供新的决策信息。100个混合案例关闭后仍全清，平均每局省72.2秒。
    post_clear_verify: bool = False
    max_clear_attempts_per_channel: int = 5  # 单频道清除尝试上限

    # ---- 决策 ----
    opportunistic_scan: bool = True        # 是否启用顺路/顺频道捎带检测
    route_replan_every_step: bool = True   # 每完成一个任务点后是否重算路径
    scan_all_channels_at_scan_points: bool = True  # 在必访扫描点是否扫全部未确认频道
    # 仅跳过两类可严格证明无收益的检测：定位区域已小到可一次清除，或整个
    # 定位区域距当前扫描点超过1500 m最大接收半径。关闭可复现优化前行为。
    certified_scan_skip: bool = True
    # 同一扫描点批量测量时优先复用后端当前频道，通常每站少一次1 s切换。
    scan_current_channel_first: bool = True
    # 已成功清除题面上限16个不同频道时，剩余频道必不存在，可安全跳过未访问扫描点。
    # 不改变普通案例路线；100例中16源局5/12提前结束，总体平均-0.31%。
    early_stop_at_max_sources: bool = True
    # 轻微延后清除可减少扫描路线被远处目标打断；当前组合初筛以 -100 m 最优。
    task_bias_clear_m: float = -100.0
    # 负值表示适度延后远距离验证任务，减少包围网遍历被来回打断。
    # 加入「终局重试」（扫描网走完后解冻冻结频道重试）之后，-500 ~ -2000 m 是平台区；
    # 配对 300 案例以 -700 m 最优（全清，相对 -300 再省约 5%），故设为默认。
    # 注意：-400 m 起会漏清是「加终局重试之前」的旧结论，已作废。见 RESULTS.md 6.3。
    task_bias_verify_m: float = -700.0
    opportunistic_radius_m: float = 600.0  # 顺捎检测的作用半径
    opportunistic_max_per_stop: int = 3    # 单次停留最多顺捎几个频道
    opportunistic_min_diameter_m: float = 40.0  # 区域已足够小就不必顺捎

    # ---- 运行 ----
    base_url: str = "http://127.0.0.1:2026"
    robot_id: str = "202623001124"
    request_timeout_s: float = 10.0
    max_requests: int = 8000              # 请求数硬上限（包围网点数多，放宽）
    virtual_time_budget_s: float = float("inf")  # 虚拟时间软预算

    def __post_init__(self) -> None:
        """规范化盲区补点：JSON 往返会把 tuple 变成 list。

        统一转回 (x, y) 元组，保证 strategy 里 scan_points.index(point)
        能按坐标匹配，也保证配置对象的相等比较稳定。
        """
        if self.scan_extra_points:
            norm = []
            for p in self.scan_extra_points:
                x, y = float(p[0]), float(p[1])
                if math.hypot(x, y) > ARENA_RADIUS_M:
                    raise ValueError("补点必须位于目标区域内")
                norm.append((x, y))
            object.__setattr__(self, "scan_extra_points", tuple(norm))

    def validate(self) -> None:
        """参数可行性检查：包围网必须覆盖，几何参数必须在合理区间。"""
        if not 0 < self.scan_inner_radius_m <= ARENA_RADIUS_M:
            raise ValueError("scan_inner_radius_m 必须位于 (0, 1800]")
        if not ARENA_RADIUS_M < self.scan_outer_radius_m <= 2000.0:
            raise ValueError("scan_outer_radius_m 必须位于 (1800, 2000]（目标区域外）")
        if self.scan_inner_count < 3:
            raise ValueError("scan_inner_count 至少为 3")
        if self.scan_outer_count < 6:
            raise ValueError("scan_outer_count 至少为 6")
        from .coverage import coverage_worst_required_radius  # 惰性导入，避免循环依赖
        worst = coverage_worst_required_radius(self, step_m=40.0, ang_step_deg=3.0)
        if worst > RECEIVER_MIN_M + 1e-9 and not self.allow_unsafe_coverage:
            raise ValueError(
                "包围网不能按最坏接收半径覆盖目标区域："
                f"需要接收半径 {worst:.3f} m > {RECEIVER_MIN_M:.3f} m"
                "（若确知风险，可设 allow_unsafe_coverage=True）"
            )
        if not 0 < self.approach_step_ratio < 1:
            raise ValueError("approach_step_ratio 必须位于 (0, 1)")
        if not self.approach_min_step_m <= self.approach_max_step_m:
            raise ValueError("approach_min_step_m 必须 <= approach_max_step_m")
        if not 0 <= self.clear_safety_margin_m < CLEAR_RADIUS_M:
            raise ValueError("clear_safety_margin_m 必须在 [0, 20)")
        if not 0 < self.first_approach_step_m <= RECEIVER_MAX_M:
            raise ValueError("first_approach_step_m 必须位于 (0, 1500]")
        if not 0 < self.bisect_fraction < 1:
            raise ValueError("bisect_fraction 必须位于 (0, 1)")
        if self.ring_clear_enabled and self.ring_clear_count < 1:
            raise ValueError("ring_clear_count 至少为 1")

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]

    @classmethod
    def from_file(cls, path: str | Path) -> "StrategyConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        known = set(cls().__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"配置含未知字段: {sorted(unknown)}")
        return replace(cls(), **data)
