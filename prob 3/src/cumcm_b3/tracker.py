"""每个频道的观测账本：把零散观测累积成"源在哪里"的凸集。

三种观测结果各有用途：
- direction：给出一条示向线，贡献一个前向楔形约束（收紧可行域）；
- near     ：源就在 <=5 m 处且在覆盖内，等价于"可以直接清除"，信息最强；
- no_signal：至少说明"该点收不到"；若该频道已在别处出现信号，
             则意味着该点距源超过源的有效接收半径（>=1000 m），
             这是可用但不便写成凸集的负面信息，此处仅作记录与诊断。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .config import ARENA_RADIUS_M, BEARING_ERROR_DEG, RECEIVER_MAX_M, StrategyConfig
from . import geometry

Point = Tuple[float, float]
Polygon = List[Point]


@dataclass
class ChannelTrack:
    channel: int
    observations: List[Tuple[Point, float]] = field(default_factory=list)
    no_signal_points: List[Point] = field(default_factory=list)
    near_points: List[Point] = field(default_factory=list)
    cleared: bool = False

    # ---- 观测登记 ----
    def add_direction(self, point: Point, bearing_deg: float) -> None:
        self.observations.append((point, float(bearing_deg) % 360.0))

    def add_no_signal(self, point: Point) -> None:
        self.no_signal_points.append(point)

    def add_near(self, point: Point) -> None:
        self.near_points.append(point)

    # ---- 推断 ----
    @property
    def known(self) -> bool:
        """该频道是否已被确认存在（至少拿到一次示向度或近距指示）。"""
        return bool(self.observations) or bool(self.near_points)

    def polygon(self, cfg: StrategyConfig) -> Polygon:
        """当前所有示向观测交出的可行域（已与地图圆域求交，保证有界）。"""
        if not self.observations:
            return []
        return geometry.locate(
            self.observations, BEARING_ERROR_DEG+cfg.bearing_rounding_slack_deg, ARENA_RADIUS_M,
            max_range_m=RECEIVER_MAX_M if cfg.received_range_bound else None
        )

    def geometry_summary(self, cfg: StrategyConfig) -> Dict[str, float]:
        """质心、直径、以及"质心到真源的最坏距离"。

        最后一项是逼近能否一次清除的判据：只要它小于光学定位半径，
        走到质心再调用清除就一定命中。
        """
        poly = self.polygon(cfg)
        if not poly:
            return {"diameter_m": float("inf"), "centroid": None,
                    "centroid_max_error_m": float("inf"), "vertices": 0}
        dia, _ = geometry.diameter(poly)
        c = geometry.centroid(poly)
        return {
            "diameter_m": dia,
            "centroid": c,
            "centroid_max_error_m": geometry.max_distance_from(c, poly),
            "area_m2": geometry.polygon_area(poly),
            "vertices": len(poly),
        }


class TrackBook:
    """全部 20 个频道的账本。"""

    def __init__(self, channels: Sequence[int]) -> None:
        self.tracks: Dict[int, ChannelTrack] = {ch: ChannelTrack(ch) for ch in channels}

    def __getitem__(self, channel: int) -> ChannelTrack:
        return self.tracks[channel]

    def known_channels(self) -> List[int]:
        return [ch for ch, t in self.tracks.items() if t.known and not t.cleared]

    def cleared_channels(self) -> List[int]:
        return [ch for ch, t in self.tracks.items() if t.cleared]

    def count_cleared(self) -> int:
        return len(self.cleared_channels())

    def apply_measure(self, channel: int, point: Point, payload: Dict) -> str:
        """登记一次检测结果，返回结果类型。"""
        track = self.tracks[channel]
        kind = payload.get("measure_result")
        if kind == "direction":
            track.add_direction(point, float(payload["svd_deg"]))
        elif kind == "near":
            track.add_near(point)
        else:
            track.add_no_signal(point)
        return str(kind)
