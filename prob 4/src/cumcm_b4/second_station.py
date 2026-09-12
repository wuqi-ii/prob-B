"""同向推进点：给"只有一次示向度"的频道生成第二个观测点。

问题 3 的"第二检测点"来自问题 2 的候选区域，取的是与首次示向方向成较大交会角的
垂直偏移点（(764, ±526)）。这个设计的前提是**全向源**：无论走到哪，只要在接收半径
内就能测到。

问题 4 引入了定向源，那个垂直偏移点很可能落在定向源的覆盖楔之外，从而测不到信号。
因此问题 4 把第二个观测点改为**沿示向度方向朝源推进**：

- 推进方向与"检测点指向源"的示向度一致，于是新观测点始终保持在覆盖楔内
  （源所在的楔是 180° 半平面，从楔内沿指向源的射线前进，整条线段都在楔内）；
- 距离源更近，仍然在接收半径内；
- 代价是两次示向度几乎共线、交会角很小，定位多边形会是一个细长楔形——这由
  逼近过程中的"边走边测"来补偿：越靠近源，±1° 误差对应的横向偏移越小。

本模块只提供坐标换算与推进点生成，不涉及候选区域约束（问题 2 结论在问题 4 不再复用）。
"""

from __future__ import annotations

import math
from typing import Tuple

from .config import StrategyConfig

Point = Tuple[float, float]


def approach_point(station: Point, bearing_deg: float,
                   step_m: float | None = None,
                   cfg: StrategyConfig | None = None) -> Point:
    """返回从 station 沿 bearing_deg 方向前进 step_m 米的点。

    step_m 缺省时取 cfg.first_approach_step_m。方位角沿用赛题约定：
    x 轴正向逆时针旋转到目标向量的角度（东为 0，北为 90）。
    """
    if step_m is None:
        if cfg is None:
            step_m = 300.0
        else:
            step_m = cfg.first_approach_step_m
    rad = math.radians(bearing_deg)
    dx = math.cos(rad) * step_m
    dy = math.sin(rad) * step_m
    return (station[0] + dx, station[1] + dy)
