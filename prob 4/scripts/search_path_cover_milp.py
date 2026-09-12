"""用离散场景集合覆盖 MILP 搜索非同心安全扫描网候选。

先在位置/朝向网格上选最少检测点，再用连续朝向判据和更细位置网格复核。
离散 MILP 只负责产生候选，任何候选在通过精核前都不会进入默认策略。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.coverage import required_radius_at_position  # noqa: E402

ARENA = 1800.0


def polar_grid(radius_max: float, radial_step: float, arc_step: float):
    points = [(0.0, 0.0)]
    r = radial_step
    while r <= radius_max + 1e-9:
        n = max(8, math.ceil(2.0 * math.pi * r / arc_step))
        for k in range(n):
            a = 2.0 * math.pi * k / n
            points.append((r * math.cos(a), r * math.sin(a)))
        r += radial_step
    return points


def candidates(angle_step_deg: float):
    pts = [(0.0, 0.0)]
    for r in (450.0, 750.0, 980.0, 1200.0, 1500.0, 1800.0, 1850.0, 1900.0, 1950.0):
        n = round(360.0 / angle_step_deg)
        for k in range(n):
            a = 2.0 * math.pi * k / n
            pts.append((r * math.cos(a), r * math.sin(a)))
    return pts


def two_opt_open(points):
    remaining = list(range(1, len(points)))
    order = [0]
    while remaining:
        j = min(remaining, key=lambda i: math.dist(points[order[-1]], points[i]))
        order.append(j)
        remaining.remove(j)

    def length(seq):
        return sum(math.dist(points[seq[i]], points[seq[i + 1]]) for i in range(len(seq) - 1))

    best = length(order)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                trial = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                value = length(trial)
                if value < best - 1e-7:
                    order, best, changed = trial, value, True
    return best, order


def sampled_worst(points, step):
    worst = 0.0
    bad = None
    for source in polar_grid(ARENA, step, step):
        value = required_radius_at_position(source, points)
        if value > worst:
            worst, bad = value, source
    return worst, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-angle", type=float, default=15.0)
    ap.add_argument("--source-step", type=float, default=180.0)
    ap.add_argument("--direction-step", type=float, default=15.0)
    ap.add_argument("--radius", type=float, default=990.0,
                    help="MILP使用的保守接收半径")
    ap.add_argument("--time-limit", type=float, default=120.0)
    args = ap.parse_args()

    cand = candidates(args.candidate_angle)
    sources = polar_grid(ARENA, args.source_step, args.source_step)
    directions = [
        (math.cos(math.radians(a)), math.sin(math.radians(a)))
        for a in np.arange(0.0, 360.0, args.direction_step)
    ]
    rows, cols = [], []
    rhs = []
    row = 0
    for gx, gy in sources:
        for ux, uy in directions:
            for j, (px, py) in enumerate(cand):
                dx, dy = px - gx, py - gy
                if dx * dx + dy * dy <= args.radius * args.radius + 1e-9 and dx * ux + dy * uy >= -1e-9:
                    rows.append(row)
                    cols.append(j)
            rhs.append(1.0)
            row += 1
    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(row, len(cand))).tocsr()
    if np.any(np.diff(matrix.indptr) == 0):
        print("候选点网格度不足：至少一个场景没有可覆盖点")
        return 2

    # 点数为主目标；极小的半径代价用于在同点数解中偏好较短、较内侧的候选。
    cost = np.ones(len(cand)) + 1e-5 * np.array([math.hypot(x, y) for x, y in cand])
    lower = np.zeros(len(cand))
    upper = np.ones(len(cand))
    lower[0] = upper[0] = 1.0
    result = milp(
        cost,
        integrality=np.ones(len(cand)),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, np.asarray(rhs), np.full(row, np.inf)),
        options={"time_limit": args.time_limit, "mip_rel_gap": 0.0},
    )
    print("status", result.status, result.message)
    if result.x is None:
        return 1
    chosen = [p for p, value in zip(cand, result.x) if value > 0.5]
    route, order = two_opt_open(chosen)
    worst40, bad40 = sampled_worst(chosen, 40.0)
    print(f"候选数={len(cand)} 场景数={row} 选中={len(chosen)}")
    print(f"open_2opt={route:.1f} m  step40_worst={worst40:.3f} @ {bad40}")
    print("points=[")
    for i in order:
        print(f"  ({chosen[i][0]:.6f}, {chosen[i][1]:.6f}),")
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
