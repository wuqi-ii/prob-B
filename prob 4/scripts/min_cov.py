"""独立复现：不含任何工程代码，纯标准库，重复跑覆盖扫描。

用于判定 TypeError('list_iterator' * float) 是解释器层面的问题还是工程代码问题。
"""
from __future__ import annotations

import math
import sys

print("python", sys.version.split()[0], flush=True)


def scan_points(n1: int, r1: float, n2: int, r2: float):
    pts = [(0.0, 0.0)]
    for k in range(n1):
        a = 2.0 * math.pi * k / n1
        pts.append((r1 * math.cos(a), r1 * math.sin(a)))
    ph = math.pi / n2
    for k in range(n2):
        a = ph + 2.0 * math.pi * k / n2
        pts.append((r2 * math.cos(a), r2 * math.sin(a)))
    return pts


def required_radius_at_position(source, points):
    gx, gy = source
    polar = []
    for px, py in points:
        dx, dy = px - gx, py - gy
        d = math.hypot(dx, dy)
        if d <= 1e-12:
            return 0.0
        polar.append((d, math.atan2(dy, dx) % (2.0 * math.pi)))
    polar.sort(key=lambda it: it[0])
    active = []
    idx = 0
    while idx < len(polar):
        d = polar[idx][0]
        while idx < len(polar) and abs(polar[idx][0] - d) <= 1e-9:
            active.append(polar[idx][1])
            idx += 1
        if len(active) < 2:
            continue
        angles = sorted(active)
        gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(angles[0] + 2.0 * math.pi - angles[-1])
        if max(gaps) <= math.pi + 1e-12:
            return d
    return math.inf


def sweep(pts, step=1.0, rmax=1800.0):
    worst = 0.0
    n = 0
    r = 0.0
    while r <= rmax + 1e-9:
        circ = 2.0 * math.pi * max(r, 1.0)
        cnt = max(8, int(circ / step))
        for k in range(cnt):
            th = 2.0 * math.pi * k / cnt
            gx, gy = r * math.cos(th), r * math.sin(th)
            req = required_radius_at_position((gx, gy), pts)
            n += 1
            if req > worst:
                worst = req
        r += step
    return n, worst


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    step = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    n1, r1, n2, r2 = 9, 980.0, 13, 1854.0
    pts = scan_points(n1, r1, n2, r2)
    for rep in range(reps):
        try:
            n, worst = sweep(pts, step=step)
        except Exception as exc:
            print(f"CRASH at rep {rep}: {type(exc).__name__}: {exc}", flush=True)
            print("  math module is:", math, flush=True)
            print("  type(math.pi) =", type(math.pi), flush=True)
            import traceback
            traceback.print_exc()
            return 1
        print(f"rep {rep}: {n} positions, worst={worst:.3f}", flush=True)
    print("no crash", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
