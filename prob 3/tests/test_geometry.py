"""几何内核的单元测试： correctness before performance."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3 import geometry  # noqa: E402


def point_in_convex_polygon(p, poly, tol=1e-7):
    """点在凸多边形内（CCW 或 CW 均可）的判定。"""
    if len(poly) < 3:
        return False
    sign = None
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cr) <= tol:
            continue
        s = cr > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


class TestDiskOuter(unittest.TestCase):
    """圆域外接多边形必须包含整个圆盘——这是"不丢源"的底线。"""

    def test_contains_disk(self):
        r = 1800.0
        poly = geometry.disk_outer_polygon(r, sides=64)
        rng = random.Random(7)
        for _ in range(2000):
            rad = r * math.sqrt(rng.random())
            th = rng.uniform(0, 2 * math.pi)
            p = (rad * math.cos(th), rad * math.sin(th))
            self.assertTrue(point_in_convex_polygon(p, poly),
                            f"圆内点 {p} 未被外接多边形包含")

    def test_semi_perimeter_is_small(self):
        r = 1800.0
        sides = 64
        poly = geometry.disk_outer_polygon(r, sides=sides)
        worst = max(math.hypot(*p) for p in poly)
        self.assertLessEqual(worst, r * (1.0 / math.cos(math.pi / sides)) + 1e-6)


class TestWedge(unittest.TestCase):
    def test_wedge_contains_true_source(self):
        """用真实方位角构造读数，编辑后的可行域必须包含源。"""
        rng = random.Random(11)
        for _ in range(300):
            src = (rng.uniform(-1500, 1500), rng.uniform(-1500, 1500))
            if math.hypot(*src) > 1800:
                continue
            station = (rng.uniform(-1800, 1800), rng.uniform(-1800, 1800))
            true = math.degrees(math.atan2(src[1] - station[1], src[0] - station[0])) % 360
            poly = geometry.disk_outer_polygon(1800.0)
            poly = geometry.clip_wedge(poly, station, true, 1.0)
            self.assertGreaterEqual(len(poly), 3)
            self.assertTrue(point_in_convex_polygon(src, poly))

    def test_wedge_excludes_backward(self):
        """背后的点必须被排除：楔形不是无向直线交点。"""
        station = (0.0, 0.0)
        poly = geometry.disk_outer_polygon(1800.0)
        poly = geometry.clip_wedge(poly, station, 0.0, 1.0)
        self.assertFalse(point_in_convex_polygon((-500.0, 0.0), poly))
        self.assertTrue(point_in_convex_polygon((500.0, 0.0), poly))


class TestLocate(unittest.TestCase):
    def two_station_locate(self):
        src = (300.0, 400.0)
        s1 = (0.0, 0.0)
        s2 = (800.0, 0.0)
        obs = []
        for st in (s1, s2):
            obs.append((st, math.degrees(math.atan2(src[1] - st[1], src[0] - st[0])) % 360))
        poly = geometry.locate(obs, 1.0, 1800.0)
        self.assertGreaterEqual(len(poly), 3)
        self.assertTrue(point_in_convex_polygon(src, poly))
        return poly

    def test_locate_contains_source(self):
        poly = self.two_station_locate()
        dia, _ = geometry.diameter(poly)
        self.assertLess(dia, 200.0, "两站交会出的区域应该在百米量级")

    def test_diameter_matches_bruteforce(self):
        rng = random.Random(3)
        for _ in range(200):
            pts = []
            while len(pts) < 12:
                ang = rng.uniform(0, 2 * math.pi)
                rad = rng.uniform(5, 300)
                pts.append((rad * math.cos(ang), rad * math.sin(ang)))
            hull = geometry.convex_hull(pts)
            if len(hull) < 3:
                continue
            hull = geometry.ensure_ccw(hull)
            # 输入必须是凸包的子集：凸包是这些点的外包，直径不小于点集直径
            d_fast, _ = geometry.diameter(hull)
            d_slow, _ = geometry.diameter_bruteforce(hull)
            self.assertAlmostEqual(d_fast, d_slow, places=6)

    def test_diameter_of_unordered_points(self):
        """散点先取凸包再求直径，结果应等于暴力解。"""
        rng = random.Random(5)
        pts = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(30)]
        hull = geometry.convex_hull(pts)
        d_fast, _ = geometry.diameter(hull)
        d_slow, _ = geometry.diameter_bruteforce(pts)
        self.assertAlmostEqual(d_fast, d_slow, places=6)

    def test_more_observations_shrink_region(self):
        """观测越多，可行域面积不应变大——这是逼近算法收敛的基础。"""
        src = (300.0, 400.0)
        stations = [(0.0, 0.0), (800.0, 0.0), (0.0, 900.0)]
        areas = []
        obs = []
        for st in stations:
            obs.append((st, math.degrees(math.atan2(src[1] - st[1], src[0] - st[0])) % 360))
            areas.append(geometry.polygon_area(geometry.locate(obs, 1.0, 1800.0)))
        self.assertGreater(areas[0], areas[1])
        self.assertGreater(areas[1], areas[2])

    def test_degenerate_line_intersection_is_preserved(self):
        """合法半平面交退化为线段时仍是有效可行域，不能返回空集。"""
        poly = geometry.locate(
            [((0.0, 0.0), 1.0), ((100.0, 0.0), 359.0)],
            1.0,
            1800.0,
        )
        self.assertGreaterEqual(len(poly), 1)
        self.assertTrue(all(abs(y) < 1e-6 for _, y in poly))
        self.assertTrue(any(x >= 500.0 for x, _ in poly))


class TestMetrics(unittest.TestCase):
    def test_centroid_of_triangle(self):
        tri = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)]
        cx, cy = geometry.centroid(tri)
        self.assertAlmostEqual(cx, 2.0, places=6)
        self.assertAlmostEqual(cy, 2.0, places=6)

    def test_max_distance_from_centroid(self):
        sq = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        c = geometry.centroid(sq)
        self.assertAlmostEqual(geometry.max_distance_from(c, sq), math.sqrt(2), places=6)

    def test_point_along_clamps(self):
        p = geometry.point_along((0.0, 0.0), (10.0, 0.0), 100.0)
        self.assertEqual(p, (10.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
