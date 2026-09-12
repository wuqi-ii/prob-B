import math
import numpy as np
import pytest
from scipy.spatial import ConvexHull
from bearing2.geometry import (clip_wedge, clip_halfplane, diameter, diameter_bruteforce,
                               local_to_global, global_to_local, bearing_interval, clip_disk_outer)
from bearing2.config import Config


@pytest.mark.parametrize("seed", range(12))
def test_calipers_against_exhaustive_vertex_pairs(seed):
    points = np.random.default_rng(seed).normal(size=(150, 2)) * [80, 20]
    poly = points[ConvexHull(points).vertices].tolist()
    assert diameter(poly)[0] == pytest.approx(diameter_bruteforce(poly)[0], abs=1e-8)


def test_parallel_edges_ties_degenerate_and_collinear():
    for p in [[], [(1,2)], [(0,0),(3,4)], [(0,0),(1,0),(2,0),(2,2),(0,2)],
              [(0,0),(1,0),(2,0),(3,0)], [(0,0),(4,0),(4,3),(0,3)]]:
        assert diameter(p)[0] == pytest.approx(diameter_bruteforce(p)[0])


def test_forward_wedge_excludes_backward_cone_and_handles_wrap():
    square = [(-2,-2),(2,-2),(2,2),(-2,2)]
    wedge = clip_wedge(square, (0,0), 0, math.radians(10))
    assert min(x for x,y in wedge) >= -1e-8
    for x,y in wedge:
        assert abs(y) <= x * math.tan(math.radians(10)) + 1e-8
    a = clip_wedge(square, (0,0), math.radians(359), math.radians(2))
    b = clip_wedge(square, (0,0), math.radians(-1), math.radians(2))
    assert np.asarray(a) == pytest.approx(np.asarray(b))


def test_known_two_wedge_intersection_is_diamond_not_rectangle():
    square = [(-10,-10),(10,-10),(10,10),(-10,10)]
    p = clip_wedge(square, (0,0), math.pi/4, math.radians(1))
    p = clip_wedge(p, (0,3), -math.pi/4, math.radians(1))
    assert len(p) == 4
    assert all(1.3 < x < 1.7 and 1.3 < y < 1.7 for x,y in p)
    for x,y in p:
        assert abs(math.atan2(y,x)-math.pi/4) <= math.radians(1)+1e-9
        assert abs(math.atan2(y-3,x)+math.pi/4) <= math.radians(1)+1e-9


def test_coordinate_round_trip():
    c = Config(first_station_m=(1250, -430), first_bearing_deg=359)
    p = np.array([[0,0],[750,500],[-30,40]])
    assert global_to_local(local_to_global(p,c),c) == pytest.approx(p)


def test_disk_polygon_is_outer_not_inner():
    sides=64
    poly=clip_disk_outer([(-2,-2),(2,-2),(2,2),(-2,2)], (0,0), 1, sides)
    assert max(math.hypot(*v) for v in poly) == pytest.approx(1/math.cos(math.pi/sides))
    # Every actual circle sample satisfies every supporting line.
    for angle in np.linspace(0,2*math.pi,301):
        p=np.array([math.cos(angle),math.sin(angle)])
        assert all(np.dot(p,[math.cos(2*math.pi*k/sides),math.sin(2*math.pi*k/sides)])<=1+1e-10 for k in range(sides))
