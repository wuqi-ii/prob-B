"""Regressions for defects found during the independent code review.

Each test corresponds to a specific finding in PLOT_REVIEW.md so that the fix
cannot silently regress.
"""
import math
from dataclasses import replace

import numpy as np
import pytest

from bearing2.config import Config
from bearing2.geometry import offset_outward, signed_area, simplify_polygon
from bearing2.optimize import COMP, DIST, normalized_lambda, station_grid
from bearing2.region import boundary_candidates, feasible_mask
from bearing2.scenarios import sample_scenarios
from bearing2.geometry import local_to_global


def test_offset_outward_actually_enlarges():
    """Clipping with shifted half-planes cannot grow a polygon; the miter must."""
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    grown = offset_outward(square, 0.1)
    assert signed_area(grown) == pytest.approx(1.44, rel=1e-9)
    assert len(grown) == 4
    # Every original vertex must stay inside the grown polygon.
    for px, py in square:
        for i in range(len(grown)):
            ax, ay = grown[i]
            bx, by = grown[(i + 1) % len(grown)]
            assert (bx - ax) * (py - ay) - (by - ay) * (px - ax) >= -1e-9


def test_simplify_never_loses_the_input_polygon():
    r = 1000.0
    circle = [(r * math.cos(t), r * math.sin(t))
              for t in np.linspace(0.0, 2 * math.pi, 361)[:-1]]
    simplified = simplify_polygon(circle, 0.5)
    assert len(simplified) < len(circle)
    # A coarse but valid enclosure: sampled circle points stay inside.
    for t in np.linspace(0.0, 2 * math.pi, 721):
        px, py = r * math.cos(t), r * math.sin(t)
        assert all(
            (simplified[(i + 1) % len(simplified)][0] - simplified[i][0]) * (py - simplified[i][1])
            - (simplified[(i + 1) % len(simplified)][1] - simplified[i][1]) * (px - simplified[i][0])
            >= -0.6
            for i in range(len(simplified)))


def test_mirror_multiplicity_weighting_keeps_the_sample_area_uniform():
    """A base draw whose mirror is also in K would otherwise be double counted."""
    config = Config(first_station_m=(1200.0, 0.0), first_bearing_deg=20.0, n_scenarios=2000)
    weighted = sample_scenarios(config)
    plain = sample_scenarios(replace(config, antithetic_mirror=False))

    def mirror_outside(points):
        mirrored = np.asarray(points) * np.array([1.0, -1.0])
        far = np.linalg.norm(local_to_global(mirrored, config)
                             - np.asarray(config.map_center_m), axis=1)
        return far > config.map_radius_m

    def weighted_fraction(scenarios):
        flags = mirror_outside(scenarios.sources).astype(float)
        return float(np.sum(scenarios.weights * flags) / np.sum(scenarios.weights))

    # The plain sample estimates the same quantity without any weighting.
    plain_share = float(mirror_outside(plain.sources).mean())
    assert weighted_fraction(weighted) == pytest.approx(plain_share, abs=0.01)
    assert not np.allclose(weighted.weights, weighted.weights[0])


def test_symmetric_case_keeps_uniform_weights_and_exact_lobe_symmetry():
    config = Config(n_scenarios=200)
    scenarios = sample_scenarios(config)
    assert np.allclose(scenarios.weights, 1.0 / len(scenarios))
    from bearing2.metrics import PointEvaluator
    from bearing2.region import source_polygon
    evaluator = PointEvaluator(source_polygon(config), scenarios, config)
    assert evaluator((750.0, 500.0)).composite_m == pytest.approx(
        evaluator((750.0, -500.0)).composite_m, abs=1e-9)


def test_lambda_ranking_hint_matches_the_degenerate_semantics():
    both_flat = np.array([[0.0, 0.0, 100.0, 50.0, 60.0, 70.0, 55.0],
                          [0.0, 0.0, 100.0, 50.0, 60.0, 70.0, 55.0]])
    info = normalized_lambda(both_flat)
    assert info["ranking"] == "distance" and info["lambda0"] == 0.0

    flat_composite = np.array([[0.0, 0.0, 100.0, 50.0, 60.0, 70.0, 55.0],
                               [10.0, 0.0, 110.0, 50.0, 60.0, 70.0, 55.0]])
    info = normalized_lambda(flat_composite)
    assert info["ranking"] == "distance" and info["lambda0"] == 0.0

    flat_distance = np.array([[0.0, 0.0, 100.0, 50.0, 60.0, 70.0, 55.0],
                              [0.0, 0.0, 100.0, 62.0, 70.0, 80.0, 66.0]])
    info = normalized_lambda(flat_distance)
    assert info["ranking"] == "composite" and math.isinf(info["lambda0"])

    mixed = np.array([[0.0, 0.0, 100.0, 50.0, 60.0, 70.0, 55.0],
                      [10.0, 0.0, 110.0, 62.0, 70.0, 80.0, 66.0]])
    info = normalized_lambda(mixed)
    assert info["ranking"] == "normalized" and info["lambda0"] == pytest.approx(10.0 / 11.0)


def test_boundary_candidates_reach_the_region_corners():
    config = Config()
    points = boundary_candidates(config)
    assert len(points) > 100
    assert feasible_mask(points, config, tol=1e-6).all()
    # The corner where the far crossing line meets the far reception circle.
    corner = np.array([666.8144759063354, 526.8109671640481])
    assert np.linalg.norm(points - corner, axis=1).min() < 1.0
    grid = station_grid(config)
    assert np.linalg.norm(grid - corner, axis=1).min() < 1.0
    assert np.all(np.linalg.norm(grid, axis=1) <= config.guaranteed_reception_m + 1e-9)
