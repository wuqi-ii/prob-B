import math
from dataclasses import replace

import numpy as np
import pytest

from bearing2.config import Config
from bearing2.geometry import clip_wedge, diameter
from bearing2.metrics import PointEvaluator, cvar, var_alpha
from bearing2.optimize import (COMP, DIST, evaluate_grid, normalized_lambda,
                               pareto_mask, refine_point, select_for_weight, station_grid)
from bearing2.region import source_polygon
from bearing2.scenarios import radial_uniformity_check, sample_scenarios


def test_scenario_sample_is_area_uniform_and_inside_region():
    config = Config(n_scenarios=2000)
    scenarios = sample_scenarios(config)
    assert len(scenarios) == 4000  # 2000 draws + valid mirrors
    deviation = radial_uniformity_check(scenarios, config, bins=8)["max_relative_deviation"]
    assert deviation < 0.25  # equal-area bins must be nearly equally populated
    global_points = np.asarray(
        [[0.0, 0.0]], dtype=float)  # placeholder to keep imports meaningful
    radius = np.linalg.norm(scenarios.sources, axis=1)
    assert radius.min() >= config.source_min_radius_m - 1e-9
    assert radius.max() <= config.max_reception_m + 1e-9
    wedge = np.abs(scenarios.sources[:, 1]) <= scenarios.sources[:, 0] * math.tan(config.eps) + 1e-9
    assert wedge.all()


def test_mirror_positions_are_valid_and_symmetric():
    from bearing2.geometry import local_to_global
    from bearing2.scenarios import Scenarios
    # Without a map boundary the sector is symmetric: every mirror is kept.
    config = Config(n_scenarios=150, use_map_boundary=False)
    scenarios = sample_scenarios(config)
    n = config.n_scenarios
    assert len(scenarios) == 2 * n
    assert scenarios.sources[n:] == pytest.approx(scenarios.sources[:n] * [1.0, -1.0])
    assert scenarios.errors[n:] == pytest.approx(-scenarios.errors[:n])
    # With the map boundary every kept position (original or mirrored) is inside.
    config = Config(first_station_m=(1200.0, 0.0), first_bearing_deg=20.0, n_scenarios=300)
    scenarios = sample_scenarios(config)
    far = np.linalg.norm(local_to_global(scenarios.sources, config) - np.asarray(config.map_center_m), axis=1)
    assert far.max() <= config.map_radius_m + 1e-9


def test_antithetic_sampling_makes_lobes_symmetric():
    config = Config(n_scenarios=300)
    poly = source_polygon(config)
    scenarios = sample_scenarios(config)
    evaluator = PointEvaluator(poly, scenarios, config)
    up = evaluator((750.0, 500.0)).composite_m
    down = evaluator((750.0, -500.0)).composite_m
    assert up == pytest.approx(down, abs=1e-9)


def test_cvar_matches_rockafellar_minimisation():
    rng = np.random.default_rng(7)
    values = rng.uniform(20.0, 200.0, 37)
    weights = rng.uniform(0.2, 1.8, 37)
    weights /= weights.sum()
    alpha = 0.82
    etas = np.linspace(values.min(), values.max(), 200001)
    objective = etas + np.sum(weights[:, None] * np.maximum(values[:, None] - etas[None, :], 0.0), axis=0) / (1 - alpha)
    assert cvar(values, weights, alpha) == pytest.approx(objective.min(), abs=1e-4)


def test_cvar_fractional_tail_uses_tail_mass_not_rounding():
    values = np.arange(1.0, 11.0)
    weights = np.full(10, 0.1)
    # alpha=0.85 -> tail mass 0.15 = all of 10 plus half of 9.
    assert cvar(values, weights, 0.85) == pytest.approx((10.0 * 0.1 + 9.0 * 0.05) / 0.15)
    # The upper tail holds mass 0.2 at value 9 and 0.1 at 10, so the boundary
    # value of the alpha=0.85 tail is 9, not 10.
    assert var_alpha(values, weights, 0.85) == pytest.approx(9.0)


def test_diameter_uses_only_the_reading_not_the_true_source():
    """Two sources producing the same reading must give the same diameter."""
    from bearing2.scenarios import Scenarios
    config = Config(n_scenarios=32)
    poly = source_polygon(config)
    point = np.array([750.0, 500.0])
    sources = np.array([[750.0, 5.0], [750.0, -8.0]])  # same ray from P, different ranges
    errors = np.array([0.0, 0.0])
    weights = np.array([0.5, 0.5])
    scenarios = Scenarios(sources=sources, errors=errors, weights=weights,
                          seed=1, attempts=2, rejected_by_map=0)
    diameters, empty = PointEvaluator(poly, scenarios, config).diameters(point)
    assert empty == 0
    assert diameters[0] == pytest.approx(diameters[1])


def test_pareto_and_lambda_degenerate_fallback():
    rows = np.array([
        [0.0, 0.0, 100.0, 60.0, 70.0, 80.0, 65.0],   # nearest AND best composite
        [10.0, 0.0, 110.0, 66.0, 75.0, 90.0, 70.5],
        [20.0, 0.0, 120.0, 72.0, 80.0, 95.0, 76.0],
    ])
    mask = pareto_mask(rows)
    assert mask.tolist() == [True, False, False]
    info = normalized_lambda(rows, mask)
    assert info["lambda0"] == 0.0 and info["degenerate_distance"] and info["degenerate_composite"]
    fallback = normalized_lambda(rows)
    assert fallback["lambda0"] == pytest.approx(20.0 / 11.0)


def test_selection_stays_feasible_and_refinement_does_not_worsen():
    config = Config(n_scenarios=80, grid_step_m=40.0, refine_budget=40)
    poly = source_polygon(config)
    scenarios = sample_scenarios(config)
    rows, evaluator = evaluate_grid(poly, scenarios, config)
    weight = 8.0
    record = select_for_weight(rows, evaluator, config, weight)
    point = np.asarray(record["point_local_m"])
    assert record["distance_m"] <= config.guaranteed_reception_m + 1e-6
    assert record["max_diameter_m"] <= 168.835 + 1e-6
    refined, value, info = refine_point(point, evaluator, config, weight)
    assert value <= record["cost_s"] + 1e-9
    assert info["evaluations"] <= config.refine_budget


def test_grid_end_to_end_smoke():
    config = Config(n_scenarios=60, grid_step_m=50.0)
    poly = source_polygon(config)
    scenarios = sample_scenarios(config)
    rows, _ = evaluate_grid(poly, scenarios, config)
    assert rows.shape[1] == 7 and len(rows) > 0
    assert np.all(rows[:, DIST] <= config.guaranteed_reception_m + 1e-9)
    assert np.all(rows[:, COMP] >= rows[:, DIST] * 0.0)
