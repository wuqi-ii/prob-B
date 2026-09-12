"""Geometry and data-integrity tests for the refined scientific figures."""
import json
from pathlib import Path

import numpy as np
import pytest

from bearing2.config import Config
from bearing2.publication_plots import region_outline, interpolate_lobe
from bearing2.region import feasible_mask
from bearing2.optimize import cost


@pytest.mark.parametrize("side", [1, -1])
def test_analytic_outline_is_feasible(side):
    outline = region_outline(Config(), side)
    assert outline.shape == (1200, 2)
    assert np.all(feasible_mask(outline, Config(), tol=1e-6))
    assert np.all(outline[:, 1] * side > 0)
    np.testing.assert_allclose(outline[0], outline[-1], atol=1e-7)
    np.testing.assert_allclose(outline[599], outline[600], atol=1e-7)


@pytest.mark.parametrize("side,expected", [(1, 10.), (-1, 90.)])
def test_interpolation_masks_exterior_and_keeps_lobes_separate(side, expected):
    cfg = Config()
    upper, lower = region_outline(cfg, 1, 100), region_outline(cfg, -1, 100)
    points = np.vstack([upper, lower])
    values = np.r_[np.full(len(upper), 10.), np.full(len(lower), 90.)]
    gx, gy, z = interpolate_lobe(points, values, cfg, side, resolution=100)
    outside = ~feasible_mask(np.column_stack([gx.ravel(), gy.ravel()]), cfg, tol=1e-8)
    assert np.all(np.isnan(z.ravel()[outside]))
    assert np.isfinite(z).sum() > 1000
    np.testing.assert_allclose(z[np.isfinite(z)], expected, atol=1e-10)


@pytest.mark.parametrize("name", ["default", "boundary"])
def test_saved_surface_matches_objective_and_selected_point(name):
    root = Path(__file__).resolve().parents[1]
    folder = root / "outputs" / name
    table_path = folder / "visual_review" / "cost_surface_samples.csv"
    if not table_path.exists():
        pytest.skip("Generate refined figures to validate cached cost samples.")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    cfg = Config(**summary["config"])
    rows = np.loadtxt(table_path, delimiter=",", skiprows=1)
    assert np.all(feasible_mask(rows[:, :2], cfg, tol=1e-6))
    expected = cost(rows[:, 2], rows[:, 5], summary["lambda"]["base_weight"], cfg)
    np.testing.assert_allclose(rows[:, 6], expected, atol=1e-6, rtol=0)
    point = np.asarray(summary["final"]["point_local_m"])
    i = np.argmin(np.linalg.norm(rows[:, :2] - point, axis=1))
    np.testing.assert_allclose(rows[i, :2], point, atol=1e-7)
    assert rows[i, 6] == pytest.approx(summary["final"]["cost_s"], abs=1e-6)
