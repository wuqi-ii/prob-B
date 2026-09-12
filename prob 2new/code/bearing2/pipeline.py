"""End-to-end solution for question 2, with audits, exports and optional plots.

Order of operations follows the method document:
fixed candidate region -> shared scenarios -> per-point metrics -> Pareto
normalisation of lambda -> weighted selection -> audits and sensitivity checks.
"""
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import optimize as opt
from .config import Config
from .geometry import clip_wedge, diameter, local_to_global, simplify_polygon
from .metrics import PointEvaluator
from .region import physical_checks, source_polygon, triangle, uniform_diameter_bound
from .risk import certify_risk
from .scenarios import sample_scenarios


def build_source_region(config):
    """Full-resolution source polygon plus the simplified copy used for speed."""
    full = source_polygon(config)
    used = simplify_polygon(full, config.simplify_tolerance_m)
    info = {
        "vertices_full": len(full),
        "vertices_used": len(used),
        "diameter_full_m": diameter(full)[0],
        "diameter_used_m": diameter(used)[0],
        "diameter_change_m": diameter(used)[0] - diameter(full)[0],
        "simplify_tolerance_m": config.simplify_tolerance_m,
        "note": "simplification widens the polygon outward, so it stays conservative",
    }
    return full, used, info


def audit_point(point, poly, config):
    """Hard constraints plus the continuous worst-reading audit."""
    checks = physical_checks(point, poly, config)
    certificate = certify_risk(point, poly, config)
    checks["worst_case_audit"] = certificate.to_dict()
    checks["uniform_diameter_upper_bound_m"] = uniform_diameter_bound(config)
    checks["audit_within_uniform_bound"] = bool(
        certificate.upper_m <= uniform_diameter_bound(config) + config.certificate_tolerance_m)
    return checks


def _independent_check(config, poly, point, seed_offset, count_factor):
    """Fresh scenarios only: a stability check, not an unbiased final evaluation."""
    cfg = replace(config, scenario_seed=config.seed + seed_offset,
                  n_scenarios=int(config.n_scenarios * count_factor))
    fresh = sample_scenarios(cfg)
    metrics = PointEvaluator(poly, fresh, cfg)(point)
    return {"scenarios": len(fresh), "seed": cfg.seed, "mean_diameter_m": metrics.mean_diameter_m,
            "cvar_diameter_m": metrics.cvar_diameter_m, "max_diameter_m": metrics.max_diameter_m,
            "composite_m": metrics.composite_m}


def _resolution_check(config, point, scenarios):
    """Doubling the circle resolution should not move the metrics materially."""
    fine = replace(config, circle_sides=config.circle_sides * 2)
    poly = source_polygon(fine)
    metrics = PointEvaluator(poly, scenarios, fine)(point)
    return {"circle_sides": fine.circle_sides, "vertices": len(poly),
            "mean_diameter_m": metrics.mean_diameter_m,
            "cvar_diameter_m": metrics.cvar_diameter_m,
            "composite_m": metrics.composite_m}


def _example_intersection(point, poly, scenarios, config):
    """One clearly hypothetical scenario, for the explanatory figure."""
    values, empty = PointEvaluator(poly, scenarios, config).diameters(point)
    index = int(np.argsort(values)[len(values) // 2])
    source = scenarios.sources[index]
    error = scenarios.errors[index]
    reading = math.atan2(source[1] - point[1], source[0] - point[0]) + error
    clipped = clip_wedge(poly, point, reading, config.eps)
    acute = math.degrees(math.acos(min(1.0, abs(math.cos(reading)))))
    return {
        "hypothetical": True,
        "source_local_m": [float(source[0]), float(source[1])],
        "error_deg": float(math.degrees(error)),
        "reading_deg_local": float(math.degrees(reading)),
        "acute_crossing_deg": float(acute),
        "diameter_m": float(diameter(clipped)[0]),
        "polygon": [[float(x), float(y)] for x, y in clipped],
    }


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def run_pipeline(root, config: Config, plots=True, sensitivity=True, progress=print):
    started = time.perf_counter()
    root = Path(root)
    name = config.name
    output_dir = root / "outputs" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    poly_full, poly, region_info = build_source_region(config)
    progress(f"Source region: {region_info['vertices_full']} vertices "
             f"(used {region_info['vertices_used']}), diameter {region_info['diameter_full_m']:.3f} m.")

    scenarios = sample_scenarios(config)
    progress(f"Scenarios: {len(scenarios)} shared samples, seed {scenarios.seed}.")

    rows, evaluator = opt.evaluate_grid(poly, scenarios, config)
    mask = opt.pareto_mask(rows)
    lambda_info = opt.normalized_lambda(rows, mask)
    lambda_info["basis"] = "Pareto set"
    if config.lambda_mode == "fixed":
        base_weight = config.weight_lambda
        lambda_info["basis"] = "fixed value from the configuration"
    elif lambda_info["ranking"] != "normalized":
        # The Pareto set collapsed (typically a single point: the nearest
        # candidate is also the most accurate one, so no distance/accuracy
        # tension exists). Two explicit conventions are available; the wider
        # full-range normalisation is preferred because it still supports a
        # meaningful sensitivity sweep, and the basis is recorded either way.
        fallback = opt.normalized_lambda(rows)
        fallback["basis"] = "all evaluated candidates"
        lambda_info["fallback"] = fallback
        if fallback["ranking"] == "normalized":
            base_weight = fallback["lambda0"]
            lambda_info["basis"] = ("all evaluated candidates (Pareto set degenerate: "
                                    f"{lambda_info['ranking']})")
        elif lambda_info["ranking"] == "distance":
            base_weight = 0.0
            lambda_info["basis"] = "composite range degenerate everywhere; J ranks by travel distance"
        else:
            base_weight = 1.0
            lambda_info["basis"] = "distance range degenerate everywhere; any positive weight ranks by S"
    else:
        base_weight = lambda_info["lambda0"]
    progress(f"Pareto set: {lambda_info['pareto_points']} points, lambda0={lambda_info['lambda0']:.4g}, "
             f"base weight={base_weight:.4g} ({lambda_info['basis']}).")

    base = opt.select_for_weight(rows, evaluator, config, base_weight)
    results = []
    if sensitivity:
        results = opt.sensitivity(rows, evaluator, config, base_weight)
    else:
        base["multiplier"] = 1.0
        results = [base]

    point = np.asarray(base["point_local_m"], dtype=float)
    global_point = local_to_global([point], config)[0]
    checks = audit_point(point, poly_full, config)
    progress(f"Final point local=({point[0]:.3f}, {point[1]:.3f}) m, "
             f"global=({global_point[0]:.3f}, {global_point[1]:.3f}) m.")
    progress(f"Audit: reception_ok={checks['reception_guaranteed']}, angle_ok={checks['angle_guaranteed']}, "
             f"worst diameter in [{checks['worst_case_audit']['lower_m']:.3f}, "
             f"{checks['worst_case_audit']['upper_m']:.3f}] m "
             f"(uniform bound {checks['uniform_diameter_upper_bound_m']:.3f} m).")

    independent = _independent_check(config, poly, point, seed_offset=1000, count_factor=1)
    resolution = _resolution_check(config, point, scenarios)
    example = _example_intersection(point, poly, scenarios, config)
    progress(f"Independent scenario check: mean {independent['mean_diameter_m']:.3f} m vs "
             f"{base['mean_diameter_m']:.3f} m on the planning sample.")
    progress(f"Resolution check ({resolution['circle_sides']} sides): composite "
             f"{resolution['composite_m']:.3f} m vs {base['composite_m']:.3f} m.")

    summary = {
        "name": name,
        "config": config.to_dict(),
        "local_frame": {
            "origin": list(config.first_station_m),
            "x_axis_bearing_deg": config.first_bearing_deg,
            "note": "local x along the first measured bearing, y to its left",
        },
        "source_region": region_info,
        "scenarios": scenarios.summary(),
        "candidate_region": {
            "parameter_R_m": float(config.far),
            "h_R_m": float(config.h),
            "constraints": ["a>=0", "a^2+b^2<=rho^2", "(a-R)^2+(|b|+h(R))^2<=rho^2",
                            "|b|>=t*a", "|b|>=t*(R-a)+h(R)"],
            "grid_step_m": config.grid_step_m,
            "evaluated_points": int(len(rows)),
            "uniform_diameter_bound_m": uniform_diameter_bound(config),
        },
        "lambda": dict(lambda_info, base_weight=float(base_weight), mode=config.lambda_mode),
        "final": {
            "point_local_m": base["point_local_m"],
            "point_global_m": [float(global_point[0]), float(global_point[1])],
            "distance_m": base["distance_m"],
            "travel_time_s": base["travel_time_s"],
            "mean_diameter_m": base["mean_diameter_m"],
            "cvar_diameter_m": base["cvar_diameter_m"],
            "max_diameter_m": base["max_diameter_m"],
            "composite_m": base["composite_m"],
            "cost_s": base["cost_s"],
            "detection_time_s": config.detection_time_s,
            "tie_rule": base["tie_rule"],
        },
        "checks": checks,
        "independent_scenario_check": independent,
        "resolution_check": resolution,
        "example_intersection": example,
        "sensitivity": results,
        "runtime_s": None,
    }

    nearest = rows[int(np.argmin(rows[:, opt.DIST]))]
    summary["nearest_point_reference"] = {
        "point_local_m": [float(nearest[opt.X]), float(nearest[opt.Y])],
        "distance_m": float(nearest[opt.DIST]),
        "mean_diameter_m": float(nearest[opt.MEAN]),
        "cvar_diameter_m": float(nearest[opt.CVAR]),
        "composite_m": float(nearest[opt.COMP]),
    }

    # Exports.
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_dir / "candidates.csv",
               list(opt.COLUMNS) + ["pareto"],
               [[float(v) for v in row] + [int(m)] for row, m in zip(rows, mask)])
    _write_csv(output_dir / "sensitivity.csv",
               ["multiplier", "weight", "x_m", "y_m", "distance_m", "mean_diameter_m",
                "cvar_diameter_m", "max_diameter_m", "composite_m", "cost_s"],
               [[r.get("multiplier", 1.0), r["weight"], *r["point_local_m"], r["distance_m"],
                 r["mean_diameter_m"], r["cvar_diameter_m"], r["max_diameter_m"],
                 r["composite_m"], r["cost_s"]] for r in results])
    np.savez(output_dir / "scenarios.npz", sources=scenarios.sources,
             errors=scenarios.errors, weights=scenarios.weights)
    np.save(output_dir / "candidates.npy", rows)
    np.save(output_dir / "pareto_mask.npy", mask)
    with (output_dir / "audit.txt").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(checks, ensure_ascii=False, indent=2))

    summary["runtime_s"] = time.perf_counter() - started
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if plots:
        try:
            from .plotting import plot_all
            summary["figures"] = plot_all(root, config, summary, rows, mask, poly, poly_full)
        except Exception as exc:  # plotting must never destroy a completed solve
            progress(f"Plotting failed: {exc}")
            summary["figures"] = {"error": str(exc)}

    progress(f"Done in {summary['runtime_s']:.2f}s; outputs in {output_dir}.")
    return summary
