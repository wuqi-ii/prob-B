"""Candidate evaluation, Pareto/lambda normalisation and weighted selection.

The hard candidate region is fixed beforehand (reception, 30-degree crossing and
the resulting diameter bound). Only the ranking inside it changed: the objective
now uses the scenario mean and the scenario CVaR instead of the single worst
reading. The worst-case search is still available in risk.py for auditing.

Search is a grid scan plus a bounded pattern-search refinement. Both are finite
sampling procedures: the returned point is the best among evaluated candidates,
which is not a certified global optimum of the continuous problem.
"""
import math
import time
import numpy as np

from .metrics import PointEvaluator
from .region import boundary_candidates, deduplicate, feasible_mask

# Column layout of the candidate table returned by evaluate_grid.
COLUMNS = ("x_m", "y_m", "distance_m", "mean_diameter_m", "cvar_diameter_m",
           "max_diameter_m", "composite_m")
X, Y, DIST, MEAN, CVAR, MAXD, COMP = range(len(COLUMNS))

# Documented tie rule: the constraints are symmetric about the local x-axis, so
# the two lobes are genuinely equivalent and the scenario sample can leave a
# residual difference of sampling origin. Candidates within a small window of the
# minimum are treated as tied; the +y lobe is then preferred, which also keeps
# the reported point stable under re-seeding. The window is bounded, so the cost
# of the tie rule is at most TIE_TOL_REL (0.1%) of the optimum.
TIE_TOL_S = 1e-6
TIE_TOL_REL = 1e-3


def cost(distance_m, composite_m, weight, config):
    """J = travel time + detection time + lambda * S / v, in seconds."""
    return distance_m / config.speed_m_s + config.detection_time_s + weight * composite_m / config.speed_m_s


def station_grid(config, step=None):
    """Interior grid plus the analytic boundary of C (corners included)."""
    step = config.grid_step_m if step is None else step
    R = config.guaranteed_reception_m
    axis = np.arange(0.0, R + 0.5 * step, step)
    xx, yy = np.meshgrid(axis, axis)
    upper = np.column_stack([xx.ravel(), yy.ravel()])
    upper = upper[feasible_mask(upper, config)]
    if not len(upper):
        raise ValueError("No feasible grid points; region may be empty or grid too coarse.")
    points = np.vstack([upper, upper * [1.0, -1.0], boundary_candidates(config)])
    return deduplicate(points, tol=1e-6)


def evaluate_grid(poly, scenarios, config, progress=print):
    """Evaluate every feasible grid point on the shared scenario set."""
    evaluator = PointEvaluator(poly, scenarios, config)
    points = station_grid(config)
    rows = np.empty((len(points), len(COLUMNS)), dtype=float)
    start = time.perf_counter()
    tick = start
    for i, point in enumerate(points):
        m = evaluator(point)
        rows[i] = [m.point_local_m[0], m.point_local_m[1], m.distance_m,
                   m.mean_diameter_m, m.cvar_diameter_m, m.max_diameter_m, m.composite_m]
        if time.perf_counter() - tick > 20:
            progress(f"Grid evaluated {i + 1}/{len(points)} candidate points.")
            tick = time.perf_counter()
    progress(f"Grid: {len(points)} feasible stations, {evaluator.calls} evaluations, "
             f"{time.perf_counter() - start:.2f}s.")
    if evaluator.cache and any(m.empty_updates for m in evaluator.cache.values()):
        progress("Warning: some scenarios produced empty localisation polygons.")
    return rows, evaluator


def pareto_mask(rows):
    """Non-dominated candidates on (travel distance, composite localisation index)."""
    order = np.lexsort((rows[:, COMP], rows[:, DIST]))
    keep = np.zeros(len(rows), dtype=bool)
    best = math.inf
    for i in order:
        if rows[i, COMP] < best - 1e-12:
            keep[i] = True
            best = float(rows[i, COMP])
    return keep


def normalized_lambda(rows, mask=None, eps_ratio=1e-6):
    """lambda0 = DeltaL / DeltaS on the Pareto set, with degenerate fallbacks.

    The returned `ranking` says which term can still discriminate:
      normalized -> lambda0 is the equal-range weight;
      distance   -> DeltaS is ~0, so J ranks by travel distance for any lambda
                    (lambda = 0 is the canonical representative);
      composite  -> DeltaL is ~0, so every positive lambda ranks by S.
    """
    use = rows if mask is None else rows[mask]
    dist, comp = use[:, DIST], use[:, COMP]
    delta_l = float(dist.max() - dist.min())
    delta_s = float(comp.max() - comp.min())
    scale_l = max(float(dist.max()), 1.0)
    scale_s = max(float(comp.max()), 1.0)
    degenerate_s = delta_s <= eps_ratio * scale_s
    degenerate_l = delta_l <= eps_ratio * scale_l
    if degenerate_s and degenerate_l:
        value, note, ranking = 0.0, "both ranges degenerate; neither term discriminates", "distance"
    elif degenerate_s:
        value, note, ranking = 0.0, "composite range degenerate; J is proportional to travel distance", "distance"
    elif degenerate_l:
        value, note, ranking = math.inf, "distance range degenerate; every positive lambda ranks by S", "composite"
    else:
        value, note, ranking = delta_l / delta_s, "equal normalised weight on the Pareto ranges", "normalized"
    return {"lambda0": value, "delta_distance_m": delta_l, "delta_composite_m": delta_s,
            "pareto_points": int(len(use)), "note": note, "ranking": ranking,
            "degenerate_distance": degenerate_l, "degenerate_composite": degenerate_s}


def refine_point(point, evaluator, config, weight, budget=None, steps=None):
    """Bounded pattern search; stays feasible and only accepts strict improvements."""
    budget = config.refine_budget if budget is None else budget
    steps = config.refine_steps_m if steps is None else steps
    best = np.asarray(point, dtype=float)
    best_value = cost(float(np.linalg.norm(best)), evaluator(best).composite_m, weight, config)
    history = [{"point": best.tolist(), "cost_s": float(best_value)}]
    used = 0
    for step in steps:
        improved = True
        while improved and used < budget:
            improved = False
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step),
                           (step, step), (step, -step), (-step, step), (-step, -step)):
                if used >= budget:
                    break
                trial = best + np.array([dx, dy])
                if not feasible_mask(trial, config, tol=1e-6):
                    continue
                value = cost(float(np.linalg.norm(trial)), evaluator(trial).composite_m, weight, config)
                used += 1
                if value < best_value - 1e-9:
                    best, best_value, improved = trial, value, True
                    history.append({"point": best.tolist(), "cost_s": float(best_value)})
                    break
    return best, best_value, {"evaluations": used, "history": history}


def select_for_weight(rows, evaluator, config, weight, refine_steps=None, budget=None,
                      refine=True):
    """Pick the best evaluated candidate for a given lambda, then refine locally."""
    scores = cost(rows[:, DIST], rows[:, COMP], weight, config)
    best = float(scores.min())
    tol = max(TIE_TOL_S, TIE_TOL_REL * abs(best))
    tied = np.flatnonzero(scores <= best + tol)
    # The tie window only decides which lobe is reported; inside a lobe the
    # cheapest candidate always wins, so the rule can never raise the cost above
    # the best candidate of the reported lobe.
    positive = [i for i in tied if rows[i, Y] > 0]
    pool = positive if positive else list(tied)
    index = int(min(pool, key=lambda i: (scores[i], -rows[i, Y], rows[i, DIST])))
    point = rows[index, :2].copy()
    record = {
        "weight": float(weight),
        "grid_point_local_m": point.tolist(),
        "grid_cost_s": float(scores[index]),
        "tie_candidates": int(len(tied)),
        "tie_rule": "tie window picks the +y lobe; inside a lobe the lowest cost wins",
        "tie_tolerance_s": float(tol),
    }
    if refine:
        refined, value, info = refine_point(point, evaluator, config, weight, budget=budget, steps=refine_steps)
        record.update({"point_local_m": refined.tolist(), "cost_s": float(value),
                       "refine_evaluations": info["evaluations"],
                       "refine_improved": bool(value < scores[index] - 1e-9)})
    else:
        refined, value = point, float(scores[index])
        record.update({"point_local_m": point.tolist(), "cost_s": float(value),
                       "refine_evaluations": 0, "refine_improved": False})
    metrics = evaluator(refined)
    record.update({
        "distance_m": metrics.distance_m,
        "mean_diameter_m": metrics.mean_diameter_m,
        "cvar_diameter_m": metrics.cvar_diameter_m,
        "max_diameter_m": metrics.max_diameter_m,
        "composite_m": metrics.composite_m,
        "travel_time_s": metrics.distance_m / config.speed_m_s,
        "selection": "minimum evaluated cost on the fixed candidate region",
    })
    return record


def sensitivity(rows, evaluator, config, lambda0, multipliers=None, progress=print):
    """Re-select under lambda = multiplier * lambda0."""
    multipliers = config.weight_multipliers if multipliers is None else multipliers
    results = []
    for factor in multipliers:
        weight = lambda0 * factor
        record = select_for_weight(rows, evaluator, config, weight)
        record["multiplier"] = float(factor)
        results.append(record)
        progress(f"lambda={weight:.4g} ({factor:g}*lambda0): P=({record['point_local_m'][0]:.2f}, "
                 f"{record['point_local_m'][1]:.2f})m, L={record['distance_m']:.2f}m, "
                 f"mean={record['mean_diameter_m']:.2f}m, CVaR={record['cvar_diameter_m']:.2f}m, "
                 f"max={record['max_diameter_m']:.2f}m, J={record['cost_s']:.3f}s.")
    return results
