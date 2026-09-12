"""Shared evaluation scenarios for candidate points.

One fixed sample of (source position, second-reading error, weight) is used for
every candidate point, so that comparisons between points are not driven by
resampling noise. The sample encodes a stated working assumption, not a fact
given in the problem statement:

  * the source is uniform by area over W1 n B(S1, R_max) n B(O, 1800);
  * the second bearing error is uniform on [-eps, +eps], independent of position.

Area-uniform sampling in the sector uses r = sqrt(r_min^2 + U (R_max^2 - r_min^2)),
never a uniform radius. Positions outside the map circle are rejected before the
sample is accepted. Scenario positions are only used to fabricate hypothetical
readings; they are never handed to the localisation update as extra constraints.

Unless disabled, each draw is mirrored in the first bearing line with its error
sign flipped (antithetic sampling). The mirror of a uniform draw on K is again
uniform on K, so the pooled sample remains a valid sample of the working
distribution and cancels the finite-sample asymmetry between the two candidate
lobes. Mirrored positions outside the map are discarded.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np

from .geometry import local_to_global


@dataclass
class Scenarios:
    sources: np.ndarray
    errors: np.ndarray
    weights: np.ndarray
    seed: int
    attempts: int
    rejected_by_map: int

    def __len__(self):
        return int(self.sources.shape[0])

    @property
    def count(self):
        return len(self)

    def readings(self, point):
        """Hypothetical second readings (radians, local frame) at a candidate point."""
        delta = self.sources - np.asarray(point, dtype=float)
        return np.arctan2(delta[:, 1], delta[:, 0]) + self.errors

    def summary(self):
        radius = np.linalg.norm(self.sources, axis=1)
        return {
            "count": self.count,
            "seed": self.seed,
            "attempts": self.attempts,
            "rejected_by_map": self.rejected_by_map,
            "radius_min_m": float(radius.min()),
            "radius_max_m": float(radius.max()),
            "radius_mean_m": float(radius.mean()),
            "error_min_deg": float(np.degrees(self.errors.min())),
            "error_max_deg": float(np.degrees(self.errors.max())),
            "uniform_weights": bool(np.allclose(self.weights, 1.0 / self.count)),
            "weight_min": float(self.weights.min()),
            "weight_max": float(self.weights.max()),
            "weight_note": ("uniform: the sample is mirror symmetric or mirroring is off"
                            if bool(np.allclose(self.weights, 1.0 / self.count))
                            else "non-uniform inside K: mirror multiplicity correction is active"),
        }

    def to_dict(self):
        return {"summary": self.summary()}


def sample_scenarios(config, count=None, seed=None, max_draw_factor=200):
    """Draw `count` accepted scenarios; returns a Scenarios object.

    max_draw_factor bounds the total number of radial draws as a multiple of the
    requested sample size, because the map circle can reject most of the sector.
    """
    n = int(config.n_scenarios if count is None else count)
    if n < 20:
        raise ValueError("At least 20 scenarios are required for a usable tail.")
    max_attempts = max(2000, max_draw_factor * n)
    rng = np.random.default_rng(config.seed if seed is None else seed)
    r_min, r_max = config.source_min_radius_m, config.max_reception_m
    eps = config.eps
    kept, attempts, rejected = [], 0, 0
    while sum(len(block) for block in kept) < n and attempts < max_attempts:
        batch = max(64, n - sum(len(block) for block in kept))
        radius = np.sqrt(r_min ** 2 + rng.random(batch) * (r_max ** 2 - r_min ** 2))
        delta = rng.uniform(-eps, eps, batch)
        points = np.column_stack([radius * np.cos(delta), radius * np.sin(delta)])
        attempts += batch
        if config.use_map_boundary:
            far = np.linalg.norm(local_to_global(points, config) - np.asarray(config.map_center_m), axis=1)
            inside = far <= config.map_radius_m
            rejected += int((~inside).sum())
            points = points[inside]
        kept.append(points)
    sources = np.vstack(kept)[:n]
    if len(sources) < n:
        raise ValueError(f"Rejection sampling produced only {len(sources)}/{n} positions "
                         f"after {attempts} draws; check the map and first station.")
    errors = rng.uniform(-eps, eps, n)
    if config.antithetic_mirror:
        # Mirror in the first bearing line (y -> -y) with the error sign flipped.
        # A mirrored draw is again a draw from the sector, so the pooled sample
        # stays valid; the mirror cancels the finite-sample asymmetry between
        # the two candidate lobes.
        #
        # Multiplicity weighting: if the mirror of a base draw also lies in K,
        # that point appears twice per draw (once as itself, once as its own
        # mirror's image), so a point in the overlap region would be double
        # counted. Each occurrence is therefore given weight 1/(1 + 1[mirror
        # in K]), which keeps the weighted sample area-uniform on K. For a
        # mirror-symmetric K the weights are uniform again, i.e. the symmetric
        # case is unchanged.
        mirrored = sources * np.array([1.0, -1.0])
        if config.use_map_boundary:
            far = np.linalg.norm(local_to_global(mirrored, config) - np.asarray(config.map_center_m), axis=1)
            keep = far <= config.map_radius_m
        else:
            keep = np.ones(len(mirrored), dtype=bool)
        sources = np.vstack([sources, mirrored[keep]])
        errors = np.concatenate([errors, -errors[keep]])
        weights = np.concatenate([1.0 / (1.0 + keep.astype(float)),
                                  np.full(int(keep.sum()), 0.5)])
    else:
        weights = np.ones(len(sources))
    weights = weights / weights.sum()
    return Scenarios(sources, errors, weights, int(config.seed if seed is None else seed), attempts, rejected)


def radial_uniformity_check(scenarios, config, bins=10):
    """Diagnostic: area-uniform sampling has radius density proportional to r."""
    radius = np.linalg.norm(scenarios.sources, axis=1)
    lo, hi = config.source_min_radius_m, config.max_reception_m
    edges = np.sqrt(np.linspace(lo ** 2, hi ** 2, bins + 1))
    counts, _ = np.histogram(radius, bins=edges)
    expected = np.full(bins, len(radius) / bins)
    return {"edges_m": edges.tolist(), "counts": counts.tolist(),
            "expected_counts": expected.tolist(),
            "max_relative_deviation": float(np.max(np.abs(counts - expected)) / expected[0])}
