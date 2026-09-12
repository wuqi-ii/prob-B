"""Typed configuration; all lengths are metres, all stored angles are degrees."""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import math


@dataclass(frozen=True)
class Config:
    name: str = "default"
    first_station_m: tuple = (0.0, 0.0)
    first_bearing_deg: float = 0.0
    map_center_m: tuple = (0.0, 0.0)
    map_radius_m: float = 1800.0
    use_map_boundary: bool = True
    bearing_error_deg: float = 1.0
    max_reception_m: float = 1500.0
    # Parameter R: the farthest source distance still possible after the
    # first bearing and map-boundary clipping.  None preserves the original
    # problem-2 assumption R=max_reception_m=1500 m.
    source_far_m: float | None = None
    guaranteed_reception_m: float = 1000.0
    min_crossing_deg: float = 30.0
    speed_m_s: float = 5.0
    detection_time_s: float = 5.0
    weight_lambda: float = 5.0
    lambda_mode: str = "normalized"
    weight_multipliers: tuple = (0.25, 0.5, 1.0, 2.0, 4.0)
    sensitivity_weights: tuple = ()
    grid_step_m: float = 10.0
    bearing_samples: int = 81
    local_starts_per_side: int = 3
    circle_sides: int = 720
    certificate_tolerance_m: float = 0.05
    certificate_max_splits: int = 4096
    random_seed: int = 20260911
    # Scenario evaluation: mean + CVaR over a shared sample of (source, error).
    tail_weight: float = 0.5
    cvar_alpha: float = 0.9
    n_scenarios: int = 400
    source_min_radius_m: float = 5.0
    scenario_seed: int = 0
    antithetic_mirror: bool = True
    simplify_tolerance_m: float = 0.02
    refine_steps_m: tuple = (25.0, 10.0, 4.0)
    refine_budget: int = 80

    def __post_init__(self):
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.name):
            raise ValueError("name must be a simple directory name (letters/numbers/_/-).")
        for key in ("first_station_m", "map_center_m"):
            value = getattr(self, key)
            if len(value) != 2 or not all(math.isfinite(x) for x in value):
                raise ValueError(f"{key} must contain two finite coordinates.")
        for key in ("map_radius_m", "max_reception_m", "guaranteed_reception_m", "speed_m_s", "grid_step_m", "certificate_tolerance_m"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite.")
        if self.source_far_m is not None:
            if not math.isfinite(self.source_far_m) or self.source_far_m <= 0:
                raise ValueError("source_far_m (parameter R) must be positive and finite.")
            if self.source_far_m > self.max_reception_m:
                raise ValueError("source_far_m cannot exceed the physical maximum reception radius.")
        if not math.isfinite(self.first_bearing_deg):
            raise ValueError("first_bearing_deg must be finite.")
        if not 0 < self.bearing_error_deg < 10:
            raise ValueError("This implementation requires 0 < bearing_error_deg < 10.")
        if not 0 < self.min_crossing_deg < 90 - self.bearing_error_deg:
            raise ValueError("Invalid minimum crossing angle.")
        if self.guaranteed_reception_m > self.max_reception_m:
            raise ValueError("Guaranteed reception radius exceeds maximum radius.")
        if self.detection_time_s < 0 or not math.isfinite(self.detection_time_s):
            raise ValueError("Invalid detection time.")
        if self.lambda_mode not in ("normalized", "fixed"):
            raise ValueError("lambda_mode must be 'normalized' or 'fixed'.")
        weights = [self.weight_lambda, *self.sensitivity_weights, *self.weight_multipliers]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Weights must be finite and nonnegative.")
        if any(w <= 0 for w in self.weight_multipliers):
            raise ValueError("weight_multipliers must be strictly positive.")
        if self.bearing_samples < 9 or self.circle_sides < 32 or self.local_starts_per_side < 1 or self.certificate_max_splits < 1:
            raise ValueError("Insufficient numerical resolution.")
        if not 0 <= self.tail_weight <= 1:
            raise ValueError("tail_weight (rho) must lie in [0, 1].")
        if not 0 < self.cvar_alpha < 1:
            raise ValueError("cvar_alpha must lie in (0, 1).")
        if self.n_scenarios < 20:
            raise ValueError("n_scenarios must be at least 20 for a meaningful tail.")
        if not 0 <= self.source_min_radius_m < self.far:
            raise ValueError("source_min_radius_m must lie in [0, source_far_m).")
        if self.simplify_tolerance_m <= 0 or self.refine_budget < 1:
            raise ValueError("Invalid simplification or refinement settings.")
        if not self.refine_steps_m or any(s <= 0 for s in self.refine_steps_m):
            raise ValueError("refine_steps_m must be a nonempty tuple of positive steps.")

    @property
    def eps(self):
        return math.radians(self.bearing_error_deg)

    @property
    def alpha(self):
        return math.radians(self.min_crossing_deg)

    @property
    def h(self):
        return self.far * math.tan(self.eps)

    @property
    def far(self):
        """Parameter R used by the source and candidate regions."""
        return self.max_reception_m if self.source_far_m is None else self.source_far_m

    @property
    def t(self):
        return math.tan(self.alpha + self.eps)

    @property
    def seed(self):
        """Scenario seed; scenario_seed=0 falls back to the global random_seed."""
        return self.scenario_seed or self.random_seed

    def to_dict(self):
        return asdict(self)


def load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for key in ("first_station_m", "map_center_m", "sensitivity_weights",
                "weight_multipliers", "refine_steps_m"):
        if key in data:
            data[key] = tuple(data[key])
    return Config(**data)
