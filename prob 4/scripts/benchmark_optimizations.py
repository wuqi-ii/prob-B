"""Paired offline benchmark for problem-4 routing/search optimizations.

This script intentionally uses the same generated cases for every configuration so
time differences are attributable to the strategy change rather than case mix.
It also decomposes virtual time into movement, measurement, channel switching and
clear-action costs.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b4.config import CLEAR_HIT_COST_S, CLEAR_MISS_COST_S, StrategyConfig  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


class LeanScanStrategy(DogStrategy):
    """Do not remeasure a known channel whose geometry is already actionable."""

    def _visit_scan_point(self, idx: int) -> None:
        point = self.scan_points[idx]
        channels = []
        for channel in range(1, 21):
            track = self.book[channel]
            if track.cleared:
                continue
            blocked_observations = self._blocked_at_obs.get(channel)
            needs_recovery_observation = (
                blocked_observations is not None
                and len(track.observations) <= blocked_observations
            )
            if not track.known or len(track.observations) < 2 or needs_recovery_observation:
                channels.append(channel)
        for channel in channels:
            payload = self.backend.measure(point[0], point[1], channel)
            kind = self.book.apply_measure(channel, point, payload)
            if kind != "no_signal":
                self._log("discover", channel=channel, at=list(point), kind=kind)
        self.visited_scan[idx] = True
        self._log(
            "scan_done",
            point=list(point),
            index=idx,
            channels=len(channels),
            found=[channel for channel in channels if self.book[channel].known],
        )


class DeferredClearStrategy(DogStrategy):
    """Finish the short scan-net route before taking localization detours."""

    def select_task(self):
        unvisited = [
            (idx, point)
            for idx, point in enumerate(self.scan_points)
            if not self.visited_scan[idx]
        ]
        if unvisited:
            current = self._pos()
            idx, point = min(unvisited, key=lambda item: math.dist(current, item[1]))
            from cumcm_b4.strategy import Task

            return Task("scan", point, None, note=f"S{idx}")
        return super().select_task()


class DeferredLeanScanStrategy(LeanScanStrategy, DeferredClearStrategy):
    pass


def run_case(seed: int, cfg: StrategyConfig, strategy_class=DogStrategy) -> dict[str, float]:
    sim = OfflineSimulator(generate_case(seed))
    stats = strategy_class(sim, cfg).run()
    position = (0.0, 0.0)
    movement_m = 0.0
    measures = 0
    switches = 0.0
    clear_cost_s = 0.0
    for event in sim.trace:
        if event["kind"] in {"measure", "clear"}:
            target = (float(event["x"]), float(event["y"]))
            movement_m += math.dist(position, target)
            position = target
        if event["kind"] == "measure":
            measures += 1
            switches += float(event.get("switch_cost_s", 0.0))
        elif event["kind"] == "clear":
            clear_cost_s += (
                CLEAR_HIT_COST_S
                if event.get("result") == "success"
                else CLEAR_MISS_COST_S
            )
    return {
        "time_s": sim.virtual_time_s,
        "movement_m": movement_m,
        "movement_s": movement_m / 5.0,
        "measure_s": measures * 5.0,
        "switch_s": switches,
        "clear_s": clear_cost_s,
        "requests": float(sim.request_count),
        "total": float(sim.case.total),
        "cleared": float(stats.cleared_count),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    result = {f"{key}_mean": statistics.fmean(row[key] for row in rows) for key in keys}
    result["perfect_cases"] = float(
        sum(row["cleared"] == row["total"] for row in rows)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=1000)
    args = parser.parse_args()

    configs = {
        "baseline": (StrategyConfig(), DogStrategy),
        "lean_scan": (StrategyConfig(), LeanScanStrategy),
        "defer_clear": (StrategyConfig(), DeferredClearStrategy),
        "defer_lean": (StrategyConfig(), DeferredLeanScanStrategy),
        "no_opportunistic": (
            replace(StrategyConfig(), opportunistic_scan=False),
            DogStrategy,
        ),
        "verify_first": (replace(StrategyConfig(), task_bias_verify_m=300.0), DogStrategy),
        "clear_first": (replace(StrategyConfig(), task_bias_clear_m=300.0), DogStrategy),
        "verify_delay_100": (
            replace(StrategyConfig(), task_bias_verify_m=-100.0),
            DogStrategy,
        ),
        "verify_delay_300": (
            replace(StrategyConfig(), task_bias_verify_m=-300.0),
            DogStrategy,
        ),
        "clear_delay_100": (
            replace(StrategyConfig(), task_bias_clear_m=-100.0),
            DogStrategy,
        ),
        "clear_delay_300": (
            replace(StrategyConfig(), task_bias_clear_m=-300.0),
            DogStrategy,
        ),
        "both_delay_100": (
            replace(
                StrategyConfig(),
                task_bias_verify_m=-100.0,
                task_bias_clear_m=-100.0,
            ),
            DogStrategy,
        ),
    }
    seeds = range(args.start_seed, args.start_seed + args.cases)
    summaries: dict[str, dict[str, float]] = {}
    # All variants below share the same already-validated 27-point layout. Running
    # the expensive coverage grid once keeps this benchmark focused on routing.
    StrategyConfig().validate()
    for name, (cfg, strategy_class) in configs.items():
        rows = [run_case(seed, cfg, strategy_class) for seed in seeds]
        summaries[name] = summarize(rows)

    base = summaries["baseline"]
    print(
        "name                 perfect    total_s   delta     move_s  measure_s "
        "switch_s  requests"
    )
    for name, summary in summaries.items():
        delta = summary["time_s_mean"] / base["time_s_mean"] - 1.0
        print(
            f"{name:20s} {int(summary['perfect_cases']):3d}/{args.cases:<3d} "
            f"{summary['time_s_mean']:9.1f} {delta:+7.2%} "
            f"{summary['movement_s_mean']:9.1f} "
            f"{summary['measure_s_mean']:10.1f} "
            f"{summary['switch_s_mean']:8.1f} "
            f"{summary['requests_mean']:9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
