"""滚动路线插入调度器的回归测试。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b3.config import StrategyConfig  # noqa: E402
from cumcm_b3.strategy import DogStrategy, Task  # noqa: E402


class TestRouteInsertScheduler(unittest.TestCase):
    def make_strategy(self, mode: str = "route_insert") -> DogStrategy:
        cfg = replace(StrategyConfig(), scheduler_mode=mode)
        return DogStrategy(OfflineSimulator(generate_case(1)), cfg)

    def test_invalid_scheduler_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scheduler_mode"):
            replace(StrategyConfig(), scheduler_mode="unknown").validate()

    def test_scan_spine_is_shortest_open_path(self):
        strat = self.make_strategy()
        scans = [Task("scan", (10.0, 0.0)), Task("scan", (20.0, 0.0)),
                 Task("scan", (30.0, 0.0))]
        route = strat._shortest_scan_spine((0.0, 0.0), scans)
        self.assertEqual([t.point for t in route],
                         [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)])

    def test_service_on_first_leg_is_inserted_before_scan(self):
        strat = self.make_strategy()
        scans = [Task("scan", (100.0, 0.0)), Task("scan", (200.0, 0.0))]
        service = Task("clear", (50.0, 0.0), 3)
        chosen = strat._route_insert_choice((0.0, 0.0), scans, [service])
        self.assertIs(chosen, service)

    def test_near_point_not_full_point_drives_nearest_cost(self):
        strat = self.make_strategy("nearest")
        verify = Task("verify", (1000.0, 0.0), 2, near_point=(20.0, 0.0))
        scan = Task("scan", (100.0, 0.0))
        self.assertIs(strat._nearest_choice((0.0, 0.0), [scan, verify]), verify)

    def test_route_scheduler_keeps_full_clear_rate(self):
        cfg = replace(StrategyConfig(), scheduler_mode="route_insert")
        for seed in range(1, 11):
            case = generate_case(seed)
            stats = DogStrategy(OfflineSimulator(case), cfg).run()
            self.assertEqual(stats.cleared_count, case.total, f"seed={seed}")
            self.assertEqual(stats.stop_reason, "all_done")


if __name__ == "__main__":
    unittest.main()
