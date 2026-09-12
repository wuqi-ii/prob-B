"""近端优先验证点（verify_near_fraction）的回归测试。

背景：标准第二检测点 (764, 526) 距首站 927 m，是"二次接收 + 交会角 ≥ 31°"的最坏情况保证。
但源必然在首个检测点 1000 m 内，因此把候选点沿"首站→标准点"连线按比例前移后，
仍**必然**落在源的接收范围内（实测 120 案例 0 次回退），却省下一大段奔赴行程。

离线批量（120 案例）实测：fraction=0.4 相对基线 **−5.73%** 总时间，清除率仍 100%。
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3.config import StrategyConfig  # noqa: E402
from cumcm_b3.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b3.strategy import DogStrategy  # noqa: E402


class TestVerifyNear(unittest.TestCase):
    def test_near_point_sits_at_requested_fraction(self):
        cfg = replace(StrategyConfig(), verify_near_fraction=0.5)
        cfg.validate()
        strat = DogStrategy(OfflineSimulator(generate_case(1)), cfg)
        strat.visited_scan = [True] * len(strat.visited_scan)  # 屏蔽扫描点候选
        strat.book[3].add_direction((0.0, 0.0), 0.0)   # 首站在原点，方位角 0°
        task = strat.select_task()
        self.assertIsNotNone(task)
        self.assertEqual(task.kind, "verify")
        self.assertIsNotNone(task.near_point)
        a, b = cfg.second_station_local_a_m, cfg.second_station_local_b_m
        full = math.hypot(a, b)
        near = math.hypot(task.near_point[0], task.near_point[1])
        self.assertAlmostEqual(near, 0.5 * full, delta=0.1)

    def test_end_to_end_clears_with_near_verify(self):
        cfg = replace(StrategyConfig(), verify_near_fraction=0.4)
        cfg.validate()
        for seed in (7, 8, 9):
            case = generate_case(seed)
            stats = DogStrategy(OfflineSimulator(case), cfg).run()
            self.assertEqual(stats.cleared_count, case.total, f"seed={seed} 未全清")
            self.assertEqual(stats.stop_reason, "all_done")
            self.assertTrue(any(e["event"] == "verify_near" for e in stats.events),
                            "应产生 verify_near 事件")

    def test_invalid_fraction_is_rejected(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                replace(StrategyConfig(), verify_near_fraction=bad).validate()

    def test_default_is_0_4(self):
        self.assertEqual(StrategyConfig().verify_near_fraction, 0.4)


if __name__ == "__main__":
    unittest.main()
