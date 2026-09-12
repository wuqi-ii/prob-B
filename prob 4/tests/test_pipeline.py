"""问题 4 的覆盖性、定向源语义与端到端行为测试。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4 import coverage  # noqa: E402
from cumcm_b4.backend_offline import OfflineSimulator, generate_case, Source  # noqa: E402
from cumcm_b4.config import (  # noqa: E402
    ARENA_RADIUS_M, RECEIVER_MAX_M, RECEIVER_MIN_M, StrategyConfig,
)
from cumcm_b4.strategy import DogStrategy  # noqa: E402


class TestSurroundingNet(unittest.TestCase):
    """包围网必须真的把任意位置、任意朝向的源都盖住，且留出安全余量。"""

    def setUp(self):
        self.cfg = StrategyConfig()

    def test_point_count_and_layout(self):
        pts = coverage.scan_points(self.cfg)
        # 中心 + 内环 + 外环（点数由配置决定，当前默认 1 + 8 + 14 = 23）
        self.assertEqual(len(pts), 1 + self.cfg.scan_inner_count + self.cfg.scan_outer_count)
        self.assertEqual(pts[0], (0.0, 0.0))
        inner = pts[1 : 1 + self.cfg.scan_inner_count]
        outer = pts[1 + self.cfg.scan_inner_count :]
        for p in inner:
            self.assertAlmostEqual(math.hypot(*p), self.cfg.scan_inner_radius_m, places=6)
        for p in outer:
            self.assertAlmostEqual(math.hypot(*p), self.cfg.scan_outer_radius_m, places=6)

    def test_outer_ring_is_outside_arena(self):
        """外环必须在目标区域外，否则贴边且楔朝外的源会漏检。"""
        self.assertGreater(self.cfg.scan_outer_radius_m, ARENA_RADIUS_M)
        for p in coverage.scan_points(self.cfg)[1 + self.cfg.scan_inner_count :]:
            self.assertGreater(math.hypot(*p), ARENA_RADIUS_M)

    def test_worst_required_radius_below_reception(self):
        worst = coverage.coverage_worst_required_radius(self.cfg, step_m=40.0, ang_step_deg=3.0)
        self.assertLess(worst, RECEIVER_MIN_M)
        self.assertGreater(RECEIVER_MIN_M - worst, 20.0)  # 至少 20 m 接收半径余量

    def test_grid_verify_covers(self):
        info = coverage.grid_verify(self.cfg, step_m=40.0, ang_step_deg=3.0)
        self.assertTrue(info["covers"])
        self.assertGreater(info["margin_m"], 20.0)

    def test_orientation_check_catches_narrow_outward_gap(self):
        """角度采样会漏掉的极窄朝外盲区，精确极角判据必须识别。"""
        from dataclasses import replace

        unsafe = replace(
            self.cfg,
            scan_outer_count=16,
            scan_outer_radius_m=1805.0,
        )
        points = coverage.scan_points(unsafe)
        required = coverage.required_radius_at_position((-1787.7, -37.4), points)
        self.assertTrue(math.isinf(required))

        safe_points = coverage.scan_points(self.cfg)
        safe_required = coverage.required_radius_at_position(
            (-1787.7, -37.4), safe_points
        )
        self.assertLess(safe_required, RECEIVER_MIN_M)

    def test_config_rejects_layout_without_guaranteed_coverage(self):
        # 外环收到区域内部（无法包围贴边源）应被拒绝
        with self.assertRaisesRegex(ValueError, "scan_outer_radius_m"):
            StrategyConfig(scan_outer_radius_m=1700.0).validate()


class TestDirectionalSemantics(unittest.TestCase):
    """定向源的检测与清除语义必须与赛题一致。"""

    def _case(self) -> OfflineSimulator:
        # 一个定向源，方向指向 +x，覆盖 [270, 90]（即右半平面）
        src = Source(channel=1, x=0.0, y=0.0, receiver_radius_m=1500.0,
                     directional=True, direction_deg=0.0)
        from cumcm_b4.backend_offline import WorldCases
        sim = OfflineSimulator(WorldCases(sources=[src], seed=0))
        sim.enter()
        return sim

    def test_directional_in_wedge_detected(self):
        sim = self._case()
        r = sim.measure(500.0, 0.0, 1)   # 正右方，在楔内
        self.assertEqual(r["measure_result"], "direction")

    def test_directional_out_of_wedge_not_detected(self):
        sim = self._case()
        r = sim.measure(-500.0, 0.0, 1)  # 正左方，在楔外
        self.assertEqual(r["measure_result"], "no_signal")

    def test_directional_near_requires_in_wedge(self):
        sim = self._case()
        r = sim.measure(3.0, 0.0, 1)     # 3 m 且在楔内 -> near
        self.assertEqual(r["measure_result"], "near")
        sim2 = self._case()
        r2 = sim2.measure(-3.0, 0.0, 1)  # 3 m 但在楔外 -> no_signal（不是 near）
        self.assertEqual(r2["measure_result"], "no_signal")

    def test_clear_ignores_wedge(self):
        """/clear 只与距离有关、与覆盖角无关：楔外也能清。"""
        sim = self._case()
        r = sim.clear(-10.0, 0.0, 1)     # 楔外，但距离 <=20 m
        self.assertEqual(r["clear_result"], "success")

    def test_omnidirectional_detected_anywhere(self):
        src = Source(channel=1, x=0.0, y=0.0, receiver_radius_m=1500.0, directional=False)
        from cumcm_b4.backend_offline import WorldCases
        sim = OfflineSimulator(WorldCases(sources=[src], seed=0))
        sim.enter()
        self.assertEqual(sim.measure(500.0, 0.0, 1)["measure_result"], "direction")
        self.assertEqual(sim.measure(-500.0, 0.0, 1)["measure_result"], "direction")

    def test_generate_case_mixes_directional(self):
        # 全向（prob=0）与全定向（prob=1）应精确符合
        all_omni = generate_case(1, directional_prob=0.0)
        self.assertEqual(all_omni.directional_count(), 0)
        all_dir = generate_case(1, directional_prob=1.0)
        self.assertEqual(all_dir.directional_count(), all_dir.total)


class TestEndToEnd(unittest.TestCase):
    """在本地模拟器上端到端验证：全部清除，且时间量级合理。"""

    def run_seed(self, seed: int, directional_prob: float = 0.5):
        cfg = StrategyConfig()
        case = generate_case(seed, directional_prob=directional_prob)
        sim = OfflineSimulator(case)
        stats = DogStrategy(sim, cfg).run()
        return case, sim, stats

    def test_all_cleared_multiple_seeds(self):
        for seed in range(1, 21):
            case, sim, stats = self.run_seed(seed)
            self.assertEqual(case.total, sim.case.count_cleared(), f"seed={seed} 未全部清除")
            self.assertEqual(stats.stop_reason, "all_done")
            self.assertTrue(10 <= case.total <= 16)

    def test_clears_across_directional_fractions(self):
        """定向源占比从 0 到 100% 都应全部清除。"""
        for prob in (0.0, 0.5, 1.0):
            for seed in (101, 102, 103):
                case, sim, stats = self.run_seed(seed, directional_prob=prob)
                self.assertEqual(case.total, sim.case.count_cleared(),
                                 f"prob={prob} seed={seed} 未全部清除")

    def test_time_is_reasonable(self):
        _, sim, _ = self.run_seed(1)
        self.assertLess(sim.virtual_time_s, 5 * 3600.0)  # 远低于 100 小时上限
        self.assertGreater(sim.virtual_time_s, 600.0)

    def test_source_count_range(self):
        for seed in range(500, 530):
            self.assertTrue(10 <= generate_case(seed).total <= 16)

    def test_channels_are_unique(self):
        case = generate_case(42)
        channels = [s.channel for s in case.sources]
        self.assertEqual(len(channels), len(set(channels)))
        self.assertTrue(all(1 <= c <= 20 for c in channels))

    def test_receiver_radius_range(self):
        for s in generate_case(43).sources:
            self.assertTrue(RECEIVER_MIN_M <= s.receiver_radius_m <= RECEIVER_MAX_M)
            self.assertLessEqual(math.hypot(s.x, s.y), ARENA_RADIUS_M)

    def test_unresolved_known_target_is_not_reported_all_done(self):
        cfg = StrategyConfig()
        strategy = DogStrategy(OfflineSimulator(generate_case(1)), cfg)
        strategy.visited_scan = [True] * len(strategy.scan_points)
        strategy.book[1].add_near((0.0, 0.0))
        strategy._give_up.add(1)
        self.assertIsNone(strategy.select_task())
        self.assertEqual(strategy._no_task_stop_reason(), "unresolved_targets")
        self.assertFalse(strategy.status()["all_clear"])


class TestOfflineSimulatorRules(unittest.TestCase):
    """模拟器本身必须忠实复现附录规则（含定向源）。"""

    def test_measure_cost_accounting(self):
        case = generate_case(99)
        sim = OfflineSimulator(case)
        sim.enter()
        t_before = sim.virtual_time_s
        sim.measure(500.0, 0.0, 1)
        self.assertAlmostEqual(sim.virtual_time_s - t_before, 500 / 5.0 + 5.0, places=6)
        sim.measure(500.0, 0.0, 2)
        self.assertAlmostEqual(sim.virtual_time_s - t_before, 100.0 + 5.0 + 6.0, places=6)

    def test_clear_does_not_switch_channel(self):
        case = generate_case(100)
        sim = OfflineSimulator(case)
        sim.enter()
        sim.measure(0.0, 0.0, 3)
        self.assertEqual(sim.current_channel, 3)
        sim.clear(0.0, 0.0, 5)
        self.assertEqual(sim.current_channel, 3)

    def test_clear_miss_cost(self):
        case = generate_case(101)
        sim = OfflineSimulator(case)
        sim.enter()
        t0 = sim.virtual_time_s
        sim.clear(0.0, 0.0, 1)
        self.assertIn(sim.virtual_time_s - t0, (3.0, 5.0))

    def test_repeated_measure_is_deterministic(self):
        case = generate_case(102)
        sim = OfflineSimulator(case)
        sim.enter()
        ch = case.sources[0].channel
        probes = [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0), (0.0, -100.0)]
        for p in probes:
            r1 = sim.measure(p[0], p[1], ch)
            r2 = sim.measure(p[0], p[1], ch)
            if r1.get("measure_result") == "direction":
                self.assertEqual(r1["svd_deg"], r2["svd_deg"])
                break


if __name__ == "__main__":
    unittest.main(verbosity=2)
