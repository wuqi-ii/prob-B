"""覆盖性、第二检测点与端到端行为的测试。"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3 import coverage, second_station  # noqa: E402
from cumcm_b3.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b3.config import (  # noqa: E402
    ARENA_RADIUS_M, RECEIVER_MAX_M, RECEIVER_MIN_M, StrategyConfig,
)
from cumcm_b3.strategy import DogStrategy  # noqa: E402


class TestCoverage(unittest.TestCase):
    """七点布局必须真的把整个圆域盖住，而且要留出安全余量。"""

    def setUp(self):
        self.cfg = StrategyConfig()

    def test_points(self):
        pts = coverage.scan_points(self.cfg)
        self.assertEqual(len(pts), 7)
        self.assertEqual(pts[0], (0.0, 0.0))
        for p in pts:
            self.assertLessEqual(math.hypot(*p), ARENA_RADIUS_M)
        for p in pts[1:]:
            self.assertAlmostEqual(math.hypot(*p), self.cfg.scan_ring_radius_m, places=6)

    def test_worst_gap_below_reception(self):
        info = coverage.worst_case_distance(self.cfg)
        self.assertAlmostEqual(info["worst_distance_m"], 988.5, places=1)
        self.assertLess(info["worst_distance_m"], RECEIVER_MIN_M)
        self.assertGreater(info["margin_m"], 10.0)
        self.assertTrue(info["covers"])

    def test_grid_matches_analysis(self):
        """独立网格校核必须与解析结论一致（允许网格离散带来的几米偏差）。"""
        ana = coverage.worst_case_distance(self.cfg)["worst_distance_m"]
        grid = coverage.grid_verify(self.cfg, step_m=10.0)
        self.assertTrue(grid["covers"])
        self.assertLess(abs(grid["grid_worst_distance_m"] - ana), 15.0)

    def test_six_points_are_not_enough(self):
        """只用六个环点会把圆心区域漏掉——这是"至少七点"的依据之一。"""
        arg = coverage.minimum_points_argument()
        self.assertFalse(arg["covers"])
        self.assertGreater(arg["shortfall_m"], 0.0)

    def test_radius_monotone_margin(self):
        """环半径从 1123 增大时余量先增后减；1150 处在安全侧。"""
        margins = []
        for d in (1123.0, 1150.0, 1300.0, 1560.0, 1700.0):
            cfg = StrategyConfig(scan_ring_radius_m=d)
            margins.append(coverage.worst_case_distance(cfg)["margin_m"])
        self.assertGreater(margins[0], -1.0)          # 1123 是临界
        self.assertTrue(all(m > 0 for m in margins))  # 其余都安全

    def test_config_rejects_layout_without_guaranteed_coverage(self):
        with self.assertRaisesRegex(ValueError, "不能按最坏接收半径覆盖"):
            StrategyConfig(scan_ring_radius_m=100.0).validate()


class TestSecondStation(unittest.TestCase):
    def setUp(self):
        self.cfg = StrategyConfig()

    def test_pair_is_mirrored(self):
        pair = second_station.candidate_pair((0.0, 0.0), 0.0, self.cfg)
        # 两点关于示向线对称，且到第一站的行程相同
        self.assertAlmostEqual(abs(pair.left[1]), abs(pair.right[1]), places=6)
        self.assertAlmostEqual(pair.left[0], pair.right[0], places=6)
        self.assertAlmostEqual(
            math.dist((0.0, 0.0), pair.left), math.dist((0.0, 0.0), pair.right), places=6
        )

    def test_pair_reuses_problem2_optimum(self):
        """默认参数下应直接复用问题 2 算出的 (764, 526)。"""
        pair = second_station.candidate_pair((0.0, 0.0), 0.0, self.cfg)
        self.assertAlmostEqual(pair.local_optimum[0], 764.0, places=6)
        self.assertAlmostEqual(pair.local_optimum[1], 526.0, places=6)
        self.assertAlmostEqual(pair.separation_m, 927.56, places=1)

    def test_pair_coordinates_are_configurable(self):
        cfg = StrategyConfig(
            second_station_local_a_m=750.0,
            second_station_local_b_m=520.0,
        )
        cfg.validate()
        pair = second_station.candidate_pair((0.0, 0.0), 0.0, cfg)
        self.assertEqual(pair.local_optimum, (750.0, 520.0))
        self.assertEqual(pair.left, (750.0, 520.0))
        self.assertEqual(pair.right, (750.0, -520.0))

    def test_config_rejects_second_station_outside_candidate_region(self):
        with self.assertRaisesRegex(ValueError, "候选区域硬约束"):
            StrategyConfig(
                second_station_local_a_m=100.0,
                second_station_local_b_m=100.0,
            ).validate()

    def test_rotation(self):
        """方位角 90° 时，局部 x 轴应指向正北。"""
        pair = second_station.candidate_pair((100.0, -50.0), 90.0, self.cfg)
        self.assertAlmostEqual(pair.right[0], 100.0 + 526.0, places=6)
        self.assertAlmostEqual(pair.right[1], -50.0 + 764.0, places=6)

    def test_local_to_global_inverse_consistency(self):
        for bearing in (0.0, 37.0, 123.0, 250.0):
            p = second_station.local_to_global((764.0, 526.0), (10.0, 20.0), bearing)
            d = math.dist((10.0, 20.0), p)
            self.assertAlmostEqual(d, math.hypot(764.0, 526.0), places=6)

    def test_dynamic_R_keeps_original_point_without_boundary_clipping(self):
        cfg = replace(self.cfg, dynamic_second_station=True)
        pair = second_station.candidate_pair((0.0, 0.0), 0.0, cfg)
        self.assertFalse(pair.boundary_clipped)
        self.assertEqual(pair.effective_far_m, 1500.0)
        self.assertEqual(pair.local_optimum, (764.0, 526.0))

    def test_dynamic_R_shortens_outward_boundary_case(self):
        cfg = replace(self.cfg, dynamic_second_station=True,
                      bearing_rounding_slack_deg=0.005001)
        pair = second_station.candidate_pair((1200.0, 0.0), 20.0, cfg)
        self.assertTrue(pair.boundary_clipped)
        self.assertGreater(pair.effective_far_m, 620.0)
        self.assertLess(pair.effective_far_m, 640.0)
        self.assertLess(pair.separation_m, 450.0)

    def test_dynamic_R_uses_whole_error_wedge_not_only_center_ray(self):
        cfg = replace(self.cfg, dynamic_second_station=True)
        far, clipped = second_station.effective_far_distance((1200.0, 0.0), 20.0, cfg)
        center = second_station._ray_exit_distance((1200.0, 0.0), math.radians(20.0))
        self.assertTrue(clipped)
        self.assertGreater(far, center)

    def test_dynamic_R_piecewise_points_remain_feasible(self):
        cfg = replace(self.cfg, dynamic_second_station=True)
        for far in range(100, 1501, 25):
            a, b = second_station.parametric_local_point(float(far), cfg)
            h = far * math.tan(math.radians(1.0))
            t = math.tan(math.radians(cfg.second_station_min_angle_deg))
            self.assertTrue(second_station._feasible(
                a, b, cfg.second_station_reach_m, float(far), h, t), far)

    def test_reusable_station_uses_problem2_hard_region(self):
        cfg = replace(self.cfg, planned_stop_reuse=True)
        safe, far = second_station.certifies_reusable_station(
            (700.0, 520.0), (0.0, 0.0), 0.0, cfg)
        self.assertTrue(safe)
        self.assertEqual(far, 1500.0)
        unsafe, _ = second_station.certifies_reusable_station(
            (300.0, 0.0), (0.0, 0.0), 0.0, cfg)
        self.assertFalse(unsafe)



class TestEndToEnd(unittest.TestCase):
    """在本地模拟器上端到端验证：全部清除，且时间量级合理。"""

    def run_seed(self, seed: int) -> tuple[int, int, float]:
        cfg = StrategyConfig()
        case = generate_case(seed)
        sim = OfflineSimulator(case)
        DogStrategy(sim, cfg).run()
        return case.total, sim.case.count_cleared(), sim.virtual_time_s

    def test_all_cleared_multiple_seeds(self):
        for seed in range(1, 21):
            total, cleared, _ = self.run_seed(seed)
            self.assertEqual(total, cleared, f"seed={seed} 未全部清除")
            self.assertGreaterEqual(total, 10)
            self.assertLessEqual(total, 16)

    def test_time_is_reasonable(self):
        _, _, t = self.run_seed(1)
        self.assertLess(t, 3 * 3600.0)   # 远低于 100 小时上限
        self.assertGreater(t, 600.0)

    def test_source_count_range(self):
        """案例生成必须落在赛题给定的 10~16 范围内。"""
        for seed in range(500, 530):
            case = generate_case(seed)
            self.assertTrue(10 <= case.total <= 16)

    def test_channels_are_unique(self):
        case = generate_case(42)
        channels = [s.channel for s in case.sources]
        self.assertEqual(len(channels), len(set(channels)))
        self.assertTrue(all(1 <= c <= 20 for c in channels))

    def test_receiver_radius_range(self):
        case = generate_case(43)
        for s in case.sources:
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

    def test_scan_can_skip_already_localized_channels(self):
        cfg = replace(StrategyConfig(), scan_all_channels_at_scan_points=False)
        sim = OfflineSimulator(generate_case(1))
        sim.enter()
        strategy = DogStrategy(sim, cfg)
        strategy.book[1].add_direction((10.0, 0.0), 0.0)
        strategy.book[1].add_direction((10.0, 10.0), 0.0)
        strategy.book[2].add_direction((10.0, 0.0), 0.0)
        for ch in range(3, 21):
            strategy.book[ch].cleared = True
        before = sim.request_count
        strategy._visit_scan_point(0)
        self.assertEqual(sim.request_count - before, 1)


class TestOfflineSimulatorRules(unittest.TestCase):
    """模拟器本身必须忠实复现附录规则。"""

    def test_measure_cost_accounting(self):
        case = generate_case(99)
        sim = OfflineSimulator(case)
        sim.enter()
        t_before = sim.virtual_time_s
        sim.measure(500.0, 0.0, 1)         # 移动 500 m + 检测 5 s（频道已是 1）
        self.assertAlmostEqual(sim.virtual_time_s - t_before, 500 / 5.0 + 5.0, places=6)
        sim.measure(500.0, 0.0, 2)         # 不移动 + 切换 1 s + 检测 5 s
        self.assertAlmostEqual(sim.virtual_time_s - t_before, 100.0 + 5.0 + 6.0, places=6)

    def test_clear_does_not_switch_channel(self):
        case = generate_case(100)
        sim = OfflineSimulator(case)
        sim.enter()
        sim.measure(0.0, 0.0, 3)
        self.assertEqual(sim.current_channel, 3)
        sim.clear(0.0, 0.0, 5)
        self.assertEqual(sim.current_channel, 3, "/clear 不应改变测向机频道")

    def test_clear_miss_cost(self):
        case = generate_case(101)
        sim = OfflineSimulator(case)
        sim.enter()
        t0 = sim.virtual_time_s
        sim.clear(0.0, 0.0, 1)
        used = sim.virtual_time_s - t0
        self.assertIn(used, (3.0, 5.0))

    def test_repeated_measure_is_deterministic(self):
        """同一地点重复检测必须给出完全相同的示向度。"""
        case = generate_case(102)
        sim = OfflineSimulator(case)
        sim.enter()
        ch = case.sources[0].channel
        # 找一个能收到信号的位置
        probes = [(100.0, 0.0), (0.0, 100.0), (-100.0, 0.0), (0.0, -100.0)]
        for p in probes:
            r1 = sim.measure(p[0], p[1], ch)
            r2 = sim.measure(p[0], p[1], ch)
            if r1.get("measure_result") == "direction":
                self.assertEqual(r1["svd_deg"], r2["svd_deg"])
                break


if __name__ == "__main__":
    unittest.main(verbosity=2)
