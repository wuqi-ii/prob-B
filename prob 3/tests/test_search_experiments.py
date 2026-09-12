"""Safety properties for continuous coverage and opt-in joint tasks."""
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cumcm_b3.backend_offline import OfflineSimulator, generate_case, WorldCases, Source
from cumcm_b3.config import StrategyConfig
from cumcm_b3.coverage import scan_points
from cumcm_b3.coverage_certificate import certifies_coverage
from cumcm_b3.experimental_strategy import ExperimentalStrategy, JointTask, make_strategy
from cumcm_b3.strategy import DogStrategy, Task


class TestCoverageCertificate(unittest.TestCase):
    def test_standard_cover_is_certified_and_holes_rejected(self):
        points = scan_points(StrategyConfig())
        self.assertTrue(certifies_coverage(points))
        self.assertFalse(certifies_coverage(points[1:]))  # hole at origin
        for i in range(1, 7):
            self.assertFalse(certifies_coverage(points[:i]+points[i+1:]))
        self.assertFalse(certifies_coverage([]))

    def test_receiving_radius_is_never_relaxed(self):
        points = scan_points(replace(StrategyConfig(), scan_ring_radius_m=765.12))
        self.assertFalse(certifies_coverage(points))

    def test_tiny_uncovered_boundary_cap_is_not_hidden_by_grid(self):
        self.assertFalse(certifies_coverage([(0.0, 0.0)], arena_radius=1000.000001))


class TestSearchExperiments(unittest.TestCase):
    def make(self, adaptive=True, joint=False):
        cfg = replace(StrategyConfig(), adaptive_search=adaptive, joint_service=joint)
        return make_strategy(OfflineSimulator(generate_case(7)), cfg)

    def test_default_is_unmodified_strategy(self):
        self.assertIs(type(self.make(False, False)), DogStrategy)

    def test_planned_or_other_channel_coverage_cannot_claim_done(self):
        strat = self.make()
        strat.visited_scan = [True]*7  # metadata alone is not evidence
        self.assertFalse(strat.status()['all_clear'])
        for ch in range(1, 20):
            strat.book[ch].no_signal_points = list(strat.scan_points)
        self.assertFalse(strat.status()['all_clear'])
        self.assertEqual(strat.status()['uncertified_channels'], [20])
        strat.book[20].no_signal_points = list(strat.scan_points)
        self.assertTrue(strat.status()['all_clear'])

    def test_unresolved_known_source_prevents_done_even_with_full_cover(self):
        strat = self.make()
        for ch in range(1, 21):
            strat.book[ch].no_signal_points = list(strat.scan_points)
        strat.book[3].add_direction((0, 0), 0)
        strat._give_up.add(3)
        self.assertFalse(strat.status()['all_clear'])

    def test_observed_maximum_count_replaces_search_but_not_clearance(self):
        strat = self.make()
        for ch in range(1, 17):
            strat.book[ch].add_near((0, 0))
        self.assertEqual(strat._unknown(), [])
        self.assertEqual(strat._scan_tasks(), [])
        self.assertFalse(strat.status()['all_clear'])
        for ch in range(1, 17):
            strat.book[ch].cleared = True
        self.assertTrue(strat.status()['all_clear'])

    def test_fifteen_observed_sources_do_not_prove_search_complete(self):
        strat = self.make()
        for ch in range(1, 16):
            strat.book[ch].add_near((0, 0))
            strat.book[ch].cleared = True
        self.assertFalse(strat.status()['all_clear'])

    def test_pruning_is_collectively_safe(self):
        strat = self.make()
        strat.visited_scan[0] = True
        for ch in range(1, 21):
            strat.book[ch].no_signal_points = [(0, 0), strat.scan_points[1]]
        remaining = strat._scan_tasks()
        self.assertLess(len(remaining), 6)
        self.assertTrue(strat._proven_with([t.point for t in remaining]))

    def test_joint_cost_includes_outgoing_target_position(self):
        task = JointTask('verify', (10, 0), 1, predicted_end=(100, 0))
        route = [Task('scan', (110, 0))]
        cost = ExperimentalStrategy._insertion_delta((0, 0), route, task, 0)
        self.assertAlmostEqual(cost, 50.0)  # zero detour + 10 s action estimate

    def test_end_to_end_all_variants(self):
        for adaptive, joint in ((True, False), (False, True), (True, True)):
            cfg = replace(StrategyConfig(), adaptive_search=adaptive, joint_service=joint)
            for seed in (7, 9):
                sim = OfflineSimulator(generate_case(seed))
                strategy = make_strategy(sim, cfg)
                stats = strategy.run()
                self.assertEqual(stats.cleared_count, sim.case.total)
                self.assertEqual(stats.stop_reason, 'all_done')

    def test_worst_reception_boundary_sources(self):
        pts = [(1800*math.cos((i+0.5)*math.pi/3), 1800*math.sin((i+0.5)*math.pi/3))
               for i in range(6)]
        pts += [(1000*math.cos((i+0.5)*math.pi/3), 1000*math.sin((i+0.5)*math.pi/3))
                for i in range(4)]
        for adaptive, joint in ((True, False), (False, True), (True, True)):
            cfg = replace(StrategyConfig(), adaptive_search=adaptive, joint_service=joint)
            case = WorldCases([Source(i+1, x, y, 1000.0) for i, (x, y) in enumerate(pts)], 1)
            stats = make_strategy(OfflineSimulator(case), cfg).run()
            self.assertEqual(stats.cleared_count, 10)
            self.assertEqual(stats.stop_reason, 'all_done')


if __name__ == '__main__':
    unittest.main()
