import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from cumcm_b3.backend_offline import OfflineSimulator, generate_case, WorldCases, Source
from cumcm_b3.config import StrategyConfig
from cumcm_b3.experimental_strategy import make_strategy
from cumcm_b3.shared_strategy import SharedStrategy, BatchTask, polish_route
from cumcm_b3.strategy import DogStrategy, Task


class TestSharedStrategy(unittest.TestCase):
    def test_open_route_polish_preserves_tasks_and_decreases_length(self):
        tasks = [Task('clear', (100, 0), 1), Task('clear', (0, 100), 2),
                 Task('clear', (100, 100), 3), Task('clear', (50, 0), 4)]
        route = polish_route((0, 0), tasks)
        self.assertEqual({id(t) for t in route}, {id(t) for t in tasks})
        self.assertLess(DogStrategy._path_length((0, 0), route), DogStrategy._path_length((0, 0), tasks))

    def test_planning_does_not_add_hypothetical_observations(self):
        cfg = replace(StrategyConfig(), shared_observations='batch')
        s = make_strategy(OfflineSimulator(generate_case(7)), cfg)
        for ch, angle in [(1, 0), (2, 15), (3, 30)]:
            s.book[ch].add_direction((0, 0), angle)
        before = {ch: list(tr.observations) for ch, tr in s.book.tracks.items()}
        s.select_task()
        self.assertEqual(before, {ch: tr.observations for ch, tr in s.book.tracks.items()})

    def test_repeat_point_has_no_information_gain(self):
        s = make_strategy(OfflineSimulator(generate_case(7)),
                          replace(StrategyConfig(), shared_observations='opportunistic'))
        s.book[1].add_direction((0, 0), 0)
        self.assertEqual(s._gain(1, (0, 0)), 0)
        s.book[1].add_no_signal((100, 100))
        self.assertEqual(s._gain(1, (100, 100)), 0)

    def test_batch_no_signal_does_not_invent_a_direction(self):
        cfg = replace(StrategyConfig(), shared_observations='batch')
        sim = OfflineSimulator(WorldCases([], 1))
        sim.enter()
        s = make_strategy(sim, cfg)
        s.execute(BatchTask('batch', (0, 0), channels=(1, 2)))
        self.assertEqual(s.book[1].observations, [])
        self.assertEqual(s.book[2].observations, [])
        self.assertFalse(s.status()['all_clear'])

    def test_variants_clear_random_and_boundary_cases(self):
        for mode, polish, fraction in [('opportunistic', False, .4), ('batch', False, .4),
                                      ('off', True, .4), ('opportunistic', False, .1),
                                      ('batch', False, .2)]:
            cfg = replace(StrategyConfig(), shared_observations=mode, route_polish=polish,
                          verify_near_fraction=fraction)
            cases = [generate_case(7), generate_case(9)]
            cases.append(WorldCases([Source(i+1, 1800*math.cos((i+.5)*math.pi/5),
                1800*math.sin((i+.5)*math.pi/5), 1000.0) for i in range(10)], 1))
            for case in cases:
                stats = make_strategy(OfflineSimulator(case), cfg).run()
                self.assertEqual(stats.cleared_count, case.total)
                self.assertEqual(stats.stop_reason, 'all_done')


if __name__ == '__main__':
    unittest.main()
