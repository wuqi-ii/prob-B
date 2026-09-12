import math
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import replace
from cumcm_b3.backend_offline import OfflineSimulator, generate_case, WorldCases, Source
from cumcm_b3.config import StrategyConfig
from cumcm_b3.experimental_strategy import make_strategy


class TestForward(unittest.TestCase):
    def test_candidate_planning_does_not_invent_observations(self):
        s = make_strategy(OfflineSimulator(generate_case(1)),
            replace(StrategyConfig(),adaptive_verify=True,route_multistart=True))
        s.book[1].add_direction((0,0),0)
        s.select_task()
        self.assertEqual(s.book[1].observations,[((0,0),0)])
        self.assertFalse(s.book[1].cleared)

    def test_boundary_and_random(self):
        for av, mr in [(True,False),(False,True),(True,True)]:
            cfg=replace(StrategyConfig(),adaptive_verify=av,route_multistart=mr,
                shared_observations='opportunistic',verify_near_fraction=.1)
            cases=[generate_case(3),generate_case(5), WorldCases([
                Source(i+1,1800*math.cos(i*math.pi/5),1800*math.sin(i*math.pi/5),1000)
                for i in range(10)],1)]
            for case in cases:
                stats=make_strategy(OfflineSimulator(case),cfg).run()
                self.assertEqual(stats.stop_reason,'all_done')
                self.assertEqual(stats.cleared_count,case.total)
