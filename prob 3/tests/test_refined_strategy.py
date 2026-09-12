import math
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cumcm_b3.config import StrategyConfig
from cumcm_b3.backend_offline import OfflineSimulator,generate_case,WorldCases,Source
from cumcm_b3.experimental_strategy import make_strategy
from cumcm_b3.refined_strategy import closest_safe_point,segment_safe_point
from cumcm_b3.strategy import Task


class TestRefined(unittest.TestCase):
    def test_worst_boundary_with_extreme_errors(self):
        cfg=replace(StrategyConfig(),route_multistart=True,shared_observations='opportunistic',
                    verify_near_fraction=.1,recovery_verify=True,scan_relocation=True,
                    safe_clear_point=True,step_replan=True,bearing_rounding_slack_deg=.005001)
        # Physical +/-1 degree followed by 0.01-degree output rounding can
        # expand the displayed discrepancy by as much as 0.005 degree.
        for error in (-1.005,1.005):
            case=WorldCases([Source(i+1,1800*math.cos((i+.5)*math.pi/3),
                1800*math.sin((i+.5)*math.pi/3),1000) for i in range(6)]+[
                Source(i+7,800*math.cos(i*math.pi/2),800*math.sin(i*math.pi/2),1000)
                for i in range(4)],1)
            with patch('cumcm_b3.backend_offline._deterministic_error',return_value=error):
                s=make_strategy(OfflineSimulator(case),cfg)
                stats=s.run()
            self.assertEqual(stats.cleared_count,case.total)
            self.assertEqual(stats.stop_reason,'all_done')

    def test_parallel_bearing_failure_recovers(self):
        for extra in ({},{'step_replan':True,'safe_clear_point':True,'scan_relocation':True}):
            cfg=replace(StrategyConfig(),route_multistart=True,shared_observations='opportunistic',
                        verify_near_fraction=.1,recovery_verify=True,**extra)
            case=generate_case(32181)
            s=make_strategy(OfflineSimulator(case),cfg)
            stats=s.run()
            self.assertEqual(stats.cleared_count,case.total)
            self.assertEqual(stats.stop_reason,'all_done')
            self.assertTrue(any(e['event']=='recovery_verify' for e in stats.events))

    def test_upper_bound_stops_without_remaining_scans(self):
        cfg=replace(StrategyConfig(),route_multistart=True,stop_at_source_upper_bound=True)
        s=make_strategy(OfflineSimulator(generate_case(1)),cfg)
        for ch in range(1,17):
            s.book[ch].cleared=True
        self.assertIsNone(s.select_task())
        self.assertTrue(s.status()['all_clear'])
        self.assertEqual(s.status()['completion_basis'],'cleared_source_count_upper_bound')
        s.book[16].cleared=False
        self.assertIsNotNone(s.select_task())

    def test_discovering_sixteen_sources_removes_only_search_tasks(self):
        cfg=replace(StrategyConfig(),route_multistart=True,
                    stop_search_when_all_sources_known=True)
        s=make_strategy(OfflineSimulator(generate_case(1)),cfg)
        for ch in range(1,17):
            s.book[ch].add_direction((0.0,0.0),float(ch))
        self.assertEqual(s._scan_tasks(),[])
        self.assertFalse(s.status()['all_clear'])
        self.assertTrue(any(t.kind=='verify' for t in s._service_tasks((0.0,0.0),[])))

    def test_relocated_scans_use_identity_not_coordinates(self):
        sim=OfflineSimulator(WorldCases([],1))
        sim.enter()
        s=make_strategy(sim,replace(StrategyConfig(),scan_relocation=True))
        s.visited_scan[0]=True
        s.scan_points[1]=s.scan_points[0]
        s.execute(Task('scan',s.scan_points[1],note='S1'))
        self.assertTrue(s.visited_scan[1])

    def test_relocated_and_range_boundary(self):
        cfg=replace(StrategyConfig(),scan_relocation=True,received_range_bound=True,
                    route_multistart=True,shared_observations='opportunistic',verify_near_fraction=.1)
        case=WorldCases([Source(i+1,1800*math.cos(i*math.pi/5),1800*math.sin(i*math.pi/5),1000)
                         for i in range(10)],1)
        s=make_strategy(OfflineSimulator(case),cfg)
        result=s.run()
        self.assertEqual(result.cleared_count,10)
        self.assertTrue(s._proven_with([]))

    def test_safe_projection(self):
        poly=[(-5,-5),(5,-5),(5,5),(-5,5)]
        p=closest_safe_point((-100,0),poly)
        self.assertIsNotNone(p)
        self.assertTrue(all(math.dist(p,v)<=18+1e-8 for v in poly))
        self.assertLess(math.dist((-100,0),p),100)
        self.assertIsNone(closest_safe_point((0,0),[(-30,0),(30,0)]))
        self.assertEqual(closest_safe_point((0,0),poly),(0,0))
        crossing=segment_safe_point((-100,0),(100,0),poly)
        self.assertIsNotNone(crossing)
        self.assertTrue(all(math.dist(crossing,v)<=18+1e-7 for v in poly))
        self.assertIsNone(segment_safe_point((-100,100),(100,100),poly))

    def test_refinements_clear(self):
        for overrides in [dict(route_time_score=True),dict(step_replan=True),dict(safe_clear_point=True),
                          dict(route_time_score=True,step_replan=True,safe_clear_point=True)]:
            cfg=replace(StrategyConfig(),shared_observations='opportunistic',verify_near_fraction=.1,
                        route_multistart=True,**overrides)
            for case in [generate_case(11),WorldCases([Source(i+1,1800*math.cos(i*math.pi/5),
                        1800*math.sin(i*math.pi/5),1000) for i in range(10)],1)]:
                result=make_strategy(OfflineSimulator(case),cfg).run()
                self.assertEqual(result.stop_reason,'all_done')
                self.assertEqual(result.cleared_count,case.total)
