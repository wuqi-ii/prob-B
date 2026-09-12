"""定向源定位路径的回归测试：同向推进、信号消失恢复、环形兜底。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.backend_offline import OfflineSimulator, generate_case, Source, WorldCases  # noqa: E402
from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.second_station import approach_point  # noqa: E402
from cumcm_b4.strategy import DogStrategy  # noqa: E402


class TestApproachPoint(unittest.TestCase):
    def test_point_along_bearing(self):
        # 方位角 0 -> 正东；90 -> 正北
        p = approach_point((0.0, 0.0), 0.0, 300.0)
        self.assertAlmostEqual(p[0], 300.0, places=6)
        self.assertAlmostEqual(p[1], 0.0, places=6)
        p = approach_point((0.0, 0.0), 90.0, 300.0)
        self.assertAlmostEqual(p[0], 0.0, places=6)
        self.assertAlmostEqual(p[1], 300.0, places=6)

    def test_default_step_from_config(self):
        cfg = StrategyConfig(first_approach_step_m=250.0)
        p = approach_point((10.0, 10.0), 45.0, cfg=cfg)
        self.assertAlmostEqual(math.dist((10.0, 10.0), p), 250.0, places=6)


class TestDirectionalLocate(unittest.TestCase):
    """构造定向源，验证策略能完成"发现->推进->清除"闭环。"""

    def _single_source_case(self, x, y, direction_deg, receiver_radius=1500.0):
        src = Source(channel=1, x=x, y=y, receiver_radius_m=receiver_radius,
                     directional=True, direction_deg=direction_deg)
        return WorldCases(sources=[src], seed=0)

    def test_max_source_count_allows_safe_early_stop(self):
        """已清除题面上限16个频道后，即使扫描点未走完也必须合法结束。"""
        sim = OfflineSimulator(WorldCases(sources=[], seed=0))
        strategy = DogStrategy(sim, StrategyConfig())
        for ch in range(1, 17):
            strategy.book[ch].cleared = True
        self.assertIsNone(strategy.select_task())
        self.assertTrue(strategy.status()["all_clear"])
        self.assertTrue(strategy.status()["early_stop_at_max_sources"])

        disabled = DogStrategy(
            OfflineSimulator(WorldCases(sources=[], seed=0)),
            StrategyConfig(early_stop_at_max_sources=False),
        )
        for ch in range(1, 17):
            disabled.book[ch].cleared = True
        self.assertIsNotNone(disabled.select_task())

    def test_wedge_edge_source_is_cleared(self):
        """楔形边缘的定向源（贴近扫描点、楔边界擦过）也必须被清除。"""
        # 源在 (711, 388)，方向 352°，几乎正对东；这是 seed=1045 曾失败的几何形态
        case = self._single_source_case(711.0, 388.0, 352.1, 1014.0)
        sim = OfflineSimulator(case)
        stats = DogStrategy(sim, StrategyConfig()).run()
        self.assertEqual(stats.cleared_count, 1, f"事件轨迹：{stats.events}")
        self.assertEqual(stats.stop_reason, "all_done")

    def test_pointing_away_source_is_cleared(self):
        """定向源背对原点（楔朝外）也能被外环扫描点捕获并清除。"""
        # 源在 (1600, 0)，方向 0°（朝正东、朝外）
        case = self._single_source_case(1600.0, 0.0, 0.0, 1500.0)
        sim = OfflineSimulator(case)
        stats = DogStrategy(sim, StrategyConfig()).run()
        self.assertEqual(stats.cleared_count, 1)
        self.assertEqual(stats.stop_reason, "all_done")

    def test_near_source_cleared_directly(self):
        """近距离定向源（首测即 near）应直接清除。"""
        case = self._single_source_case(1.0, 1.0, 45.0, 1500.0)
        sim = OfflineSimulator(case)
        stats = DogStrategy(sim, StrategyConfig()).run()
        self.assertEqual(stats.cleared_count, 1)

    def test_terminal_retry_recovers_last_frozen_target(self):
        """扫描结束后的冻结目标必须获得有上限的收尾重试机会。"""
        sim = OfflineSimulator(generate_case(1, directional_prob=0.5))
        strat = DogStrategy(sim, StrategyConfig())
        strat.visited_scan = [True] * len(strat.visited_scan)
        track = strat.book[6]
        track.add_direction((0.0, 0.0), 0.0)
        track.add_direction((0.0, 100.0), 315.0)
        strat._blocked_at_obs[6] = len(track.observations)

        task = strat.select_task()
        self.assertIsNotNone(task)
        self.assertEqual(task.channel, 6)
        self.assertEqual(task.kind, "clear")
        self.assertNotIn(6, strat._blocked_at_obs)
        self.assertTrue(any(e["event"] == "terminal_retry" for e in strat.stats.events))


class TestRingClearFallback(unittest.TestCase):
    def test_ring_clear_hits_source_within_radius(self):
        """环形兜底：源在中心 15 m 附近时，7 点必有一个命中。"""
        cfg = StrategyConfig()
        src = Source(channel=5, x=10.0, y=8.0, receiver_radius_m=1500.0,
                     directional=True, direction_deg=90.0)  # 楔朝上，中心在其楔外
        sim = OfflineSimulator(WorldCases(sources=[src], seed=0))
        strat = DogStrategy(sim, cfg)
        strat.enter_done = True  # 无操作，仅为可读
        sim.enter()
        self.assertTrue(strat._ring_clear((0.0, 0.0), 5))
        self.assertTrue(src.cleared)


if __name__ == "__main__":
    unittest.main(verbosity=2)
