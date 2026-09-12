"""配置守卫与扫描任务解析的回归测试。

覆盖两处容易被"静默绕过"的地方：
1. ``scan_relocation`` 会拦截 ``scan`` 任务的执行分支，必须与覆盖/批量实验互斥；
2. ``RelocatedStrategy`` 把 ``scan`` 任务解析成扫描点下标时不能依赖 note 的固定格式。
"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3.backend_offline import OfflineSimulator, generate_case  # noqa: E402
from cumcm_b3.config import StrategyConfig  # noqa: E402
from cumcm_b3.experimental_strategy import make_strategy  # noqa: E402
from cumcm_b3.relocated_strategy import RelocatedStrategy  # noqa: E402
from cumcm_b3.strategy import Task  # noqa: E402


class TestMutuallyExclusiveFlags(unittest.TestCase):
    def test_validate_rejects_conflicting_flags(self):
        base = replace(StrategyConfig(), scan_relocation=True)
        for kwargs in (
            {"adaptive_search": True},
            {"joint_service": True},
            {"shared_observations": "batch"},
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, "分开使用"):
                replace(base, **kwargs).validate()

    def test_valid_combination_still_passes(self):
        replace(StrategyConfig(), scan_relocation=True).validate()

    def test_factory_is_self_guarding(self):
        """直接调用工厂（不经过入口脚本）也必须拒绝互斥配置。"""
        cfg = replace(StrategyConfig(), scan_relocation=True, adaptive_search=True)
        with self.assertRaisesRegex(ValueError, "分开使用"):
            make_strategy(OfflineSimulator(generate_case(1)), cfg)


class TestScanTaskResolution(unittest.TestCase):
    def _strategy(self):
        cfg = replace(StrategyConfig(), scan_relocation=True)
        cfg.validate()
        strategy = make_strategy(OfflineSimulator(generate_case(1)), cfg)
        self.assertIsInstance(strategy, RelocatedStrategy)
        return strategy

    def test_standard_note_resolves(self):
        strategy = self._strategy()
        for idx in (0, 3, 6):
            task = Task("scan", strategy.scan_points[idx], note=f"S{idx}")
            self.assertEqual(strategy._scan_index(task), idx)

    def test_note_without_index_falls_back_to_point(self):
        strategy = self._strategy()
        task = Task("scan", strategy.scan_points[2], note="unknown")
        self.assertEqual(strategy._scan_index(task), 2)

    def test_out_of_range_note_falls_back_to_point(self):
        strategy = self._strategy()
        task = Task("scan", strategy.scan_points[1], note="S99")
        self.assertEqual(strategy._scan_index(task), 1)

    def test_unresolvable_task_raises_clear_error(self):
        strategy = self._strategy()
        task = Task("scan", (1e9, 1e9), note="adaptive:1150:0:0")
        with self.assertRaisesRegex(ValueError, "无法把扫描任务定位到扫描点"):
            strategy._scan_index(task)


if __name__ == "__main__":
    unittest.main()
