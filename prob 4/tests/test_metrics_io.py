"""产物序列化的严格性测试。

2026-09-11 发现：``json.dumps`` 默认允许 ``Infinity`` / ``NaN`` 字面量，
而它们**不是** RFC 8259 允许的写法——JS 的 ``JSON.parse`` 会直接抛错。
本工程多处参数取 ``inf``（虚拟时间软预算）或可能为 ``nan``（清除数为 0 时的平均时间），
所以写盘前必须统一把非有限浮点收口成 ``null``。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b4.config import StrategyConfig  # noqa: E402
from cumcm_b4.metrics import append_jsonl, dumps_strict, write_json  # noqa: E402


def _strict_load(text: str):
    """严格解析：遇到 Infinity / NaN 直接判为非法。"""
    def _reject(constant: str):
        raise ValueError(f"非法 JSON 常量 {constant}")

    return json.loads(text, parse_constant=_reject)


class TestStrictJsonArtifacts(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="strict_json_"))

    def test_effective_config_is_strict_json(self):
        record = StrategyConfig().to_dict()
        record["robot_id"] = "***"          # 与 run_drill 的脱敏写法一致
        self.assertEqual(record["virtual_time_budget_s"], float("inf"))

        path = self.tmp / "effective_config.json"
        write_json(path, record)
        parsed = _strict_load(path.read_text(encoding="utf-8"))
        self.assertIsNone(parsed["virtual_time_budget_s"], "inf 必须写成 null")
        self.assertEqual(parsed["scan_inner_radius_m"], 980.0)
        self.assertEqual(parsed["scan_outer_radius_m"], 1840.0)
        self.assertEqual(parsed["scan_inner_count"], 8)
        self.assertEqual(parsed["scan_outer_count"], 18)

    def test_drill_index_line_is_strict_jsonl(self):
        path = self.tmp / "drill_index.jsonl"
        append_jsonl(path, {"tag": "t1", "mean_clear_time_s": float("nan"),
                            "cleared_ratio": None, "requests": 3})
        append_jsonl(path, {"tag": "t2", "mean_clear_time_s": 320.3})
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertEqual(len(lines), 2)
        for line in lines:
            _strict_load(line)
        self.assertIsNone(_strict_load(lines[0])["mean_clear_time_s"])

    def test_nested_non_finite_is_sanitised(self):
        payload = {"a": [1.0, float("inf")], "b": {"c": float("-inf"), "d": float("nan")}}
        text = dumps_strict(payload)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)
        parsed = _strict_load(text)
        self.assertEqual(parsed, {"a": [1.0, None], "b": {"c": None, "d": None}})

    def test_finite_values_are_untouched(self):
        payload = {"v": 0.0, "w": -1.5, "n": 7, "s": "x", "b": True, "nil": None}
        self.assertEqual(_strict_load(dumps_strict(payload)), payload)


if __name__ == "__main__":
    unittest.main()
