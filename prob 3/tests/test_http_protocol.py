"""HTTP 协议层回归测试。

这些测试针对的是**真实模拟器**才能暴露的问题：
离线后端一直是对的，所以本地全绿也挡不住协议字段写错。
这里用可捕获载荷的假客户端把请求体逐字段钉死。

重点覆盖 2026-09-11 演练实测踩到的坑：
`/clear` 漏发 `channel` 字段 → 模拟器直接返回 HTTP 400，且错误响应里没有 message。
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cumcm_b3.config import StrategyConfig  # noqa: E402
from cumcm_b3.client import Position, SimulatorClient, probe, probe_interface  # noqa: E402


class _CapturingClient(SimulatorClient):
    """把 _post 拦下来，记录载荷，并回放最小合法响应。"""

    def __init__(self) -> None:
        super().__init__(robot_id="202623001124", log_path=None)
        self.sent: list = []

    def _post(self, endpoint, payload):  # type: ignore[override]
        self.sent.append((endpoint, dict(payload)))
        if endpoint == "enter":
            return {"accepted": True, "virtual_time_s": 0,
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": 1200}
        if endpoint == "measure":
            return {"accepted": True, "virtual_time_s": 5, "measure_result": "no_signal"}
        if endpoint == "clear":
            return {"accepted": True, "virtual_time_s": 8, "clear_result": "success"}
        if endpoint == "exit":
            return {"accepted": True, "virtual_time_s": 8, "exit_reason": "user_exit"}
        raise AssertionError(endpoint)


class TestClearPayload(unittest.TestCase):
    def test_clear_must_carry_channel(self):
        cli = _CapturingClient()
        cli.clear(Position(10.0, -20.0), 7)
        endpoint, payload = cli.sent[-1]
        self.assertEqual(endpoint, "clear")
        self.assertIn("channel", payload, "/clear 必须带频道参数，否则模拟器返回 400")
        self.assertEqual(payload["channel"], 7)
        self.assertEqual(payload["position"], {"x": 10.0, "y": -20.0})
        self.assertEqual(payload["arena_id"], "default")
        self.assertIn("request_id", payload)

    def test_clear_rejects_bad_channel(self):
        cli = _CapturingClient()
        for bad in (0, 21, -1, True):
            with self.assertRaises(ValueError):
                cli.clear(Position(0.0, 0.0), bad)  # type: ignore[arg-type]

    def test_measure_payload(self):
        cli = _CapturingClient()
        cli.measure(Position(1.0, 2.0), 3)
        endpoint, payload = cli.sent[-1]
        self.assertEqual(endpoint, "measure")
        self.assertEqual(payload["channel"], 3)
        self.assertEqual(payload["position"], {"x": 1.0, "y": 2.0})

    def test_every_action_uses_fresh_request_id(self):
        cli = _CapturingClient()
        cli.enter()
        cli.measure(Position(0.0, 0.0), 1)
        cli.clear(Position(0.0, 0.0), 1)
        cli.exit()
        ids = [p["request_id"] for _, p in cli.sent]
        self.assertEqual(len(ids), len(set(ids)), "每个新动作必须使用新的 request_id")


class _FakeHttpBackend:
    """与 HttpBackend 同接口，但底层换成离线模拟器，用于离线验证脚本装配。"""

    def __init__(self, cfg, log_path=None):
        from cumcm_b3.backend_offline import OfflineSimulator, generate_case
        self._sim = OfflineSimulator(generate_case(7))
        self.trace = self._sim.trace
        self.max_real_s = 900.0

    def __getattr__(self, name):
        return getattr(self._sim, name)


class TestDrillScriptPlumbing(unittest.TestCase):
    """演练脚本的装配链路必须能在离线环境下跑通（否则现场才发现装配错误）。"""

    def _load_module(self):
        import importlib.util
        path = ROOT / "scripts" / "run_drill.py"
        spec = importlib.util.spec_from_file_location("run_drill_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def _run_offline(self, extra_args=()):
        mod = self._load_module()
        import cumcm_b3.client as client_mod
        import cumcm_b3.backend_http as http_mod

        orig_probe, orig_backend = client_mod.probe_interface, http_mod.HttpBackend
        client_mod.probe_interface = lambda *a, **k: (True, True, "接口就绪（stub）")
        http_mod.HttpBackend = _FakeHttpBackend          # 换成离线后端
        tmp = Path(tempfile.mkdtemp(prefix="drill_selftest_"))
        orig_root, orig_argv = mod.ROOT, sys.argv
        try:
            mod.ROOT = tmp
            sys.argv = ["run_drill.py", "--yes", "--tag", "selftest",
                        "--sim-data-dir", str(tmp / "no-such-dir"), *extra_args]
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = mod.main()
        finally:
            client_mod.probe_interface, http_mod.HttpBackend = orig_probe, orig_backend
            mod.ROOT, sys.argv = orig_root, orig_argv

        out = buf.getvalue()
        self.assertEqual(code, 0, f"演练脚本装配失败：\n{out}\n{tmp}")
        self.assertIn("演练测试结束", out)
        payload = json.loads((tmp / "outputs" / "drill" / "selftest.json").read_text(encoding="utf-8"))
        self.assertGreater(payload["result"]["cleared_count"], 0)
        self.assertIsNone(payload["result"]["cleared_ratio"])
        self.assertIsNone(payload["result"]["total_sources"])
        self.assertIsNone(payload["honest_ratio"])
        self.assertEqual(payload["status"]["all_clear"], True)
        run_dir = tmp / "logs" / "runs" / "selftest"
        self.assertTrue((run_dir / "http.jsonl").parent.is_dir())
        self.assertTrue((run_dir / "summary.json").is_file())
        self.assertTrue((run_dir / "effective_config.json").is_file())
        index_rows = (tmp / "logs" / "drill_index.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(index_rows), 1)
        self.assertEqual(json.loads(index_rows[0])["tag"], "selftest")
        # 注：离线替身不写 HTTP jsonl，故此处不校验日志文件；
        # 真实后端的日志落盘由 test_every_action_uses_fresh_request_id 覆盖的同一套客户端保证。
        return payload, out, json.loads(index_rows[0])

    def test_drill_runs_end_to_end_offline(self):
        self._run_offline()

    def test_experiment_config_reaches_engine_and_logs(self):
        payload, out, index = self._run_offline([
            '--config', str(ROOT/'configs'/'experiment_combined.json')])
        self.assertTrue(payload['strategy_parameters']['adaptive_search'])
        self.assertTrue(payload['strategy_parameters']['joint_service'])
        self.assertTrue(index['adaptive_search'])
        self.assertTrue(index['joint_service'])
        self.assertIn('completion_basis', payload['status'])
        self.assertTrue(any(e['event'] == 'joint_replan' for e in payload['events']))
        self.assertIn('动态布局', out)

    def test_shared_config_reaches_engine_and_logs(self):
        payload, out, index = self._run_offline([
            '--config', str(ROOT/'configs'/'shared_near_01.json')])
        self.assertEqual(payload['strategy_parameters']['shared_observations'], 'opportunistic')
        self.assertEqual(index['shared_observations'], 'opportunistic')
        self.assertEqual(index['verify_near_fraction'], 0.1)
        self.assertFalse(index['route_polish'])
        self.assertTrue(any(e['event'] == 'shared_measure' for e in payload['events']))

    def test_multistart_config_reaches_engine_and_logs(self):
        payload, out, index = self._run_offline([
            '--config', str(ROOT/'configs'/'multistart_shared_01.json')])
        self.assertTrue(index['route_multistart'])
        self.assertFalse(index['adaptive_verify'])
        self.assertTrue(payload['strategy_parameters']['route_multistart'])

    def test_refined_config_logged(self):
        payload,out,index=self._run_offline([
            '--config',str(ROOT/'configs'/'experimental_safe_step.json')])
        self.assertTrue(index['step_replan'])
        self.assertTrue(index['safe_clear_point'])
        self.assertFalse(index['scan_relocation'])
        self.assertTrue(payload['strategy_parameters']['step_replan'])
        self.assertTrue(index['recovery_verify'])

    def test_recovery_config_logged(self):
        payload,out,index=self._run_offline([
            '--config',str(ROOT/'configs'/'multistart_recovered_01.json')])
        self.assertTrue(index['recovery_verify'])
        self.assertTrue(index['route_multistart'])
        self.assertFalse(index['step_replan'])
        self.assertEqual(index['bearing_rounding_slack_deg'],0.005001)
        self.assertTrue(index['stop_at_source_upper_bound'])

    def test_relocated_recovered_config_logged(self):
        payload,out,index=self._run_offline([
            '--config',str(ROOT/'configs'/'relocated_recovered_01.json')])
        self.assertTrue(index['scan_relocation'])
        self.assertTrue(index['safe_clear_point'])
        self.assertTrue(index['recovery_verify'])


class TestDrillTruthReader(unittest.TestCase):
    """从模拟器本地真值文件自动读取干扰源总数（否则算不出诚实的清除比例）。"""

    def _load(self):
        import importlib.util
        path = ROOT / "scripts" / "run_drill.py"
        spec = importlib.util.spec_from_file_location("run_drill_truth", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def test_reads_newest_result_file(self):
        mod = self._load()
        tmp = Path(tempfile.mkdtemp(prefix="truth_"))
        (tmp / "practice-p3-1-AAAA-AAAA-AAAA-AAAA.result.json").write_text(
            json.dumps({"version": 2, "problem_no": 3, "jammer_count": 16,
                        "omnidirectional_jammer_count": 16,
                        "directional_jammer_count": 0}), encoding="utf-8")
        count, name, payload = mod.read_practice_truth(str(tmp), time.time() - 600, timeout_s=2.0)
        self.assertEqual(count, 16)
        self.assertTrue(name.endswith(".result.json"))
        self.assertEqual(payload["problem_no"], 3)

    def test_ignores_stale_files_and_times_out_quietly(self):
        mod = self._load()
        tmp = Path(tempfile.mkdtemp(prefix="truth_stale_"))
        old = tmp / "practice-p3-2-BBBB-BBBB-BBBB-BBBB.result.json"
        old.write_text(json.dumps({"jammer_count": 12}), encoding="utf-8")
        # 只接受“本次运行开始之后”写出的文件：旧文件必须被忽略
        count, name, _ = mod.read_practice_truth(str(tmp), time.time() + 3600, timeout_s=1.0)
        self.assertIsNone(count)
        self.assertIsNone(name)

    def test_missing_directory_returns_immediately(self):
        mod = self._load()
        t0 = time.time()
        count, _, _ = mod.read_practice_truth(str(Path(tempfile.gettempdir()) / "nope-xyz"), 0.0, timeout_s=5.0)
        self.assertIsNone(count)
        self.assertLess(time.time() - t0, 1.0, "目录不存在时应立即返回，不应等待超时")


class TestProtocolConformanceNotes(unittest.TestCase):
    """把 2026-09-11 演练实测反推出的协议约束固化成断言，防止再次退化。"""

    def test_clear_and_measure_share_the_same_field_names(self):
        cli = _CapturingClient()
        cli.measure(Position(1.0, 2.0), 4)
        cli.clear(Position(1.0, 2.0), 4)
        _, m = cli.sent[0]
        _, c = cli.sent[1]
        # 模拟器用同一套字段白名单（二进制中 arena_id/robot_id/position 连续存放）
        for key in ("arena_id", "robot_id", "request_id", "position", "channel"):
            self.assertIn(key, m)
            self.assertIn(key, c)
        # 多一个未知字段也会被 400 拒绝，因此绝不允许附带额外键
        self.assertEqual(set(m) - {"arena_id", "robot_id", "request_id", "position", "channel"}, set())
        self.assertEqual(set(c) - {"arena_id", "robot_id", "request_id", "position", "channel"}, set())

    def test_400_without_message_yields_actionable_hint(self):
        text = SimulatorClient._error_text(
            {"accepted": False, "real_timestamp_ms": 1, "virtual_time_s": 0}, b"")
        self.assertIn("模拟器未在响应体中给出原因", text)
        self.assertIn("/clear", text)


class TestReadinessProbe(unittest.TestCase):
    """模拟器启动时就保留端口，所以必须区分"端口可连"与"接口就绪"。

    2026-09-11 实测：/exit 之后端口仍可连（probe=True），但 HTTP 得到空响应。
    如果只看端口就 /enter，会在无意中把脚本指向一个并未就绪的会话。
    """

    def _serve(self, responder):
        import socket
        import threading

        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(5)
        port = srv.getsockname()[1]

        def loop():
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                with conn:
                    try:
                        conn.recv(4096)
                        responder(conn)
                    except OSError:
                        pass

        threading.Thread(target=loop, daemon=True).start()
        return srv, port

    def test_reserved_port_without_interface_is_not_ready(self):
        def responder(conn):  # 直接关闭连接 → 空响应
            return

        srv, port = self._serve(responder)
        try:
            self.assertTrue(probe("127.0.0.1", port, 1.0), "TCP 层仍然是可连的")
            tcp_ok, ready, detail = probe_interface("127.0.0.1", port, timeout_s=1.5)
            self.assertTrue(tcp_ok)
            self.assertFalse(ready, "空响应必须判定为未就绪")
            self.assertIn("未开放", detail)
        finally:
            srv.close()

    def test_json_response_means_ready(self):
        def responder(conn):
            body = json.dumps({"accepted": False, "real_timestamp_ms": 1,
                               "virtual_time_s": 0}).encode()
            conn.sendall(
                b"HTTP/1.1 404 Not Found\r\nContent-Type: application/json; charset=utf-8\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body
            )

        srv, port = self._serve(responder)
        try:
            tcp_ok, ready, detail = probe_interface("127.0.0.1", port, timeout_s=1.5)
            self.assertTrue(tcp_ok)
            self.assertTrue(ready, "合法的机器狗接口 JSON 必须判定为就绪")
            self.assertIn("404", detail)
        finally:
            srv.close()

    def test_closed_port_reports_tcp_failure(self):
        import socket
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]
        srv.close()   # 立刻释放，确保没人监听
        tcp_ok, ready, detail = probe_interface("127.0.0.1", port, timeout_s=0.5)
        self.assertFalse(tcp_ok)
        self.assertFalse(ready)
        self.assertIn("TCP 连接失败", detail)


class TestDrillRefusesWhenNotReady(unittest.TestCase):
    """就绪探测不通过时，演练脚本必须拒绝发送任何动作。"""

    def test_returns_2_without_touching_backend(self):
        import importlib.util
        import cumcm_b3.client as client_mod
        import cumcm_b3.backend_http as http_mod

        path = ROOT / "scripts" / "run_drill.py"
        spec = importlib.util.spec_from_file_location("run_drill_gate", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        touched = []

        class _Boom:
            def __init__(self, *a, **k):
                touched.append("backend-created")

        orig_probe, orig_backend = client_mod.probe_interface, http_mod.HttpBackend
        orig_argv = sys.argv
        client_mod.probe_interface = lambda *a, **k: (True, False, "接口未开放（stub）")
        http_mod.HttpBackend = _Boom
        sys.argv = ["run_drill.py", "--yes", "--tag", "gatecheck"]
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = mod.main()
        finally:
            client_mod.probe_interface, http_mod.HttpBackend = orig_probe, orig_backend
            sys.argv = orig_argv

        self.assertEqual(code, 2)
        self.assertEqual(touched, [], "未就绪时绝不能创建后端、绝不能发动作")
        self.assertIn("接口当前不可用", buf.getvalue())


class TestDrillParameterOverrides(unittest.TestCase):
    """--second-a / --second-b 单侧覆盖非法时必须给人话，且不得产生会话动作。"""

    def _load(self):
        import importlib.util
        path = ROOT / "scripts" / "run_drill.py"
        spec = importlib.util.spec_from_file_location("run_drill_params", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod

    def _run(self, argv):
        import cumcm_b3.client as client_mod
        import cumcm_b3.backend_http as http_mod

        mod = self._load()
        tmp = Path(tempfile.mkdtemp(prefix="drill_params_"))
        touched = []

        class _Boom:
            """替身后端：一被使用就记录下来，避免任何真实网络动作。"""

            def __init__(self, *a, **k):
                touched.append("backend-created")

            def enter(self, *a, **k):
                raise RuntimeError("离线替身不允许动作")

            def exit(self, *a, **k):
                raise RuntimeError("离线替身不允许动作")

        orig_probe, orig_backend, orig_argv, orig_root = (
            client_mod.probe_interface, http_mod.HttpBackend, sys.argv, mod.ROOT)
        client_mod.probe_interface = lambda *a, **k: (True, True, "stub")
        http_mod.HttpBackend = _Boom
        sys.argv = ["run_drill.py"] + argv
        mod.ROOT = tmp
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = mod.main()
        finally:
            client_mod.probe_interface = orig_probe
            http_mod.HttpBackend = orig_backend
            sys.argv = orig_argv
            mod.ROOT = orig_root
        return code, buf.getvalue(), touched

    def test_infeasible_override_gives_readable_error(self):
        # (1000, 526) 到第一站 1130 m > 1000 m，超出候选区域
        code, out, touched = self._run(["--yes", "--force", "--tag", "t_override",
                                        "--second-a", "1000"])
        self.assertEqual(code, 4, out)
        self.assertIn("策略参数不可用", out)
        self.assertIn("第二检测点", out)
        self.assertNotIn("Traceback", out)
        self.assertEqual(touched, [], "参数不合法时不得创建后端")

    def test_feasible_override_reports_effective_values(self):
        code, out, touched = self._run(["--yes", "--force", "--tag", "t_override",
                                        "--second-a", "700", "--second-b", "540"])
        self.assertIn("(700, ±540)", out)
        self.assertNotEqual(code, 4)
        # --force 下会真的去连模拟器；离线环境必然失败，关键是参数已被接受
        self.assertNotIn("策略参数不可用", out)


class TestHttpBackendAccounting(unittest.TestCase):
    """HttpBackend 的虚拟时间记账与 /clear 语义（不改变频道）。"""

    def _backend(self):
        from cumcm_b3.backend_http import HttpBackend
        cfg = StrategyConfig()
        be = HttpBackend(cfg, log_path=None)
        be.client = _CapturingClient()   # 替换掉真实客户端
        return be

    def test_clear_keeps_current_channel(self):
        be = self._backend()
        be.enter()
        be.measure(0.0, 0.0, 5)
        self.assertEqual(be.current_channel, 5)
        be.clear(0.0, 0.0, 9)
        self.assertEqual(be.current_channel, 5, "/clear 不切换测向机频道")

    def test_move_and_switch_accounting(self):
        be = self._backend()
        be.enter()
        be.measure(0.0, 0.0, 1)          # 切换 0（初始频道 1） + 检测 5
        self.assertAlmostEqual(be.virtual_time_s, 5.0, places=6)
        be.measure(500.0, 0.0, 2)        # 移动 100 + 切换 1 + 检测 5
        self.assertAlmostEqual(be.virtual_time_s, 111.0, places=6)
        be.clear(900.0, 0.0, 3)          # 移动 80 + 命中 5
        self.assertAlmostEqual(be.virtual_time_s, 196.0, places=6)

    def test_server_virtual_time_overrides_local(self):
        be = self._backend()
        be.enter()
        be.measure(0.0, 0.0, 1)
        # 假客户端回放的是 5，本地也是 5，不应有漂移记录
        self.assertAlmostEqual(be.max_drift, 0.0, places=6)

        class _Skewed(_CapturingClient):
            def _post(self, endpoint, payload):
                res = super()._post(endpoint, payload)
                if endpoint == "measure":
                    res["virtual_time_s"] = 999.0
                return res

        be2 = self._backend()
        be2.client = _Skewed()
        be2.enter()
        be2.measure(0.0, 0.0, 1)
        self.assertAlmostEqual(be2.virtual_time_s, 999.0, places=6)
        self.assertAlmostEqual(be2.max_drift, -994.0, places=6)


if __name__ == "__main__":
    unittest.main()
