from __future__ import annotations

import json
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulator_client import (  # noqa: E402
    Position,
    SimulatorClient,
    SimulatorProtocolError,
    SimulatorRejected,
)


class MockState:
    requests: list[tuple[str, bytes]] = []
    drop_first_measure = False
    dropped = False


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        MockState.requests.append((self.path, body))

        if self.path == "/measure" and MockState.drop_first_measure and not MockState.dropped:
            MockState.dropped = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return

        payload = json.loads(body.decode("utf-8"))
        common = {"accepted": True, "virtual_time_s": 0.0, "request_id": payload["request_id"]}
        if self.path == "/enter":
            result = {**common, "max_virtual_duration_s": 360000, "max_real_duration_s": 1200, "remaining_real_duration_s": 1199}
        elif self.path == "/measure":
            result = {**common, "virtual_time_s": 5.0, "measure_result": "direction", "svd_deg": 42.5}
        elif self.path == "/clear":
            result = {**common, "virtual_time_s": 3.0, "clear_result": "no_target_in_range"}
        elif self.path == "/exit":
            result = {**common, "exit_reason": "user_exit"}
        elif self.path == "/reject":
            result = {"accepted": False, "virtual_time_s": 0, "message": "mock rejection"}
        else:
            self.send_error(404)
            return
        encoded = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class SimulatorClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        MockState.requests = []
        MockState.drop_first_measure = False
        MockState.dropped = False
        self.client = SimulatorClient(
            "TEAM-001",
            base_url=self.base_url,
            retries=1,
            retry_delay_s=0,
            log_path=None,
        )

    def test_full_happy_path(self):
        self.assertEqual(self.client.enter()["remaining_real_duration_s"], 1199)
        self.assertEqual(self.client.measure(Position(10, 20), 3)["svd_deg"], 42.5)
        self.assertEqual(self.client.clear(Position(10, 20), 3)["clear_result"], "no_target_in_range")
        self.assertEqual(self.client.exit()["exit_reason"], "user_exit")
        self.assertEqual([path for path, _ in MockState.requests], ["/enter", "/measure", "/clear", "/exit"])
        clear_payload = json.loads(MockState.requests[2][1].decode("utf-8"))
        self.assertEqual(clear_payload["channel"], 3)

    def test_retry_is_byte_identical(self):
        MockState.drop_first_measure = True
        result = self.client.measure(Position(0, 0), 1, request_id="fixed-id")
        self.assertEqual(result["measure_result"], "direction")
        self.assertEqual(len(MockState.requests), 2)
        self.assertEqual(MockState.requests[0], MockState.requests[1])

    def test_validation_happens_before_network(self):
        with self.assertRaises(ValueError):
            self.client.measure(Position(0, 0), 0)
        with self.assertRaises(ValueError):
            self.client.measure(Position(float("nan"), 0), 1)
        with self.assertRaises(ValueError):
            self.client.clear(Position(2_000_001, 0), 1)
        self.assertEqual(MockState.requests, [])

    def test_protocol_check(self):
        original = MockHandler.do_POST

        def malformed(handler):
            length = int(handler.headers.get("Content-Length", "0"))
            handler.rfile.read(length)
            encoded = b'{"unexpected":true}'
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(encoded)))
            handler.end_headers()
            handler.wfile.write(encoded)

        MockHandler.do_POST = malformed
        try:
            with self.assertRaises(SimulatorProtocolError):
                self.client.enter()
        finally:
            MockHandler.do_POST = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
