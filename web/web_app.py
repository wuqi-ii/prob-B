"""Local web console and same-origin proxy for the official simulator."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from simulator_client import Position, SimulatorClient, SimulatorError, probe


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
ACTION_LOCK = threading.Lock()


class WebHandler(BaseHTTPRequestHandler):
    server_version = "CUMCMSimulatorConsole/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {fmt % args}")

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
        elif self.path == "/styles.css":
            self._file(STATIC / "styles.css", "text/css; charset=utf-8")
        elif self.path == "/app.js":
            self._file(STATIC / "app.js", "text/javascript; charset=utf-8")
        elif self.path == "/api/status":
            base_url = self.server.simulator_base_url  # type: ignore[attr-defined]
            parts = urlsplit(base_url)
            reachable = probe(parts.hostname or "127.0.0.1", parts.port or 2026)
            self._json(200, {"ok": True, "reachable": reachable, "base_url": base_url})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/action":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 32_768:
                raise ValueError("请求体大小无效")
            data = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("请求必须是 JSON 对象")
            if data.get("confirm_exercise") is not True:
                self._json(403, {"ok": False, "error": "为保护正式次数，只允许明确确认演练测试后发送动作"})
                return
            robot_id = data.get("robot_id")
            action = data.get("action")
            if action not in {"enter", "measure", "clear", "exit"}:
                raise ValueError("未知动作")
            client = SimulatorClient(
                robot_id=robot_id,
                base_url=self.server.simulator_base_url,  # type: ignore[attr-defined]
                log_path=ROOT / "logs" / "simulator.jsonl",
            )
            with ACTION_LOCK:
                if action == "enter":
                    result = client.enter()
                elif action == "measure":
                    result = client.measure(Position(data.get("x"), data.get("y")), data.get("channel"))
                elif action == "clear":
                    result = client.clear(
                        Position(data.get("x"), data.get("y")), data.get("channel")
                    )
                else:
                    result = client.exit()
            self._json(200, {"ok": True, "action": action, "result": result})
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._json(400, {"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        except SimulatorError as exc:
            self._json(502, {"ok": False, "error_type": type(exc).__name__, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web console for the CUMCM B simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--simulator-url", default="http://127.0.0.1:2026")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), WebHandler)
    server.simulator_base_url = args.simulator_url  # type: ignore[attr-defined]
    print(f"本地控制台：http://{args.host}:{args.port}")
    print(f"官方模拟机：{args.simulator_url}")
    print("按 Ctrl+C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
