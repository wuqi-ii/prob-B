"""CUMCM 2026 B-problem simulator HTTP client.

Uses only the Python standard library.  Requests are deliberately serialized:
the official simulator does not accept concurrent actions.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SimulatorError(RuntimeError):
    """Base class for simulator client errors."""


class SimulatorUnavailable(SimulatorError):
    """The simulator is not accepting connections."""


class SimulatorHTTPError(SimulatorError):
    """The simulator returned a non-2xx HTTP response."""

    def __init__(self, status: int, message: str, payload: Any = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.payload = payload


class SimulatorRejected(SimulatorError):
    """The request was valid HTTP but accepted=false."""

    def __init__(self, payload: Mapping[str, Any]):
        reason = payload.get("message") or payload.get("error") or "request rejected"
        super().__init__(str(reason))
        self.payload = dict(payload)


class SimulatorProtocolError(SimulatorError):
    """The response did not follow the documented JSON protocol."""


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def as_json(self) -> dict[str, float]:
        for name, value in (("x", self.x), ("y", self.y)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"position.{name} must be a number")
            if not math.isfinite(value) or abs(value) > 2_000_000:
                raise ValueError(f"position.{name} must be finite and |value| <= 2,000,000")
        return {"x": self.x, "y": self.y}


class JsonlLogger:
    """Append request/response events to JSON Lines, without credentials."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, event: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        record = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


class SimulatorClient:
    """Sequential client for /enter, /measure, /clear and /exit."""

    ENDPOINTS = frozenset({"enter", "measure", "clear", "exit"})

    def __init__(
        self,
        robot_id: str,
        base_url: str = "http://127.0.0.1:2026",
        arena_id: str = "default",
        timeout_s: float = 10.0,
        retries: int = 2,
        retry_delay_s: float = 0.5,
        log_path: str | Path | None = "logs/simulator.jsonl",
    ) -> None:
        if not isinstance(robot_id, str) or not robot_id.strip():
            raise ValueError("robot_id is required and must match the logged-in team number")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self.robot_id = robot_id.strip()
        self.arena_id = arena_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self.logger = JsonlLogger(log_path)
        self._action_lock = threading.Lock()

    @staticmethod
    def new_request_id() -> str:
        return str(uuid.uuid4())

    def _base_payload(self, request_id: str | None) -> dict[str, Any]:
        rid = request_id or self.new_request_id()
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError("request_id must be a non-empty string")
        return {"arena_id": self.arena_id, "robot_id": self.robot_id, "request_id": rid}

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if endpoint not in self.ENDPOINTS:
            raise ValueError(f"unsupported endpoint: {endpoint}")
        # Encoding once is important: retries resend byte-for-byte identical JSON.
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        url = f"{self.base_url}/{endpoint}"
        request_id = payload.get("request_id")

        with self._action_lock:
            logged_payload = dict(payload)
            logged_payload["robot_id"] = "***"
            self.logger.write({"kind": "request", "endpoint": endpoint, "request_id": request_id, "payload": logged_payload})
            for attempt in range(self.retries + 1):
                request = Request(url, data=body, method="POST", headers={"Content-Type": "application/json; charset=utf-8"})
                try:
                    with urlopen(request, timeout=self.timeout_s) as response:
                        raw = response.read()
                        status = response.status
                except HTTPError as exc:
                    raw = exc.read()
                    parsed = self._decode_optional_json(raw)
                    self.logger.write({"kind": "http_error", "endpoint": endpoint, "request_id": request_id, "status": exc.code, "response": parsed})
                    raise SimulatorHTTPError(exc.code, self._error_text(parsed, raw), parsed) from exc
                except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
                    self.logger.write({"kind": "transport_error", "endpoint": endpoint, "request_id": request_id, "attempt": attempt + 1, "error": str(exc)})
                    if attempt >= self.retries:
                        raise SimulatorUnavailable(
                            "simulator unavailable; start an exercise test, wait for the 5-second countdown, and verify the base URL"
                        ) from exc
                    time.sleep(self.retry_delay_s * (2**attempt))
                    continue

                if not 200 <= status < 300:
                    raise SimulatorHTTPError(status, raw.decode("utf-8", errors="replace"))
                if not raw:
                    raise SimulatorProtocolError("simulator closed the connection or returned an empty response")
                try:
                    result = json.loads(raw.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SimulatorProtocolError("response is not valid UTF-8 JSON") from exc
                if not isinstance(result, dict) or not isinstance(result.get("accepted"), bool):
                    raise SimulatorProtocolError("response must be a JSON object containing boolean 'accepted'")
                self.logger.write({"kind": "response", "endpoint": endpoint, "request_id": request_id, "status": status, "response": result})
                if not result["accepted"]:
                    raise SimulatorRejected(result)
                return result

        raise AssertionError("unreachable")

    @staticmethod
    def _decode_optional_json(raw: bytes) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _error_text(parsed: Any, raw: bytes) -> str:
        if isinstance(parsed, dict):
            msg = parsed.get("message") or parsed.get("error")
            if msg:
                return str(msg)
            # 模拟器的 400 响应体只有 accepted/real_timestamp_ms/virtual_time_s，
            # 不带任何原因字段（诊断信息只显示在模拟器界面的"指令与反馈"里）。
            # 这种情况下必须自己给出可操作的提示，否则现场无法定位。
            return (
                f"{parsed} —— 模拟器未在响应体中给出原因。请检查："
                "① 请求字段名与格式是否与附件 1 一致（严格白名单，多一个字段也会被拒）；"
                "② /measure 与 /clear 都必须同时给出 position 与 channel；"
                "③ 也可在模拟器界面「指令与反馈」一栏查看本次请求的诊断信息。"
            )
        return str(parsed or raw.decode("utf-8", errors="replace") or "empty response")

    def enter(self, request_id: str | None = None) -> dict[str, Any]:
        result = self._post("enter", self._base_payload(request_id))
        for field in ("max_virtual_duration_s", "max_real_duration_s", "remaining_real_duration_s"):
            if field not in result:
                raise SimulatorProtocolError(f"/enter response missing {field}")
        return result

    def measure(self, position: Position, channel: int, request_id: str | None = None) -> dict[str, Any]:
        if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
            raise ValueError("channel must be an integer from 1 to 20")
        payload = self._base_payload(request_id)
        payload.update({"position": position.as_json(), "channel": channel})
        result = self._post("measure", payload)
        if result.get("measure_result") not in {"no_signal", "near", "direction"}:
            raise SimulatorProtocolError("unknown or missing measure_result")
        if result["measure_result"] == "direction" and "svd_deg" not in result:
            raise SimulatorProtocolError("direction result is missing svd_deg")
        return result

    def clear(self, position: Position, channel: int, request_id: str | None = None) -> dict[str, Any]:
        # 附件 1 第 2.3 节：/clear 必须同时给出位置参数与**目标频道参数**。
        # 缺少 channel 会被模拟器判为 400，且错误响应不含 message，很难定位。
        if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= 20:
            raise ValueError("channel must be an integer from 1 to 20")
        payload = self._base_payload(request_id)
        payload.update({"position": position.as_json(), "channel": channel})
        result = self._post("clear", payload)
        if result.get("clear_result") not in {"success", "no_target_in_range"}:
            raise SimulatorProtocolError("unknown or missing clear_result")
        return result

    def exit(self, request_id: str | None = None) -> dict[str, Any]:
        result = self._post("exit", self._base_payload(request_id))
        if "exit_reason" not in result:
            raise SimulatorProtocolError("/exit response missing exit_reason")
        return result


def probe(host: str = "127.0.0.1", port: int = 2026, timeout_s: float = 1.0) -> bool:
    """Return True if the simulator TCP port is open; sends no simulator action.

    .. warning::
       模拟器**启动时就会保留端口**，所以"端口可连"并不等于"接口已开放"。
       会话未开始或已结束时，TCP 仍可连接，但服务端会直接关闭连接（空响应）。
       需要判断"接口是否真的能收动作"请用 :func:`probe_interface`。
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def probe_interface(host: str = "127.0.0.1", port: int = 2026,
                    timeout_s: float = 1.5) -> tuple:
    """判断机器狗接口是否真的在服务。返回 ``(tcp_ok, ready, detail)``。

    只发送一个 ``GET /``：它不是四个动作接口之一，不会消耗任何测试资源，
    被服务端当作未知路径拒绝。判据是**能否收到符合协议的 JSON 响应**：

    * 会话进行中 → 返回 404 + ``{"accepted": false, ...}`` → ready=True；
    * 会话未开始/已结束 → 空响应（连接被关闭）→ ready=False。

    这样就能避免"看到端口开着就 /enter"这一类误判。
    """
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except OSError as exc:
        return False, False, f"TCP 连接失败：{exc}"

    request = (
        f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    try:
        with sock:
            sock.sendall(request)
            chunks = []
            total = 0
            while total < 65536:
                data = sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
                total += len(data)
    except OSError as exc:
        # 端口被保留但服务端并未在服务时会 RST / 直接关闭连接
        return True, False, f"端口可连但被对端立即关闭（接口未开放）：{exc}"

    raw = b"".join(chunks)
    if not raw:
        return True, False, "端口已被模拟器保留，但接口未开放（会话未开始或已结束）"
    head, _, body = raw.partition(b"\r\n\r\n")
    if not head.startswith(b"HTTP/"):
        return True, False, "已连接但不是 HTTP 响应"
    status_line = head.split(b"\r\n", 1)[0].decode("latin-1", errors="replace")
    try:
        payload = json.loads(body.decode("utf-8-sig").strip() or "null")
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("accepted"), bool):
        return True, True, f"接口就绪（{status_line}）"
    return True, False, f"已连接但响应不是机器狗接口的 JSON（{status_line}）"


def _main() -> int:
    parser = argparse.ArgumentParser(description="CUMCM 2026 B simulator client")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe", help="check the TCP port without sending an action")
    for name in ("enter", "exit"):
        p = sub.add_parser(name)
        p.add_argument("--robot-id", required=True)
    for name in ("measure", "clear"):
        p = sub.add_parser(name)
        p.add_argument("--robot-id", required=True)
        p.add_argument("--x", type=float, required=True)
        p.add_argument("--y", type=float, required=True)
        p.add_argument("--channel", type=int, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        from urllib.parse import urlsplit

        parts = urlsplit(args.base_url)
        host = parts.hostname or "127.0.0.1"
        port = parts.port or 2026
        tcp_ok, ready, detail = probe_interface(host, port)
        print(json.dumps({"tcp_reachable": tcp_ok, "interface_ready": ready,
                          "detail": detail, "base_url": args.base_url}, ensure_ascii=False))
        return 0 if ready else 2

    client = SimulatorClient(args.robot_id, base_url=args.base_url)
    try:
        if args.command == "enter":
            result = client.enter()
        elif args.command == "measure":
            result = client.measure(Position(args.x, args.y), args.channel)
        elif args.command == "clear":
            result = client.clear(Position(args.x, args.y), args.channel)
        else:
            result = client.exit()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except SimulatorError as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
