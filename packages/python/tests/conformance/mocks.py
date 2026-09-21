"""Mock Claude Messages and Codex Responses APIs that answer with conformance mock steps.

Steps are documented in docs/fixtures/CONFORMANCE.md. Each mock records every model
request (method, path, lowercase headers, JSON body) and serves the next step; the
last step repeats.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class MockApi:
    """A local HTTP server that records model requests and answers with scripted steps."""

    base_path = ""

    def __init__(self, steps: list[dict[str, Any]] | None = None) -> None:
        self.steps: list[dict[str, Any]] = steps or [{"text": "ok"}]
        self.requests: list[dict[str, Any]] = []
        self._served = 0
        self._lock = threading.Lock()
        self._release = threading.Event()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._server.block_on_close = False
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}{self.base_path}"

    def start(self) -> MockApi:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._release.set()
        self._server.shutdown()
        self._server.server_close()

    def set_steps(self, steps: list[dict[str, Any]]) -> None:
        with self._lock:
            self.steps = steps
            self._served = 0

    def hang(self, seconds: float) -> None:
        self._release.wait(seconds)

    def _take_step(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self._lock:
            self.requests.append(request)
            index = self._step_index(request)
            return index, self.steps[min(index, len(self.steps) - 1)]

    def _step_index(self, request: dict[str, Any]) -> int:
        index = self._served
        self._served += 1
        return index

    def is_model_request(self, path: str) -> bool:
        raise NotImplementedError

    def respond(self, handler: Handler, index: int, step: dict[str, Any], body: Any) -> None:
        raise NotImplementedError

    def other(self, handler: Handler, path: str) -> None:
        handler.send(404, {"error": {"message": "not mocked"}})

    def _handler(self) -> type[Handler]:
        mock = self

        class Bound(Handler):
            api = mock

        return Bound


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    api: MockApi

    def log_message(self, *args: Any) -> None:
        pass

    def send(
        self,
        status: int,
        body: Any,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
    ) -> None:
        if isinstance(body, str) and content_type == "application/json":
            content_type = "text/plain; charset=utf-8"
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def sse(
        self, events: list[dict[str, Any]], headers: dict[str, str] | None, truncate: bool
    ) -> None:
        """Send server-sent events; ``truncate`` closes the stream after the first."""

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("connection", "close")
        self.end_headers()
        for event in events[:1] if truncate else events:
            self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:
        self.api.other(self, self.path)

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        path = self.path.split("?")[0]
        if not self.api.is_model_request(path):
            self.api.other(self, path)
            return
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        request = {
            "method": "POST",
            "path": self.path,
            "headers": {key.lower(): value for key, value in self.headers.items()},
            "body": body,
        }
        index, step = self.api._take_step(request)
        if "hang" in step:
            self.api.hang(step["hang"])
            self.close_connection = True
            return
        if "status" in step:
            self.send(step["status"], step.get("body", {}), headers=step.get("headers"))
            return
        self.api.respond(self, index, step, body)


class MockClaude(MockApi):
    """The Claude Messages API. ANTHROPIC_BASE_URL is ``base_url``, without ``/v1``.

    The CLI retries a failed stream once without streaming; that request replays
    the failed step, so the fault reaches the run instead of the next step.
    """

    def __init__(self, steps: list[dict[str, Any]] | None = None) -> None:
        super().__init__(steps)
        self._stream_step: int | None = None

    def is_model_request(self, path: str) -> bool:
        return path == "/v1/messages"

    def _step_index(self, request: dict[str, Any]) -> int:
        body = request["body"] if isinstance(request["body"], dict) else {}
        if not body.get("stream") and self._stream_step is not None:
            return self._stream_step
        index = super()._step_index(request)
        self._stream_step = index
        return index

    def other(self, handler: Handler, path: str) -> None:
        if path.endswith("/count_tokens"):
            handler.send(200, {"input_tokens": 10})
            return
        handler.send(404, {"type": "error", "error": {"type": "not_found_error", "message": path}})

    def respond(self, handler: Handler, index: int, step: dict[str, Any], body: Any) -> None:
        model = body.get("model", "claude-haiku-4-5") if isinstance(body, dict) else ""
        if isinstance(body, dict) and body.get("stream"):
            handler.sse(
                _claude_events(index, model, step), step.get("headers"), step.get("truncate")
            )
            return
        if "stream_error" in step:
            error = step["stream_error"]
            status = {"overloaded_error": 529, "rate_limit_error": 429}.get(error.get("type"), 500)
            handler.send(status, {"type": "error", "error": error})
            return
        if step.get("truncate"):
            handler.send_response(200)
            handler.send_header("content-type", "application/json")
            handler.send_header("content-length", "1000")
            handler.end_headers()
            handler.wfile.write(b'{"id": "msg_')
            handler.close_connection = True
            return
        message = _claude_message(index, model, step)
        message["content"] = _claude_blocks(index, step)
        handler.send(200, message, headers=step.get("headers"))


def _tool_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
    tool = step.get("tool", [])
    return tool if isinstance(tool, list) else [tool]


def _claude_message(index: int, model: str, step: dict[str, Any]) -> dict[str, Any]:
    input_tokens, output_tokens = step.get("usage", (100, 10))[:2]
    return {
        "id": f"msg_{index}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": _claude_stop_reason(step),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _claude_stop_reason(step: dict[str, Any]) -> str:
    calls = "tool" in step or "shell" in step
    return step.get("stop_reason") or ("tool_use" if calls else "end_turn")


def _claude_blocks(index: int, step: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if "thinking" in step:
        blocks.append({"type": "thinking", "thinking": step["thinking"], "signature": "c2ln"})
    if "text" in step:
        blocks.append({"type": "text", "text": step["text"]})
    calls = _tool_calls(step)
    if "shell" in step:
        calls.append({"name": "Bash", "input": {"command": step["shell"]}})
    for n, call in enumerate(calls):
        blocks.append(
            {
                "type": "tool_use",
                "id": f"toolu_{index}_{n}",
                "name": call["name"],
                "input": call.get("input", {}),
            }
        )
    return blocks


def _claude_events(index: int, model: str, step: dict[str, Any]) -> list[dict[str, Any]]:
    message = _claude_message(index, model, step)
    output_tokens = message["usage"]["output_tokens"]
    message["usage"]["output_tokens"] = 1
    message["stop_reason"] = None
    events: list[dict[str, Any]] = [{"type": "message_start", "message": message}]
    for n, block in enumerate(_claude_blocks(index, step)):
        if block["type"] == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif block["type"] == "text":
            start = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": block["text"]}]
        else:
            start = {**block, "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        events.append({"type": "content_block_start", "index": n, "content_block": start})
        events += [{"type": "content_block_delta", "index": n, "delta": d} for d in deltas]
        if "stream_error" not in step:
            events.append({"type": "content_block_stop", "index": n})
    if "stream_error" in step:
        return [*events, {"type": "error", "error": step["stream_error"]}]
    return [
        *events,
        {
            "type": "message_delta",
            "delta": {"stop_reason": _claude_stop_reason(step), "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
        {"type": "message_stop"},
    ]


class MockCodex(MockApi):
    """The Codex Responses API; a custom model provider's ``base_url`` is ``base_url``."""

    base_path = "/v1"

    def is_model_request(self, path: str) -> bool:
        return path == "/v1/responses"

    def respond(self, handler: Handler, index: int, step: dict[str, Any], body: Any) -> None:
        handler.sse(_codex_events(index, step), step.get("headers"), step.get("truncate"))


def _codex_events(index: int, step: dict[str, Any]) -> list[dict[str, Any]]:
    response_id = f"resp_{index}"
    items: list[dict[str, Any]] = []
    if "thinking" in step:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{index}",
                "summary": [{"type": "summary_text", "text": step["thinking"]}],
            }
        )
    if "tool_search" in step:
        items.append(
            {
                "type": "tool_search_call",
                "id": f"ts_{index}",
                "call_id": f"ts_{index}",
                "execution": "client",
                "status": "completed",
                "arguments": {"query": step["tool_search"]},
            }
        )
    calls = _tool_calls(step)
    if "shell" in step:
        calls.append({"name": "exec_command", "input": {"cmd": step["shell"]}})
    items += [_codex_call(f"{index}_{n}", **call) for n, call in enumerate(calls)]
    if "text" in step:
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "id": f"msg_{index}",
                "content": [{"type": "output_text", "text": step["text"]}],
            }
        )
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": response_id}},
        *({"type": "response.output_item.done", "item": item} for item in items),
    ]
    if "stream_error" in step:
        failed = {"id": response_id, "status": "failed", "error": step["stream_error"]}
        return [*events, {"type": "response.failed", "response": failed}]
    input_tokens, output_tokens, reasoning_tokens = [*step.get("usage", (100, 10)), 0][:3]
    usage = {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        "total_tokens": input_tokens + output_tokens,
    }
    return [
        *events,
        {"type": "response.completed", "response": {"id": response_id, "usage": usage}},
    ]


def _codex_call(call_id: str, name: str, input: dict[str, Any] | None = None) -> dict[str, Any]:
    """A function call; ``mcp__<server>__<tool>`` becomes Codex's namespaced MCP call."""

    call: dict[str, Any] = {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": f"call_{call_id}",
        "name": name,
        "arguments": json.dumps(input or {}),
    }
    if name.startswith("mcp__"):
        _, server, tool = name.split("__", 2)
        call.update(name=tool, namespace=f"mcp__{server}")
    return call
