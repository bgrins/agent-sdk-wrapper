"""HTTP + SSE server for the remote-control protocol.

Endpoints (see docs/remote-control.md):

    GET    /                                        web UI
    GET    /api/sessions                            list sessions
    POST   /api/sessions                            start/resume a session
    GET    /api/sessions/{id}                       session status
    DELETE /api/sessions/{id}                       close session
    GET    /api/sessions/{id}/events?since=N        SSE event stream
    POST   /api/sessions/{id}/messages              send (idle) / steer (running)
    POST   /api/sessions/{id}/interrupt             interrupt current turn
    POST   /api/sessions/{id}/permissions/{req_id}  answer a permission prompt

Requires the ``remote`` extra (aiohttp). Binds localhost by default; there is
no authentication — do not expose beyond localhost as-is.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

from ..errors import AgentSdkWrapperError, ConfigError
from .sessions import RemoteSessionError, SessionManager

_STATIC_DIR = Path(__file__).parent / "static"
_HEARTBEAT_SEC = 15.0


def _json_error(status: int, message: str):
    from aiohttp import web

    return web.json_response({"error": message}, status=status)


def create_app(manager: SessionManager | None = None):
    from aiohttp import web

    manager = manager or SessionManager()
    app = web.Application()
    app["manager"] = manager

    async def index(request: web.Request) -> web.StreamResponse:
        return web.FileResponse(_STATIC_DIR / "index.html")

    async def list_sessions(request: web.Request) -> web.StreamResponse:
        return web.json_response({"sessions": manager.list()})

    async def create_session(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _json_error(400, "request body must be JSON")
        provider = body.get("provider")
        if not provider:
            return _json_error(400, "'provider' is required")
        try:
            session = await manager.create(
                provider,
                model=body.get("model"),
                cwd=body.get("cwd"),
                resume=body.get("resume"),
                permission_mode=body.get("permission_mode"),
            )
        except (ConfigError, RemoteSessionError) as exc:
            return _json_error(400, str(exc))
        except AgentSdkWrapperError as exc:
            return _json_error(502, str(exc))
        return web.json_response(session.describe(), status=201)

    def _session(request: web.Request):
        return manager.get(request.match_info["session_id"])

    async def get_session(request: web.Request) -> web.StreamResponse:
        try:
            return web.json_response(_session(request).describe())
        except KeyError:
            return _json_error(404, "unknown session")

    async def delete_session(request: web.Request) -> web.StreamResponse:
        try:
            session = _session(request)
        except KeyError:
            return _json_error(404, "unknown session")
        info = session.describe()
        await manager.close(session.id)
        return web.json_response(info)

    async def post_message(request: web.Request) -> web.StreamResponse:
        try:
            session = _session(request)
        except KeyError:
            return _json_error(404, "unknown session")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _json_error(400, "request body must be JSON")
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return _json_error(400, "'text' is required")
        try:
            await session.send(text)
        except RemoteSessionError as exc:
            return _json_error(409, str(exc))
        return web.json_response(session.describe(), status=202)

    async def post_interrupt(request: web.Request) -> web.StreamResponse:
        try:
            session = _session(request)
        except KeyError:
            return _json_error(404, "unknown session")
        try:
            await session.interrupt()
        except RemoteSessionError as exc:
            return _json_error(409, str(exc))
        return web.json_response(session.describe(), status=202)

    async def post_permission(request: web.Request) -> web.StreamResponse:
        try:
            session = _session(request)
        except KeyError:
            return _json_error(404, "unknown session")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return _json_error(400, "request body must be JSON")
        behavior = body.get("behavior")
        try:
            session.respond_permission(
                request.match_info["request_id"], behavior, body.get("message")
            )
        except RemoteSessionError as exc:
            return _json_error(409, str(exc))
        return web.json_response(session.describe(), status=202)

    async def get_events(request: web.Request) -> web.StreamResponse:
        try:
            session = _session(request)
        except KeyError:
            return _json_error(404, "unknown session")
        since = 0
        raw_since = request.query.get("since") or request.headers.get("Last-Event-ID")
        if raw_since:
            try:
                since = int(raw_since)
            except ValueError:
                return _json_error(400, "'since' must be an integer")

        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        await response.prepare(request)

        async def _stream() -> None:
            async for entry in session.events_since(since):
                data = json.dumps(entry, ensure_ascii=False)
                await response.write(
                    f"id: {entry['seq']}\ndata: {data}\n\n".encode()
                )

        stream_task = asyncio.ensure_future(_stream())
        try:
            while not stream_task.done():
                await asyncio.wait({stream_task}, timeout=_HEARTBEAT_SEC)
                if not stream_task.done():
                    await response.write(b": heartbeat\n\n")
        except ConnectionResetError:
            pass
        finally:
            # Always retrieve the task result so client disconnects mid-write
            # don't log "Task exception was never retrieved".
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stream_task
        return response

    app.router.add_get("/", index)
    app.router.add_get("/api/sessions", list_sessions)
    app.router.add_post("/api/sessions", create_session)
    app.router.add_get("/api/sessions/{session_id}", get_session)
    app.router.add_delete("/api/sessions/{session_id}", delete_session)
    app.router.add_get("/api/sessions/{session_id}/events", get_events)
    app.router.add_post("/api/sessions/{session_id}/messages", post_message)
    app.router.add_post("/api/sessions/{session_id}/interrupt", post_interrupt)
    app.router.add_post(
        "/api/sessions/{session_id}/permissions/{request_id}", post_permission
    )

    async def _cleanup(app: Any) -> None:
        await manager.close_all()

    app.on_cleanup.append(_cleanup)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-sdk-wrapper serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "Directory for provider session state: <dir>/claude becomes "
            "CLAUDE_CONFIG_DIR and <dir>/codex becomes CODEX_HOME for spawned "
            "runtimes. Default: the providers' native home directories."
        ),
    )
    args = parser.parse_args(argv)

    try:
        from aiohttp import web
    except ImportError:
        print(
            "aiohttp is required for the remote-control server. "
            "Install with: uv sync --extra remote"
        )
        return 1

    app = create_app(SessionManager(state_dir=args.state_dir))
    print(f"remote-control server on http://{args.host}:{args.port}")
    if args.state_dir:
        print(f"provider state under {args.state_dir}")
    web.run_app(app, host=args.host, port=args.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
