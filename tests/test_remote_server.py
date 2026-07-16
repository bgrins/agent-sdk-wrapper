import asyncio
import json

import pytest

aiohttp = pytest.importorskip("aiohttp")

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402
from test_remote_sessions import FakeRemoteSession  # noqa: E402

from agent_sdk_wrapper.remote.server import create_app  # noqa: E402
from agent_sdk_wrapper.remote.sessions import SessionManager  # noqa: E402


class PermissionSession(FakeRemoteSession):
    async def send(self, text):
        from agent_sdk_wrapper.remote.protocol import SessionState, UserMessage

        self._emit(UserMessage(text=text))
        self._set_state(SessionState.RUNNING)
        self._create_permission_request(tool="Bash", tool_input={"command": text})


@pytest.fixture
async def client():
    manager = SessionManager(
        factories={"fake": FakeRemoteSession, "perm": PermissionSession}
    )
    app = create_app(manager)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    yield client
    await client.close()


async def _create(client, provider="fake", **kwargs):
    resp = await client.post("/api/sessions", json={"provider": provider, **kwargs})
    assert resp.status == 201, await resp.text()
    return await resp.json()


async def test_create_list_get_delete(client):
    info = await _create(client, model="m1")
    sid = info["session_id"]
    assert info["provider"] == "fake"
    assert info["model"] == "m1"
    assert info["state"] == "idle"

    resp = await client.get("/api/sessions")
    listed = (await resp.json())["sessions"]
    assert [s["session_id"] for s in listed] == [sid]

    resp = await client.get(f"/api/sessions/{sid}")
    assert resp.status == 200

    resp = await client.delete(f"/api/sessions/{sid}")
    assert resp.status == 200
    resp = await client.get(f"/api/sessions/{sid}")
    assert resp.status == 404


async def test_create_validation(client):
    resp = await client.post("/api/sessions", json={})
    assert resp.status == 400
    resp = await client.post("/api/sessions", json={"provider": "unknown"})
    assert resp.status == 400


async def test_message_and_sse_events(client):
    info = await _create(client)
    sid = info["session_id"]

    resp = await client.post(f"/api/sessions/{sid}/messages", json={"text": "hi"})
    assert resp.status == 202

    resp = await client.get(f"/api/sessions/{sid}/events?since=0")
    assert resp.status == 200
    assert resp.headers["Content-Type"].startswith("text/event-stream")

    events = []
    async for chunk in resp.content:
        line = chunk.decode().strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
            if events[-1]["event"]["type"] == "text":
                break
    types = [e["event"]["type"] for e in events]
    assert "user_message" in types
    assert "text" in types
    resp.close()


async def test_message_validation(client):
    info = await _create(client)
    sid = info["session_id"]
    resp = await client.post(f"/api/sessions/{sid}/messages", json={})
    assert resp.status == 400
    resp = await client.post("/api/sessions/nope/messages", json={"text": "x"})
    assert resp.status == 404


async def test_permission_roundtrip(client):
    info = await _create(client, provider="perm")
    sid = info["session_id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"text": "rm -rf /"})

    resp = await client.get(f"/api/sessions/{sid}")
    info = await resp.json()
    assert info["state"] == "awaiting_permission"
    assert len(info["pending_permissions"]) == 1
    request_id = info["pending_permissions"][0]

    resp = await client.post(
        f"/api/sessions/{sid}/permissions/{request_id}", json={"behavior": "allow"}
    )
    assert resp.status == 202
    info = await (await client.get(f"/api/sessions/{sid}")).json()
    assert info["pending_permissions"] == []

    resp = await client.post(
        f"/api/sessions/{sid}/permissions/{request_id}", json={"behavior": "allow"}
    )
    assert resp.status == 409


async def test_interrupt(client):
    info = await _create(client)
    sid = info["session_id"]
    resp = await client.post(f"/api/sessions/{sid}/interrupt")
    assert resp.status == 202


async def test_sse_replay_since(client):
    info = await _create(client)
    sid = info["session_id"]
    await client.post(f"/api/sessions/{sid}/messages", json={"text": "one"})

    session = client.app["manager"].get(sid)
    total = len(session.events)
    close_task = asyncio.get_running_loop().create_task(_close_soon(session))

    resp = await client.get(f"/api/sessions/{sid}/events?since={total - 1}")
    seqs = []
    async for chunk in resp.content:
        line = chunk.decode().strip()
        if line.startswith("data: "):
            seqs.append(json.loads(line[len("data: "):])["seq"])
    await close_task
    assert seqs[0] == total
    assert all(s >= total for s in seqs)


async def _close_soon(session):
    await asyncio.sleep(0.05)
    await session.close()


async def test_index_served(client):
    resp = await client.get("/")
    assert resp.status == 200
    body = await resp.text()
    assert "remote control" in body
