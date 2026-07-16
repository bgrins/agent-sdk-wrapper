import asyncio

import pytest

from agent_sdk_wrapper.events import Text
from agent_sdk_wrapper.remote.protocol import SessionState, UserMessage
from agent_sdk_wrapper.remote.sessions import (
    BaseRemoteSession,
    RemoteSessionError,
    SessionManager,
)


class FakeRemoteSession(BaseRemoteSession):
    provider = "fake"

    def __init__(self, session_id, **kwargs):
        super().__init__(session_id, **kwargs)
        self.interrupts = 0

    async def start(self):
        self.native_session_id = self.resume or f"native-{self.id}"
        self._set_state(SessionState.IDLE)

    async def send(self, text):
        if self.state == SessionState.CLOSED:
            raise RemoteSessionError("closed")
        self._emit(UserMessage(text=text))
        self._set_state(SessionState.RUNNING)
        self._emit(Text(text=f"echo: {text}"))
        self._set_state(SessionState.IDLE)

    async def interrupt(self):
        self.interrupts += 1
        self._deny_pending_permissions("interrupted")
        self._set_state(SessionState.IDLE)

    async def close(self):
        self._deny_pending_permissions("closed")
        self._set_state(SessionState.CLOSED)


@pytest.fixture
def manager():
    return SessionManager(factories={"fake": FakeRemoteSession})


async def test_session_lifecycle_and_events(manager):
    session = await manager.create("fake")
    assert session.state == SessionState.IDLE
    assert session.native_session_id == f"native-{session.id}"

    await session.send("hello")
    types = [e["event"]["type"] for e in session.events]
    assert types == [
        "state_changed",  # idle
        "user_message",
        "state_changed",  # running
        "text",
        "state_changed",  # idle
    ]
    assert [e["seq"] for e in session.events] == [1, 2, 3, 4, 5]

    await manager.close(session.id)
    assert session.state == SessionState.CLOSED
    with pytest.raises(KeyError):
        manager.get(session.id)


async def test_events_since_replays_and_follows(manager):
    session = await manager.create("fake")
    await session.send("one")
    seen = []

    async def consume():
        async for entry in session.events_since(0):
            seen.append(entry["event"]["type"])

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    replayed = len(seen)
    assert replayed == len(session.events)

    await session.send("two")
    await session.close()
    await asyncio.wait_for(task, timeout=2)
    assert len(seen) > replayed
    assert seen[-1] == "state_changed"


async def test_events_since_partial_replay(manager):
    session = await manager.create("fake")
    await session.send("one")
    await session.close()
    entries = [e async for e in session.events_since(2)]
    assert [e["seq"] for e in entries] == list(
        range(3, len(session.events) + 1)
    )


async def test_permission_flow(manager):
    session = await manager.create("fake")
    request_id, future = session._create_permission_request(
        tool="Bash", tool_input={"command": "ls"}
    )
    assert session.state == SessionState.AWAITING_PERMISSION
    assert session.pending_permission_ids() == [request_id]

    session.respond_permission(request_id, "allow")
    decision = await future
    assert decision.allowed
    assert session.state == SessionState.RUNNING
    types = [e["event"]["type"] for e in session.events]
    assert "permission_request" in types
    assert "permission_resolved" in types


async def test_permission_invalid_behavior_and_unknown_id(manager):
    session = await manager.create("fake")
    request_id, _ = session._create_permission_request(tool="Bash", tool_input={})
    with pytest.raises(RemoteSessionError):
        session.respond_permission(request_id, "maybe")
    with pytest.raises(RemoteSessionError):
        session.respond_permission("nope", "allow")


async def test_respond_permission_does_not_resurrect_closed_session(manager):
    session = await manager.create("fake")
    request_id, _ = session._create_permission_request(tool="Bash", tool_input={})
    # Simulate the underlying turn dying while the permission is pending.
    session._set_state(SessionState.CLOSED)
    session.respond_permission(request_id, "allow")
    assert session.state == SessionState.CLOSED


async def test_interrupt_denies_pending_permissions(manager):
    session = await manager.create("fake")
    _, future = session._create_permission_request(tool="Bash", tool_input={})
    await session.interrupt()
    decision = await future
    assert not decision.allowed
    assert session.pending_permission_ids() == []


async def test_manager_resume_and_unknown_provider(manager):
    from agent_sdk_wrapper.errors import ConfigError

    session = await manager.create("fake", resume="native-abc")
    assert session.native_session_id == "native-abc"
    with pytest.raises(ConfigError):
        await manager.create("nope")


async def test_state_dir_plumbing(tmp_path):
    manager = SessionManager(
        factories={"fake": FakeRemoteSession}, state_dir=str(tmp_path / "state")
    )
    session = await manager.create("fake")
    assert session.state_dir == str(tmp_path / "state")

    env = session._state_env("claude", "CLAUDE_CONFIG_DIR")
    expected = (tmp_path / "state" / "claude").resolve()
    assert env == {"CLAUDE_CONFIG_DIR": str(expected)}
    assert expected.is_dir()

    unset = await SessionManager(factories={"fake": FakeRemoteSession}).create("fake")
    assert unset._state_env("codex", "CODEX_HOME") == {}


async def test_manager_close_all(manager):
    a = await manager.create("fake")
    b = await manager.create("fake")
    await manager.close_all()
    assert a.state == SessionState.CLOSED
    assert b.state == SessionState.CLOSED
    assert manager.list() == []
