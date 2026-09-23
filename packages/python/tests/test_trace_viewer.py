"""Coverage for the static trace viewer artifact."""

from __future__ import annotations

import re
import typing
from pathlib import Path

from agent_sdk_wrapper.events import AgentEvent

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "docs" / "trace-viewer.html").is_file()
)
VIEWER = ROOT / "docs" / "trace-viewer.html"


def test_conversation_view_handles_every_event_type() -> None:
    """The timeline shows any event; the conversation drops unhandled types."""
    html = VIEWER.read_text(encoding="utf-8")
    start = html.index("function buildConversation(")
    body = html[start : html.index("\n    }\n", start)]
    handled = set(re.findall(r'event\.type === "([a-z_]+)"', body))

    assert {member.type for member in typing.get_args(AgentEvent)} - handled == set()
