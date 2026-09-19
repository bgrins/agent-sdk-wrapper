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


def viewer_html() -> str:
    return VIEWER.read_text(encoding="utf-8")


def test_trace_viewer_is_static_and_dependency_free() -> None:
    html = viewer_html()

    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert "https://" not in html
    assert "http://" not in html


def test_conversation_view_handles_every_event_type() -> None:
    """The timeline shows any event; the conversation drops unhandled types."""
    html = viewer_html()
    start = html.index("function buildConversation(")
    body = html[start : html.index("\n    }\n", start)]
    handled = set(re.findall(r'event\.type === "([a-z_]+)"', body))

    assert {member.type for member in typing.get_args(AgentEvent)} - handled == set()
