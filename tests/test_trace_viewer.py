"""Coverage for the static trace viewer artifact."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "docs" / "trace-viewer.html"


def viewer_html() -> str:
    return VIEWER.read_text(encoding="utf-8")


def test_trace_viewer_is_static_and_dependency_free() -> None:
    html = viewer_html()

    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert "https://" not in html
    assert "http://" not in html


def test_trace_viewer_accepts_artifact_directories_and_core_files() -> None:
    html = viewer_html()

    assert 'id="artifactFiles"' in html
    assert "webkitdirectory" in html
    assert "directory" in html
    assert 'id="looseFiles"' in html
    assert "manifest.json" in html
    assert "trace.jsonl" in html
    assert "result.json" in html


def test_trace_viewer_auto_discovers_served_results_tree() -> None:
    html = viewer_html()

    for marker in (
        'id="resultsSidebar"',
        'class="results-scroll"',
        'id="refreshResults"',
        'id="runs"',
        '"/results/"',
        "manifest.json",
        "Serve the repository root to scan /results/.",
    ):
        assert marker in html


def test_trace_viewer_results_sidebar_is_collapsible_and_persisted() -> None:
    html = viewer_html()

    for marker in (
        'id="toggleResults"',
        'aria-controls="resultsSidebar"',
        "RESULTS_SIDEBAR_STORAGE_KEY",
        "localStorage.getItem",
        "localStorage.setItem",
        'document.body.classList.toggle("results-collapsed"',
        "overflow: auto",
    ):
        assert marker in html


def test_trace_viewer_renders_trace_artifact_concepts() -> None:
    html = viewer_html()

    for event_type in (
        "text",
        "thinking",
        "tool_call",
        "tool_result",
        "session_info",
        "warning",
        "error",
    ):
        assert event_type in html

    assert "Structured Output" in html
    assert "Usage" in html
    assert re.search(r"URL\.createObjectURL\(file\)", html)


def test_trace_viewer_has_conversation_first_information_architecture() -> None:
    html = viewer_html()

    tab_markers = (
        'role="tablist"',
        'data-view="conversation-view"',
        'data-view="timeline-view"',
        'data-view="events-view"',
        'data-view="files-view"',
    )
    for marker in tab_markers:
        assert marker in html

    assert html.index('data-view="conversation-view"') < html.index(
        'data-view="timeline-view"'
    )
    assert 'id="conversation"' in html
    assert 'id="events"' in html
    assert ".tool-panel" in html
    assert ".msg-role" in html
