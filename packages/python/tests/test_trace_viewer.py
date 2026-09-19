"""Coverage for the static trace viewer artifact."""

from __future__ import annotations

from pathlib import Path

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
