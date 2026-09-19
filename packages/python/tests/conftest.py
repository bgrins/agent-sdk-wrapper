import os

import pytest


@pytest.fixture(autouse=True)
def _claude_credentials(monkeypatch):
    """The Claude adapter refuses to launch without API credentials; tests fake the runtime."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
