import os

import pytest


@pytest.fixture(autouse=True)
def _claude_credentials(request, monkeypatch):
    """The Claude adapter refuses to launch without API credentials; tests fake the runtime.

    Live tests skip without a real key, so they keep the real environment.
    """
    if "test_live_" in request.node.nodeid:
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
