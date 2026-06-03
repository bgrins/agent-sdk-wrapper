"""CLI behavior."""

from __future__ import annotations

import json

import pytest

from agent_sdk_wrapper import Error, Text, cli
from agent_sdk_wrapper.providers import base
from agent_sdk_wrapper.providers import openai_provider as op_mod


class FakeProvider(base.ProviderAdapter):
    name = "openai"

    async def stream(self, req):  # type: ignore[override]
        yield Text(text="hello")


def test_stream_defaults_to_text_output(monkeypatch, capsys):
    monkeypatch.setattr(op_mod, "OpenAIProvider", FakeProvider)

    rc = cli.main(["run", "--provider", "openai", "--prompt", "ignored", "--stream"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "hello\n"
    assert '"text"' not in captured.out


def test_stream_rejects_jsonl_output(capsys):
    rc = cli.main(
        [
            "run",
            "--provider",
            "openai",
            "--prompt",
            "ignored",
            "--stream",
            "--output",
            "jsonl",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "--stream cannot be combined with --output jsonl" in captured.err


def test_jsonl_output_returns_nonzero_on_stream_error(monkeypatch, capsys):
    class ErrorProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="partial")
            yield Error(message="provider failed", error_type="result_error")

    monkeypatch.setattr(op_mod, "OpenAIProvider", ErrorProvider)

    rc = cli.main(["run", "--provider", "openai", "--prompt", "ignored"])

    captured = capsys.readouterr()
    assert rc == 1
    assert '"text"' in captured.out
    assert '"error"' in captured.out


def test_text_stream_returns_nonzero_on_stream_error(monkeypatch, capsys):
    class ErrorProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            yield Text(text="partial")
            yield Error(message="provider failed", error_type="result_error")

    monkeypatch.setattr(op_mod, "OpenAIProvider", ErrorProvider)

    rc = cli.main(["run", "--provider", "openai", "--prompt", "ignored", "--stream"])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == "partial\n"


def test_run_forwards_extended_cli_options(monkeypatch, capsys):
    seen_requests = []
    seen_provider_options = []

    class CapturingProvider(base.ProviderAdapter):
        name = "openai"

        def __init__(self, **options):
            seen_provider_options.append(options)

        async def stream(self, req):  # type: ignore[override]
            seen_requests.append(req)
            yield Text(text="ok")

    monkeypatch.setattr(op_mod, "OpenAIProvider", CapturingProvider)

    rc = cli.main(
        [
            "run",
            "--provider",
            "openai",
            "--prompt",
            "ignored",
            "--output",
            "text",
            "--effort",
            "max",
            "--builtin-tool",
            "Read",
            "--builtin-tool",
            "Grep",
            "--allowed-tool",
            "repo.read_file",
            "--disallowed-tool",
            "repo.write_file",
            "--session-id",
            "sess-123",
            "--continue-session",
            "--env",
            "MODE=test",
            "--env",
            "TOKEN=a=b",
            "--provider-option",
            'api_key="test-key"',
            "--provider-option",
            'config.codex_bin="/bin/codex"',
            "--extra-option",
            'sandbox={"mode":"workspace-write"}',
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "ok\n"
    assert seen_provider_options == [
        {"api_key": "test-key", "config": {"codex_bin": "/bin/codex"}}
    ]

    req = seen_requests[0]
    assert req.effort == "xhigh"
    assert req.builtin_tools == ["Read", "Grep"]
    assert req.allowed_tools == ["repo.read_file"]
    assert req.disallowed_tools == ["repo.write_file"]
    assert req.session_id == "sess-123"
    assert req.continue_session is True
    assert req.env == {"MODE": "test", "TOKEN": "a=b"}
    assert req.extra_options == {"sandbox": {"mode": "workspace-write"}}


def test_run_rejects_bad_cli_assignment(capsys):
    rc = cli.main(["run", "--provider", "openai", "--prompt", "ignored", "--env", "MODE"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "--env expects KEY=VALUE" in captured.err


def test_run_web_tools_flags_map_to_request(monkeypatch, capsys):
    seen_requests = []

    class CapturingProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            seen_requests.append(req)
            yield Text(text="ok")

    monkeypatch.setattr(op_mod, "OpenAIProvider", CapturingProvider)

    assert cli.main(
        ["run", "--provider", "openai", "--prompt", "x", "--no-web-tools"]
    ) == 0
    assert cli.main(
        ["run", "--provider", "openai", "--prompt", "x", "--web-tools"]
    ) == 0
    assert cli.main(["run", "--provider", "openai", "--prompt", "x"]) == 0
    capsys.readouterr()

    assert [r.web_tools for r in seen_requests] == [False, True, None]


def test_run_rejects_conflicting_web_tools_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "run",
                "--provider",
                "openai",
                "--prompt",
                "ignored",
                "--web-tools",
                "--no-web-tools",
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--web-tools" in captured.err and "--no-web-tools" in captured.err


def test_run_rejects_conflicting_builtin_tool_flags(capsys):
    rc = cli.main(
        [
            "run",
            "--provider",
            "openai",
            "--prompt",
            "ignored",
            "--builtin-tool",
            "Read",
            "--no-builtin-tools",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "--no-builtin-tools cannot be combined with --builtin-tool" in captured.err


def test_run_rejects_bad_json_cli_option(capsys):
    rc = cli.main(
        [
            "run",
            "--provider",
            "openai",
            "--prompt",
            "ignored",
            "--provider-option",
            "api_key=test-key",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "--provider-option expects KEY=JSON" in captured.err


def test_run_loads_toml_config_file(monkeypatch, tmp_path, capsys):
    seen_requests = []
    seen_provider_options = []
    config_path = tmp_path / "agent-sdk-wrapper.toml"
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    config_path.write_text(
        """
provider = "openai"
model = "gpt-5"
system_prompt = "system from config"
cwd = "."
max_retries = 0
effort = "high"
continue_session = true
builtin_tools = "none"
allowed_tools = ["repo.read_file"]
disallowed_tools = ["repo.delete_file"]
env = { MODE = "config", TOKEN = "config-token" }
provider_options = { api_key = "from-config" }

[extra_options.sandbox]
mode = "workspace-write"

[[mcp_servers]]
type = "stdio"
name = "repo"
command = "python"
args = ["-m", "repo_mcp"]
cwd = "tools"
env = { REPO_MODE = "test" }
env_passthrough = ["FIREFOX_BINARY"]
enabled_tools = ["read_file"]
required = true

[[mcp_servers]]
type = "http"
name = "remote"
url = "http://127.0.0.1:3000/mcp"
headers = { X_MODE = "test" }
env_http_headers = { Authorization = "REMOTE_TOKEN" }
bearer_token_env_var = "REMOTE_BEARER"
default_tools_approval_mode = "prompt"
tool_approval_modes = { search = "approve" }
""",
        encoding="utf-8",
    )

    class CapturingProvider(base.ProviderAdapter):
        name = "openai"

        def __init__(self, **options):
            seen_provider_options.append(options)

        async def stream(self, req):  # type: ignore[override]
            seen_requests.append(req)
            yield Text(text="ok")

    monkeypatch.setattr(op_mod, "OpenAIProvider", CapturingProvider)

    rc = cli.main(
        [
            "run",
            "--config",
            str(config_path),
            "--prompt",
            "ignored",
            "--output",
            "text",
            "--env",
            "TOKEN=cli-token",
            "--allowed-tool",
            "repo.search",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "ok\n"
    assert seen_provider_options == [{"api_key": "from-config"}]

    req = seen_requests[0]
    assert req.model == "gpt-5"
    assert req.system_prompt == "system from config"
    assert req.cwd == tmp_path
    assert req.max_retries == 0
    assert req.effort == "high"
    assert req.continue_session is True
    assert req.builtin_tools == "none"
    assert req.allowed_tools == ["repo.read_file", "repo.search"]
    assert req.disallowed_tools == ["repo.delete_file"]
    assert req.env == {"MODE": "config", "TOKEN": "cli-token"}
    assert req.extra_options == {"sandbox": {"mode": "workspace-write"}}
    assert len(req.mcp_servers) == 2
    stdio_server = req.mcp_servers[0]
    assert stdio_server.name == "repo"
    assert stdio_server.command == "python"
    assert stdio_server.args == ["-m", "repo_mcp"]
    assert stdio_server.cwd == tools_dir
    assert stdio_server.env == {"REPO_MODE": "test"}
    assert stdio_server.env_passthrough == ["FIREFOX_BINARY"]
    assert stdio_server.enabled_tools == ["read_file"]
    assert stdio_server.required is True

    http_server = req.mcp_servers[1]
    assert http_server.name == "remote"
    assert http_server.url == "http://127.0.0.1:3000/mcp"
    assert http_server.headers == {"X_MODE": "test"}
    assert http_server.env_http_headers == {"Authorization": "REMOTE_TOKEN"}
    assert http_server.bearer_token_env_var == "REMOTE_BEARER"
    assert http_server.default_tools_approval_mode == "prompt"
    assert http_server.tool_approval_modes == {"search": "approve"}


def test_run_loads_json_config_prompt(monkeypatch, tmp_path, capsys):
    seen_prompts = []
    config_path = tmp_path / "agent-sdk-wrapper.json"
    config_path.write_text(
        json.dumps({"provider": "openai", "prompt": "prompt from config"}),
        encoding="utf-8",
    )

    class CapturingProvider(base.ProviderAdapter):
        name = "openai"

        async def stream(self, req):  # type: ignore[override]
            seen_prompts.append(req.prompt)
            yield Text(text="ok")

    monkeypatch.setattr(op_mod, "OpenAIProvider", CapturingProvider)

    rc = cli.main(["run", "--config", str(config_path), "--output", "text"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "ok\n"
    assert seen_prompts == ["prompt from config"]


def test_run_rejects_unknown_config_field(tmp_path, capsys):
    config_path = tmp_path / "agent-sdk-wrapper.toml"
    config_path.write_text('provider = "openai"\nunknown = true\n', encoding="utf-8")

    rc = cli.main(["run", "--config", str(config_path), "--prompt", "ignored"])

    captured = capsys.readouterr()
    assert rc == 2
    assert "unknown config field(s): unknown" in captured.err
