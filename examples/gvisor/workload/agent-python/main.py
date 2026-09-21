import asyncio
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from agent_sdk_wrapper import Agent


async def bridge(reader, writer):
    upstream_reader, upstream_writer = await asyncio.open_unix_connection("/inference/gateway.sock")

    async def copy(source, destination):
        try:
            while data := await source.read(65536):
                destination.write(data)
                await destination.drain()
        finally:
            destination.close()

    await asyncio.gather(copy(reader, upstream_writer), copy(upstream_reader, writer))


async def main():
    if "gvisor" not in platform.release() or os.getuid() == 0:
        raise RuntimeError("Verified non-root gVisor agent required")
    os.makedirs(os.environ["CODEX_HOME"], exist_ok=True)
    request = dict(provider=os.environ["PROVIDER"], model=os.environ["GVISOR_MODEL"])
    request.update(json.loads(os.environ.get("JOB_REQUEST") or "{}"))
    subprocess.run(["node", "/example/shared/project.mjs", "prepare"], check=True)
    prompts = request.get("prompts", json.loads(Path("/example/shared/prompts.json").read_text()))
    server = await asyncio.start_server(bridge, "127.0.0.1", 0)
    base_url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    token = os.environ["GATEWAY_TOKEN"]
    options = dict(
        provider=request["provider"],
        model=request["model"],
        cwd="/job/work",
        continue_session=True,
        timeout=150,
    )
    if request["provider"] == "anthropic":
        options.update(
            permission_mode="bypassPermissions",
            builtin_tools=["Read", "Edit", "Write", "Bash"],
            max_turns=8,
            env={"ANTHROPIC_API_KEY": token, "ANTHROPIC_BASE_URL": base_url},
            setting_sources=[],
        )
    else:
        options.update(
            provider_options={
                "api_key": token,
                "sandbox": "full-access",
                "approval_mode": "deny_all",
                "config": {
                    "config_overrides": (f"openai_base_url={json.dumps(base_url + '/v1')}",)
                },
            },
            web_tools=False,
        )
    async with server:
        agent = Agent(**options)
        trace_prefix = f"/job/output/{time.time_ns() // 1_000_000}-{uuid4()}"
        # Both calls share one worker and session.
        async with asyncio.timeout(150):
            for turn, prompt in enumerate(prompts):
                result = await agent.run(prompt, trace_file=f"{trace_prefix}-{turn:04}.trace.jsonl")
                print(json.dumps({"kind": "result", "result": result.to_dict()}), flush=True)
                if not result.ok:
                    return 1
        subprocess.run(["node", "/example/shared/project.mjs", "check"], check=True)
        return 0


raise SystemExit(asyncio.run(main()))
