"""Live smoke test for the remote-control server.

Drives a running `agent-sdk-wrapper serve` instance end to end:
create session -> send a message that needs a permission -> allow it ->
interrupt a long task -> close -> resume by native id -> recall check.

Usage:
    uv run python scripts/remote_control_smoke.py --provider anthropic --model claude-haiku-4-5
    uv run python scripts/remote_control_smoke.py --provider openai --model gpt-5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

BASE = "http://127.0.0.1:8765"


class Smoke:
    def __init__(self, http: aiohttp.ClientSession, base: str) -> None:
        self.http = http
        self.base = base
        self.events: list[dict] = []
        self.cursor = 0
        self.sse_task: asyncio.Task | None = None
        self.session_id: str | None = None

    async def api(self, method: str, path: str, **kwargs):
        async with self.http.request(method, f"{self.base}{path}", **kwargs) as resp:
            body = await resp.json()
            if resp.status >= 400:
                raise RuntimeError(f"{method} {path} -> {resp.status}: {body}")
            return body

    async def follow_events(self, session_id: str) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
        async with self.http.get(
            f"{self.base}/api/sessions/{session_id}/events?since=0", timeout=timeout
        ) as resp:
            async for chunk in resp.content:
                line = chunk.decode().strip()
                if line.startswith("data: "):
                    entry = json.loads(line[len("data: "):])
                    self.events.append(entry)
                    event = entry["event"]
                    detail = {
                        k: v for k, v in event.items() if k != "type"
                    }
                    text = json.dumps(detail, ensure_ascii=False)
                    print(f"  [sse #{entry['seq']}] {event['type']}: {text[:200]}")

    async def start_session(self, **body) -> dict:
        info = await self.api("POST", "/api/sessions", json=body)
        self.session_id = info["session_id"]
        self.events = []
        self.cursor = 0
        self.sse_task = asyncio.create_task(self.follow_events(self.session_id))
        print(f"session {self.session_id} state={info['state']}")
        return info

    async def send(self, text: str) -> None:
        print(f"> send: {text}")
        await self.api(
            "POST", f"/api/sessions/{self.session_id}/messages", json={"text": text}
        )

    async def wait_for(self, predicate, what: str, timeout: float = 180.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            while self.cursor < len(self.events):
                entry = self.events[self.cursor]
                self.cursor += 1
                if predicate(entry["event"]):
                    return entry["event"]
            await asyncio.sleep(0.1)
        raise TimeoutError(f"timed out waiting for {what}")

    async def wait_state(self, state: str, timeout: float = 180.0) -> None:
        await self.wait_for(
            lambda e: e["type"] == "state_changed" and e["state"] == state,
            f"state={state}",
            timeout,
        )
        print(f"  state -> {state}")

    async def stop_sse(self) -> None:
        if self.sse_task:
            self.sse_task.cancel()
            try:
                await self.sse_task
            except (asyncio.CancelledError, aiohttp.ClientError):
                pass
            self.sse_task = None


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--cwd", default="/tmp/rc-spike-test")
    parser.add_argument(
        "--permission-mode",
        default=None,
        help="e.g. 'untrusted' to force Codex approval prompts for every command",
    )
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--skip-interrupt", action="store_true")
    parser.add_argument("--skip-resume", action="store_true")
    args = parser.parse_args()

    create_body = {"provider": args.provider, "cwd": args.cwd}
    if args.model:
        create_body["model"] = args.model
    if args.permission_mode:
        create_body["permission_mode"] = args.permission_mode

    async with aiohttp.ClientSession() as http:
        smoke = Smoke(http, args.base)

        print("== 1. start session ==")
        await smoke.start_session(**create_body)

        print("== 2. permission round-trip ==")
        await smoke.send(
            "Run exactly this shell command: echo hello-from-remote > smoke.txt "
            "-- then reply with just: WROTE"
        )
        request = await smoke.wait_for(
            lambda e: e["type"] == "permission_request", "permission_request"
        )
        print(f"  approving {request['request_id']} tool={request.get('tool')}")
        await smoke.api(
            "POST",
            f"/api/sessions/{smoke.session_id}/permissions/{request['request_id']}",
            json={"behavior": "allow"},
        )
        await smoke.wait_state("idle")
        texts = [e["event"] for e in smoke.events if e["event"]["type"] == "text"]
        print(f"  final text: {texts[-1]['text'][:100] if texts else '<none>'}")

        if not args.skip_interrupt:
            print("== 3. interrupt ==")
            await smoke.send(
                "Count slowly from 1 to 50, one number per line of output text. "
                "Do not use any tools."
            )
            await smoke.wait_state("running", timeout=30)
            await asyncio.sleep(3)
            await smoke.api("POST", f"/api/sessions/{smoke.session_id}/interrupt")
            print("  interrupt sent")
            await smoke.wait_state("idle", timeout=60)

        info = await smoke.api("GET", f"/api/sessions/{smoke.session_id}")
        native_id = info["native_session_id"]
        print(f"== native session id: {native_id} ==")

        print("== 4. close ==")
        await smoke.api("DELETE", f"/api/sessions/{smoke.session_id}")
        await smoke.stop_sse()

        if not args.skip_resume:
            print("== 5. resume ==")
            resume_body = dict(create_body, resume=native_id)
            await smoke.start_session(**resume_body)
            await smoke.send(
                "Without using any tools: what filename did I ask you to write "
                "earlier in this conversation? Reply with just the filename."
            )
            await smoke.wait_state("running", timeout=30)
            await smoke.wait_state("idle")
            texts = [e["event"] for e in smoke.events if e["event"]["type"] == "text"]
            answer = texts[-1]["text"] if texts else "<none>"
            print(f"  recall answer: {answer[:100]}")
            ok = "smoke.txt" in answer
            print(f"  resume recall: {'PASS' if ok else 'FAIL'}")
            await smoke.api("DELETE", f"/api/sessions/{smoke.session_id}")
            await smoke.stop_sse()
            if not ok:
                return 1

        print("SMOKE PASS")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
