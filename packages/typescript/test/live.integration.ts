import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { Agent, collectRun } from "../src/index.js";
import type { EventEnvelope, Provider } from "../src/index.js";

const enabled = process.env.AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION === "1";
for (const provider of [
  "anthropic",
  "openai",
] as const satisfies readonly Provider[]) {
  const key = provider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY";
  const skip = !enabled
    ? "Set AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1"
    : !process.env[key]
      ? `${key} is required`
      : false;
  test(`live ${provider}: stream, collect, continue and resume in another Agent`, {
    skip,
    timeout: 180_000,
  }, async () => {
    const cwd = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-ts-"));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 170_000);
    try {
      const providerOptions =
        provider === "anthropic"
          ? {
              provider,
              options: {
                tools: [],
                permissionMode: "dontAsk" as const,
                settingSources: [],
              },
            }
          : {
              provider,
              thread: {
                skipGitRepoCheck: true,
                sandboxMode: "read-only" as const,
                approvalPolicy: "never" as const,
                webSearchMode: "disabled" as const,
                networkAccessEnabled: false,
              },
            };
      const model =
        process.env[
          provider === "anthropic"
            ? "AGENT_SDK_WRAPPER_TS_ANTHROPIC_MODEL"
            : "AGENT_SDK_WRAPPER_TS_OPENAI_MODEL"
        ] || undefined;
      const defaults = {
        provider,
        providerOptions,
        model,
        cwd,
        maxRetries: 0,
        continueSession: true,
        signal: controller.signal,
      };
      const agent = new Agent(defaults);
      const events: EventEnvelope[] = [];
      const first = await collectRun(
        agent.stream(
          "Remember the token NATIVE_TWIN_42 for later. Reply READY.",
        ),
        (event) => {
          events.push(event);
        },
      );
      assert.equal(first.status, "success", first.error ?? "");
      assert.ok(first.session_id);
      assert.ok(first.final_text.trim());
      assert.equal(events[0]?.event.type, "run_started");
      assert.equal(events.at(-1)?.event.type, "run_finished");
      const second = await agent.run(
        "What token did I ask you to remember? Return only that token.",
      );
      assert.equal(second.status, "success", second.error ?? "");
      assert.equal(second.session_id, first.session_id);
      assert.match(second.final_text, /NATIVE_TWIN_42/);
      const resumed = await new Agent({
        ...defaults,
        sessionId: first.session_id,
      }).run("Repeat the remembered token.");
      assert.equal(resumed.status, "success", resumed.error ?? "");
      assert.equal(resumed.session_id, first.session_id);
      assert.match(resumed.final_text, /NATIVE_TWIN_42/);
    } finally {
      clearTimeout(timer);
      controller.abort();
      await rm(cwd, { recursive: true, force: true });
    }
  });
}
