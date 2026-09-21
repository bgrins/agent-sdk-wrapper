import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { Agent } from "../src/index.js";

// Drives the bundled Codex runtime against a local mock Responses API.
function responses(steps: Record<string, unknown>[][]) {
  let posts = 0;
  const server = createServer((req, res) => {
    req.resume();
    req.on("end", () => {
      if (req.method !== "POST") {
        res.writeHead(404).end("{}");
        return;
      }
      const index = posts++;
      const items = steps[Math.min(index, steps.length - 1)] ?? [];
      const id = `resp_${index}`;
      const events = [
        { type: "response.created", response: { id } },
        ...items.map((item) => ({ type: "response.output_item.done", item })),
        {
          type: "response.completed",
          response: {
            id,
            usage: {
              input_tokens: 10,
              input_tokens_details: { cached_tokens: 0 },
              output_tokens: 1,
              output_tokens_details: { reasoning_tokens: 0 },
              total_tokens: 11,
            },
          },
        },
      ];
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end(
        events
          .map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`)
          .join(""),
      );
    });
  });
  return server;
}
async function files(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { recursive: true, withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath, entry.name));
}

test("Codex runs keep the API key out of CODEX_HOME and shell commands", async (t) => {
  const key = "sk-wrapper-test-SECRET";
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-codex-home-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const home = join(root, "home");
  const cwd = join(root, "work");
  await mkdir(home);
  await mkdir(cwd);
  // A shell call can snapshot the login shell's environment; printing it would copy the
  // key into the rollout.
  const server = responses([
    [
      {
        type: "function_call",
        id: "fc_0",
        call_id: "call_0",
        name: "exec_command",
        arguments: JSON.stringify({ cmd: "env" }),
      },
    ],
    [
      {
        type: "message",
        role: "assistant",
        id: "msg_1",
        content: [{ type: "output_text", text: "done" }],
      },
    ],
  ]);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());
  const { port } = server.address() as AddressInfo;
  await writeFile(
    join(home, "config.toml"),
    `model_provider = "mock"\n[model_providers.mock]\nname = "mock"\nbase_url = "http://127.0.0.1:${port}/v1"\nwire_api = "responses"\nrequires_openai_auth = true\nrequest_max_retries = 0\nstream_max_retries = 0\nsupports_websockets = false\n`,
  );
  const dead = "http://127.0.0.1:9";
  const run = await new Agent({
    provider: "codex",
    model: "gpt-5.4",
    cwd,
    providerOptions: {
      provider: "openai",
      client: {
        env: {
          PATH: process.env.PATH ?? "/usr/bin:/bin",
          HOME: home,
          CODEX_HOME: home,
          OPENAI_API_KEY: key,
          NO_PROXY: "127.0.0.1,localhost",
          HTTPS_PROXY: dead,
          HTTP_PROXY: dead,
          ALL_PROXY: dead,
          https_proxy: dead,
          http_proxy: dead,
          all_proxy: dead,
        },
      },
      thread: { skipGitRepoCheck: true, approvalPolicy: "never" },
    },
  }).run("Run env.");
  assert.equal(run.status, "success", run.error ?? "");
  const output = run.events.flatMap((env) =>
    env.event.type === "tool_result" ? [env.event.output ?? ""] : [],
  );
  assert.ok(
    output.some((text) => text.includes("PATH=")),
    "env ran",
  );
  assert.ok(!output.some((text) => text.includes(key)));
  const leaked = [];
  for (const path of await files(home))
    if ((await readFile(path)).includes(key)) leaked.push(path);
  assert.deepEqual(leaked, []);
});
