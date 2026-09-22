import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { type TestContext, test } from "node:test";
import { Codex, type ThreadEvent } from "@openai/codex-sdk";
import { Agent, type AgentDefaults } from "../src/index.js";
import { type Step, startMock } from "./conformance/mock.js";

// Drives the bundled Codex runtime against a local mock Responses API.
async function scratch(t: TestContext, steps: Step[]) {
  const root = await mkdtemp(join(tmpdir(), "agent-sdk-wrapper-codex-home-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const home = join(root, "home");
  const cwd = join(root, "work");
  await mkdir(home);
  await mkdir(cwd);
  const mock = await startMock("codex", steps);
  t.after(() => mock.close());
  await writeFile(
    join(home, "config.toml"),
    `model_provider = "mock"\n[model_providers.mock]\nname = "mock"\nbase_url = "${mock.url}"\nwire_api = "responses"\nrequires_openai_auth = true\nrequest_max_retries = 0\nstream_max_retries = 0\nsupports_websockets = false\n`,
  );
  const dead = "http://127.0.0.1:9";
  const env = {
    PATH: process.env.PATH ?? "/usr/bin:/bin",
    HOME: home,
    CODEX_HOME: home,
    NO_PROXY: "127.0.0.1,localhost",
    HTTPS_PROXY: dead,
    HTTP_PROXY: dead,
    ALL_PROXY: dead,
    https_proxy: dead,
    http_proxy: dead,
    all_proxy: dead,
  };
  return { root, home, cwd, env, mock };
}
const thread = { skipGitRepoCheck: true, approvalPolicy: "never" } as const;
async function files(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { recursive: true, withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile())
    .map((entry) => join(entry.parentPath, entry.name));
}
const alive = (pid: number) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

test("Codex runs keep the API key out of CODEX_HOME and shell commands", async (t) => {
  const key = "sk-wrapper-test-SECRET";
  // A shell call can snapshot the login shell's environment; printing it would copy the
  // key into the rollout.
  const { home, cwd, env } = await scratch(t, [
    { shell: "env" },
    { text: "done" },
  ]);
  const run = await new Agent({
    provider: "codex",
    model: "gpt-5.4",
    cwd,
    providerOptions: {
      provider: "openai",
      client: { env: { ...env, OPENAI_API_KEY: key } },
      thread,
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

test("an async provider-event rejection stops codex exec and propagates", {
  skip: process.platform === "win32",
}, async (t) => {
  const commandPid = join(tmpdir(), `agent-sdk-wrapper-command-${process.pid}`);
  const { root, cwd, env, mock } = await scratch(t, [
    { shell: `echo $$ > ${commandPid}; exec sleep 20` },
    { text: "done" },
  ]);
  // Record the runtime's PID: exec keeps it for the real binary.
  const codex = join(root, "codex");
  const binary = (
    new Codex() as unknown as { exec: { executablePath: string } }
  ).exec.executablePath;
  await writeFile(
    codex,
    `#!/bin/sh\necho $$ > "$0.pid"\nexec "${binary}" "$@"\n`,
    { mode: 0o755 },
  );
  const error = new Error("async callback failed");
  const defaults: AgentDefaults = {
    provider: "codex",
    model: "gpt-5.4",
    cwd,
    // Without the rejection stopping it, the run would end cancelled instead.
    signal: AbortSignal.timeout(10_000),
    providerOptions: {
      provider: "openai",
      client: {
        env: { ...env, OPENAI_API_KEY: "sk-mock" },
        codexPathOverride: codex,
      },
      thread: { ...thread, sandboxMode: "danger-full-access" },
    },
    onProviderEvent: async (event) => {
      if ((event as ThreadEvent).type === "item.started") throw error;
    },
  };
  try {
    await assert.rejects(
      new Agent(defaults).run("Run it."),
      (thrown) => thrown === error,
    );
    const pid = Number(await readFile(`${codex}.pid`, "utf8"));
    for (let wait = 0; alive(pid) && wait < 50; wait++) await sleep(100);
    assert.equal(alive(pid), false, "codex exec is still running");
    assert.equal(mock.requests.length, 1, "the run continued past the command");
  } finally {
    // Model commands outlive codex exec (a documented limit); stop this one.
    await sleep(500);
    const pid = Number(await readFile(commandPid, "utf8").catch(() => ""));
    if (pid && alive(pid)) process.kill(pid, "SIGKILL");
    await rm(commandPid, { force: true });
  }
});
