import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync } from "node:fs";
import { basename, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";
import { createTraceServer } from "../../../scripts/trace-viewer.mjs";

const docker = (args) =>
  execFileSync("docker", args, { encoding: "utf8", timeout: 10000 }).trim();
for (const [signal, status] of [
  ["SIGTERM", 143],
  ["SIGINT", 130],
  [null, 124],
]) {
  test(`${signal ?? "deadline"} stops the job and leaves its live trace readable`, {
    timeout: 30000,
  }, async (t) => {
    const root = resolve("results/gvisor-output/tests");
    const server = createTraceServer(root, { depth: 1 });
    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    t.after(() => {
      server.closeAllConnections();
      server.close();
    });
    const base = `http://127.0.0.1:${server.address().port}`;
    const child = spawn("bash", ["examples/gvisor/tests/run.sh", "fixture"], {
      env: {
        ...process.env,
        LANGUAGE: "typescript",
        PROVIDER: "openai",
        GVISOR_OUTPUT_DIR: root,
        TEST_TIMEOUT_SECONDS: signal ? "180" : "12",
        JOB_REQUEST: '{"hang":true}',
      },
      stdio: ["ignore", "ignore", "pipe"],
    });
    let stderr = "";
    child.stderr.on("data", (data) => {
      stderr += data;
    });
    const done = once(child, "close");
    t.after(async () => {
      if (child.exitCode === null) {
        child.kill("SIGTERM");
        await done;
      }
    });
    let output;
    for (let i = 0; i < 100; i++) {
      output = stderr.match(/^Output: (.+)$/m)?.[1];
      if (output && existsSync(`${output}/fixture.trace.jsonl`)) break;
      if (child.exitCode !== null) break;
      await delay(100);
    }
    assert.ok(output && existsSync(`${output}/fixture.trace.jsonl`), stderr);
    const id = basename(output);
    const [container] = JSON.parse(docker(["inspect", `${id}-agent`]));
    assert.equal(container.HostConfig.Runtime, "runsc-agent");
    assert.equal(container.HostConfig.NetworkMode, "none");
    assert.equal(container.HostConfig.ReadonlyRootfs, true);
    assert.equal(container.HostConfig.Memory, 2 * 1024 ** 3);
    assert.equal(container.HostConfig.PidsLimit, 256);
    const runs = await (await fetch(`${base}/api/runs`)).json();
    const run = runs.find((entry) => entry.trace?.includes(id));
    assert.ok(run, "Viewer discovers the running job without a copy step");
    const trace = await (await fetch(base + run.trace)).text();
    assert.equal(JSON.parse(trace).event.type, "run_started");
    if (signal) child.kill(signal);
    assert.equal((await done)[0], status, stderr);
    if (!signal) assert.match(stderr, /Job deadline exceeded/);
    const filter = ["--filter", `label=com.docker.compose.project=${id}`];
    assert.equal(docker(["ps", "-aq", ...filter]), "");
    assert.equal(docker(["volume", "ls", "-q", ...filter]), "");
    assert.equal(await (await fetch(base + run.trace)).text(), trace);
    assert.equal(existsSync(`${output}/fix.patch`), false);
  });
}
