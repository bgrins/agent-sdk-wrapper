import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { readOutputFile } from "./output-file.mjs";

const mode = process.argv[2] ?? "--offline";
assert.ok(
  [
    "--offline",
    "--live",
    "--security",
    "--fixture",
    "--bad-patch",
    "--missing-output",
  ].includes(mode),
);
const env = {
  ...process.env,
  LANGUAGE: process.env.LANGUAGE ?? "typescript",
  GVISOR_OUTPUT_DIR:
    process.env.GVISOR_OUTPUT_DIR ?? "results/gvisor-output/tests",
};
const request = {};
let worker = mode === "--security" ? "security" : "sdk";
if (["--fixture", "--bad-patch", "--missing-output"].includes(mode)) {
  worker = "fixture";
  request.broken = mode === "--bad-patch";
  request.missing = mode === "--missing-output";
}
const token = randomUUID();
if (mode === "--live") {
  const flag =
    env.LANGUAGE === "python"
      ? "AGENT_SDK_WRAPPER_RUN_INTEGRATION"
      : "AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION";
  assert.equal(env[flag], "1", `Set ${flag}=1 for live tests`);
}
if (["--offline", "--live"].includes(mode)) {
  request.prompts = JSON.parse(
    readFileSync(new URL("../workload/shared/prompts.json", import.meta.url)),
  );
  // Non-ASCII checks that results survive the launcher's ASCII-only output.
  request.prompts[0] += ` Remember token ${token} (café).`;
  request.prompts[1] += " Include the token I asked you to remember.";
}
env.JOB_REQUEST = JSON.stringify(request);
const { status, stdout, stderr, error } = spawnSync(
  "bash",
  mode === "--live"
    ? ["examples/gvisor/workload/run.sh"]
    : ["examples/gvisor/tests/run.sh", worker],
  {
    env,
    encoding: "utf8",
    timeout: 210000,
    maxBuffer: 16 * 1024 * 1024,
  },
);
if (error) throw error;
const output = stderr.match(/^Output: (.+)$/m)?.[1];
assert.ok(output, "Launcher reports the mounted output directory");
const failed = mode === "--offline" || mode === "--missing-output";
assert.equal(status, failed ? 1 : 0, stdout + stderr);
const envelopes = readdirSync(output)
  .filter((name) => name.endsWith(".trace.jsonl"))
  .sort()
  .flatMap((name) =>
    readOutputFile(`${output}/${name}`)
      .trim()
      .split("\n")
      .filter(Boolean)
      .map(JSON.parse),
  );
assert.equal(
  existsSync(`${output}/fix.patch`),
  !failed && mode !== "--security",
);
if (["--offline", "--live"].includes(mode)) {
  const results = stdout
    .trim()
    .split("\n")
    .map(JSON.parse)
    .filter((record) => record.kind === "result")
    .map((record) => record.result);
  assert.equal(results.length, failed ? 1 : 2);
  for (const result of results) {
    assert.equal(result.status, failed ? "failure" : "success");
    const events = envelopes.filter((event) => event.run_id === result.run_id);
    assert.equal(events[0].event.type, "run_started");
    assert.equal(events.at(-1).event.type, "run_finished");
    assert.deepEqual(events, result.events);
  }
  if (failed) assert.match(JSON.stringify(results[0]), /Offline gateway probe/);
  else {
    assert.ok(results[0].session_id);
    assert.equal(results[0].session_id, results[1].session_id);
    assert.ok(results[1].final_text.includes(token));
  }
} else if (mode !== "--security") {
  assert.equal(envelopes[0].event.type, "run_started");
  if (mode === "--missing-output")
    assert.match(stderr, /Missing or empty output patch/);
  else assert.equal(envelopes.at(-1).event.type, "run_finished");
  if (mode === "--bad-patch")
    assert.match(readOutputFile(`${output}/fix.patch`), /milliseconds \/ 10\)/);
  for (const text of [stdout, stderr]) {
    assert.match(text, /forged/);
    assert.doesNotMatch(text.replace(/[\t\n]/g, ""), /\p{Cc}/u);
  }
}
console.log(
  `${env.LANGUAGE} ${env.PROVIDER ?? "anthropic"} ${mode}: ${output}`,
);
