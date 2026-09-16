// Offline lifecycle fixture: no SDK or inference calls.
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { closeSync, openSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";

assert.match(os.release(), /gvisor/);
assert.notEqual(process.getuid(), 0);
execFileSync("node", ["/example/shared/project.mjs", "prepare"]);
const baseline = spawnSync("node", ["--test"], {
  cwd: "/job/work",
  encoding: "utf8",
  timeout: 10000,
});
assert.equal(baseline.status, 1);
assert.match(baseline.stdout, /ERR_ASSERTION/);
const request = JSON.parse(process.env.JOB_REQUEST || "{}");
const run_id = randomUUID();
const started = Date.now();
let sequence = 0;
const trace = openSync("/job/output/fixture.trace.jsonl", "w");
const event = (event) =>
  writeFileSync(
    trace,
    `${JSON.stringify({
      run_id,
      sequence: sequence++,
      timestamp: new Date().toISOString(),
      event,
    })}\n`,
  );
event({
  type: "run_started",
  provider: "fixture",
  model: "offline",
  prompt: "Offline lifecycle check",
});
if (request.hang) await new Promise(() => setInterval(() => {}, 1000));
const path = "/job/work/src/deadline.js";
writeFileSync(
  path,
  readFileSync(path, "utf8").replace(
    "milliseconds / 1000",
    request.broken ? "milliseconds / 10" : "milliseconds",
  ),
);
const patch = execFileSync("git", ["diff", "--", "src/deadline.js"], {
  cwd: "/job/work",
});
if (!request.missing) writeFileSync("/job/output/fix.patch", patch);
event({ type: "text", text: "Offline fixture wrote a patch." });
execFileSync("node", ["/example/shared/project.mjs", "check"]);
event({
  type: "run_finished",
  status: "success",
  ended_reason: "success",
  duration_ms: Date.now() - started,
});
closeSync(trace);
