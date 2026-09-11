import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { AgentEvent, RunResult, TokenUsage } from "../src/index.js";

const root = new URL("../../../../docs/", import.meta.url);
const read = (path: string) =>
  JSON.parse(readFileSync(new URL(path, root), "utf8"));

// Exhaustive mapped keys make a new public event/field require a schema decision.
const eventTypes = {
  run_started: true,
  text: true,
  thinking: true,
  tool_call: true,
  tool_result: true,
  usage: true,
  session_info: true,
  warning: true,
  error: true,
  run_finished: true,
} satisfies Record<AgentEvent["type"], true>;
const usageFields = {
  input_tokens: true,
  output_tokens: true,
  total_tokens: true,
  cache_read_tokens: true,
  cache_write_tokens: true,
  reasoning_output_tokens: true,
  requests: true,
} satisfies Record<keyof TokenUsage, true>;
const resultFields = {
  run_id: true,
  provider: true,
  model: true,
  status: true,
  ended_reason: true,
  final_text: true,
  structured_output: true,
  usage: true,
  cost_usd: true,
  duration_ms: true,
  session_id: true,
  artifacts_dir: true,
  error: true,
  events: true,
} satisfies Record<keyof RunResult, true>;

test("every TypeScript event has a shared schema branch and replay fixture", () => {
  const schema = read(
    "schemas/agent-sdk-wrapper.event-envelope-jsonl.v1.schema.json",
  );
  const branches = new Set(
    schema.properties.event.oneOf.map(
      (branch: { $ref: string }) => branch.$ref,
    ),
  );
  const fixtures: RunResult[] = read("fixtures/native-twin-v1.json");
  const covered = new Set(
    fixtures.flatMap((run) => run.events.map(({ event }) => event.type)),
  );
  assert.deepEqual([...covered].sort(), Object.keys(eventTypes).sort());
  for (const type of Object.keys(eventTypes)) {
    assert.ok(
      branches.has(`#/$defs/${type}`),
      `Missing schema branch: ${type}`,
    );
    assert.equal(schema.$defs[type].properties.type.const, type);
  }
  assert.deepEqual(
    Object.keys(usageFields).sort(),
    schema.$defs.token_usage.required.toSorted(),
  );
});

test("TypeScript result fields and terminal variants remain within the shared schema", () => {
  const schema = read("schemas/agent-sdk-wrapper.run-result.v1.schema.json");
  assert.deepEqual(
    Object.keys(resultFields).sort(),
    schema.required.toSorted(),
  );
  const statuses = {
    success: true,
    failure: true,
    cancelled: true,
  } satisfies Record<RunResult["status"], true>;
  const reasons = {
    success: true,
    error: true,
    max_turns: true,
    refused: true,
    cancelled: true,
  } satisfies Record<RunResult["ended_reason"], true>;
  for (const status of Object.keys(statuses))
    assert.ok(schema.properties.status.enum.includes(status));
  for (const reason of Object.keys(reasons))
    assert.ok(schema.properties.ended_reason.enum.includes(reason));
});
