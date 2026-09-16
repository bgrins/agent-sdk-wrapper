import assert from "node:assert/strict";
import { test } from "node:test";
import { deadlineAfter } from "../src/deadline.js";

const start = new Date("2026-01-01T00:00:00.000Z");
test("adds seconds without losing precision", () => {
  assert.equal(
    deadlineAfter(start, "1.5s").toISOString(),
    "2026-01-01T00:00:01.500Z",
  );
});
test("crosses a minute boundary", () => {
  assert.equal(
    deadlineAfter(start, "90s").toISOString(),
    "2026-01-01T00:01:30.000Z",
  );
});
test("does not mutate the input", () => {
  deadlineAfter(start, "2m");
  assert.equal(start.toISOString(), "2026-01-01T00:00:00.000Z");
});
test("rejects invalid and negative durations", () => {
  assert.throws(() => deadlineAfter(start, "bad duration"), TypeError);
  assert.throws(() => deadlineAfter(start, "-1s"), TypeError);
});
