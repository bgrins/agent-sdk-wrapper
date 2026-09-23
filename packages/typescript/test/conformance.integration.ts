import { describe, test } from "node:test";
import {
  cases,
  liveSkip,
  type Plan,
  resolveCase,
  runCase,
} from "./conformance/runner.js";

// Billed: runs each case's live section against the real APIs.
describe("conformance cases, live", () => {
  for (const c of cases) {
    if (!c.live) continue;
    const plan = resolveCase(c, "live");
    test(c.id, {
      skip: typeof plan === "string" ? plan : liveSkip(c.provider),
      timeout: 180_000 * (1 + (c.runs?.length ?? 0)),
    }, async () => {
      await runCase(plan as Plan, "live");
    });
  }
});
