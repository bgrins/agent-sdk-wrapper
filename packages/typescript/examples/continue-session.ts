import { mkdir, writeFile } from "node:fs/promises";
import { Agent, collectRun, normalizeProvider } from "../src/index.js";

// Run the compiled example after npm test (which compiles examples too).
const provider = normalizeProvider(process.env.PROVIDER ?? "anthropic");
const defaults = {
  provider,
  model: process.env.MODEL,
  cwd: process.cwd(),
  continueSession: true,
};
const agent = new Agent(defaults);
const first = await collectRun(
  agent.stream("Remember the token NATIVE_TWIN_42. Reply READY."),
  (envelope) => {
    if (envelope.event.type === "text")
      process.stdout.write(envelope.event.text);
  },
);
if (first.status !== "success" || !first.session_id)
  throw new Error(first.error ?? "No session returned");
await mkdir("results/typescript", { recursive: true });
await writeFile(
  "results/typescript/session.json",
  JSON.stringify({ provider, sessionId: first.session_id }),
);
await writeFile(
  "results/typescript/trace.jsonl",
  `${first.events.map((event) => JSON.stringify(event)).join("\n")}\n`,
);
const next = await agent.run("What token did I ask you to remember?");
if (next.status !== "success")
  throw new Error(next.error ?? "Continuation failed");
console.log("\n", next.status, next.final_text);
const resumed = new Agent({ ...defaults, sessionId: first.session_id });
const third = await resumed.run("Repeat the remembered token.");
if (third.status !== "success") throw new Error(third.error ?? "Resume failed");
console.log(third.final_text);
