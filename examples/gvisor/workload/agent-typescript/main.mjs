import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdirSync, readFileSync } from "node:fs";
import net from "node:net";
import os from "node:os";
import { Agent } from "agent-sdk-wrapper";

if (!os.release().includes("gvisor") || process.getuid() === 0)
  throw new Error("Verified non-root gVisor agent required");
mkdirSync(process.env.CODEX_HOME, { recursive: true });
const request = {
  provider: process.env.PROVIDER,
  model: process.env.GVISOR_MODEL,
  ...JSON.parse(process.env.JOB_REQUEST || "{}"),
};
execFileSync("node", ["/example/shared/project.mjs", "prepare"]);
const prompts =
  request.prompts ??
  JSON.parse(readFileSync("/example/shared/prompts.json", "utf8"));
const bridge = net.createServer((client) => {
  const upstream = net.connect("/inference/gateway.sock");
  client.on("error", () => upstream.destroy());
  upstream.on("error", () => client.destroy());
  client.pipe(upstream).pipe(client);
});
await new Promise((resolve) => bridge.listen(0, "127.0.0.1", resolve));
const baseUrl = `http://127.0.0.1:${bridge.address().port}`;
const token = process.env.GATEWAY_TOKEN;
const providerOptions =
  request.provider === "anthropic"
    ? {
        provider: "anthropic",
        options: {
          env: {
            ...process.env,
            ANTHROPIC_API_KEY: token,
            ANTHROPIC_BASE_URL: baseUrl,
          },
          tools: ["Read", "Edit", "Write", "Bash"],
          permissionMode: "bypassPermissions",
          allowDangerouslySkipPermissions: true,
          settingSources: [],
          maxTurns: 8,
        },
      }
    : {
        provider: "openai",
        client: { apiKey: token, baseUrl: `${baseUrl}/v1` },
        thread: {
          sandboxMode: "danger-full-access",
          approvalPolicy: "never",
          skipGitRepoCheck: true,
          webSearchMode: "disabled",
        },
      };
try {
  const agent = new Agent({
    provider: request.provider,
    model: request.model,
    cwd: "/job/work",
    sessionId: request.session_id,
    continueSession: true,
    maxRetries: 0,
    signal: AbortSignal.timeout(150000),
    providerOptions,
  });
  const tracePrefix = `/job/output/${Date.now()}-${randomUUID()}`;
  // Both calls share one worker and session.
  for (const [turn, prompt] of prompts.entries()) {
    const result = await agent.run({
      prompt,
      traceFile: `${tracePrefix}-${String(turn).padStart(4, "0")}.trace.jsonl`,
    });
    // The launcher passes only printable ASCII, like Python's json.dumps output.
    console.log(
      JSON.stringify({ kind: "result", result }).replace(
        /[^\x00-\x7e]/g,
        (char) => `\\u${char.charCodeAt(0).toString(16).padStart(4, "0")}`,
      ),
    );
    if (result.status !== "success") {
      process.exitCode = 1;
      break;
    }
  }
  if (!process.exitCode)
    execFileSync("node", ["/example/shared/project.mjs", "check"]);
} finally {
  bridge.close();
}
