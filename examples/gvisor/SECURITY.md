# Security

Trust the host, launcher and gateway. Treat agent code, patches, traces and worker
output as untrusted. SDK permission bypasses require gVisor isolation.

- Workers have no root, capabilities, network beyond loopback or Docker socket.
  Their only outbound channel is the provider API (below).
- The gateway runs as a non-root user without Node, git or setuid binaries. It
  replaces a job token with the provider key on exact API paths.
- Each job has separate state and one writable host output mount. Keep its parent
  host-controlled. The viewer's `--depth 1` skips agent-created directories;
  file reads reject symlinks, hardlinks, FIFOs and files over 16 MiB.
- The launcher prints only printable ASCII from worker stdout and stderr.
  Sandboxed code can still forge result lines, as it can forge traces.
- CPU, memory, processes, Docker logs and individual files are bounded.
  The launcher stops jobs after 180 seconds and removes containers/session volumes.

## Provider API

The agent can send its own requests through the gateway. Provider-run tools then
reach the network for it: Anthropic web search, web fetch and MCP connector, and
OpenAI hosted web search and remote MCP. These tools can send job data off the
host, including to attacker URLs.

The gateway accepts only the `anthropic-beta` values the pinned SDKs send, which
blocks the Anthropic MCP connector. It does not filter request bodies: other
server tools need no beta header, and Codex can stream Responses over WebSocket.
Close the rest at the provider: give jobs a dedicated key, turn off web search and
web fetch for the Claude Console organization, and deny hosted tools in the
OpenAI project's hosted tool permissions. Add a beta to
`infra/gateway/Caddyfile` when an SDK upgrade or new option sends one.

No total disk quota or spend limit. Never execute output on the host. SIGKILL,
host failure or an unresponsive Docker daemon requires external cleanup. Session
state lasts one job.

See [gVisor security](https://gvisor.dev/docs/architecture_guide/security/) and
[credential gateways](https://code.claude.com/docs/en/agent-sdk/secure-deployment).
