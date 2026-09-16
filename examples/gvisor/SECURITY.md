# Security

Trust the host, launcher and gateway. Treat agent code, patches and traces as untrusted.
SDK permission bypasses require gVisor isolation.

- Workers have no root, capabilities, external network or Docker socket.
- The gateway replaces a job token with the provider key on fixed API routes.
- Each job has separate state and one writable host output mount. Keep its parent
  host-controlled. The viewer's `--depth 1` skips agent-created directories;
  file reads reject symlinks, hardlinks, FIFOs and files over 16 MiB.
- CPU, memory, processes, Docker logs and individual files are bounded.
  The launcher stops jobs after 180 seconds and removes containers/session volumes.

No total disk quota, spend limit or inference-payload filtering. Traces can be
forged. Never execute output on the host. SIGKILL, host failure or an unresponsive
Docker daemon requires external cleanup. Session state is not a durable checkpoint.

See [gVisor security](https://gvisor.dev/docs/architecture_guide/security/) and
[credential gateways](https://code.claude.com/docs/en/agent-sdk/secure-deployment).
