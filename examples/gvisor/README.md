# gVisor example

A Python or TypeScript worker edits a Node app and writes a patch and SDK traces.
Two agent calls share one session; the native SDK runs tools and the agent loop.

```text
Bash → gVisor worker → mounted output → trace viewer
              ↓
       credential gateway → provider API
```

## Run

Requires Bash and Linux Docker with Compose. Install `curl`, `bzip2` and gVisor
on the Docker host: `bash examples/gvisor/infra/install-runsc.sh`.

From the repository root, with API keys exported:

```sh
bash examples/gvisor/scripts/build.sh
LANGUAGE=typescript PROVIDER=anthropic bash examples/gvisor/workload/run.sh
LANGUAGE=python PROVIDER=codex bash examples/gvisor/workload/run.sh
```

Load a trusted, shell-formatted `.env` with `set -a; source .env; set +a`.
Override the model with `GVISOR_MODEL` and the output root with `GVISOR_OUTPUT_DIR`.

Find `fix.patch` and one `*.trace.jsonl` per SDK call in `results/gvisor-output/<job>/`.
Interrupted jobs keep partial output. Checks cover resource access and output presence,
not patch correctness.

## View traces

Requires Node 22.14+:

```sh
npm run trace-viewer -- results/gvisor-output --depth 1
```

Open <http://127.0.0.1:8765> and select a trace; it refreshes automatically.
`--depth 1` skips agent-created subdirectories. Use `results/gvisor-output/tests`
for test runs.

## Adapt

- `workload/run.sh`: SDK, model and credentials.
- `workload/agent-{python,typescript}/`: your program and local SDK dependency.
- `workload/shared/` and `workload/project/`: sample task and app setup.
- `infra/` and `compose.yaml`: lifecycle, images, gateway and isolation.
- `tests/run.sh` and `tests/compose.yaml`: test endpoint and workers.

Workers take `PROVIDER`, `GVISOR_MODEL` and optional `JOB_REQUEST` JSON for prompts
or a session ID. They write results to stdout and files to `/job/output`.
Cleanup removes session state. Cloud deployment is not included.

```sh
bash examples/gvisor/scripts/test.sh  # offline; builds test images; requires Node
AGENT_SDK_WRAPPER_RUN_INTEGRATION=1 AGENT_SDK_WRAPPER_TS_RUN_INTEGRATION=1 \
  bash examples/gvisor/scripts/test.sh --live
bash examples/gvisor/workload/run.sh cleanup  # remove leftover example resources
```

[Security limits](SECURITY.md).

## macOS setup

Install Docker CLI, Compose and Colima, then run:

```sh
mkdir -p results/gvisor-output
colima start agent-sdk-gvisor --runtime docker --cpus 4 --memory 6 --disk 30 \
  --mount "$PWD/results/gvisor-output:w" --activate=false \
  --ssh-agent=false --ssh-config=false --port-forwarder none
export DOCKER_CONTEXT=colima-agent-sdk-gvisor
colima --profile agent-sdk-gvisor ssh -- sudo apt-get update
colima --profile agent-sdk-gvisor ssh -- sudo apt-get install -y curl bzip2
colima --profile agent-sdk-gvisor ssh -- bash -s < examples/gvisor/infra/install-runsc.sh
```

Rerun the last command after restarting the VM.
