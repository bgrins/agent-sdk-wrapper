# Multi-stage audit example

`examples/auditor_style.py` runs planning, analysis, verification and fix-plan
stages using separate agents, structured output and read-only MCP tools.

From the repository root:

```sh
docker compose run --rm python-example-auditor-style
```

Set `PROVIDER` and optionally `MODEL`; provider credentials are required.
Outputs are under `packages/python/results/<provider>/auditor_style/<timestamp>/artifacts/`,
including per-stage traces and a final report. Open them with `docs/trace-viewer.html`.
