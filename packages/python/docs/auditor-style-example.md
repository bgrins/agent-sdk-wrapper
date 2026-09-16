# Multi-stage audit

Run separate agents for planning, analysis, verification and reporting:

```sh
docker compose run --rm python-example-auditor-style
```

Run from the repository root with provider credentials. Set `PROVIDER` and optional `MODEL`.
Find traces and reports under `packages/python/results/<provider>/auditor_style/<timestamp>/artifacts/`.
View them with `npm run trace-viewer -- packages/python/results`.
