# Auditor-Style Example Notes

`examples/auditor_style.py` exercises a structured auditor workflow using
`agent-sdk-wrapper`.

The example covers:

- multiple role-specific `Agent` instances
- structured output contracts between stages
- a verifier-approved follow-up stage that produces a non-code fix plan
- provider/model switching through `PROVIDER` and `MODEL`
- read-only stdio MCP tools for project facts and artifact policy
- per-stage `max_turns`
- top-level workflow event and usage aggregation
- per-stage `trace.jsonl`, `manifest.json`, and `result.json`
- top-level JSON artifacts for plan, analysis, verification, fix plan, context,
  stats, and final report

The example intentionally does not recreate the downstream audit MCP harness or
write source patches. Its MCP server is a small read-only facts source, and the
post-verification branch writes a rollout/fix plan instead of a diff.

Run it with:

```bash
docker compose run --rm example-auditor-style
```

Outputs land under
`results/<provider>/auditor_style/<timestamp>/artifacts/`.

The stable replay surface is still `manifest.json` plus `trace.jsonl`.
Provider side files are diagnostic only.
