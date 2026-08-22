# compare-providers

Run the same NegotiationToM pilot on two providers on Quest and table the speed difference from
their own logs.

```
Workflow({scriptPath: ".claude/workflows/compare-providers.js",
          args: {a: {dir: "NEG_Gemma", script: "run_pilot.sh", label: "Nebius"},
                 b: {dir: "NEG_Gemma_DeepInfra", script: "run_pilot.sh", label: "DeepInfra"},
                 rows: 300}})
```

## Input

| Field | Required | Notes |
|---|---|---|
| `a`, `b` | yes | each `{dir, script, label}` — all three fields, or it aborts |
| `rows` | yes | pilot size per side; both sides get the same |

## Output

`{status, jobs, watch, headline, table (markdown), caveats, report}`

`report` is the path of the written comparison. `caveats` is not optional reading — two providers
serving "the same" model rarely serve the same decoding config.

## Preflight

- The shared core is verified before either transfer, and the workflow **gates on that
  verification** — both sides must run the same code or the comparison measures the diff instead.
- Both target directories exist on Quest with their own `.env` intact.

## When it fails

| Return | Means | Do |
|---|---|---|
| `error: need {a:{dir,script,label}, …}` | argument shape wrong | re-send with all three fields per side |
| `status: blocked_at_sync` | verification failed; nothing was launched | read `sync` and `checks` |
| `status: launch_failed` | sync passed, submit did not | only the submit needs redoing |
| `status: finished_with_<state>` | one side did not finish | a one-sided result is not a comparison; say so rather than reporting the half that ran |
