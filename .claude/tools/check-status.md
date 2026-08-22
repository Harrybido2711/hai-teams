# check-status

Read-only snapshot of a running model, with a finish estimate and whatever needs a decision.
Two agents, so it is cheap enough to repeat.

```
Workflow({scriptPath: ".claude/workflows/check-status.js",
          args: {model: "NEG_Gemma", sinceMinutes: 60}})
```

Safe to run alongside a supervising workflow: it never submits, cancels, edits or transfers.

## Input

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | directory name |
| `expected` | no | NegotiationToM's counts | **the script's default is NegotiationToM-specific** — pass this explicitly for any other benchmark, from its page |
| `sinceMinutes` | no | `0` (all) | rate over the last N minutes, so a bad first hour stops dragging the average |

## Output

`{model, running, workers, rows{by task}, rows_total, progress_pct, rows_per_min, hang_rate,
empty_rows, healthy, eta_hours, eta_confidence, eta_basis, ceiling_ok, halt_markers,
stalled_shards, needs_decision[], summary}`

`needs_decision` is the field to act on. `eta_confidence` and `eta_basis` exist so an ETA computed
from four minutes of data is not mistaken for a measurement.

## Preflight

Nothing. This is the one tool with no prerequisites — that is the point of having it.

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: args.model is required` | argument problem | re-send |
| `outcome: NO OBSERVATION` | the watcher returned nothing | re-run before concluding anything; absence of observation is not evidence of a stall |
| `healthy: false` | judged unhealthy | this tool only reports — the fix is `fix-broken-run` |
| `eta_confidence` low | window too short | pass a larger `sinceMinutes`, or wait |
