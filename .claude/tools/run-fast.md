# run-fast

One model as three parallel per-task jobs instead of one sequential array, supervised, with a
supervise phase that may repair rather than only report.

```
Workflow({scriptPath: ".claude/workflows/run-fast.js",
          args: {model: "NEG_Gemma", reason: "last model, needed today"}})
```

It replaces the single `--task all` array with one array per task and **does not change
`--total-shards`**, so every row already written stays valid and falling back costs nothing.

## Input

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | directory name |
| `reason` | yes | — | one sentence; reaches `ISSUES.md` |
| `tasks` | no | `["desire","belief","intention"]` | one job array per entry |
| `gateMinutes` | no | `20` | longer than `run-model` — three arrays have to all come up |
| `syncIntervalMin` | no | `120` | |
| `maxHours` | no | `30` | |
| `push` | no | `true` | |

## Output

`{outcome, model, jobs[], workers, usable, metrics, row_counts, empty_rate,
vs_reasoning_on_pilot, comparability_caveats, problems, repairs[], measured_latency_p99,
ceiling_verdict, issues_updated, cycles, next}`

`outcome` is `COMPLETE` or `COMPLETE BUT NOT USABLE`. `repairs` lists what the supervise phase did
without being asked — pruned empty rows, resubmitted a dead shard, re-merged.

## Preflight

- Same as `run-model`, plus: the provider must tolerate 3x the concurrency. If that is the open
  question, use `scale-shards` instead — it measures the ceiling instead of assuming one.
- `comparability_caveats` exists because a faster config is often a different config. Read it
  before putting the number in a cross-model table.

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: …` | argument problem, nothing was submitted | as `run-model` |
| `outcome: LAUNCH FAILED` | the per-task arrays did not start; `detail` says why | fall back to `run-model`; no row is invalidated by having tried |
| `workers` below the number of tasks | Quest started fewer arrays than asked | the run is still valid, just not three-wide; decide whether to wait |
| `usable: false` | numbers not trustworthy | read `problems`; `repairs` may already have pruned |
