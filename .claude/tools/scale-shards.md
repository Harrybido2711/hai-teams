# scale-shards

Find the shard count a provider actually sustains, then run the model at it.

```
Workflow({scriptPath: ".claude/workflows/scale-shards.js",
          args: {model: "NEG_Gemma", ladder: [5, 10, 20], reason: "throughput-bound on the provider"}})
```

It **climbs** the ladder from the lowest rung upward, whatever order the caller writes it in. A
dynamic rate limiter is shaped by recent traffic, so one oversized burst first would depress the
quota for every rung measured after it. The highest rung that stayed healthy wins.

## Input

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | directory name |
| `reason` | yes | — | one sentence |
| `ladder` | yes | — | positive integers, e.g. `[5, 10, 20]`; sorted ascending before use |
| `callTimeout` | no | script default | per-call SIGALRM ceiling in seconds |
| `gateMinutes` | no | `15` | how long each rung is judged for |
| `syncIntervalMin` | no | `180` | |
| `maxHours` | no | `20` | |
| `push` | no | `true` | |

## Output

`{outcome, model, sustained_shards, shards_actually_running, quest_capped, ladder_attempts[], job,
usable, metrics, row_counts, empty_rate, problems, comparability_caveats, needs_prune,
issues_updated, cycles, next}`

`quest_capped: true` means the limit found was Quest's, not the provider's — a different fact, and
it does not generalise to the other models.

## Preflight

- The question really is throughput, not correctness. If the shard count is already settled, this
  is `run-model`.
- The prepare phase **clears the checkpoint** and stops anything running for this model. Rows
  written under a different config must be archived first, not resumed into.
- Shard outputs must carry a shard tag (`{model}_shard{N}of{M}.jsonl`); without one each shard
  overwrites the last.

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: args.ladder must be positive integers` | bad ladder | fix and re-send |
| `outcome: BLOCKED in prepare` | could not stop the old job or clean the checkpoint | resolve by hand; nothing was submitted |
| every rung unhealthy | the floor of the ladder is already too high | re-run with a lower first rung |
| `usable: false` | the ladder finished but the data is bad | read `problems`; the shard count finding may still be valid |
