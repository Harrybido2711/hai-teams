# run-model

Full lifecycle for one model: check local -> sync to Quest -> launch -> gate -> supervise -> audit
-> record. The default way to start or restart a run.

```
Workflow({scriptPath: ".claude/workflows/run-model.js",
          args: {model: "NEG_Gemma", reason: "reasoning-off rerun"}})
```

## Input

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | the directory: `NEG_Gemma`, not "the Gemma run" |
| `reason` | yes | — | one sentence; it reaches `ISSUES.md` |
| `pilot` | no | `false` | `run_pilot.sh` (10% of data) vs `run_negotiation.sh` |
| `gateMinutes` | no | `5` | how long to watch before trusting the launch |
| `syncIntervalMin` | no | `60` | pull results down, commit and push, this often |
| `maxHours` | no | `24` | stop supervising; the job itself keeps running |
| `push` | no | `true` | `false` pulls results down without publishing them |

## Output

`{model, mode, job, gate{healthy,rows,empty}, cycles, usable, metrics, row_counts, empty_rate,
problems, needs_rerun, needs_prune, proposed_fix, sync{local_matches_quest,pushed,
anything_only_on_quest}, issues_updated, next}`

Branch on `usable`, then `needs_rerun`. The workflow never retries itself: `needs_rerun` arrives
with `proposed_fix` because the fix is usually a code change and belongs in front of a human before
another run spends hours on it. `needs_prune` means rows already written must go before the rerun,
or one result set ends up holding two configurations.

## Preflight

- The code that should run exists locally and is verified — phase 1 reviews it, it does not write it.
- Nothing is already running for this model (`squeue -u uwr0681`). If there is, this is
  `fix-broken-run`, not this.
- If the prompt or decoding config changed, decide archive-vs-prune first
  (`../references/quest-cluster.md`).

## When it fails

| Return | Means | Do |
|---|---|---|
| `aborted: args arrived as a string that is not JSON` | the caller's object was serialised and is malformed | re-send `args` as a real object |
| `aborted: args.model is required` | `model` missing, or `args` was a string that parsed to something else | check `args` shape before blaming the caller |
| `outcome: BLOCKED in local check` | the reviewer refused; `blockers` says why | fix locally, rerun. Nothing was transferred or submitted |
| `outcome: launch failed` | `sbatch` did not return a job id | read `launched`; the sync already happened, so only the submit needs redoing |
| `usable: false` | it ran and the numbers cannot be trusted | read `problems` before publishing anything |
