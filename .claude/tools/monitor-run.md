# monitor-run

Phase 4: is a running job in trouble, when will it finish, what does it cost — for one run or several
of one benchmark side by side. **Read-only**, so it can be repeated on a timer beside anything else.

```
Workflow({scriptPath: ".claude/workflows/monitor-run.js", args: {
  benchmark: "emobench", model: "EMO_GPT_5.6_Luna", jobId: 4511556}})

Workflow({scriptPath: ".claude/workflows/monitor-run.js", args: {
  benchmark: "emobench",
  runs: [{label: "Google",     dir: "EMO_Gemini_Flash3.5lite_Google",     jobId: 3810331},
         {label: "OpenRouter", dir: "EMO_Gemini_Flash3.5lite_OpenRouter", jobId: 3810332,
          priceIn: 0.30, priceOut: 2.50},
         {label: "effort=low", dir: "EMO_GPT_5.6_Luna", resultsDir: "results_eLow", jobId: 4511556}]}})
```

## Phases

| Phase | Agent | Skill | Stops the workflow when |
|---|---|---|---|
| Observe | `watcher`, one per run | [check-run](../skills/check-run/SKILL.md) | every observer failed |
| Judge | `evaluator` | [audit-results](../skills/audit-results/SKILL.md), the in-progress checks | — |

## Input

| Field | Required | Notes |
|---|---|---|
| `benchmark` | yes | page stem |
| `model` + `jobId`, or `runs` | yes | `runs: [{label, dir, jobId, resultsDir?, priceIn?, priceOut?}]` |
| `questDir`, `expected` | no | otherwise read from the page |
| `sinceMinutes` | no | rate over the last N minutes, so a bad first hour stops dragging the ETA |

`resultsDir` separates sweep arms of one model that share a folder; two runs resolving to one path
are refused, since identical rows would read as agreement. Prices are $/M tokens — **omit rather
than guess**, and cost reads *not established*.

## Output

`{benchmark, perRun: {label: watcher STATUS}, status: trust, recommendation, report, next}` —
`report` holds the verdict, finish-time and cost tables and what needs a decision.

## Adapting

Change freely: add a third phase that pulls results ([pull-results](../skills/pull-results/SKILL.md))
on a long run; drop Judge for a quick "is it alive". Keep: read-only — no `sbatch`, `scancel`, edits or
provider calls, because a probe spends quota against the very run being measured; rows from the
`.jsonl`, never the CSV; under 20 rows is `too-early`.

## When it fails

| Symptom | Cause |
|---|---|
| a healthy run reported as stalled | judged from `log.txt` size — these runners do not flush stdout |
| a count of zero, or above the row count | a guessed field name; the observer must print one row's keys first |
| every cost cell *not established* | no prices supplied, or the runner records no per-call usage (both EmoBench flash-lite runners keep only `thinking_tokens`) |
| the ETA is far out | it assumes the current rate holds, which a retrying run will not |
