# watch-live-runs

Several runs of one benchmark are executing side by side, and the question is two questions at once:
**is either in trouble**, and **which is cheaper and faster**. Read-only, so it can be run on a timer
and beside a supervising workflow.

`check-status` answers the first question for one model, and only on NegotiationToM.
`compare-providers` answers the second by *launching* a fresh pilot. This one measures runs that are
already going, which is the only way to compare the actual job you are paying for.

## Input

```
Workflow({scriptPath: ".claude/workflows/watch-live-runs.js", args: {
  benchmark: "EmoBench",
  questDir:  "/gpfs/projects/p32983/Interpersonal_processes_benchmarks/EmoBench",
  runs: [
    {label: "Google",     dir: "EMO_Gemini_Flash3.5lite_Google",     jobId: 3810331},
    {label: "OpenRouter", dir: "EMO_Gemini_Flash3.5lite_OpenRouter", jobId: 3810332,
     priceIn: 0.30, priceOut: 2.50},
  ],
  expected: {EU: 200, EA: 200},
  sinceMinutes: 0,
}})
```

`runs` needs at least two entries — for one, use `check-status`. `priceIn`/`priceOut` are $/M tokens
and are optional: **omit them rather than guessing**, and the cost cell reports *not established*
instead of a number that looks measured.

## Output

```
{ benchmark, tasks, expectedPerRun,
  perRun: {Google: "healthy", OpenRouter: "healthy"},
  status: "trustworthy" | "partial" | "untrustworthy" | "cannot-tell",
  recommendation: "continue" | "kill" | "kill-and-archive" | "prune-and-resume" | "publish" | "needs-human",
  report }
```

`report` holds the four tables: per-run verdict, finish-time, cost, and what needs a decision.

## Preflight

- **Nothing it does may write.** No `sbatch`, `scancel`, edit, transfer, commit, or provider API
  call. The prompts say so explicitly, because these jobs are live and a probe spends real quota
  against the run being measured. `srun --overlap` is permitted and used only to read `/proc`.
- **Rows come from the `.jsonl`, never the CSV.** Model output contains embedded newlines, and
  counting CSV lines has already produced a false alarm here.
- **Fewer than 20 rows is `too-early`, not a stall.** These runners checkpoint every 20 items, so
  before the first checkpoint the results file legitimately does not exist.

## When it fails

| Symptom | Cause |
|---|---|
| every cost cell says *not established* | `priceIn`/`priceOut` were omitted, or the runner recorded no token counts — the second is the usual one, see below |
| "args.runs needs at least two entries" | one run was passed; `check-status` is the tool for that |
| a healthy run reported as stalled | judged from `log.txt` size. **These runners do not flush stdout**, so a 0-byte log is normal all the way through a working run; the workflow's prompts forbid this inference, but a hand check can still make it |
| the finish-time projection is far out | it assumes the current rate holds, which a run that has started retrying will not do |

**The standing gap this workflow cannot close:** neither EmoBench flash-lite runner records
per-call `usage` — not prompt tokens, not completion tokens, and not OpenRouter's per-call
`usage.cost`, which its API returns and the runner discards. Only `thinking_tokens` is kept. So cost
is *derived* from supplied prices and assumed token counts, and every derived number is labelled as
such. Fixing this means adding usage capture to the runners — which must not be done mid-run, since
one result set would then hold two record shapes.
