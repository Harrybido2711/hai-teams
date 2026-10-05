# watch-live-runs

One run, or several runs of one benchmark side by side, are executing on Quest, and the question is
**is it in trouble and when will it finish** — plus, with several, **which is cheaper and faster**.
Any benchmark. Read-only, so it can be run on a timer.

It measures runs that are already going, which is the only way to judge the actual job you are
paying for. The same checks by hand are in `../references/quest-cluster.md` § Reading the live state.

## Input

```
Workflow({scriptPath: ".claude/workflows/watch-live-runs.js", args: {
  benchmark: "EmoBench",
  questDir:  "/gpfs/projects/p32983/Interpersonal_processes_benchmarks/EmoBench",
  runs: [
    {label: "Google",     dir: "EMO_Gemini_Flash3.5lite_Google",     jobId: 3810331},
    {label: "OpenRouter", dir: "EMO_Gemini_Flash3.5lite_OpenRouter", jobId: 3810332,
     priceIn: 0.30, priceOut: 2.50},
    // sweep arms of ONE model share a directory and differ only by --tag:
    {label: "effort=low", dir: "EMO_GPT_5.6_Luna", resultsDir: "results_eLow", jobId: 4511556},
  ],
  expected: {EU: 200, EA: 200},
  sinceMinutes: 0,
}})
```

`runs` takes one entry or several. `priceIn`/`priceOut` are $/M tokens
and are optional: **omit them rather than guessing**, and the cost cell reports *not established*
instead of a number that looks measured.

`resultsDir` defaults to `results`. **Sweep arms of one model need it**, because they live in the
same folder and are separated only by the runner's `--tag`. Two runs resolving to the same path are
refused up front: identical rows would otherwise be reported as two arms agreeing, which reads as a
finding rather than as the misconfiguration it is.

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
| "args.runs needs at least one entry" | `runs` was empty or not an array |
| a count that is zero or larger than the row count | the observer counted a guessed field name. The prompt makes it print one real row's keys first; a hand check has to do the same |
| a healthy run reported as stalled | judged from `log.txt` size. **These runners do not flush stdout**, so a 0-byte log is normal all the way through a working run; the workflow's prompts forbid this inference, but a hand check can still make it |
| the finish-time projection is far out | it assumes the current rate holds, which a run that has started retrying will not do |

**The gap this workflow cannot close:** a runner that does not record per-call `usage` — prompt
tokens, completion tokens, OpenRouter's per-call `usage.cost` — leaves cost to be *derived* from
supplied prices and assumed token counts, and every derived number is labelled as such. Both
EmoBench flash-lite runners are like this: they keep only `thinking_tokens`. Fixing this means adding usage capture to the runners — which must not be done mid-run, since
one result set would then hold two record shapes.
