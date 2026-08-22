# Tools

What can be dispatched, and when. **Index only — open a detail file once a row matches.** Each has
the same four sections: Input, Output, Preflight, When it fails.

## Workflows

Multi-agent procedures, committed so they outlive the session. Invoke by path:
`Workflow({scriptPath: ".claude/workflows/<name>.js", args: {...}})`.

| Tool | Does | Use when | Detail |
|---|---|---|---|
| `run-model` | check local, sync, launch, gate, supervise, audit | one model, end to end. **The default** | [→](run-model.md) |
| `run-fast` | the same, as three per-task jobs, repairing as it runs | it must finish today and survive unattended | [→](run-fast.md) |
| `fix-broken-run` | kill, fix, resync, gate, resubmit, record | a job is *already running* and its output is unusable | [→](fix-broken-run.md) |
| `check-status` | read-only snapshot, ETA, what needs a decision | "how is it going". Repeatable, safe beside a supervisor | [→](check-status.md) |
| `verify-change` | attack a change for the paths where it does nothing | a guard/retry/scoring fix is written, not yet proven wrong | [→](verify-change.md) |
| `scale-shards` | climb a shard ladder, keep the highest healthy rung | throughput-bound on the provider, parallelism unknown | [→](scale-shards.md) |
| `compare-providers` | one pilot on two providers, tabled from their logs | which provider is actually faster | [→](compare-providers.md) |
| `watch-live-runs` | health of several live runs at once, plus cost and finish-time | two+ jobs of one benchmark are running and one may be in trouble | [→](watch-live-runs.md) |
| `harvest-patterns` | sweep GitHub, refute, record adopted and rejected | looking outside this repo. Proposes only | [→](harvest-patterns.md) |
| `create-workflow` | the constraints a new workflow must satisfy | writing or editing a workflow | [→](create-workflow.md) |

`check-status` and `watch-live-runs` report, `fix-broken-run` kills, `run-model` starts.

## Agents

Dispatched with the `Agent` tool, and only by the planner — no subagent has it, and each starts with
no memory of the last. What a dispatch must carry and the `STATUS:` vocabulary each returns:
[handoffs.md](../references/handoffs.md).

| Tool | Does | Use when | Detail |
|---|---|---|---|
| `watcher` | live job state: queue, rows, stalls. Observes only | what is happening on Quest right now | [→](../agents/watcher.md) |
| `evaluator` | are the numbers believable, what they cost. Advises only | output exists and must be judged | [→](../agents/evaluator.md) |
| `executor` | edit, transfer, sbatch, scancel, verify | the change is decided. Give it the decision, not the problem | [→](../agents/executor.md) |
| `reviewer` | read the diff, hunt the silent failures. Reports only | after executor, before anything reaches Quest | [→](../agents/reviewer.md) |
| `tracker` | write the problem and its fix into `ISSUES.md` | a problem is resolved, or "have we hit this before" | [→](../agents/tracker.md) |
| `summarizer` | read many files, return the conclusion only | much reading, none of which belongs in context | [→](../agents/summarizer.md) |

Adding a workflow means adding its row and its detail file in the same edit; a tool nothing routes
to is never used.
