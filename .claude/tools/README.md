<!-- size-budget: 6000 -->
<!-- The index of three layers — workflows, skills, agents — one row each. It grows when a tool is added. -->
# Tools

What can be dispatched, and when. **Index only — open a detail file once a row matches.** Three
layers, each with one job:

- **Skills** are the steps — one reusable procedure each, benchmark-agnostic, read from the
  benchmark's page at run time. `.claude/skills/<name>/SKILL.md`. Any agent can follow one: Claude
  Code invokes it by name, a subagent or Codex reads the file.
- **Agents** are the roles — who may observe, judge, act, review, record. `.claude/agents/<name>.md`.
- **Workflows** are baselines — which agent runs each phase and which skill it follows, with the
  gates that may refuse. Five, one per stretch of the project's loop. **Adapt them to the task** —
  each detail page has an *Adapting* section saying what may change and what must not;
  [create-workflow.md](create-workflow.md) is how.

## Workflows

Invoke by path — `Workflow({scriptPath: ".claude/workflows/<name>.js", args: {...}})` — or read the
script, change it for the task, and pass it inline as `script`.

| Workflow | Phases (agent · skill) | Use when | Detail |
|---|---|---|---|
| `new-runner` | brief (summarizer) → write (executor · write-runner) → review (reviewer) → commit (sync-pass) | adding a model to a benchmark, or changing a runner. **Stops before Quest** | [→](new-runner.md) |
| `launch-run` | sync (executor · quest-sync) → gate (reviewer) → submit (executor · submit-run) → first rows (watcher · check-run) | the user has verified the scripts and a pilot or full run should start | [→](launch-run.md) |
| `monitor-run` | observe (watcher · check-run) → judge (evaluator · audit-results). Read-only | a job of any benchmark is running: is it in trouble, when will it finish, what does it cost | [→](monitor-run.md) |
| `fix-run` | observe → stop → sync → gate → resubmit → confirm → record (kill-and-resync) | a running job's data is unusable and the fix is decided. Standing authorisation | [→](fix-run.md) |
| `finish-run` | pull (executor · pull-results) → audit (evaluator · audit-results) → *user confirms* → record (record-results, sync-pass) | a pilot or full run has finished | [→](finish-run.md) |

`monitor-run` only reports; `fix-run` kills; `launch-run` starts; `finish-run` records, and only
after the user confirms.

## Skills

| Skill | Does | Use when |
|---|---|---|
| `analyse-benchmark` | paths, counts, scoring, traps from the code → the benchmark's page | a benchmark has no page, or a field a runner needs is unestablished |
| `write-runner` | one model's runner, up to a local `--limit` smoke test and a review | adding or changing a runner |
| `quest-sync` | code up to Quest, md5 on both sides, joined by filename | before any submit; whenever "does Quest match local?" |
| `submit-run` | checkpoint disposition, `sbatch`, first-rows gate | the user has verified the scripts |
| `check-run` | halt markers → rows → quality → errors → hang → stale code → ETA. Read-only | "how is it going"; before deciding to kill |
| `kill-and-resync` | scancel → fix → review → sync → disposition → resubmit → record | the data is bad, not merely slow |
| `pull-results` | results and logs down, verified, committed | during a long run, and when it ends |
| `audit-results` | schema → counts → usability → one configuration → score → verdict | after a pilot or a run, before any number is used |
| `rescore-offline` | stored rows rescored with the one lenient matcher, flips read | a scorer changed, or two matchers in one benchmark |
| `record-results` | page + both workbooks + Provenance in one edit, cell-diffed | the user has confirmed a finished run |
| `sync-pass` | doc check → explicit-path commit → push both → Quest md5 if needed | the end of every finished change |

## Agents

Dispatched with the `Agent` tool, or by a workflow phase's `agentType` — only by the planner; no
subagent holds the `Agent` tool, and each starts with no memory of the last. What a dispatch must
carry and the `STATUS:` vocabulary each returns: [handoffs.md](../references/handoffs.md).

| Agent | Does | Use when | Detail |
|---|---|---|---|
| `watcher` | live job state: queue, rows, stalls. Observes only | what is happening on Quest right now | [→](../agents/watcher.md) |
| `evaluator` | are the numbers believable, what they cost. Advises only | output exists and must be judged | [→](../agents/evaluator.md) |
| `executor` | edit, transfer, sbatch, scancel, verify | the change is decided. Give it the decision, not the problem | [→](../agents/executor.md) |
| `reviewer` | read the diff, hunt the silent failures. Reports only | after executor, before anything reaches Quest | [→](../agents/reviewer.md) |
| `tracker` | write the problem and its fix into `ISSUES.md` | a problem is resolved, or "have we hit this before" | [→](../agents/tracker.md) |
| `summarizer` | read many files, return the conclusion only | much reading, none of which belongs in context | [→](../agents/summarizer.md) |

Adding a workflow or a skill means adding its row here in the same edit; the doc check fails
otherwise. A tool nothing routes to is never used.
