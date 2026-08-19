# Handoffs

How work is passed to an agent, and what comes back. This exists because a subagent starts with no
memory of the last one and cannot dispatch another: everything it needs has to arrive in its prompt
or be readable at a path it is given.

## What a dispatch must carry

The planner passes **what to accomplish** and **where**. The agent decides **how**.

1. **The target and its paths.** `NEG_Gemma`, not "the Gemma run". `Interpersonal_processes_benchmarks/NegotiationToM/neg_eval_core.py`,
   not "the shared core". An agent that has to guess which file you meant will guess.
2. **The decision already made, when there is one.** `executor` is given a decision, not a problem;
   if the instruction is ambiguous or looks wrong it stops and says so rather than re-opening it.
3. **The hard rules that apply.** Which model is in scope and that nothing else may be touched; that
   no provider API may be called directly; who owns `sbatch`/`scancel` in this task. A reviewer once
   wrote four probe scripts and spent real quota because its prompt did not forbid it.
4. **The output shape** — the status token below, plus whatever fields the caller will branch on.
5. **What to do when blocked.** The default is: stop and report with the concrete blocker. Guessing
   is never the fallback.

## Status tokens

Every agent ends its report with a single line beginning `STATUS:`, so the planner can branch on it
without re-reading prose. The vocabularies match the enums the workflows in `.claude/workflows/`
already use — do not invent a third wording for the same state.

| Agent | Line | Values |
|---|---|---|
| `watcher` | `STATUS: <verdict>` | `healthy` · `too-early` · `degraded` · `failed` · `stalled` · `quota-blocked` · `stale-code` · `cannot-tell` |
| `evaluator` | `STATUS: <trust> / <recommendation>` | trust: `trustworthy` · `partial` · `untrustworthy` · `cannot-tell`. recommendation: `continue` · `kill` · `kill-and-archive` · `prune-and-resume` · `publish` · `needs-human` |
| `reviewer` | `STATUS: <verdict>` | `safe-to-run` · `needs-change` · `unsafe` |
| `executor` | `STATUS: <state>` + job ids | `done` · `partial` · `blocked` |
| `summarizer` | `STATUS: <confidence>` | `verified` · `partly-inferred` · `insufficient-evidence` |
| `tracker` | `STATUS: <action>` | `added` · `updated` · `already-recorded` · `no-entry-needed` |

`cannot-tell` and `too-early` are real answers. A verdict invented to fill the field is worse than
an admission that the window was too short, because the planner acts on it.

Two states carry a required companion field:

- **`kill`, `kill-and-archive`, `prune-and-resume`** — say whether the existing checkpoint is
  resumed, pruned or archived, and why. Rows written under an old prompt or decoding config must not
  be mixed into the new run.
- **`blocked`** — name the concrete blocker and what input would clear it, not "could not proceed".

## Writing the prompt

Rewrite these before sending. Each left-hand form leaves a decision to an agent that has less
context than you do.

| Ambiguous form | Rewrite as |
|---|---|
| "check the results" | the path, the expected row count, and what makes them wrong |
| "if needed" / "as appropriate" | the triggering condition and the required action |
| "fix it" | the change already decided on, or a dispatch to the agent that decides |
| "the usual checks" | the named checks, or the reference row that lists them |
| "make sure it works" | the observable condition that proves it — an exit code, a row count, an md5 match |
| "look at the recent runs" | the directories, and the date or job ids that bound "recent" |

## Reporting back

Evidence over assertion. Include the command output rather than claiming success — a partial result
described accurately beats a claim of completion. Rank findings by whether they can produce **wrong
numbers that look right**; those come first, then crashes, then style. Where you are uncertain, say
what observation would settle it instead of hedging.
