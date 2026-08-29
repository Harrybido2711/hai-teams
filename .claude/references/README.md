<!-- size-budget: 6000 -->
<!-- An index of every reference: it grows when the directory does, and splitting an index
     defeats its purpose. -->
# References

Knowledge loaded **on demand**, so cost scales with the task. An agent definition holds its role and
its judgement; everything a task might or might not need lives here. `executor.md` was 13 KB and
entered the context in full even for a one-line `scancel`; splitting it means that task pays for a
routing table and nothing else, while writing a new runner still gets every line of the provider
notes.

This file is the map. It holds no knowledge of its own.

## Three tiers

| Tier | What | When |
|---|---|---|
| **auto-loaded** | `CLAUDE.md` (rules) · personal memory | arrives with every session; nothing to decide |
| **every task** | [`../INDEX.md`](../INDEX.md) (orientation) · [`shared-context.md`](shared-context.md) (who is authoritative on what) | unconditional, ~8 KB together |
| **on demand** | the routing table below | only when a trigger matches |

## Routing table

**Not advisory.** Read every row whose trigger appears in the task, *before* acting. Each line in
these files was paid for by a run that failed. If two rows match, read both — reading the wrong one
and proceeding anyway is the failure this table exists to prevent.

| Triggered by | Read | Cost |
|---|---|---|
| anything at all, before deciding what is true about this repo | [shared-context.md](shared-context.md) | 4 KB — always |
| a named benchmark — its paths, expected counts, tasks, output layout, run order, traps | [benchmarks/](benchmarks/README.md), then that benchmark's group and page | 3 KB + page |
| `ssh`, transfer, `md5sum`, `sbatch`, `scancel`, `squeue`, shard, array, "pull results", a Quest path | [quest-cluster.md](quest-cluster.md) | 7 KB |
| a provider name (GPT, Gemini, Gemma, Qwen, Deepseek, grok), client, timeout, empty response, halt marker | [provider-gotchas.md](provider-gotchas.md) | 3 KB |
| **writing a runner for a model** — which client, `base_url`, key, model id, non-optional parameters | [model-calls.md](model-calls.md) — the invocation recipe, measured where we have run it | 5 KB |
| **writing or changing any runner** — the decoding, thinking and output limits it must set, and the settled per-model configs | [model-parameters.md](model-parameters.md) | 6 KB |
| a reasoning bill that looks too high, or how a cap backfires | [reasoning-cost.md](reasoning-cost.md) | 3 KB |
| the model offers **no thinking or output parameter**, so the cap goes in the prompt | [prompt-ceiling.md](prompt-ceiling.md) | 1 KB |
| writing or changing an eval script — retries, checkpoints, resume, scoring, a shared core | [script-skeleton.md](script-skeleton.md) | 5 KB |
| whether a change needs a Quest sync, or what the three sync layers oblige | [sync-and-consistency.md](sync-and-consistency.md) | 4 KB |
| a commit blocked by the doc check, two copies of a fact, "what does this change touch" | [doc-check.md](doc-check.md) | 3 KB |
| dispatching a subagent or reading its report — what to pass, `STATUS:` values, vague instructions | [handoffs.md](handoffs.md) | 4 KB |
| what can be dispatched at all — the workflow and agent list | [../tools/README.md](../tools/README.md) | 3 KB |
| borrowing a pattern from outside, or "was this screened and rejected" | [external-patterns.md](external-patterns.md) | **28 KB** — the most expensive here; for a sweep only |

Sizes are a cost hint for deciding what to open, not a checksum; they drift as the files grow.

**Retrieving everything about one model costs four greps, by design** — limits, invocation, client
failures and cluster ceilings live in four files, and that split is right. Do not pay it by hand:
`python3 .claude/scripts/check_docs.py --model <id>` prints all four at once and names any gap.
`--impact <term>` is the same idea for a term rather than a model, and is the work list before an
edit rather than after one.

## Authority when two documents disagree

1. **[shared-context.md](shared-context.md)** — first, because it decides *which* document is
   authoritative on a subject. Where it names one (`negotiation.md` for NegotiationToM findings,
   `DATA_NOTES.md` for dataset traps, `JUDGE_RECORD.md` for judges), that one wins on that subject.
2. **[PLAN.md](../../PLAN.md)** — repo layout, which benchmark needs a judge, what state each run is in.
3. A **benchmark page** wins on its own benchmark — counts, paths and task names are never inherited
   from a generic file or from another benchmark.
4. Everything else, on its own subject only.

**Never pick silently.** If a reference contradicts what you were told, or another document, say so
and name both. A contradiction resolved quietly is how a reproduction ends up measuring something
nobody described.

## Keeping these honest

When a run exposes something a reference should have said, edit the reference. A fact that lives
only in a transcript is lost when the session ends; a fact here reaches every future agent whose
task matches the row.

- **A reference nothing routes to is never read.** Adding a file means adding its row in the same
  edit, phrased as a condition an agent can recognise in its own task — not as a topic name.
- **A fact lives in exactly one place.** When something moves here, delete it from where it was; do
  not leave a summary behind. Two copies drift, and then the agent has to decide which is right.
- **Past its budget, consider splitting — but the number does not decide.** ~5 KB is the default
  nudge; a file that is legitimately longer sets its own with `<!-- size-budget: N -->`. The signal
  worth acting on is a file doing two jobs, not one crossing a threshold. A reference that keeps
  growing does become the thing this directory exists to prevent — a file everyone loads and nobody
  needs all of — but shaving prose to fit a constant produces exactly the same file, minus the
  reasoning.
