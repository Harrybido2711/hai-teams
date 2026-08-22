# hai-teams — start here

**Goal, in one line:** evaluate LLMs against the *team-process taxonomy* (transition / action /
interpersonal processes, plus general task ability), running on Northwestern's **Quest** SLURM
cluster against six commercial providers, with every reported number landing in `Results.xlsx`.

Read this file, then only what the task needs. Nothing else is loaded up front — that is the design,
not an omission.

## What the work is — five phases

Every task in this repo sits in one of these. Find yours, then read what its row points at.

| # | Phase | What it means | Read first |
|---|---|---|---|
| 1 | **Analyse a new benchmark** | read the vendored repo, establish paths, counts, scoring, traps; write its page | [`references/benchmarks/`](references/benchmarks/README.md) — the page template is in its README |
| 2 | **Write the per-model scripts** | one runner per model from prior experience, plus the template answer the scoring depends on | [`references/script-skeleton.md`](references/script-skeleton.md) · [`references/model-parameters.md`](references/model-parameters.md) · [`references/provider-gotchas.md`](references/provider-gotchas.md) |
| 3 | **Upload and run on Quest** | **only after the user has finished and verified every script.** Then sync, submit, gate on the first minutes | [`references/quest-cluster.md`](references/quest-cluster.md) · [`tools/run-model.md`](tools/run-model.md) |
| 4 | **Monitor the run** | dispatch the agents, and build a workflow when the procedure repeats | [`tools/README.md`](tools/README.md) — the roster of both |
| 5 | **Keep everything in sync** | not a phase so much as the thing that runs through all four | [`references/sync-and-consistency.md`](references/sync-and-consistency.md) |

Phase 3 has a gate in front of it that is not technical: **the scripts are the user's to verify, and
nothing is transferred or submitted before they say so.** Phases 1, 2 and 5 are local work and need
no such permission.

Two files are read on every task regardless of phase:
[`references/README.md`](references/README.md), the map of what to read when, and
`references/shared-context.md`, which says which document is authoritative on what.

**Working on a specific benchmark? Read its page and its group page** —
[`references/benchmarks/`](references/benchmarks/README.md). The rest of `.claude/` is deliberately
benchmark-agnostic; every number, path and task name belonging to one benchmark lives on its own
page, and carrying one across benchmarks is the mistake that split is there to prevent.

## Where the project is

- **Working scope right now (2026-08-22): the local tree.** Quest is not being checked or synced;
  the transfer happens once local is settled. That does not relax any rule about *how* a sync is
  done when it happens — it means the sync has not happened yet.
- **Running now:** nothing is assumed. Check, don't remember — `check-status`, or `squeue -u uwr0681`.
- **All ten benchmarks have a knowledge-base page**; three of them have no runner at all, so work
  there starts at phase 1 or 2 rather than 3.
- **No runner currently complies with the model-parameter rule** — none sets a thinking or output
  cap. See [`references/model-parameters.md`](references/model-parameters.md).
- Per-benchmark state, provider coverage and open work: `PLAN.md`.

## Terms this project uses in a specific way

| Term | Means |
|---|---|
| **kill-and-resync** | standing authorisation to `scancel` a known-bad job, fix locally, overwrite on Quest with `md5sum` confirmation, resubmit — without asking first |
| **sync check** | proving every code file on Quest matches local before a submit. A `PreToolUse` hook runs it automatically and **fails open**, so a stale path silently protects nothing. Contract and the two ways the check lies: `references/quest-cluster.md` |
| **gate** | a workflow phase that is allowed to refuse — `fix-broken-run` returns without submitting when the reviewer says no. Why they are built that way: `tools/create-workflow.md` |
| **`STATUS:` line** | the fixed vocabulary every agent ends its report with, so a dispatch can be branched on without re-reading prose. `references/handoffs.md` |
| **pilot** | a small fraction of the data run first and reviewed before the full run commits hours to a config. The script name is on the benchmark's page |
| **shard tag** | `{model}_shard{N}of{M}.jsonl` in an output filename. Without it every shard overwrites the last |
| **halt marker** | `BILLING_HALT` / `QUOTA_HALT` / `FAILURE_HALT` in a model folder — the cheapest signal there is, cleared at the start of each run so one that exists is about the current run |
| **checkpoint** | resume skips any UID already present. After a prompt or decoding change, **archive** it rather than resuming, or one result set holds two configurations |

## Last major change

**2026-08-19** — benchmarks reorganised into the four team-process folders (`269bbfe`). **Quest's
layout did not move**: `/gpfs/projects/p32983/NegotiationToM` is still correct and must not be
"fixed" to match local. Any path remembered from before that date is stale — and this move is what
blinded the pre-submit gate, which found zero files and exited 2 while still looking wired up.

**2026-08-22** — agent-facing docs split three ways: this file (orientation), `references/`
(knowledge), `tools/` (what to dispatch). `CLAUDE.md` holds rules only.
