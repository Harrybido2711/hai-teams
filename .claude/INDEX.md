# hai-teams — start here

**Goal, in one line:** evaluate LLMs against the *team-process taxonomy* (transition / action /
interpersonal processes, plus general task ability), running on Northwestern's **Quest** SLURM
cluster against six commercial providers, with every reported number landing in `Results.xlsx`.

Read this file, then only what the task needs. Nothing else is loaded up front — that is the design,
not an omission.

## The three that are always first

| Read | For |
|---|---|
| [tools/README.md](tools/README.md) | what can be dispatched — workflows and agents. Check here **before** hand-rolling a procedure; the common ones already exist |
| [references/README.md](references/README.md) | the document map: which reference a task triggers, what each costs, which document wins in a conflict |
| [../PLAN.md](../PLAN.md) | the repo map — what each folder is, which benchmark needs a judge, what state each run is in |

`references/shared-context.md` is the one reference read on *every* task: it says which committed
document is authoritative on what. Everything else is on demand.

**Working on a specific benchmark? Read its page first, and its group page with it** —
[`references/benchmarks/`](references/benchmarks/README.md). The rest of `.claude/` is deliberately
benchmark-agnostic; every number, path and task name belonging to one benchmark lives on its own
page, and carrying one across benchmarks is the mistake that split is there to prevent.

## Where the project is

- **Working scope right now (2026-08-22): the local tree.** Quest is not being checked or synced;
  the transfer happens once local is settled. That does not relax any rule about *how* a sync is
  done when it happens — it means the sync has not happened yet.
- **Running now:** nothing is assumed. Check, don't remember — `check-status`, or `squeue -u uwr0681`.
- **NegotiationToM** — the active benchmark, six providers with results
  ([its page](references/benchmarks/interpersonal/negotiationtom.md)).
- **EmoBench** six providers · **bbh**, **mmlu** seven · **DocVQA** two · the other four have no
  results and, in three cases, no runner.
- **All ten have a knowledge-base page**, grouped by process folder under
  [`references/benchmarks/`](references/benchmarks/README.md).
- **Judge record written 2026-08-19** for the three judged benchmarks (Wonderbread, MultiChallenge,
  AwareBench); AwareBench still has three open blockers. `LLM_as_judge/JUDGE_RECORD.md`.
- **No runner exists** for Multi-party Goal Tracking, Wonderbread or MultiChallenge — vendored only.
- Open work in full: `PLAN.md` § Open work.

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
