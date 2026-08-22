# Benchmark knowledge bases

The rest of `.claude/` is deliberately benchmark-agnostic: it describes Quest, providers, the runner
shape and the handoff protocol in terms that hold for any of the ten benchmarks. **Everything true of
one benchmark and not the others lives here** — one page per benchmark, grouped by the process folder
it belongs to, with a group page for what its neighbours genuinely share.

A number carried from one page to another is the failure this split exists to prevent. 14,138 rows is
NegotiationToM's; 200 scored items per task is EmoBench's; neither is a property of the suite. Read
the count off the page, and check what the runner filters — EmoBench's files hold twice what it
scores.

| Group | Benchmarks |
|---|---|
| [transition/](transition/README.md) — mission analysis, strategy, goal specification | [AwareBench](transition/awarebench.md) · [PlanBench](transition/planbench.md) · [mpgt-eval](transition/mpgt.md) |
| [action/](action/README.md) — monitoring progress, coordination | [Wonderbread](action/wonderbread.md) · [MultiChallenge](action/multichallenge.md) |
| [interpersonal/](interpersonal/README.md) — conflict, affect | [NegotiationToM](interpersonal/negotiationtom.md) · [EmoBench](interpersonal/emobench.md) |
| [tasks/](tasks/README.md) — general task ability | [DocVQA](tasks/docvqa.md) · [BIG-Bench Hard](tasks/bbh.md) · [MMLU](tasks/mmlu.md) |

**Read the group page too, not only the benchmark's.** The group page carries the layout convention,
the scoring family and the state its members share — which is where "how do I start on this one"
usually gets answered.

## What a page is, and what it is not

A page is an **index into that benchmark's own knowledge**, plus the operating facts that exist
nowhere else: paths (local *and* Quest — they differ), layout, verified counts, output naming, run
order, and its own traps. Where a benchmark already has a committed document of its own —
`AWARENESS_NOTES.md`, `OPENAI_EVAL_NOTES.md`, `negotiation.md`, `DATA_NOTES.md`, `ISSUES.md` — the
page points at it and does not restate it. Two copies drift.

Counts marked *verified* were measured from the local tree on 2026-08-22. Anything not established
is written as not established, never inferred from what a benchmark "probably" does.

**The Quest rows are the weakest field on every page.** Three remote paths are known from
`quest-cluster.md` — NegotiationToM, EmoBench (remote name `EmoBench-master`) and DocVQA. The other
seven say *not verified*, because Quest itself was not checked: the current working scope is the
local tree, and the sync happens once local is settled. Do not upgrade a "not verified" to a "not
present" without looking.

## Known gap: the problem log is still single-benchmark

`tracker`, and the `run-model` / `fix-broken-run` / `verify-change` workflows, all write to
`Interpersonal_processes_benchmarks/NegotiationToM/ISSUES.md` by name. A problem hit while working on
any other benchmark has nowhere of its own to go. Until that changes, say in the entry which
benchmark it concerns, and never read a NegotiationToM entry as a statement about the suite.

## Adding to a page

Fill only what has been verified, and delete anything you move out of a generic reference rather than
leaving a copy behind. The seven fields worth having, in the order they get used:

1. **Paths** — local, and on Quest. The 2026-08-19 reorganisation moved only the local side.
2. **Layout** — the per-model folders or the flat scripts, the shared code they import, the data.
3. **Expected counts** — per task, and what a wrong count means. This is the field that catches
   silent bugs.
4. **Scoring** — the metric, what is excluded from the denominator, and whether a judge is involved.
5. **Output and logs** — including the shard-tag pattern and whether log names collide. Say which
   result directories are ours: **a directory of ours carries the model's name**, so an unnamed
   `results/` shipped with a vendored copy is upstream's and is not ours to report.
6. **Run order** — the exact scripts, in sequence.
7. **Its own traps** — the ones that produce wrong numbers that look right.
