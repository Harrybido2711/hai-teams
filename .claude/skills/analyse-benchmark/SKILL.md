---
name: analyse-benchmark
description: Establish a benchmark's paths, counts, scoring, reasoning visibility and traps from its code, and write its knowledge-base page. Use when a benchmark has no page yet, or a runner is about to depend on a field its page marks unestablished.
---

# analyse-benchmark

Phase 1 · run by the planner, with `summarizer` for the wide reading · used by `new-runner` when
the page is missing. Changes nothing but the page and its group-index row.

## Steps

1. **Find the entry point** — the upstream run script, CLI or config that actually produces scores.
   Not the file whose name looks right.
2. **Establish each field from the code**; the paper only explains what the code does.
   - Paths — local, and Quest. Quest is "not verified" until checked; never inferred.
   - Layout — data files, per-model folders or flat scripts, any shared core module.
   - Expected counts per task — count the data (`wc -l`, `len(json.load(…))`), then what the scorer
     filters out of the denominator.
   - Scoring — the metric, the exclusions, judge or not (a judge also needs
     `LLM_as_judge/JUDGE_DOCUMENTATION_RULE.md`).
   - Output naming and the shard tag; which `results/` folders are upstream's (no model in the name).
   - What the README says about showing reasoning — quote the line.
   - Traps that would produce wrong numbers that look right.
3. **Write the page** `.claude/references/benchmarks/<group>/<name>.md` from the template, **and its
   row in the group README, in the same edit.**

## Done when

- Every field is either verified — with the command that verified it — or written as unestablished.
- `python3 .claude/scripts/check_docs.py` passes; its `benchmarks` check fails on an unindexed page.

## Never

- Fill a field from what a benchmark "probably" does.
- Carry a count, path or task name over from another benchmark's page.

## Detail

[benchmarks/README.md](../../references/benchmarks/README.md) § Adding to a page.
