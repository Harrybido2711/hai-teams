---
name: rescore-offline
description: Rescore stored model responses with the benchmark's one shared lenient matcher, without rerunning any model. Use when a scorer changes, when one benchmark's models were scored by different matchers, or when scores look like they measure formatting rather than answers.
---

# rescore-offline

Local work, no Quest, no provider calls · run by the planner or `executor`, checked by `reviewer`.

## Steps

1. **Find the matcher in the shared core** (`<bench>_eval_core.py` — `score_response`,
   `clean_surface` or equivalent). Runners with their own scorers: move it into the core first.
2. **Rescore into new output, never over the stored rows.** Print old and new per model and per
   task.
3. **Read the rows whose verdict flipped, in both directions.** Every flip should be a difference of
   form (`B` vs `(B)`, case, punctuation), never of content. Measure each matcher branch separately
   — a branch that never fires, or fires on content, is the finding.
4. **Write the rescored files with a scorer tag**; keep the originals.
5. **Changed numbers go to `record-results`** — after the user confirms them.

## Done when

A per-model before/after table, and a sample of flips from every branch read and explained.

## Never

- Rerun a model to fix a scoring problem.
- Normalise gold labels — only model output.
- Leave two matchers in one benchmark.

## Detail

[script-skeleton.md](../../references/script-skeleton.md) §7 · the benchmark's page (its scorer
and exclusions)
