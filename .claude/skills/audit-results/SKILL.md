---
name: audit-results
description: Decide whether a pilot or finished result set can be believed before any of its numbers are used. Use after a pilot (before the full run), after a full run (before record-results), and whenever a score looks surprising.
---

# audit-results

Run by `evaluator` · used by `monitor-run` and `finish-run`. Reads only.

## Steps

1. **Schema first.** Print one real row and name the response, prediction, gold and uid fields.
   Every count below uses those names, not guessed ones.
2. **Counts against the benchmark's page:** rows per task, unique uids from the `.jsonl`, duplicates
   across shards, and the merged file against the sum of its shards — never shards and merged file
   counted together.
3. **Usability:** empty responses, null predictions, `MAX_TOKENS` / no-answer-marker rows, and
   predictions outside the label set — a model problem or a normaliser gap? Say which.
4. **One configuration:** file mtimes after the code's; one seed, effort and reasoning setting across
   all rows; nothing resumed from an older checkpoint.
5. **The score:** it comes from the shared core's lenient matcher; the denominator is the
   `{task}_scored_rows` the page's exclusions predict. Recompute the headline from the per-task files
   and compare with the `_overall.csv`.
6. **Beyond accuracy.** Report criteria 1, 2, 3 and 6 of `evaluation-criteria.md` from the
   per-row fields. Rows without the fields are reported as lacking them, never estimated.
7. **Verdict,** with the number that would have to change for the verdict to change.

## Done when

The report ends `STATUS: <trustworthy|partial|untrustworthy|cannot-tell> / <continue|kill|kill-and-archive|prune-and-resume|publish|needs-human>`,
and a recommendation that stops a run says what happens to its checkpoint.

## Never

- Treat a finished job, or a plausible total, as evidence.
- Compare models before steps 2–4 hold.

## Detail

[shared-context.md](../../references/shared-context.md) § Counting rows ·
[script-skeleton.md](../../references/script-skeleton.md) §7 ·
[evaluation-criteria.md](../../references/evaluation-criteria.md) · the benchmark's page
