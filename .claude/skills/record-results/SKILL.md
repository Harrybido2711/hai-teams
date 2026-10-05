---
name: record-results
description: Put a finished, user-confirmed model's numbers into its benchmark page and both workbooks in one edit, rebuilt from the result files on disk. Use only after the user has confirmed the run (sync layer 4).
---

# record-results

Layer 4 · run by `executor` or the planner · used by `finish-run`.

## Steps

0. **Gate: the user has confirmed this run.** Rows on disk are not a result until they say so.
1. **Which workbook takes this model:** `PLAN.md` § Two workbooks. The wide one takes every model;
   the reported one takes only the selected models, and a selected model that has not run stays
   blank — never the number of the model that held the slot before.
2. **Recompute from the per-task result files on disk**, not from a roll-up another run wrote.
3. **Edit by exact value.** Copy the workbook first; locate cells by exact header and row value,
   never by substring — `Provenance` column A holds compound values such as `MMLU · DocVQA`. After
   the edit, diff every cell against the copy: only the intended ones changed.
4. **`Provenance` row in the same edit:** source file, scorer tag, coverage, unusable-row counts.
5. **The benchmark page's results table, same edit.** Then `check_docs.py --impact <benchmark>`:
   coverage sentences in `PLAN.md` and `INDEX.md` that this run just made false.
6. **`sync-pass`.**

## Done when

Page, both workbooks and `Provenance` agree, the cell diff shows only intended changes, and the
commit is on both remotes.

## Never

- Record before the user confirms.
- Edit a workbook in place without a before-copy to diff against.
- Slice a document or sheet by assumed position — use explicit anchors.

## Detail

[sync-and-consistency.md](../../references/sync-and-consistency.md) § Layer 4 · `PLAN.md` § Two
workbooks
