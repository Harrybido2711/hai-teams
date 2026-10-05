---
name: kill-and-resync
description: Stop a Quest job that is producing unusable data, put the fix on Quest, decide what happens to its rows, and resubmit. Standing authorisation — no need to ask first. Use when check-run shows bad data (empty rows, stale code, a config error on every call), not when a job is merely slow.
---

# kill-and-resync

Phase 4 · the standing authorisation in `CLAUDE.md` · run by `executor`, gated by `reviewer`,
confirmed by `watcher`, logged by `tracker` · the whole of the `fix-run` workflow.

## Steps

1. **Evidence first.** `check-run` shows the data is bad — cite the number. Slow is not bad.
2. **`scancel <id>`** — that job only. `squeue` afterwards: every other job is still RUNNING.
3. **The fix is in the local tree** — decided by the planner, compiled, and smoke-tested with
   `--limit` if it touches a call or the parsing.
4. **`reviewer` on the diff** — `safe-to-run`, or stop.
5. **`quest-sync`** the whole change set.
6. **Decide the rows already written** (`submit-run` step 2): archive after a config change, prune
   empties, or resume. Say which, with counts before and after.
7. **Resubmit** (`submit-run` steps 4–5) and gate on the new job's first rows, compared with the
   numbers of the job you stopped.
8. **`tracker`** records it: why stopped, before/after, the disposition, the new job id.

## Done when

The new job's first rows are healthy where the old job's were not — both sets of numbers in the
report.

## Never

- Transfer under the live job and hope it picks the change up.
- Use this on a job that is only slow.
- Let a known-bad job run to its walltime because hours were already spent on it.

## Detail

[quest-cluster.md](../../references/quest-cluster.md) § Replacing the code under a broken run ·
`CLAUDE.md` § Standing authorisation
