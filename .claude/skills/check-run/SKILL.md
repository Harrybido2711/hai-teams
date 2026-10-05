---
name: check-run
description: Read-only check of a live or just-finished Quest job for any benchmark — is it producing usable rows, is it stalled or quota-blocked, is it on stale code, when will it finish. Use for "how is it going", before deciding to kill a job, and as the first-rows gate after a submit.
---

# check-run

Phase 4 · run by `watcher` · used by the `monitor-run`, `launch-run` and `fix-run` workflows. Changes nothing.

Cheapest signal first. Paths, task names and expected counts come from the benchmark's page.

## Steps

1. **Halt markers:** `ssh quest "cat $Q/<model>/*_HALT.txt 2>/dev/null"`. Quote whatever is there,
   including its line about pruning.
2. **Queue:** `squeue -u uwr0681 -o "%.12i %.16j %.9P %.9T %.10M"`; a job that has left it →
   `sacct -X -j <id> -o JobID,JobName%18,State,ExitCode,Elapsed`. `COMPLETED 0:0` proves nothing.
3. **Rows:** lines per task in the `.jsonl` (never the CSV) and each file's mtime. Two samples
   minutes apart give rows/min. Under 20 rows is `too-early` — the first checkpoint is at 20.
4. **Quality:** print one real row's keys first, then count empty responses, null predictions, and
   finish reasons containing `MAX_TOKENS`.
5. **Errors and quota:** `grep -cE "API error|Retrying|Traceback|429|insufficient_quota|requests per day|billing"`
   on the logs; quote the last two. Logs are unflushed — "no errors" may mean "none visible yet".
6. **Hung?** Rows flat across two samples → `srun --jobid=<id> --overlap` and read
   `/proc/<pid>/wchan` and CPU time. Flat CPU means a call is hung; `hrtimer_nanosleep` is the
   runner's own sleep.
7. **Stale code?** The md5 comparison from `quest-sync` step 2, read-only. A mismatch is a finding
   on its own.
8. **ETA** = remaining rows ÷ rows/min over a window of at least 15 minutes; state the window.

## Done when

The report gives rows/expected per task, rows/min, empty count, halt markers, and ends
`STATUS: healthy | too-early | degraded | failed | stalled | quota-blocked | stale-code | cannot-tell`.

## Never

- Submit, cancel, edit, transfer, or call a provider.
- Infer a stall from log size or from a window too short to contain a checkpoint.

## Detail

[quest-cluster.md](../../references/quest-cluster.md) § Reading the live state ·
[shared-context.md](../../references/shared-context.md) § Counting rows
