---
name: submit-run
description: Submit a pilot or full run of one model on Quest, decide what happens to rows already on disk, and confirm the first rows are usable. Use after quest-sync has proven the code matches and the user has verified the scripts.
---

# submit-run

Phase 3 · run by `executor` · first rows judged by `watcher` · used by the `launch-run` and `fix-run` workflows.

## Steps

0. **Gate: the user has verified these scripts** (`CLAUDE.md` phase 3). Without that, stop here.
1. **Read the page's run order** — preflight, pilot and full scripts, task names. Shards: 5 unless
   the page says otherwise (`quest-cluster.md` § SLURM). Per-task job arrays add parallelism without
   changing the shard count, so rows already written stay valid.
2. **Decide the checkpoint before submitting**, and say which:
   - prompt or decoding config changed since those rows → archive to
     `results_archive_<what changed>_<UTC>`;
   - rows are good but some are empty → prune the empties (resume would skip them forever);
   - nothing written, config unchanged → resume.
   Never delete. Show row counts before and after.
3. **Preflight:** `.env` exists in the Quest directory (`test -f`); the page's preflight script, if
   it has one, passes with the real prompt builders.
4. **Submit from the model's own directory:** `ssh quest "cd $Q/<model> && sbatch <script>"`.
   Record the job id. Then `squeue` — exactly the new job plus whatever was already running.
5. **Gate on the first rows** with `check-run`, two samples several minutes apart (sleep on the
   Quest side: `ssh quest 'sleep 300; …'`). Rows growing, empties ≈ 0, no halt marker → go.
   Anything else → `kill-and-resync` now, not at the walltime.
6. **A pilot is audited before the full run is submitted** (`audit-results`).

## Done when

A job id, and first rows that are non-empty and growing — output pasted.

## Never

- Submit before the user has verified the scripts.
- Resume into rows written under a different config.
- Raise shards above 5 on a headline rate limit; requests-per-day is the limit that bit.

## Detail

[quest-cluster.md](../../references/quest-cluster.md) § SLURM, § Replacing the code ·
[script-skeleton.md](../../references/script-skeleton.md) §6 (checkpoints)
