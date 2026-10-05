---
name: pull-results
description: Bring one model's results and logs down from Quest into the local tree and into git, for any benchmark. Use periodically during a long run, so results never live only on the cluster, and once a run finishes.
---

# pull-results

Phase 4 · run by `executor` · used by `finish-run`. Results flow down; code never does.

## Steps

1. **Paths from the benchmark's page.** Pull results and logs only:

   ```bash
   ssh quest "cd $Q && tar cf - <model>/results* <model>/log*.txt" | tar xf - -C "$L"
   ```

   NegotiationToM has a script for every `NEG_*` folder at once:
   `bash .claude/scripts/pull_quest_results.sh`.
2. **Verify locally:** rows per task match what `check-run` saw on Quest. A short count means the
   transfer or the run is incomplete — say which.
3. **Commit explicit paths only** — `git add "$L/<model>/results…" "$L/<model>/log….txt"` — with the
   row count in the message, then push to `origin` and `backup`.

## Done when

Local row counts equal Quest's, and the commit is on both remotes — output pasted.

## Never

- Pull code, configs or `.env` down.
- `git add -A`, or stage anything outside this model's results and logs.

## Detail

[quest-cluster.md](../../references/quest-cluster.md) § Pulling results down
