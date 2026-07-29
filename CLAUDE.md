# hai-teams — planner instructions

This file is for the **main session**, which acts as the planner. Operating detail lives in
`.claude/agents/*.md`; do not copy provider gotchas, SLURM syntax or dataset traps back here.

## Who does what

The planner is this session, not a subagent. No subagent has the `Agent` tool, so none can dispatch
another, and each one starts with no memory of the last. Sequencing, and the decision to stop or
start a job, stay here.

| Agent | Use it for |
|---|---|
| `watcher` | live job state — queue, rows written, stalls. Observes only |
| `evaluator` | whether the numbers can be believed, what they mean, cost audit. Recommends only |
| `executor` | a change that has already been decided — edit, transfer, sbatch, scancel |
| `reviewer` | read the diff before it reaches Quest |
| `tracker` | record a problem and its resolution in `NegotiationToM/ISSUES.md` |
| `summarizer` | read a lot of files, return only the conclusion |

## A broken run gets killed, not waited out

Standing authorisation from the user (2026-07-29). Do not ask before doing this:

1. `scancel` the affected job.
2. Fix and verify locally.
3. Overwrite the file(s) on Quest, confirming with `md5sum`.
4. Resubmit and let it run.

Letting a known-bad job run to the wall wastes the quota *and* the wall-clock slot. A Qwen pilot
burned 3h10m for 315 rows and 105 empty responses because the real fix
(`reasoning={"enabled": False}`) had never left the laptop.

**Before resubmitting, decide whether the existing checkpoint is still valid.** Resume skips any UID
already present, so rows written by the old code survive into the new run. If the fix changed the
prompt or the decoding config, archive the checkpoint to a timestamped directory rather than
resuming — one result set containing two configurations is worse than redoing the rows.

## Verify local↔Quest sync before every submit

Assuming Quest was current is what caused the incident above, and the drift was never limited to the
model being worked on: a check on 2026-07-29 found 6 of 32 files stale, including the shared
`neg_eval_core.py`. Because the runners import from that core, transferring a runner without it
fails at import. **Sync the core and the runners together, or not at all.**

```bash
cd NegotiationToM
setopt null_glob
FILES=(*.py NEG_*/*.py NEG_*/*.sh); FILES=(${(u)FILES})
md5 -r "${FILES[@]}" | awk '{print $2, $1}' | sort -k1,1 > /tmp/l.md5
ssh quest "cd /gpfs/projects/p32983/NegotiationToM && md5sum ${FILES[*]}" \
  | awk 'NF==2{print $2, $1}' | sort -k1,1 > /tmp/q.md5
join -j1 -o 0,1.2,2.2 /tmp/l.md5 /tmp/q.md5 | awk '$2!=$3{print "DIFFER  " $1}'
join -v1 -j1 /tmp/l.md5 /tmp/q.md5      | awk '{print "MISSING ON QUEST  " $1}'
```

Two ways this check lies, both hit while writing it:

- **The shell is zsh, which does not word-split an unquoted `$FILES`.** `md5 -r $FILES` then treats
  the whole list as one filename, both sides come back empty, and `diff` reports "in sync". Use an
  array and quote the expansion.
- **`join` requires input sorted on the join field.** Sorting by hash makes it emit nonsense —
  every file listed as simultaneously missing from both sides. Sort by filename (`-k1,1`).

Always print the row count of both files before trusting the comparison.

## Shared context

- `NegotiationToM/negotiation.md` — current results, dataset traps, reasoning-token cost, the
  silent-failure catalogue
- `NegotiationToM/ISSUES.md` — problems already hit, what was rejected, what shipped, false alarms
- `NegotiationToM/DATA_NOTES.md` — cutoff tiling, the `"None"` sentinel, expected row counts
