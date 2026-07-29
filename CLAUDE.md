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

## Saved workflows

`.claude/workflows/*.js` holds the multi-agent procedures this project has already worked out. They
are committed, so they survive the session that wrote them — a workflow passed inline to the tool
does not, and is lost the moment the session ends.

Invoke one by path, which always works:

```
Workflow({scriptPath: ".claude/workflows/fix-broken-run.js",
          args: {model: "NEG_Gemma", reason: "...", pilot: true}})
Workflow({scriptPath: ".claude/workflows/verify-change.js",
          args: {change: "...", files: ["neg_eval_core.py"]}})
```

`{name: "fix-broken-run"}` also resolves, but only from a session that started *after* the file
existed — the registry is built once at startup, so a workflow written mid-session is invisible to
it until the next session. `scriptPath` has no such delay, and is the safe form when in doubt.

Two constraints the tool enforces, both of which cost a launch to discover:

- **`meta` must be a pure literal.** No string concatenation, no variables, no template
  interpolation — `'a' + 'b'` in a field is rejected as a BinaryExpression.
- **Validate `args` at the top and `return` early.** A missing argument then costs 16 ms and zero
  agents instead of spawning a fleet that discovers the problem one at a time.

| Workflow | Use it when |
|---|---|
| `run-model` | starting or restarting one model, end to end: check local → sync to Quest → launch → gate on the first minutes → supervise with hourly local+git sync → audit → record. This is the default |
| `fix-broken-run` | a job is *already* running and needs killing — stale code, a provider refusing every call, rows arriving empty. Not for a job that is merely slow |
| `verify-change` | a change meant to prevent a class of failure has been written and not yet proven wrong. Run it *before* trusting the change in a real run |

### Both sync directions are steps, not habits

`run-model` makes them explicit phases because each has already cost this project real work:

- **local ↔ Quest.** A Qwen pilot spent 3h10m producing 315 rows and 105 empty responses because
  the fix existed only on the laptop; a check that day found 6 of 32 files stale on Quest. Code
  flows up (`check_quest_sync.py`), results flow down (`pull_quest_results.sh`), never the reverse.
- **local ↔ git.** After the unattended watcher was killed, four full runs — 56,774 rows — lived
  only on the cluster for hours, because the pull had been part of that watcher and nothing
  replaced it. Pulling protects data and should be automatic; pushing publishes it and is a
  deliberate step.

When staging for git, stage explicit paths. Never `git add -A`: the watcher did, and swept
unreviewed work into commits named "watcher checkpoint" that went to both remotes.

The loop is meant to repeat. `run-model` returns `needs_rerun` with `proposed_fix` and
`needs_prune` rather than retrying by itself — the fix is usually a code change, which belongs in
front of a human before another run spends hours on it.

**Improving one is editing a file, not writing a new script.** When a run exposes something a
workflow should have caught, add the check to the workflow rather than remembering to do it by hand.
Two rules keep them useful:

- **Every prompt carries the hard rules.** Each workflow states what its agents must not do — no
  provider API calls, no `sbatch`/`scancel` outside the phase that owns it, no edits outside the
  target. These are not decoration: a reviewer once wrote four probe scripts and spent real quota
  because its prompt did not forbid it.
- **The gate must be able to say no.** `fix-broken-run` returns without submitting when the reviewer
  refuses. A verification phase that cannot block is a formality.

## Shared context

- `NegotiationToM/negotiation.md` — current results, dataset traps, reasoning-token cost, the
  silent-failure catalogue
- `NegotiationToM/ISSUES.md` — problems already hit, what was rejected, what shipped, false alarms
- `NegotiationToM/DATA_NOTES.md` — cutoff tiling, the `"None"` sentinel, expected row counts
