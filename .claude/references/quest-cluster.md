# Quest

<!-- size-budget: 8000 -->

Northwestern's cluster. Key-based SSH to `uwr0681@login.quest.northwestern.edu`
(`~/.ssh/id_ed25519`); project space `/gpfs/projects/p32983/`. **Each benchmark's remote path is on
its page in [benchmarks/](benchmarks/README.md)** — the remote layout is flat and did not follow the
2026-08-19 local reorganisation, so a path inferred from the local tree is wrong. Never ask for or
use the NetID password. `client_global_hostkeys_prove_confirm ... libcrypto` on connect is cosmetic; filter
it out.

**Hard boundary:** under `/projects/p32983` touch only directories owned by `uwr0681` —
`NegotiationToM/`, `EmoBench-master/`, `DocVQA/`. The rest belong to other project members.

## Transferring

Transfer with `ssh quest "cat > $REMOTE/$f" < $LOCAL/$f`, then **verify with `md5sum`**. Never
assume a transfer landed. **Never overwrite `.env` on Quest** and never copy it out — it exists only
there; if it goes missing, `cp ../EmoBench-master/.env .env`.

**Sync a benchmark's shared core together with its runners, always.** The runners import from the
core, so a runner transferred without it dies at import, and a core transferred without the runners
breaks whichever runner used a signature that changed. **Check the whole set before submitting, not
just the file you edited** — drift is never confined to the model being worked on.

### The pre-submit gate, and how it goes blind

`.claude/settings.json` wires a `PreToolUse` hook on `Bash` to
`.claude/scripts/sbatch_sync_gate.py`. It runs the sync check before any command containing the
submit keyword, and its contract is:

| checker exit | gate |
|---|---|
| 0 · in sync | silent, command proceeds |
| 1 · drift | **blocks**, printing which files differ |
| 2 · check could not run | **allows, with a warning** — it fails open on purpose |

Three things about it are worth knowing before trusting it:

- **It only covers NegotiationToM.** `check_quest_sync.py` resolves that one directory and globs
  `NEG_*`; there is no per-benchmark argument. A submit for EmoBench or DocVQA therefore gets
  exit 0 — *in sync* — on the strength of a comparison that never looked at the benchmark being
  submitted. Verified 2026-08-22: 41 files compared, all NegotiationToM's. For anything else, run the
  comparison by hand from that benchmark's page before submitting.

- **A stale local path makes it protect nothing, silently.** The 2026-08-19 reorganisation moved
  `NegotiationToM/` under `Interpersonal_processes_benchmarks/`; the checker then found zero code
  files, exited 2, and every submit took the allow-with-warning branch while the gate still looked
  wired up. `check_quest_sync.py` now *searches* for the folder and honours a `NEG_LOCAL_DIR`
  override, but the lesson generalises: after any move, run the checker by hand and confirm it
  reports a non-zero file count on both sides.
- **It triggers on the keyword appearing anywhere in the command text**, including inside a quoted
  commit message or a heredoc that merely mentions it. That is a false positive, not a bug to fix by
  narrowing the match — a gate that under-triggers is invisible, while one that over-triggers is
  merely annoying. Reword the command and move on.

`python3 .claude/scripts/check_quest_sync.py` does this comparison and exits 1 on drift. The manual
form — hash both sides, sort **by filename**, `join` — is written out per benchmark on its page,
because the file globs and the remote directory differ.

**Print the row count of both `.md5` files before believing the result.** This check has two silent
failure modes, both hit in practice:

- zsh does not word-split an unquoted `$FILES`, so `md5 -r $FILES` treats the list as a single
  filename and both sides come back empty — which `diff` happily calls "in sync";
- `join` needs input sorted on the join field, so sorting by hash makes it report every file as
  missing from both sides at once.

## Replacing the code under a broken run

The sequence is fixed: **`scancel` first, then transfer, then resubmit.** Do not transfer under a
live job and hope it picks the change up — a running Python process has already imported its modules
and will finish the run on the old code regardless of what is on disk.

**Stale checkpoints after a config change.** If the fix altered the prompt or the decoding config,
the rows already in the checkpoint were produced under the old one and resume will keep them.
Archive to a timestamped directory instead — a result set holding two configurations is worse than
redoing the rows. Say which you did and why.

## SLURM

```bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1 --ntasks=1 --mem=8GB
#SBATCH --time=7-00:00:00

module purge
export PYTHONUNBUFFERED=1        # or the log stays empty while the job runs
/projects/p32983/pythonenvs/hai-teams/bin/python <script>.py --task all --save-every 20
```

Measured partition ceilings (`sinfo`): `short` 4h, `normal` 2 days, `long` 7 days.
`sbatch` / `squeue -u uwr0681` / `sacct -X` / `scancel <id>`.

Prefer a single job. Shard only when a run is genuinely too long. `--array=0-4` is convention, not a
limit — measured `MaxJobsPU` is **5000**, and 22 jobs have run concurrently. Shard outputs **must**
carry a shard tag (`{model}_shard{N}of{M}.jsonl`); writing an `_overall.csv` without one made each
shard overwrite the last, leaving only one category's results.

**Run order:** preflight → pilot (a small fraction of the data, reviewed before anything else) →
full run → merge if sharded. The exact script names are on the benchmark's page.

## Reading the live state

Output lives under `<model folder>/results/` — pilot and full-run paths, the task names and the log
filenames are on the benchmark's page.

```bash
squeue -u uwr0681 -o "%.12i %.16j %.9P %.9T %.10M"           # queued and running
sacct -X -j <ids> -o JobID,JobName%18,State,ExitCode,Elapsed # finished
```

**Judge progress by rows written, not by job state.** SLURM reports RUNNING for a process hung
inside an API call. Gemma once sat that way for over two hours with an empty log while the queue
looked perfectly healthy.

```bash
for d in <BENCH>_*; do                      # model folders, e.g. NEG_* / EMO_*
  for t in <task> <task>; do                # task names come from the benchmark's page
    f=$(ls $d/results/pilot/$t/*.jsonl 2>/dev/null | head -1)
    [ -n "$f" ] && echo "$d/$t $(wc -l < $f) rows, written $(date -r $f +%H:%M:%S)"
  done
done
```

A file untouched for longer than a checkpoint interval (20 items) plausibly takes is a stall,
whatever the queue says.

**Halt markers are the cheapest signal there is** — check them before grepping any log:

```bash
ls <BENCH>_*/BILLING_HALT.txt <BENCH>_*/QUOTA_HALT.txt <BENCH>_*/FAILURE_HALT.txt 2>/dev/null
```

`BILLING_HALT` needs a human to top the account up; `QUOTA_HALT` clears when the provider's daily
window resets; `FAILURE_HALT` means the provider was failing outright or intermittently. Each file
says whether the checkpoint needs pruning before a resubmit — quote that line rather than guessing.
Markers are cleared at the start of every run, so one that exists is always about the current run.

## Pulling results down

`bash .claude/scripts/pull_quest_results.sh`. Code flows up, results flow down, never the reverse.
After the unattended watcher was killed, four full runs — 56,774 rows — lived only on the cluster
for hours because the pull had been part of that watcher and nothing replaced it.
