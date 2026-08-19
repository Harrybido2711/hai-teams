# Quest

Northwestern's cluster. Key-based SSH to `uwr0681@login.quest.northwestern.edu`
(`~/.ssh/id_ed25519`); repo at `/gpfs/projects/p32983/NegotiationToM`. Never ask for or use the
NetID password. `client_global_hostkeys_prove_confirm ... libcrypto` on connect is cosmetic; filter
it out.

**Hard boundary:** under `/projects/p32983` touch only directories owned by `uwr0681` —
`NegotiationToM/`, `EmoBench-master/`, `DocVQA/`. The rest belong to other project members.

## Transferring

Transfer with `ssh quest "cat > $REMOTE/$f" < $LOCAL/$f`, then **verify with `md5sum`**. Never
assume a transfer landed. **Never overwrite `.env` on Quest** and never copy it out — it exists only
there; if it goes missing, `cp ../EmoBench-master/.env .env`.

**Sync `neg_eval_core.py` together with the runners, always.** The runners import from it
(`record_usage` was added there on 2026-07-29), so a runner transferred without the core dies at
import, and a core transferred without the runners breaks whichever runner used a signature that
changed. Check the whole set before submitting, not just the file you edited — on 2026-07-29 a check
found 6 of 32 files stale on Quest when only Qwen was suspected.

`python3 .claude/scripts/check_quest_sync.py` does this comparison and exits 1 on drift. The manual
form, when you need to see it:

```bash
cd Interpersonal_processes_benchmarks/NegotiationToM
setopt null_glob
FILES=(*.py NEG_*/*.py NEG_*/*.sh); FILES=(${(u)FILES})
md5 -r "${FILES[@]}" | awk '{print $2, $1}' | sort -k1,1 > /tmp/l.md5
ssh quest "cd /gpfs/projects/p32983/NegotiationToM && md5sum ${FILES[*]}" \
  | awk 'NF==2{print $2, $1}' | sort -k1,1 > /tmp/q.md5
join -j1 -o 0,1.2,2.2 /tmp/l.md5 /tmp/q.md5 | awk '$2!=$3{print "DIFFER  " $1}'
join -v1 -j1 /tmp/l.md5 /tmp/q.md5      | awk '{print "MISSING ON QUEST  " $1}'
```

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

**Run order:** `preflight.py` → `sbatch run_pilot.sh` (10% of data, output under `results/pilot/`)
→ review → `sbatch run_<bench>.sh` → `bash run_merge.sh` if sharded.

## Reading the live state

Output lives in `<folder>/results/pilot/<task>/*.jsonl` (pilot) and
`<folder>/results/<task>/*_shard*.jsonl` (full). Logs: `log_pilot.txt` / `log_shard%a.txt` and
`.err`.

```bash
squeue -u uwr0681 -o "%.12i %.16j %.9P %.9T %.10M"           # queued and running
sacct -X -j <ids> -o JobID,JobName%18,State,ExitCode,Elapsed # finished
```

**Judge progress by rows written, not by job state.** SLURM reports RUNNING for a process hung
inside an API call. Gemma once sat that way for over two hours with an empty log while the queue
looked perfectly healthy.

```bash
for d in NEG_*; do
  for t in desire belief intention; do
    f=$(ls $d/results/pilot/$t/*.jsonl 2>/dev/null | head -1)
    [ -n "$f" ] && echo "$d/$t $(wc -l < $f) rows, written $(date -r $f +%H:%M:%S)"
  done
done
```

A file untouched for longer than a checkpoint interval (20 items) plausibly takes is a stall,
whatever the queue says.

**Halt markers are the cheapest signal there is** — check them before grepping any log:

```bash
ls NEG_*/BILLING_HALT.txt NEG_*/QUOTA_HALT.txt NEG_*/FAILURE_HALT.txt 2>/dev/null
```

`BILLING_HALT` needs a human to top the account up; `QUOTA_HALT` clears when the provider's daily
window resets; `FAILURE_HALT` means the provider was failing outright or intermittently. Each file
says whether the checkpoint needs pruning before a resubmit — quote that line rather than guessing.
Markers are cleared at the start of every run, so one that exists is always about the current run.

## Pulling results down

`bash .claude/scripts/pull_quest_results.sh`. Code flows up, results flow down, never the reverse.
After the unattended watcher was killed, four full runs — 56,774 rows — lived only on the cluster
for hours because the pull had been part of that watcher and nothing replaced it.
