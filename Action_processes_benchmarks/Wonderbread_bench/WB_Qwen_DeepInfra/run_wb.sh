#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_di_wb
#SBATCH --output=log_%j.txt
#SBATCH --error=log_%j.err

# Wonderbread — Qwen3.5-9B on DeepInfra, full set (--set full, the default): 120 QA answers, then 162
# rank-1 demos x 3 ablations = 486 SOPs, in ONE job: both runs share this folder (their halt
# markers are named per set and task, so neither clears the other's), and sequential runs keep the
# request rate to one stream. Needs the 255 demos under ../data/demos/ first
# (`python ../fetch_sample_demos.py --set full` on the login node); the runner checks before any call.
# Submit from inside this folder. Arguments reach both runs: `<submit> run_wb.sh --limit 6` is a
# smoke run into smoke/. Exits non-zero unless both runs did — the judge job depends on that (afterok).
# 24 h, not the other models' 12: DeepInfra served this model at 17–111 s a call on 2026-11-09 (Gemma
# on the same key: 0.3 s), so 606 units come to ~9–10 h. A wall-clock kill loses nothing — rows are
# written per unit and resume picks up — but it would cost a resubmit.

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/projects/p32983/pythonenvs/hai-teams/bin/python

"$PY" qwen_di_wb_eval.py --task qa "$@"; qa=$?
"$PY" qwen_di_wb_eval.py --task sop "$@"; sop=$?
echo "exit codes: qa=$qa sop=$sop"
[ "$qa" -eq 0 ] && [ "$sop" -eq 0 ]
