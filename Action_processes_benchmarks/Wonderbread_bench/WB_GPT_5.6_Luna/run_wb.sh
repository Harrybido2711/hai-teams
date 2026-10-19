#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --job-name=luna_wb
#SBATCH --output=log_%j.txt
#SBATCH --error=log_%j.err

# Wonderbread judge pilot — gpt-5.6-luna on the sample: 12 QA answers, then 16 demos x 3 ablations
# = 48 SOPs, in ONE job: both runs share this folder (their halt markers are named per task, so
# neither clears the other's), and sequential runs keep the request rate to one stream.
# Needs data/demos/ filled first (`python ../fetch_sample_demos.py` on the login node); this checks.
# Submit from inside this folder. Arguments reach both runs: `<submit> run_wb.sh --limit 6` is a
# smoke run into smoke/ (6 QA items — the 6th is the largest request, 30 screenshots — and 6 demos).

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/projects/p32983/pythonenvs/hai-teams/bin/python

"$PY" ../fetch_sample_demos.py --check || exit 1
"$PY" gpt56luna_wb_eval.py --task qa "$@"; qa=$?
"$PY" gpt56luna_wb_eval.py --task sop "$@"; sop=$?
echo "exit codes: qa=$qa sop=$sop"
[ "$qa" -eq 0 ] && [ "$sop" -eq 0 ]
