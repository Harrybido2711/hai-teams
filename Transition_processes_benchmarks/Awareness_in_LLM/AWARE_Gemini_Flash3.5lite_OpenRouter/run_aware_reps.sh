#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=04:00:00
#SBATCH --job-name=gem35_aware_rep
#SBATCH --output=log_rep%a.txt
#SBATCH --error=log_rep%a.err

# The five repeats behind stability and robustness (evaluation-criteria.md). Array task K asks the
# committed subsample (../subsample_uids.json: 300 rows, 30 from each task) once, at exactly the full
# run's config, into results_rep<K>/ — never results/. Five at once is this project's ceiling of
# concurrent streams per provider. A repeat is whole when its log ends "300/300 subsample rows,
# whole". Re-check one by hand with:
#   python gemini35lite_aware_eval.py --model google/gemini-3.5-flash-lite --rep 1 --score-only

module purge
export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_aware_eval.py \
    --model google/gemini-3.5-flash-lite \
    --rep "${SLURM_ARRAY_TASK_ID}" "$@"
