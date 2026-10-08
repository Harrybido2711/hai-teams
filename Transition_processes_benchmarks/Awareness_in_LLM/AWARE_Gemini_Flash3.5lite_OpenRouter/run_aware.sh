#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --array=0-9%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=12:00:00
#SBATCH --job-name=gem35_aware
#SBATCH --output=log_%a.txt
#SBATCH --error=log_%a.err

# One array task per AwareEval task, at most 5 running at once — 5 concurrent streams is this
# project's measured ceiling (quest-cluster.md). Unsharded, so every file keeps its plain name and
# there is no merge step: array task N writes results/<task>/<model>.jsonl. Largest task first, so
# the throttle never leaves the 966-row task to start last.
#
# Submit from inside this folder. Every array task rewrites the summary files when it exits; the
# run is whole when results/<model>_awareness_overall.csv says COMPLETE 1. Rebuild by hand with:
#   python gemini35lite_aware_eval.py --model google/gemini-3.5-flash-lite --score-only

module purge
export PYTHONUNBUFFERED=1

TASKS=(mission_explicit perspective_mcq capability culture mission_implicit emotion \
       perspective_story_2nd perspective_story_1st perspective_story_reality perspective_story_memory)

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_aware_eval.py \
    --model google/gemini-3.5-flash-lite \
    --task "${TASKS[$SLURM_ARRAY_TASK_ID]}"
