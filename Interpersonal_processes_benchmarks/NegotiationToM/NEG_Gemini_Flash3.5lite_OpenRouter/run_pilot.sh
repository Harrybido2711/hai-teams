#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --job-name=negpilot_gem35lite
#SBATCH --output=log_pilot.txt
#SBATCH --error=log_pilot.err

# Pilot: 10% of dialogues (238/2380), all three tasks, ~1,414 API calls.
# Writes to results/pilot/ so a full run's output is never touched.
#
# RUN THIS FIRST AND READ THE BUDGET BLOCK. Both caps in the runner are chosen, not measured:
# `truncated` must be at or near 0 before the full run commits 14,138 rows to them. The pilot also
# prints the call-latency percentiles the 200s watchdog is sized against.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_neg_eval.py \
    --model google/gemini-3.5-flash-lite \
    --task all \
    --pilot \
    --pilot-frac 0.10 \
    --save-every 20
