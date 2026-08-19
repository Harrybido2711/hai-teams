#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --job-name=negpilot_gemini
#SBATCH --output=log_pilot.txt
#SBATCH --error=log_pilot.err

# Pilot: 10% of dialogues (238/2380), all three tasks, ~1,414 API calls.
# Writes to results/pilot/ so a full run's output is never touched.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemini_neg_eval.py \
    --model gemini-2.5-flash \
    --task all \
    --pilot \
    --pilot-frac 0.10 \
    --save-every 20
