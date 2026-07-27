#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=16:00:00
#SBATCH --job-name=negpilot_qwen
#SBATCH --output=log_pilot.txt
#SBATCH --error=log_pilot.err

# Pilot: 10% of dialogues (238/2380), all three tasks, ~1,414 API calls.
# Qwen3.5 is a thinking model with a large token budget (32768), so calls are much slower than
# GPT/Gemini — this runs on 'normal' rather than the 4h 'short' partition.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python qwen_neg_eval.py \
    --model Qwen/Qwen3.5-9B \
    --task all \
    --pilot \
    --pilot-frac 0.10 \
    --save-every 20
