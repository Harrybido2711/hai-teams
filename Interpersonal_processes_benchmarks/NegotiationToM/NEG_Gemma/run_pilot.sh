#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=16:00:00
#SBATCH --job-name=negpilot_gemma
#SBATCH --output=log_pilot.txt
#SBATCH --error=log_pilot.err

# Pilot: 10% of dialogues (238/2380), all three tasks, ~1,414 API calls.
# Measured throughput on the first attempt was ~3.6 rows/min, i.e. ~6.5h for the full pilot, which
# overran the 4h 'short' partition — hence 'normal'.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemma_neg_eval.py \
    --model google/gemma-4-31B-it \
    --task all \
    --pilot \
    --pilot-frac 0.10 \
    --save-every 20
