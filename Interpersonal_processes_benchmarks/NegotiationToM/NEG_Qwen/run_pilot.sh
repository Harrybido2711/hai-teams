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
# Qwen3.5 is the slowest of the six — roughly 3 minutes per item once retries are counted — so it
# runs on 'normal' rather than the 4h 'short' partition.
#
# --save-every 5, not the usual 20. At ~3 min/item a 20-item interval means the checkpoint only
# advances once an hour, and any restart in between discards everything since the last write. The
# monitor's stall detector reads row counts, so it saw an hour of real work as "no progress",
# cancelled the job, and the run livelocked at 160 rows across three restarts. Committing every 5
# items caps that loss at ~15 minutes.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python qwen_neg_eval.py \
    --model Qwen/Qwen3.5-9B \
    --task all \
    --pilot \
    --pilot-frac 0.10 \
    --save-every 5
