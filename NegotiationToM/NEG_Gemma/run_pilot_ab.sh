#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=8:00:00
#SBATCH --job-name=gemma_ab
#SBATCH --output=log_pilot_ab.txt
#SBATCH --error=log_pilot_ab.err

# Accuracy arm of the reasoning A/B.
#
# --pilot-frac 0.03 is 71 of 2,380 dialogues, and it is a STRICT SUBSET of the archived 0.10
# pilot: run_cli shuffles with a fixed seed and takes a prefix, so every uid produced here also
# exists in pilot_archive_reasoning_on_20260804/. That is what makes this a paired comparison —
# the same items, the same scoring code, one variable changed.
#
# The one variable is reasoning={"enabled": False} in gemma_neg_eval.py. Everything else is held:
# max_tokens stays 8192, the watchdog stays at the shared 200s, the prompts are untouched. The
# ceiling is a known second lever worth about 2x on its own, but changing it here would confound
# the accuracy question this job exists to answer.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemma_neg_eval.py \
    --model google/gemma-4-31B-it \
    --task all \
    --pilot \
    --pilot-frac 0.03 \
    --save-every 10
