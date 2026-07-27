#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=16:00:00
#SBATCH --job-name=negpilot_deepseek
#SBATCH --output=log_pilot.txt
#SBATCH --error=log_pilot.err

# deepseek-reasoner is a reasoning model and can take tens of seconds per call, so this pilot uses
# 3% (71 dialogues, ~420 calls) instead of 10% and runs on the 'normal' partition.
# The sampler shuffles with a fixed seed and takes a prefix, so these 71 dialogues are a strict
# subset of the 238 the other five models see — the scores stay comparable.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python deepseek_neg_eval.py \
    --model deepseek-reasoner \
    --task all \
    --pilot \
    --pilot-frac 0.03 \
    --save-every 10
