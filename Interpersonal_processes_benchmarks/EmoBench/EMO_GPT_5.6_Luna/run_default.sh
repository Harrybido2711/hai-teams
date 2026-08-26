#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=8:00:00
#SBATCH --job-name=luna_default
#SBATCH --output=log_default.txt
#SBATCH --error=log_default.err

# The model's OWN default reasoning effort, run in full to see what it does.
# Passed explicitly rather than omitted: rule 4 — a default belongs to the provider and can move,
# a pinned value records what this run actually used.
# max-tokens is raised to 4096 because reasoning counts toward max_completion_tokens, and a cap
# that thinking exhausts returns a billed empty row rather than an error.
module purge

/projects/p32983/pythonenvs/hai-teams/bin/python -u gpt56luna_emo_eval.py \
    --model gpt-5.6-luna \
    --task all \
    --seed 42 \
    --save-every 20 \
    --reasoning-effort medium \
    --tag default \
    --max-tokens 4096
