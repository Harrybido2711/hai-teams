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

# DeepSeek V4 Flash runs with thinking explicitly disabled for this classification benchmark.
# Use the same fixed 10% subset as the other models so accuracy and token cost are comparable.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python deepseek_neg_eval.py \
    --model deepseek-reasoner \
    --task all \
    --pilot \
    --pilot-frac 0.10 \
    --save-every 10
