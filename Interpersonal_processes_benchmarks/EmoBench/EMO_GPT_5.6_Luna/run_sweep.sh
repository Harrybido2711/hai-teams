#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=4:00:00
#SBATCH --job-name=luna_sweep
#SBATCH --output=log_sweep_%x_%j.txt
#SBATCH --error=log_sweep_%x_%j.err

# One arm of the reasoning-effort sweep. EFFORT and TAG come from --export.
# seed is pinned on every arm so a score difference is the parameter, not the sampler.
module purge

/projects/p32983/pythonenvs/hai-teams/bin/python -u gpt56luna_emo_eval.py \
    --model gpt-5.6-luna \
    --task EU \
    --seed 42 \
    --save-every 25 \
    --reasoning-effort "$EFFORT" \
    --tag "$TAG" \
    --max-tokens 4096
