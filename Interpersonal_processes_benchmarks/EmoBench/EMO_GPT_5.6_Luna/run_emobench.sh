#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=gpt56luna_emobench
#SBATCH --output=log.txt
#SBATCH --error=log.err

# -u so the log is readable while the job runs. Without it SLURM block-buffers stdout and a run
# that is retrying looks identical to a run that is working.
module purge

/projects/p32983/pythonenvs/hai-teams/bin/python -u gpt56luna_emo_eval.py \
    --model gpt-5.6-luna \
    --task all \
    --seed 42 \
    --save-every 20 \
    --reasoning-effort low \
    --max-tokens 2048
