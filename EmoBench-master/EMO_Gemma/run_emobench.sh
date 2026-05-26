#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=gemma_emobench
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python gemma_emo_eval.py \
    --model google/gemma-4-31B-it \
    --task all \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 4 \
    --save-every 20
