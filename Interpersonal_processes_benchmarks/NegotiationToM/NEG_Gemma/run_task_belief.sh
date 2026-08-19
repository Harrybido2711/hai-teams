#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=7-00:00:00
#SBATCH --job-name=neg_gemma_belief
#SBATCH --output=log_belief_shard%a.txt
#SBATCH --error=log_belief_shard%a.err
module purge

export PYTHONUNBUFFERED=1
/projects/p32983/pythonenvs/hai-teams/bin/python gemma_neg_eval.py --model google/gemma-4-31B-it --task belief --shard "$SLURM_ARRAY_TASK_ID" --total-shards 5 --save-every 20
