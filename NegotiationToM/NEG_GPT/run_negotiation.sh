#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=7-00:00:00
#SBATCH --job-name=neg_gpt
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

# Full NegotiationToM run: 2380 dialogues split into 5 shards.
# Quest allows at most 5 parallel array jobs, so all 5 fit in one submission.
# Each job runs all 3 tasks (desire + belief + intention) for its shard.
#
# Run run_pilot.sh first and check log_pilot.txt before submitting this.
# After all 5 shards finish: bash run_merge.sh
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gpt_neg_eval.py \
    --model gpt-4o-mini \
    --task all \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5 \
    --save-every 20
