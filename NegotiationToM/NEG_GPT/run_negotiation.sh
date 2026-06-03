#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=openai_negotiation
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

# Full NegotiationToM run: 2380 samples split into 5 shards (~476 each).
# Quest allows max 5 parallel jobs — all 5 fit in one submission.
# Each job runs all 3 tasks (desire + belief + intention) for its shard.
#
# Before submitting:
#   1. Extract the dataset: unzip -P "NegotiationToM" ../NegotiationToM.zip -d ../
#   2. Set OPENAI_API_KEY in your .env file
#   3. Submit: sbatch run_negotiation.sh

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python openai_neg_eval.py \
    --model gpt-4o-mini \
    --task all \
    --data ../NegotiationToM.json \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5 \
    --save-every 20
