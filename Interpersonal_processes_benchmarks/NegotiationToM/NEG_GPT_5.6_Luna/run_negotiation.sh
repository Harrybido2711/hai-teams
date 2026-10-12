#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=7-00:00:00
#SBATCH --job-name=neg_gpt56
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

# Full NegotiationToM run: 2380 dialogues split into 5 shards, each running all three tasks
# (desire + belief + intention). Expected total across the five: 14,138 rows
# (desire 4,760 + belief 4,760 + intention 4,618). An intention count of 4,760 means a known
# bug has returned — see .claude/references/benchmarks/interpersonal/negotiationtom.md.
#
# 5 shards, not more: it matches the other six model folders, and for the OpenAI slot it is also
# the DocVQA fix — requests-per-day, not concurrency, is that key's binding limit, and 10 shards
# is what exhausted it there (.claude/references/quest-cluster.md).
#
# Run run_pilot.sh first and check log_pilot.txt before submitting this.
# After all 5 shards finish: bash run_merge.sh
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gpt56luna_neg_eval.py \
    --model gpt-5.6-luna \
    --task all \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5 \
    --save-every 20
