#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=luna_docvqa
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

# 5 shards, and here that is a FIX rather than a convention. At 10 shards this benchmark exhausted
# the key's daily request cap and left 3,021 of 5,349 rows empty — OPENAI_EVAL_NOTES.md has the
# arithmetic. Do not raise it. The 2.5 s sleep is the other half of that fix.
#
# ~1,070 images a shard. Submit from inside this folder. When all five finish, merge:
#   python ../merge_docvqa_shards.py --model gpt-5.6-luna --model-dir . --total-shards 5
# The merge reports which shard is missing and exits 1, so a partial number is never mistaken for
# a whole one.

module purge
export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gpt56luna_docvqa_eval.py \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5 \
    --sleep 2.5
