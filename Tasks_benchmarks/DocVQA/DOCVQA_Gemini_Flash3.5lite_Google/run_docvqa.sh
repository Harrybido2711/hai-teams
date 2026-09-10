#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=gemini35lite_docvqa
#SBATCH --output=log_shard%a.txt
#SBATCH --error=log_shard%a.err

# 5 shards, matching the OpenAI slot so the two runs are the same shape. The daily-cap incident that
# set that number was on the OpenAI key (OPENAI_EVAL_NOTES.md); this route has its own quota, but
# the same count is kept because it is the benchmark's established one and 5,349 images do not need
# more.
#
# ~1,070 images a shard. Submit from inside this folder. When all five finish, merge:
#   python ../merge_docvqa_shards.py --model gemini-3.5-flash-lite --model-dir . --total-shards 5

module purge
export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_docvqa_eval.py \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --total-shards 5
