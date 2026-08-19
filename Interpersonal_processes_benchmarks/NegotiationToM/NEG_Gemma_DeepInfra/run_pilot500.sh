#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=8:00:00
#SBATCH --job-name=neg500_di_gemma
#SBATCH --output=log_pilot500.txt
#SBATCH --error=log_pilot500.err

# Speed comparison against NEG_Gemma/run_pilot500.sh: the same checkpoint (gemma-4-31B-it) served
# by DeepInfra (bf16) instead of Together (FP8).
#
# --pilot-frac 0.035 -> 83 of 2380 dialogues -> ~166 desire + ~166 belief + ~161 intention ~= 493
# rows. run_cli shuffles with a FIXED seed (42) and takes a prefix, so this frac selects exactly the
# same dialogues in both runners. Changing the frac on one side only would compare different data.
#
# Sequential, one call at a time — the same shape as Together's per-shard production numbers, so
# the comparison measures the provider rather than the harness.
#
# There is no --concurrency flag: a version of it existed briefly and was reverted after review
# found it reports a 0% hang rate and an N-times-low throughput above 1 worker. It is parked at
# .claude/patches/concurrency-wip.patch. Do not add the flag to this invocation expecting it to
# work — argparse will reject it and the job dies on the first line.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemma_di_neg_eval.py \
    --model google/gemma-4-31B-it \
    --task all \
    --pilot \
    --pilot-frac 0.035 \
    --save-every 20
