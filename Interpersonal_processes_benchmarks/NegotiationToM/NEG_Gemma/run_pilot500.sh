#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=8:00:00
#SBATCH --job-name=neg500_tg_gemma
#SBATCH --output=log_pilot500.txt
#SBATCH --error=log_pilot500.err

# Together half of the DeepInfra-vs-Together speed comparison. Paired with
# NEG_Gemma_DeepInfra/run_pilot500.sh, which must keep the identical --pilot-frac: run_cli shuffles
# with a fixed seed (42) and takes a prefix, so the same frac selects the same 83 dialogues on both
# sides and the two runs answer the same ~493 questions.
#
# This writes to results/pilot/, which already holds the 10% pilot from the full-run work. Those use
# the same stem, so this job would resume on top of them and finish in seconds having called nothing
# — the stale-checkpoint failure the skeleton warns about. The launcher archives results/pilot/
# before submitting; if you run this by hand, archive it yourself first.
module purge

export PYTHONUNBUFFERED=1

/projects/p32983/pythonenvs/hai-teams/bin/python gemma_neg_eval.py \
    --model google/gemma-4-31B-it \
    --task all \
    --pilot \
    --pilot-frac 0.035 \
    --save-every 20
