#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=16:00:00
#SBATCH --job-name=gpt56luna_bbh
#SBATCH --output=log.txt
#SBATCH --error=log.err

# Submit from inside this folder: the log paths above are relative to the submit directory.
# The runner resolves its own data and output paths from __file__.
#
# NOT YET PILOTED. Run this first and read results/*/*_overall.csv for `no_marker`:
#   python gpt56luna_bbh_eval.py --task boolean_expressions,word_sorting --limit 20
# max_completion_tokens counts reasoning tokens, and a cap that truncates bills for a response
# with no answer in it.

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python gpt56luna_bbh_eval.py \
    --model gpt-5.6-luna \
    --task all
