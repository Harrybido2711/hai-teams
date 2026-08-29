#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=16:00:00
#SBATCH --job-name=gemini35lite_bbh
#SBATCH --output=log.txt
#SBATCH --error=log.err

# Submit from inside this folder: the log paths above are relative to the submit directory.
# The runner resolves its own data and output paths from __file__.
#
# NOT YET PILOTED. Run this first and read results/*/*_overall.csv for `no_marker`:
#   python gemini35lite_bbh_eval.py --task boolean_expressions,word_sorting --limit 20
# The model this replaces, gemini-2.5-flash, failed exactly there: 62% of its responses were cut
# off before the answer, and the job still reported success.

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_bbh_eval.py \
    --model google/gemini-3.5-flash-lite \
    --task all
