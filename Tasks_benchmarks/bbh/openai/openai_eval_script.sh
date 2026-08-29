#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=2:30:00
#SBATCH --job-name=openai_bbh
#SBATCH --output=openai/openai_outlog
#SBATCH --error=openai/openai_errlog

# Submit from the bbh ROOT (sbatch openai/openai_eval_script.sh): every path here and
# the --output/--error paths above are relative to the directory you submit from.

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python openai/openai_eval.py
