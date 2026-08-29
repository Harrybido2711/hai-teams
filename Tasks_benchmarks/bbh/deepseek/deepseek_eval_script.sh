#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=28:00:00
#SBATCH --job-name=deepseek_bbh
#SBATCH --output=deepseek/deepseek_outlog
#SBATCH --error=deepseek/deepseek_errlog

# Submit from the bbh ROOT (sbatch deepseek/deepseek_eval_script.sh): every path here and
# the --output/--error paths above are relative to the directory you submit from.

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python deepseek/deepseek_eval.py
