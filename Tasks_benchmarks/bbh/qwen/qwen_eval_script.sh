#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_bbh
#SBATCH --output=qwen/qwen_outlog
#SBATCH --error=qwen/qwen_errlog

# Submit from the bbh ROOT (sbatch qwen/qwen_eval_script.sh): every path here and
# the --output/--error paths above are relative to the directory you submit from.

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python qwen/qwen_finish.py
