#!/bin/bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --time=24:00:00
#SBATCH --job-name=llama_bbh
#SBATCH --output=llama/llama_outlog
#SBATCH --error=llama/llama_errlog

# Submit from the bbh ROOT (sbatch llama/llama_eval_script.sh): every path here and
# the --output/--error paths above are relative to the directory you submit from.

module purge

eval "$(conda shell.bash hook)"

conda activate /projects/p32983/pythonenvs/hai-teams

python llama/llama_eval.py
