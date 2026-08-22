#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=gemini35lite_or_emobench
#SBATCH --output=log.txt
#SBATCH --error=log.err

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python -u gemini35lite_or_emo_eval.py \
    --model google/gemini-3.5-flash-lite \
    --task all \
    --save-every 20 \
    --max-tokens 2048
