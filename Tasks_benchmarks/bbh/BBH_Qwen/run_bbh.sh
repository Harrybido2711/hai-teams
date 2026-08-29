#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_bbh
#SBATCH --output=log.txt
#SBATCH --error=log.err

# Submit from inside this folder: the log paths above are relative to the submit directory.
# The runner resolves its own data and output paths from __file__, so its reads and writes do not
# depend on where the job was submitted from.

module purge

/projects/p32983/pythonenvs/hai-teams/bin/python qwen_bbh_eval.py \
    --model Qwen/Qwen3.5-9B \
    --task all
