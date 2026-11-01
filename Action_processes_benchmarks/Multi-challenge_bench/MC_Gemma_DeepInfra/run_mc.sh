#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=04:00:00
#SBATCH --job-name=gemma_mc
#SBATCH --output=log_%j.txt
#SBATCH --error=log_%j.err

# MultiChallenge — gemma-4-31B-it responses for all 273 items (--set full, the default), into
# results/full/. One job, one stream. Submit from inside this folder. Arguments reach the runner: a
# smoke run is `<submit> run_mc.sh --limit 3`, written to smoke/, never results/. Exits non-zero
# unless every item has a usable response, and 2 if any hit the output cap — the judge job depends
# on that (afterok).

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1

/projects/p32983/pythonenvs/hai-teams/bin/python gemma_di_mc_eval.py "$@"
