#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --job-name=crit_gem35lite
#SBATCH --output=log_criteria_%a.txt
#SBATCH --error=log_criteria_%a.err

# Criteria 4 and 5 (.claude/references/evaluation-criteria.md): seven runs over the committed
# subsample (../criteria_subsample.json - 150 dialogues, 300/300/295 rows), each into its own
# results/<tag>/ so none touches the main checkpoint or another run's:
#   array 0-4  rep1..rep5          the reported config, unperturbed - stability
#   array 5    perturb-shuffle     item order / intent-label order reshuffled - robustness
#   array 6    perturb-paraphrase  instruction reworded, answer format verbatim - robustness
# Same model, decoding and caps as run_negotiation.sh, which produced the reported number.
module purge

export PYTHONUNBUFFERED=1

TAGS=(rep1 rep2 rep3 rep4 rep5 perturb-shuffle perturb-paraphrase)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
EXTRA=()
case "$TAG" in
    perturb-*) EXTRA=(--perturb "${TAG#perturb-}") ;;
esac

/projects/p32983/pythonenvs/hai-teams/bin/python gemini35lite_neg_eval.py \
    --model google/gemini-3.5-flash-lite \
    --task all \
    --subsample ../criteria_subsample.json \
    --tag "$TAG" "${EXTRA[@]}" \
    --save-every 20
