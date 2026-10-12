#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=short
#SBATCH --array=0-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --job-name=crit_emo_gpt56
#SBATCH --output=log_criteria_%a.txt
#SBATCH --error=log_criteria_%a.err

# Criteria 4 and 5 (.claude/references/evaluation-criteria.md): seven runs over all 400 English
# items (EmoBench's whole set is already the ~300-per-benchmark subsample), each into its own
# results_<tag>/ so none touches results/ or another run's:
#   array 0-4  rep1..rep5          the reported config, unperturbed - stability
#   array 5    perturb-shuffle     every choice list reordered (EU's two independently) - robustness
#   array 6    perturb-paraphrase  instruction paragraph reworded, answer format verbatim - robustness
# Same model, decoding and caps as run_emobench.sh, which produced the reported number.
module purge

TAGS=(rep1 rep2 rep3 rep4 rep5 perturb-shuffle perturb-paraphrase)
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}
EXTRA=()
case "$TAG" in
    perturb-*) EXTRA=(--perturb "${TAG#perturb-}") ;;
esac

/projects/p32983/pythonenvs/hai-teams/bin/python -u gpt56luna_emo_eval.py \
    --model gpt-5.6-luna \
    --task all \
    --seed 42 \
    --save-every 20 \
    --reasoning-effort low \
    --max-tokens 2048 \
    --tag "$TAG" "${EXTRA[@]}"
