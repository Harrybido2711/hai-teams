#!/usr/bin/env bash
#SBATCH --account=p32983
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --time=02:00:00
#SBATCH --job-name=judge_wb_merge
#SBATCH --output=judge_wb_merge_log_%j.txt
#SBATCH --error=judge_wb_merge_log_%j.err

# Fold the five shard files of run_judge_wb_shards.sh into the canonical judge file, then run the plain
# judge command once: with every row present it makes 0 calls and its exit code says whether the set is
# complete (a missing or failed row would otherwise be judged serially, so the 2 h limit is a backstop).
# Only after this exits 0 is the canonical file the whole result — Results/build_judge_summary.py scores
# whatever the canonical file holds and would report a partial file as complete. Submit from this folder,
# after the shard array: `sbatch --export=ALL,PRODUCER=WB_Qwen_DeepInfra,TASK=sop --dependency=afterok:<array job id> run_judge_wb_merge.sh`.

module purge
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR" || exit 1
PY=/projects/p32983/pythonenvs/hai-teams/bin/python
# No defaults: they must name the same producer and task the shard array was submitted with
# (`sbatch --export=ALL,PRODUCER=WB_Qwen_DeepInfra,TASK=sop --dependency=afterok:<array id> ...`),
# else a merge of the wrong set would report "already merged" and exit 0 beside unmerged shards.
PRODUCER=${PRODUCER:?set PRODUCER, as for the shard array}
TASK=${TASK:?set TASK, as for the shard array}
SHARDS=5
echo "producer=$PRODUCER task=$TASK shards=$SHARDS"
"$PY" Wonderbread_bench/WB_Judges/wb_judge.py --task "$TASK" --set full --producer "$PRODUCER" --merge-shards "$SHARDS" || exit 1
"$PY" Wonderbread_bench/WB_Judges/wb_judge.py --task "$TASK" --set full --producer "$PRODUCER"
