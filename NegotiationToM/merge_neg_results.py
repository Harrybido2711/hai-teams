"""Merge NegotiationToM shard checkpoints and reproduce Output_template CSVs."""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neg_eval_core import scorable  # noqa: E402


TEMPLATE_COLUMNS = {
    "desire": ["uid", "dialogue_id", "agent", "gold_desire", "pred", "raw_response", "desire_em"],
    "belief": ["uid", "dialogue_id", "agent", "opponent", "gold_high", "gold_med", "gold_low",
               "pred", "belief_em", "raw_response"],
    "intention": ["uid", "dialogue_id", "utt_idx", "target_utterance", "gold_intent",
                  "gold_bitmask", "pred_intents", "pred_bitmask", "raw_response"],
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_task(results_root, task, model_slug, total_shards):
    task_dir = os.path.join(results_root, task)
    rows, seen = [], set()
    missing = []
    for shard in range(total_shards):
        path = os.path.join(task_dir, f"{model_slug}_shard{shard}of{total_shards}.jsonl")
        if not os.path.exists(path):
            missing.append(path)
            continue
        for row in load_jsonl(path):
            if row["uid"] not in seen:
                seen.add(row["uid"])
                rows.append(row)
    if missing:
        raise FileNotFoundError("Missing shard files:\n" + "\n".join(missing))
    if not rows:
        raise RuntimeError(f"No rows found for {task}")

    merged_jsonl = os.path.join(task_dir, f"{model_slug}_all.jsonl")
    with open(merged_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    frame = pd.DataFrame(rows)
    frame.reindex(columns=TEMPLATE_COLUMNS[task]).to_csv(
        os.path.join(task_dir, f"{model_slug}_all.csv"), index=False
    )
    # Same exclusion rule as neg_eval_core.write_task_outputs: unannotated rows (all slots the
    # sentinel string "None") are unanswerable and must not count as wrong answers.
    scored, dropped = scorable(task, rows)
    if dropped:
        print(f"[{task}] excluded {dropped}/{len(rows)} unannotated rows from the metrics")
    sframe = pd.DataFrame(scored)
    if task == "desire":
        metrics = [{"metric": "Desire_EM", "score": sframe["desire_em"].mean()}]
    elif task == "belief":
        metrics = [{"metric": "Belief_EM", "score": sframe["belief_em"].mean()}]
    else:
        gold = list(sframe["gold_bitmask"])
        pred = list(sframe["pred_bitmask"])
        metrics = [
            {"metric": "Intent_Micro_F1", "score": f1_score(gold, pred, average="micro", zero_division=0)},
            {"metric": "Intent_Macro_F1", "score": f1_score(gold, pred, average="macro", zero_division=0)},
        ]
    metrics.append({"metric": f"{task}_scored_rows", "score": len(scored)})
    pd.DataFrame(metrics, columns=["metric", "score"]).to_csv(
        os.path.join(task_dir, f"{model_slug}_all_overall.csv"), index=False
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-shards", type=int, default=5)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    slug = args.model.split("/")[-1].replace(".", "_").replace("/", "-")
    metrics = []
    for task in ("desire", "belief", "intention"):
        metrics.extend(merge_task(args.results_root, task, slug, args.total_shards))
    output = os.path.join(args.results_root, f"{slug}_negotiation_overall.csv")
    pd.DataFrame(metrics, columns=["metric", "score"]).to_csv(output, index=False)
    print(f"Merged output written to {output}")


if __name__ == "__main__":
    main()
