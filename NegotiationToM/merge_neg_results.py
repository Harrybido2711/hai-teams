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
    return metrics, scored


def all_em(scored_by_task):
    """All (Exact Match): a dialogue counts only if desire, belief and intention are ALL correct.

    The information unit is the dialogue_id — one turn-cutoff sample. Desire and belief contribute
    two rows each (one per agent) and intention one or two, and every one of them must be right,
    so this is AND logic over the whole unit rather than an average of the three task scores.

    A dialogue is skipped when any task has no scorable row for it (i.e. it was one of the
    unannotated samples), keeping this consistent with the per-task exclusion rule.
    """
    per_dialogue = {}
    for task, rows in scored_by_task.items():
        flag = {"desire": "desire_em", "belief": "belief_em"}.get(task)
        for row in rows:
            entry = per_dialogue.setdefault(row["dialogue_id"], {})
            if flag:
                ok = bool(row[flag])
            else:
                ok = row["gold_bitmask"] == row["pred_bitmask"]
            entry[task] = entry.get(task, True) and ok

    complete = [v for v in per_dialogue.values() if len(v) == 3]
    if not complete:
        return 0.0, 0
    correct = sum(1 for v in complete if all(v.values()))
    return correct / len(complete), len(complete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--total-shards", type=int, default=5)
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    slug = args.model.split("/")[-1].replace(".", "_").replace("/", "-")
    metrics = []
    scored_by_task = {}
    for task in ("desire", "belief", "intention"):
        task_metrics, scored = merge_task(args.results_root, task, slug, args.total_shards)
        metrics.extend(task_metrics)
        scored_by_task[task] = scored

    score, n = all_em(scored_by_task)
    print(f"[all] All_EM over {n} dialogues where all three tasks are scorable")
    metrics.append({"metric": "All_EM", "score": score})
    metrics.append({"metric": "All_EM_dialogues", "score": n})

    output = os.path.join(args.results_root, f"{slug}_negotiation_overall.csv")
    pd.DataFrame(metrics, columns=["metric", "score"]).to_csv(output, index=False)
    print(f"Merged output written to {output}")


if __name__ == "__main__":
    main()
