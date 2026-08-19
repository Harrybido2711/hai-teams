"""Merge NegotiationToM shard checkpoints and reproduce Output_template CSVs."""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Interpersonal_processes_benchmarks.NegotiationToM.neg_eval_core import scorable  # noqa: E402


TEMPLATE_COLUMNS = {
    "desire": ["uid", "dialogue_id", "agent", "gold_desire", "pred", "raw_response", "desire_em"],
    "belief": ["uid", "dialogue_id", "agent", "opponent", "gold_high", "gold_med", "gold_low",
               "pred", "belief_em", "raw_response"],
    "intention": ["uid", "dialogue_id", "utt_idx", "target_utterance", "gold_intent",
                  "gold_bitmask", "pred_intents", "pred_bitmask", "raw_response"],
}


# Full-run row counts on the real data. desire and belief are 2 x 2,380 dialogues; intention is
# 4,618 rather than 4,760 because odd-length dialogues annotate one target utterance, not two.
EXPECTED_ROWS = {"desire": 4760, "belief": 4760, "intention": 4618}


def load_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_task(results_root, task, model_slug, total_shards, allow_partial=False):
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

    # A present-but-short shard is the failure shape the halt guards created. Before them, a broken
    # run was full-length and scored zero — obvious. Now a run that stops itself part-way leaves a
    # well-formed shorter .jsonl, and every metric below would be computed over it without a word.
    # Only the file's absence was ever checked, so assert the count too.
    expected = None if allow_partial else EXPECTED_ROWS.get(task)
    if expected is not None and len(rows) != expected:
        raise RuntimeError(
            f"{task}: merged {len(rows)} rows but expected {expected}. A shard stopped early — "
            f"check NEG_*/BILLING_HALT.txt, QUOTA_HALT.txt and FAILURE_HALT.txt for the reason, "
            f"resume the run, and merge again. Refusing to publish a partial result as a complete "
            f"one. Pass --allow-partial if a short merge is genuinely what you want."
        )

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
        # Headline metric excludes unannotated rows; the _all variant keeps them, for lining up
        # against a published table that may have scored them.
        metrics = [{"metric": "Desire_EM", "score": sframe["desire_em"].mean()},
                   {"metric": "Desire_EM_all", "score": frame["desire_em"].mean()}]
    elif task == "belief":
        metrics = [{"metric": "Belief_EM", "score": sframe["belief_em"].mean()},
                   {"metric": "Belief_EM_all", "score": frame["belief_em"].mean()}]
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
    parser.add_argument("--allow-partial", action="store_true",
                        help="merge even when a task is short of its expected row count")
    args = parser.parse_args()
    # A pilot merges a seeded 10% subset, so the full-run counts do not apply to it.
    is_pilot = os.path.basename(os.path.normpath(args.results_root)) == "pilot"
    allow_partial = args.allow_partial or is_pilot
    if is_pilot:
        print("[merge] pilot results root — skipping the full-run row-count assertion")
    slug = args.model.split("/")[-1].replace(".", "_").replace("/", "-")
    metrics = []
    scored_by_task = {}
    for task in ("desire", "belief", "intention"):
        task_metrics, scored = merge_task(
            args.results_root, task, slug, args.total_shards, allow_partial)
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
