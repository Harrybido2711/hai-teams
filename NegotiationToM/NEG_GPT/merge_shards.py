"""
Merge shard outputs from run_negotiation.sh and compute final scores.

Run this after all SLURM array jobs finish:
    python merge_shards.py --model gpt-4o-mini --total-shards 5

Reads:  results/<task>/<model>_shard{0..N-1}of{N}.jsonl
Writes: results/<task>/<model>_all.jsonl
        results/<task>/<model>_all.csv
        results/<task>/<model>_final_overall.csv
"""
import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

INTENT_LABELS = [
    "Build-Rapport", "Callout-Fairness", "Describe-Need", "Discover-Preference",
    "No-Intention", "No-Need", "Promote-Coordination", "Show-Empathy", "Undermine-Requirements",
]

_ITEM_NORM = {
    "food": "Food", "water": "Water", "firewood": "Firewood",
    "not given": "Not Given", "none": "None",
}


def _norm_item(s):
    if not isinstance(s, str):
        return ""
    return _ITEM_NORM.get(s.strip().lower(), s.strip().title())


def _pred_tuple(pred):
    """Return normalized (high, medium, low) from a pred dict for consistency checks."""
    if not pred or not isinstance(pred, dict):
        return ("", "", "")
    def get(key):
        val = pred.get(key) or pred.get(key.title()) or ""
        return _norm_item(val)
    return (get("high"), get("medium"), get("low"))


# ── helpers ───────────────────────────────────────────────────────────────────

def load_shard(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_missing_shards(task, model_name, total_shards, results_root):
    missing = []
    for s in range(total_shards):
        p = os.path.join(results_root, task, f"{model_name}_shard{s}of{total_shards}.jsonl")
        if not os.path.exists(p):
            missing.append(p)
    return missing


def merge_task(task, model_name, total_shards, results_root):
    out_dir = os.path.join(results_root, task)
    print(f"\n[{task}] Merging {total_shards} shards...")

    # ── collect rows ──────────────────────────────────────────────────────────
    missing = check_missing_shards(task, model_name, total_shards, results_root)
    if missing:
        print(f"  WARNING: {len(missing)} shard file(s) not found:")
        for p in missing:
            print(f"    {p}")
        print("  Proceeding with available shards — final scores will be partial.")

    all_rows = []
    seen_uids = set()
    for s in range(total_shards):
        path = os.path.join(out_dir, f"{model_name}_shard{s}of{total_shards}.jsonl")
        if not os.path.exists(path):
            continue
        rows = load_shard(path)
        for r in rows:
            uid = r.get("uid", "")
            if uid in seen_uids:
                # duplicate from checkpoint overlap — skip
                continue
            seen_uids.add(uid)
            all_rows.append(r)
        print(f"  shard {s}: {len(rows)} rows loaded")

    if not all_rows:
        print(f"  ERROR: No rows found for task={task}. Skipping.")
        return None

    # ── save merged JSONL ─────────────────────────────────────────────────────
    merged_jsonl = os.path.join(out_dir, f"{model_name}_all.jsonl")
    with open(merged_jsonl, "w", encoding="utf-8") as f:
        for r in all_rows:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    # ── build DataFrame; reconstruct list fields that CSV may have stringified ──
    df = pd.DataFrame(all_rows)

    # bitmask columns are proper lists in JSONL (from json.loads) — but double-check
    for col in ["gold_bitmask", "pred_bitmask"]:
        if col in df.columns and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(json.loads)

    # ── save merged CSV ───────────────────────────────────────────────────────
    merged_csv = os.path.join(out_dir, f"{model_name}_all.csv")
    df.to_csv(merged_csv, index=False)
    print(f"  Total rows merged: {len(df)}")

    # ── compute final scores ──────────────────────────────────────────────────
    if task == "desire":
        overall = df["desire_em"].mean()
        result = {"metric": "Desire_EM", "score": overall}
        print(f"  Desire EM: {overall:.4f}  ({df['desire_em'].sum()}/{len(df)})")

    elif task == "belief":
        overall = df["belief_em"].mean()
        result = {"metric": "Belief_EM", "score": overall}
        print(f"  Belief EM: {overall:.4f}  ({df['belief_em'].sum()}/{len(df)})")

    elif task == "intention":
        golds = list(df["gold_bitmask"])
        preds = list(df["pred_bitmask"])
        micro = f1_score(golds, preds, average="micro", zero_division=0)
        macro = f1_score(golds, preds, average="macro", zero_division=0)
        result = [
            {"metric": "Intent_Micro_F1", "score": micro},
            {"metric": "Intent_Macro_F1", "score": macro},
        ]
        print(f"  Intent Micro F1: {micro:.4f}  |  Macro F1: {macro:.4f}")

    else:
        print(f"  Unknown task '{task}' — skipping scoring.")
        return df

    rows_out = result if isinstance(result, list) else [result]
    overall_path = os.path.join(out_dir, f"{model_name}_final_overall.csv")
    pd.DataFrame(rows_out).to_csv(overall_path, index=False)
    print(f"  Saved → {merged_csv}")
    print(f"  Saved → {overall_path}")

    return df


# ── composite metrics ─────────────────────────────────────────────────────────

def compute_all_em(df_desire, df_belief, df_intention):
    """All_EM: desire + belief + intention all correct for the same dialogue_id.
    - desire:    both agents must have desire_em=1
    - belief:    both agents must have belief_em=1
    - intention: both utterances must exactly match (pred_bitmask == gold_bitmask)
    """
    desire_ok   = df_desire.groupby("dialogue_id")["desire_em"].min()
    belief_ok   = df_belief.groupby("dialogue_id")["belief_em"].min()

    df_i = df_intention.copy()
    df_i["intent_em"] = df_i.apply(
        lambda r: int(list(r["pred_bitmask"]) == list(r["gold_bitmask"])), axis=1
    )
    intention_ok = df_i.groupby("dialogue_id")["intent_em"].min()

    common = set(desire_ok.index) & set(belief_ok.index) & set(intention_ok.index)
    if not common:
        return 0.0

    scores = [
        int(desire_ok[did] == 1 and belief_ok[did] == 1 and intention_ok[did] == 1)
        for did in common
    ]
    return sum(scores) / len(scores)


def compute_consistency(df):
    """Consistency: fraction of (dialogue, agent) groups whose predictions are
    identical across ALL turn cutoffs of the same dialogue."""
    df = df.copy()
    df["dialogue"] = df["dialogue_id"].str.split("-").str[0]
    df["pred_tuple"] = df["pred"].apply(_pred_tuple)

    consistent = total = 0
    for _, group in df.groupby(["dialogue", "agent"]):
        tuples = group["pred_tuple"].tolist()
        if len(set(tuples)) == 1:
            consistent += 1
        total += 1

    return consistent / total if total > 0 else 0.0


# ── summary ───────────────────────────────────────────────────────────────────

def _collect_scores(dfs):
    """Compute all metrics from merged DataFrames and return as a dict."""
    scores = {}
    have_desire    = "desire"    in dfs and dfs["desire"]    is not None
    have_belief    = "belief"    in dfs and dfs["belief"]    is not None
    have_intention = "intention" in dfs and dfs["intention"] is not None

    if have_desire:
        scores["Desire_EM"] = round(dfs["desire"]["desire_em"].mean(), 4)
        scores["Consistency_Desire"] = round(compute_consistency(dfs["desire"]), 4)

    if have_belief:
        scores["Belief_EM"] = round(dfs["belief"]["belief_em"].mean(), 4)
        scores["Consistency_Belief"] = round(compute_consistency(dfs["belief"]), 4)

    if have_intention:
        golds = list(dfs["intention"]["gold_bitmask"])
        preds = list(dfs["intention"]["pred_bitmask"])
        scores["Intent_Micro_F1"] = round(f1_score(golds, preds, average="micro", zero_division=0), 4)
        scores["Intent_Macro_F1"] = round(f1_score(golds, preds, average="macro", zero_division=0), 4)

    if have_desire and have_belief and have_intention:
        scores["All_EM"] = round(compute_all_em(dfs["desire"], dfs["belief"], dfs["intention"]), 4)

    return scores


def print_summary(dfs, model_name):
    scores = _collect_scores(dfs)
    print(f"\n{'='*55}")
    print(f"FINAL SCORES — {model_name}")
    print(f"{'='*55}")
    for metric, val in scores.items():
        print(f"  {metric:<25} {val:.4f}")
    print(f"{'='*55}\n")


def write_results_csv(dfs, model_name, results_root):
    scores = _collect_scores(dfs)
    rows = [{"Score": k, model_name: v} for k, v in scores.items()]
    out_path = os.path.join(results_root, f"negotiation_{model_name}_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Summary CSV saved → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="gpt-4o-mini")
    parser.add_argument("--total-shards", type=int, default=5)
    parser.add_argument("--task",         default="all",
                        choices=["desire", "belief", "intention", "all"])
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    model_name = args.model.split("/")[-1].replace(".", "_")
    tasks = ["desire", "belief", "intention"] if args.task == "all" else [args.task]

    dfs = {}
    for task in tasks:
        dfs[task] = merge_task(task, model_name, args.total_shards, args.results_root)

    if args.task == "all":
        print_summary(dfs, model_name)
        write_results_csv(dfs, model_name, args.results_root)
