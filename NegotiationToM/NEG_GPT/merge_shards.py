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


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(dfs, model_name):
    print(f"\n{'='*55}")
    print(f"FINAL SCORES — {model_name}")
    print(f"{'='*55}")

    if "desire" in dfs and dfs["desire"] is not None:
        df = dfs["desire"]
        print(f"  Desire EM:       {df['desire_em'].mean():.4f}  ({df['desire_em'].sum()}/{len(df)})")

    if "belief" in dfs and dfs["belief"] is not None:
        df = dfs["belief"]
        print(f"  Belief EM:       {df['belief_em'].mean():.4f}  ({df['belief_em'].sum()}/{len(df)})")

    if "intention" in dfs and dfs["intention"] is not None:
        df = dfs["intention"]
        golds = list(df["gold_bitmask"])
        preds = list(df["pred_bitmask"])
        micro = f1_score(golds, preds, average="micro", zero_division=0)
        macro = f1_score(golds, preds, average="macro", zero_division=0)
        print(f"  Intent Micro F1: {micro:.4f}")
        print(f"  Intent Macro F1: {macro:.4f}")

    print(f"{'='*55}\n")


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
