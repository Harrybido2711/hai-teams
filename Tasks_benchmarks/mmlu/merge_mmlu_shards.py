"""Merge sharded MMLU results back into one file per subject.

    python merge_mmlu_shards.py --model gpt-5.6-luna --total-shards 5 --model-dir MMLU_GPT_5.6_Luna

Modelled on `NegotiationToM/NEG_GPT/merge_shards.py`, including the three behaviours that make it
safe rather than merely convenient:

* **A missing shard is reported and the merge proceeds partial**, loudly. Failing outright would
  discard four good shards because one job died; failing silently would report a partial number as
  a whole one. Neither is acceptable, so it prints exactly which files are absent and stamps the
  row count it actually merged.
* **Duplicate rows are dropped on `idx`.** A shard that was resumed can re-emit a row it already
  wrote; the merged file must contain each item once.
* **The merged file is written under the plain, untagged name**, which is what every reader —
  the workbook updater, the rescorer — already expects.
"""

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmlu_eval_core as core  # noqa: E402


def merge_subject(model_dir, subject, model_id, total_shards):
    slug = core.model_slug(model_id)
    d = os.path.join(model_dir, "results", subject)
    if not os.path.isdir(d):
        return None, [f"{d} (no such directory)"]

    missing, rows, seen = [], [], set()
    for s in range(total_shards):
        path = os.path.join(d, f"{slug}_shard{s}of{total_shards}.jsonl")
        if not os.path.exists(path):
            missing.append(os.path.basename(path))
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["idx"] in seen:      # a resumed shard can re-emit a row
                continue
            seen.add(r["idx"])
            rows.append(r)
    if not rows:
        return None, missing or ["no rows in any shard"]

    rows.sort(key=lambda r: r["idx"])
    cfg = core.parse_config(rows[0].get("config", "")) or None
    summary = core.write_subject_results(model_dir, subject, model_id, rows, config=cfg)
    expected = len(core.load_subject(subject))
    if len(rows) != expected:
        missing.append(f"MERGED {len(rows)} rows but {subject} has {expected}")
    return summary, missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge sharded MMLU results.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=5)
    ap.add_argument("--subject", default="all")
    args = ap.parse_args()

    subjects = core.SUBJECTS if args.subject == "all" else [s.strip() for s in args.subject.split(",")]
    summaries, problems = [], []
    for subj in subjects:
        s, miss = merge_subject(args.model_dir, subj, args.model, args.total_shards)
        if miss:
            problems.append((subj, miss))
        if s:
            summaries.append(s)
            print(f"  {subj:28s} n={s['n']:4d} score={s['average_score']} "
                  f"no_marker={s['no_marker']} empty={s['empty_response']}")
    if summaries:
        core.write_overall(args.model_dir, args.model, summaries)
        means = [x["average_score"] for x in summaries if x["average_score"] != ""]
        print(f"\nmerged {len(summaries)} subject(s), macro-avg {sum(means)/len(means):.4f}")
    if problems:
        print("\nPROBLEMS — the numbers above are partial:")
        for subj, miss in problems:
            for m in miss:
                print(f"  {subj}: {m}")
        sys.exit(1)
