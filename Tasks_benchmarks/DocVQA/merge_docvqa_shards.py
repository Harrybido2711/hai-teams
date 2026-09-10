"""Merge sharded DocVQA results back into one file.

    python merge_docvqa_shards.py --model gpt-5.6-luna --total-shards 5 \
                                  --model-dir DOCVQA_GPT_5.6_Luna

Modelled on `mmlu/merge_mmlu_shards.py`, with the three behaviours that make it safe rather than
merely convenient:

* **A missing shard is reported and the merge proceeds partial**, loudly, and the exit code is 1.
  Failing outright would discard four good shards because one job died; failing silently would
  report a partial number as a whole one.
* **Duplicate rows are dropped on `questionId`.** A resumed shard can re-emit a row it already
  wrote; the merged file must hold each question once.
* **The merged file is written under the plain, untagged name**, which is what the workbook
  updater and any rescorer expect to find.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import docvqa_eval_core as core  # noqa: E402


def merge(model_dir, model_id, total_shards):
    slug = core.model_slug(model_id)
    d = os.path.join(model_dir, "results")
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
            qid = str(r["questionId"])
            if qid in seen:
                continue
            seen.add(qid)
            rows.append(r)
    if not rows:
        return None, missing or ["no rows in any shard"]

    cfg = core.parse_config(rows[0].get("config", "")) or None
    summary = core.write_results(model_dir, model_id, rows, config=cfg)
    expected = len(core.load_data())
    if len(rows) != expected:
        missing.append(f"MERGED {len(rows)} rows but the validation set has {expected}")
    return summary, missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-dir", dest="model_dir", required=True)
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=5)
    args = ap.parse_args()

    summary, missing = merge(args.model_dir, args.model, args.total_shards)
    if summary:
        print("  n=%(n)s  overall_accuracy=%(overall_accuracy)s  anls=%(anls)s  "
              "no_marker=%(no_marker)s  empty=%(empty_response)s" % summary)
    for m in missing:
        print("  MISSING/SHORT:", m)
    sys.exit(1 if missing or not summary else 0)
