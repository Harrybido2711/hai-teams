#!/usr/bin/env python3
"""Drop failed rows from checkpoints so a re-run actually re-runs them.

`load_checkpoint` adds every row's uid to the done set, including rows whose call failed and were
written with `raw_response: ""` and a zeroed prediction. A plain resume therefore skips them,
finishes in seconds, and reproduces the identical broken numbers — the stale-checkpoint failure
wearing a new hat. Run this first, then resubmit.

    python3 prune_failed_rows.py                    # dry run by default — touches nothing
    python3 prune_failed_rows.py --apply            # rewrite, keeping a .bak of each file
    python3 prune_failed_rows.py --apply --only NEG_XAI NEG_Gemini
"""

import argparse
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["NEG_GPT", "NEG_Gemini", "NEG_XAI", "NEG_Qwen", "NEG_Gemma", "NEG_Deepseek"]


def is_failed(row):
    """A row that recorded a non-answer rather than a wrong answer.

    Empty raw_response is the signal. A parsed-but-wrong prediction is a genuine result and must be
    kept — deleting those would quietly inflate the score.
    """
    if (row.get("raw_response") or "").strip():
        return False
    if row.get("pred") is not None:
        return False
    return True


def process(path, apply):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keep = [r for r in rows if not is_failed(r)]
    dropped = len(rows) - len(keep)
    if dropped and apply:
        shutil.copy2(path, path + ".bak")
        with open(path, "w", encoding="utf-8") as handle:
            for row in keep:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite files (default is a dry run)")
    ap.add_argument("--only", nargs="+", choices=MODELS, help="restrict to these model folders")
    args = ap.parse_args()

    targets = args.only or MODELS
    print(f"{'APPLY' if args.apply else 'DRY RUN'} — {', '.join(targets)}\n")
    print("%-46s %8s %8s %8s" % ("file", "rows", "dropped", "keep"))
    print("-" * 74)

    grand_rows = grand_dropped = 0
    for model in targets:
        pattern = os.path.join(ROOT, model, "results", "**", "*.jsonl")
        for path in sorted(glob.glob(pattern, recursive=True)):
            if path.endswith(".bak"):
                continue
            total, dropped = process(path, args.apply)
            grand_rows += total
            grand_dropped += dropped
            if dropped:
                print("%-46s %8d %8d %8d" % (
                    os.path.relpath(path, ROOT), total, dropped, total - dropped))
    print("-" * 74)
    print("%-46s %8d %8d %8d" % ("TOTAL", grand_rows, grand_dropped, grand_rows - grand_dropped))

    if grand_dropped and not args.apply:
        print("\nNothing was modified. Re-run with --apply to rewrite (a .bak is kept per file).")
    elif not grand_dropped:
        print("\nNo failed rows found — a resume would be safe as-is.")


if __name__ == "__main__":
    main()
