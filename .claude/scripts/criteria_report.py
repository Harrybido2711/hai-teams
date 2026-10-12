#!/usr/bin/env python3
"""Stability (criterion 4) and robustness (criterion 5) from a model folder's seven tagged runs.

    python3 .claude/scripts/criteria_report.py neg   Interpersonal_processes_benchmarks/NegotiationToM/NEG_GPT_5.6_Luna
    python3 .claude/scripts/criteria_report.py emo   Interpersonal_processes_benchmarks/EmoBench/EMO_GPT_5.6_Luna

Reads only. Every rule it applies is in .claude/references/criteria-runs.md § Analysing it:
content not letters, a failed row is never an answer, all seven tags must hold the same items, and
robustness is flips in excess of the rep1-vs-rep2 baseline. Writes <folder>/criteria_summary.json
beside the runs it summarised, and prints the same numbers.
"""
import argparse
import glob
import json
import os
import random
import sys
from math import comb

REPS = ["rep1", "rep2", "rep3", "rep4", "rep5"]
PERTURBS = ["perturb-shuffle", "perturb-paraphrase"]
BOOTSTRAP = 2000


# ── loading: one dict per task, {item key: (content or None, correct 0/1, cluster)} ──────────────

def load_neg(folder):
    sys.path.insert(0, os.path.dirname(os.path.abspath(folder)))
    import neg_eval_core as core

    def content(task, row):
        if not (row.get("raw_response") or "").strip():
            return None
        if task == "intention":
            return row["pred_bitmask"] if row.get("pred_intents") is not None else None
        if row.get("pred") is None:
            return None
        return tuple(core.pred_item(row["pred"], k) for k in ("high", "medium", "low"))

    def correct(task, row):
        if task == "intention":
            # The reported metric is F1 over the set, so a row has no "correct" of its own. Exact
            # bitmask match is the proposed definition (criteria-runs.md: open for the user).
            return int(row["gold_bitmask"] == row["pred_bitmask"])
        return int(bool(row[f"{task}_em"]))

    runs = {}
    for tag in REPS + PERTURBS:
        runs[tag] = {}
        for task in ("desire", "belief", "intention"):
            files = glob.glob(os.path.join(folder, "results", tag, task, "*.jsonl"))
            if len(files) != 1:
                raise SystemExit(f"{tag}/{task}: expected one jsonl, found {len(files)}")
            rows = [json.loads(line) for line in open(files[0], encoding="utf-8")]
            scored, _ = core.scorable(task, rows)
            runs[tag][task] = {r["uid"]: (content(task, r), correct(task, r),
                                          r["dialogue_id"].split("-")[0]) for r in scored}
    return runs


def load_emo(folder):
    runs = {}
    for tag in REPS + PERTURBS:
        runs[tag] = {}
        for task in ("EU", "EA"):
            files = glob.glob(os.path.join(folder, f"results_{tag}", task, "*_en.jsonl"))
            if len(files) != 1:
                raise SystemExit(f"results_{tag}/{task}: expected one jsonl, found {len(files)}")
            out = {}
            for r in map(json.loads, open(files[0], encoding="utf-8")):
                if task == "EA":
                    text = r.get("answer_text")
                    ok = str(r["label"]).strip().upper() == str(r["answer"]).strip().upper()
                else:
                    pair = (r.get("emo_answer_text"), r.get("cause_answer_text"))
                    text = None if None in pair else pair
                    ok = (str(r["emo_label"]).strip().upper() == str(r["emo_answer"]).strip().upper()
                          and str(r["cause_label"]).strip().upper()
                          == str(r["cause_answer"]).strip().upper())
                out[str(r["qid"])] = (text, int(ok), str(r["qid"]))     # each item its own cluster
            runs[tag][task] = out
    return runs


# ── statistics ────────────────────────────────────────────────────────────────────────────────

def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 lists: (a only, b only, p)."""
    ab = sum(1 for x, y in zip(a, b) if x and not y)
    ba = sum(1 for x, y in zip(a, b) if y and not x)
    n = ab + ba
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(ab, ba) + 1)) / 2 ** n) if n else 1.0
    return ab, ba, p


def cluster_bootstrap(per_item, clusters, seed=42):
    """95% interval of a mean of per-item values, resampling whole clusters — cutoffs of one
    negotiation are not independent items."""
    groups = {}
    for value, cluster in zip(per_item, clusters):
        groups.setdefault(cluster, []).append(value)
    keys = list(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP):
        sample = [v for k in (rng.choice(keys) for _ in keys) for v in groups[k]]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def differs(x, y):
    """A flip. A failed row (None) is never the same answer as anything, itself included."""
    return x is None or y is None or x != y


def summarise(task_runs):
    keys = sorted(task_runs["rep1"])
    for tag, rows in task_runs.items():
        if sorted(rows) != keys:
            raise SystemExit(f"{tag} holds a different item set from rep1 "
                             f"({len(rows)} vs {len(keys)}) — fix the runs before computing anything")
    get = lambda tag, i: task_runs[tag][keys[i]]               # noqa: E731
    clusters = [task_runs["rep1"][k][2] for k in keys]
    n = len(keys)

    stable = [int(all(not differs(get("rep1", i)[0], get(t, i)[0]) for t in REPS[1:])) for i in range(n)]
    accs = {t: sum(get(t, i)[1] for i in range(n)) / n for t in REPS + PERTURBS}
    out = {"items": n, "clusters": len(set(clusters)),
           "stability": {
               "identical_all_5": sum(stable) / n,
               "pass@1": sum(accs[t] for t in REPS) / len(REPS),
               "pass^5": sum(int(all(get(t, i)[1] for t in REPS)) for i in range(n)) / n,
               "accuracy_min": min(accs[t] for t in REPS),
               "accuracy_max": max(accs[t] for t in REPS),
               "failed_rows_per_rep": {t: sum(1 for i in range(n) if get(t, i)[0] is None) for t in REPS},
           }}
    baseline = [int(differs(get("rep1", i)[0], get("rep2", i)[0])) for i in range(n)]
    out["robustness"] = {"baseline_flip_rep1_vs_rep2": sum(baseline) / n}
    for p in PERTURBS:
        flips = [int(differs(get("rep1", i)[0], get(p, i)[0])) for i in range(n)]
        excess = [f - b for f, b in zip(flips, baseline)]
        fa, fb, fp = mcnemar(flips, baseline)
        ra = [get("rep1", i)[1] for i in range(n)]
        pa = [get(p, i)[1] for i in range(n)]
        ca, cb, cp = mcnemar(ra, pa)
        lo, hi = cluster_bootstrap(excess, clusters)
        out["robustness"][p] = {
            "flip_vs_rep1": sum(flips) / n,
            "excess_flip": sum(excess) / n,
            "excess_flip_95ci_cluster": [lo, hi],
            "excess_mcnemar": {"flip_only_under_perturbation": fa, "flip_only_between_reps": fb, "p": fp},
            "accuracy_rep1": accs["rep1"], "accuracy_perturbed": accs[p],
            "accuracy_mcnemar": {"rep1_only": ca, "perturbed_only": cb, "p": cp},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["neg", "emo"])
    ap.add_argument("folder", help="the model folder holding the seven tagged runs")
    args = ap.parse_args()
    runs = (load_neg if args.benchmark == "neg" else load_emo)(args.folder)
    tasks = list(runs["rep1"])
    summary = {"folder": args.folder,
               "definitions": "criteria-runs.md § Analysing it; seed pinned in every rep",
               "tasks": {t: summarise({tag: runs[tag][t] for tag in runs}) for t in tasks}}
    path = os.path.join(args.folder, "criteria_summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    for task, s in summary["tasks"].items():
        st, rb = s["stability"], s["robustness"]
        print(f"\n== {task}: {s['items']} items, {s['clusters']} clusters")
        print(f"  stability  identical in all 5 {st['identical_all_5']:.3f} | pass@1 {st['pass@1']:.3f} "
              f"| pass^5 {st['pass^5']:.3f} | accuracy {st['accuracy_min']:.3f}-{st['accuracy_max']:.3f} "
              f"| failed rows {st['failed_rows_per_rep']}")
        print(f"  baseline   rep1 vs rep2 flip {rb['baseline_flip_rep1_vs_rep2']:.3f}")
        for p in PERTURBS:
            r = rb[p]
            print(f"  {p:19s} flip {r['flip_vs_rep1']:.3f}, excess {r['excess_flip']:+.3f} "
                  f"(95% CI {r['excess_flip_95ci_cluster'][0]:+.3f}..{r['excess_flip_95ci_cluster'][1]:+.3f}, "
                  f"McNemar p={r['excess_mcnemar']['p']:.3f}) | accuracy {r['accuracy_rep1']:.3f} -> "
                  f"{r['accuracy_perturbed']:.3f} (p={r['accuracy_mcnemar']['p']:.3f})")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
