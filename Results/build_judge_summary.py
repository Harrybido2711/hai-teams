"""Haiku 5.5 judge summary — MultiChallenge and Wonderbread, five models, rebuilt from the row files.

Reads the judge rows under Action_processes_benchmarks/{Multi-challenge_bench,Wonderbread_bench}/
.../results/full/haiku-5.5/ and the generation rows beside them; writes
Results/Haiku_judge_summary.md and .csv. Nothing is typed in by hand: a number is a function of the
files on disk. Scoring is the benchmark's own (mc_eval_core.score, wb_eval_core.sop_metrics).

    python3 Results/build_judge_summary.py [--exclude-qid <MC qid> ...]

`--exclude-qid` drops an MC item from EVERY model (so all are scored on the same items); without it
each model keeps every item it has a verdict for.
"""
import argparse
import collections
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AP = os.path.join(ROOT, "Action_processes_benchmarks")
sys.path[:0] = [AP, os.path.join(AP, "Multi-challenge_bench"), os.path.join(AP, "Wonderbread_bench"),
                os.path.join(AP, "Wonderbread_bench", "WB_Judges")]

import mc_eval_core as mc  # noqa: E402
import wb_eval_core as wb  # noqa: E402

JUDGE = "haiku-5.5"
# (label, MC folder, WB folder or None, judge file stem)
MODELS = [
    ("gpt-5.6-luna", "MC_GPT_5.6_Luna", "WB_GPT_5.6_Luna", "gpt-5.6-luna"),
    ("gemini-3.5-flash-lite", "MC_Gemini_Flash3.5lite_OpenRouter", "WB_Gemini_Flash3.5lite_OpenRouter",
     "google-gemini-3.5-flash-lite"),
    ("Qwen3.5-9B", "MC_Qwen", "WB_Qwen", "Qwen-Qwen3.5-9B"),
    ("gemma-4-31B-it", "MC_Gemma_DeepInfra", "WB_Gemma_DeepInfra", "google-gemma-4-31B-it"),
    ("deepseek-flash", "MC_Deepseek", None, "deepseek-flash"),
]


def rows(path):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                out[r["uid"]] = r
    return out


def mc_score(stem, exclude):
    path = os.path.join(AP, "Multi-challenge_bench", "MC_Judges", "results", "full", JUDGE, stem + ".jsonl")
    rs = [r for r in rows(path).values() if r["status"] == "ok" and r["qid"] not in exclude]
    if not rs:
        return None
    by = collections.defaultdict(list)
    for r in rs:
        by[r["qid"]].append(r["passed"])
    s = mc.score({q: mc.majority(v) for q, v in by.items()})
    return {"macro": s["overall_macro"], "micro": s["overall_micro"], "n": s["n_judged"],
            "unjudged": s["n_unjudged"], "axes": s["axis_scores"]}


def gen_counts(folder, task=None):
    """ok / over_image_limit / other, from the generation rows."""
    d = os.path.join(AP, folder.split("_")[0] == "MC" and "Multi-challenge_bench" or "Wonderbread_bench",
                     folder, "results", "full")
    if task:
        d = os.path.join(d, task)
    files = [f for f in os.listdir(d) if f.endswith(".jsonl")] if os.path.isdir(d) else []
    if not files:
        return None
    return collections.Counter(r["status"] for r in rows(os.path.join(d, files[0])).values())


def same_producer(judge_rows, wb_folder, task):
    """Refuse to score judge rows against a generation folder that did not produce them.

    The judge file is named by the model id alone, and two folders share one: WB_Qwen (Together, cap
    10) and WB_Qwen_DeepInfra (cap 30) both write Qwen-Qwen3.5-9B.jsonl. Paired with the wrong folder,
    SOP precision/recall still compute — over another run's lines — so this is checked, not assumed:
    every judge row's `responses_config` must equal the generation rows' own config."""
    d = os.path.join(AP, "Wonderbread_bench", wb_folder, "results", "full", task)
    files = [f for f in os.listdir(d) if f.endswith(".jsonl")] if os.path.isdir(d) else []
    if len(files) != 1:
        raise SystemExit(f"{d}: expected one generation file, found {files or 'none'}")
    gen = {json.dumps(r["config"], sort_keys=True) for r in rows(os.path.join(d, files[0])).values()}
    judged = {json.dumps(r["config"]["responses_config"], sort_keys=True) for r in judge_rows}
    if len(gen) != 1 or judged != gen:
        raise SystemExit(f"Wonderbread {task}: the judge rows were not produced from {wb_folder}'s "
                         f"generations (configs differ) — point MODELS at the folder that made them.")


def wb_qa(stem, wb_folder):
    path = os.path.join(AP, "Wonderbread_bench", "WB_Judges", "results", "full", JUDGE, "qa", stem + ".jsonl")
    if os.path.exists(path):
        same_producer(rows(path).values(), wb_folder, "qa")
    rs = [r for r in rows(path).values() if r["status"] == "ok" and r["score"] is not None]
    if not rs:
        return None
    out = {c: statistics.mean(r["score"] for r in rs if r["criterion"] == c) for c in wb.QA_CRITERIA}
    out["n_answers"] = len({r["uid"].split("|rep")[0].rsplit("|", 1)[0] for r in rs})
    return out


def wb_sop(stem, wb_folder):
    import wb_judge
    units, _, _ = wb_judge.sop_units("full", False, 0, wb_folder)
    path = os.path.join(AP, "Wonderbread_bench", "WB_Judges", "results", "full", JUDGE, "sop", stem + ".jsonl")
    if os.path.exists(path):
        same_producer(rows(path).values(), wb_folder, "sop")
    rs = [r for r in rows(path).values() if r["status"] == "ok"]
    if not rs:
        return None
    per_rep = collections.defaultdict(list)
    for rep in range(3):
        for uid, pred, gold in units:
            p, g = wb.sop_lines(pred), wb.sop_lines(gold)
            got = {(r["direction"], r["line"]): r["index"] for r in rs if r["pair"] == uid and r["rep"] == rep}
            m = wb.sop_metrics([got.get(("pred", i)) for i in range(len(p))],
                               [got.get(("gold", j)) for j in range(len(g))])
            if m:
                per_rep[rep].append(m)
    keys = ("precision", "recall", "f1", "ordering")
    out = {k: statistics.mean(statistics.mean(m[k] for m in ms if m[k] is not None) for ms in per_rep.values())
           for k in keys}
    out["n_sops"] = len(units)
    return out


SKIPPED_QID = "6765fc5e0af93a24e8fbe4ee"   # Gemini's RECITATION item: not scored anywhere (user, 2026-11-07)
AXES = ["INFERENCE_MEMORY", "INSTRUCTION_RETENTION", "RELIABLE_VERSION_EDITING", "SELF_COHERENCE"]


def fmt(v, nd=3):
    return "N/A" if v is None else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-qid", action="append", default=[SKIPPED_QID],
                    help="MC item dropped for EVERY model (default: Gemini's RECITATION item)")
    args = ap.parse_args()
    exclude = set(args.exclude_qid)
    T = []
    for label, mcf, wbf, stem in MODELS:
        T.append({"model": label, "mc": mc_score(stem, exclude), "gen_mc": gen_counts(mcf),
                  "qa": wb_qa(stem, wbf) if wbf else None, "sop": wb_sop(stem, wbf) if wbf else None,
                  "gen_qa": gen_counts(wbf, "qa") if wbf else None,
                  "gen_sop": gen_counts(wbf, "sop") if wbf else None})

    def n_of(g, k="ok"):
        return None if g is None else g.get(k, 0)

    L = ["# Haiku 5.5 judge summary — MultiChallenge and Wonderbread", "",
         "Judge: Claude Haiku 5.5, 3 replicates per unit. Every number is rebuilt from the row files on "
         "disk by `Results/build_judge_summary.py`. **Preliminary: not yet audited, and not yet in "
         "`Tempo_results.xlsx` / `Final_result.xlsx`.** N/A = the model cannot take that benchmark.", "",
         "## MultiChallenge (higher is better, %)", "",
         "| Model | Macro (4 axes) | Micro | Items scored | " + " | ".join(a.replace("_", " ").title() for a in AXES) + " |",
         "|---|---|---|---|" + "---|" * len(AXES)]
    for r in T:
        m = r["mc"]
        L.append(f"| {r['model']} | {fmt(m['macro'], 1)} | {fmt(m['micro'], 1)} | {m['n']} / 273 | "
                 + " | ".join(fmt(m['axes'].get(a), 1) for a in AXES) + " |")
    L += ["", f"All five models are scored on the same {T[0]['mc']['n']} items: item `{SKIPPED_QID}` is left "
          "out for every model, because Gemini's answer to it was blocked by Google's RECITATION filter "
          "(5 of 5 attempts) and the user decided it is not scored (2026-11-07). Macro = mean of the four "
          "axes; an item's label is the majority of its 3 replicates.", "",
          "## Wonderbread (N/A = text-only model)", "",
          "QA: judge scores 1–3, **1 is best**, mean over answers and 3 replicates. SOP: precision / recall / "
          "F1 / ordering, higher is better.", "",
          "| Model | Completeness | Soundness | Clarity | Compactness | SOP precision | SOP recall | SOP F1 | SOP ordering |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in T:
        q, s_ = r["qa"], r["sop"]
        L.append(f"| {r['model']} | " + " | ".join(fmt(q and q[c], 2) for c in wb.QA_CRITERIA) + " | "
                 + " | ".join(fmt(s_ and s_[k]) for k in ("precision", "recall", "f1", "ordering")) + " |")
    L += ["", "## Coverage — how many units each model was scored on", "",
          "| Model | MC items | WB QA answers | WB SOPs | WB QA not run (image cap) | WB SOP not run (image cap) |",
          "|---|---|---|---|---|---|"]
    for r in T:
        na = r["gen_qa"] is None
        L.append(f"| {r['model']} | {r['mc']['n']} / 273 | "
                 + (" N/A | N/A | N/A | N/A |" if na else
                    f" {n_of(r['gen_qa'])} / 120 | {n_of(r['gen_sop'])} / 486 | "
                    f"{n_of(r['gen_qa'], 'over_image_limit')} | {n_of(r['gen_sop'], 'over_image_limit')} |"))
    cov = {r["model"]: r for r in T}

    def scored(label):
        r = cov[label]
        return f"{n_of(r['gen_qa'])} of 120 QA and {n_of(r['gen_sop'])} of 486 SOP units"

    L += ["", "## Two things to read the table with", "",
          "**(a) A model's own limit means some questions could not be run.** Wonderbread questions carry "
          "many screenshots, and Qwen3.5-9B and gemma-4-31B-it accept only a fixed number of images per "
          "request. A question that needs more is never sent — it is recorded as `over_image_limit`, "
          f"not judged and not counted as a wrong answer. Qwen is scored on {scored('Qwen3.5-9B')}; "
          f"Gemma on {scored('gemma-4-31B-it')}. The other models are scored on every unit, so "
          "a Wonderbread number is not computed over the same questions for every model; the Coverage "
          "table shows the denominator.", "",
          "**(b) A text-only model cannot complete every benchmark.** deepseek-flash takes no images, so "
          "it cannot run Wonderbread at all and shows N/A there. It is compared with the others only on "
          "MultiChallenge, and its row is not comparable across the whole suite. The same applies to any "
          "text-only model.", ""]
    out = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(out, "Haiku_judge_summary.md"), "w").write("\n".join(L))
    import csv
    with open(os.path.join(out, "Haiku_judge_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mc_macro", "mc_micro", "mc_items"] + [f"mc_{a.lower()}" for a in AXES]
                   + [f"wb_qa_{c}" for c in wb.QA_CRITERIA] + ["wb_sop_precision", "wb_sop_recall", "wb_sop_f1",
                   "wb_sop_ordering", "wb_qa_answers", "wb_sop_units"])
        for r in T:
            m, q, s_ = r["mc"], r["qa"], r["sop"]
            w.writerow([r["model"], fmt(m["macro"], 2), fmt(m["micro"], 2), m["n"]]
                       + [fmt(m["axes"].get(a), 2) for a in AXES]
                       + [fmt(q and q[c], 3) for c in wb.QA_CRITERIA]
                       + [fmt(s_ and s_[k], 4) for k in ("precision", "recall", "f1", "ordering")]
                       + [q["n_answers"] if q else "N/A", s_["n_sops"] if s_ else "N/A"])
    print("\n".join(L))


if __name__ == "__main__":
    main()
