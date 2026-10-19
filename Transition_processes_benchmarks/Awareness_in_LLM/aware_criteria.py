"""The six evaluation criteria (`.claude/references/evaluation-criteria.md`) for every AwareBench
model, from the rows on disk. Read-only over results; writes criteria_awareness.csv beside this file,
which `.claude/scripts/build_criteria_workbook.py` lays out as a workbook.

    python aware_criteria.py            # all five model folders

1 cost, 2 token efficiency, 3 latency and 6 reliability come from the full run (results/), through
`.claude/scripts/criteria_report.py`'s usage_criteria() and PRICES — the computation and the price
table every benchmark uses. 4 stability needs all five repeats (results_rep1..5/, on the committed subsample). 5 robustness:
AwareBench's data already holds 2-3 option orders per MCQ question, so the shuffle flip rate comes
from the full run; the noise floor it is corrected by — the rep1-vs-rep2 flip rate on the same
rows — needs two repeats. A paraphrase perturbation is not defined here: the prompts are the
dataset's own and no instruction text is ours (`criteria-runs.md`).

Rules shared with `criteria-runs.md`:
- every row is re-scored from `raw_response` with the core's current scorer, so a scorer change
  can never leave accuracy on one version and the criteria on another;
- answers are compared by CONTENT: a permuted row's letter is mapped to its option text, so a
  shuffle that moves the gold letter is not a flip;
- a failure is not an answer: a parse failure never matches anything, itself included, so five
  failures are not "stable" and two failing orderings are a flip.
"""

import csv
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aware_eval_core as core  # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "..", ".claude", "scripts"))
import criteria_report  # noqa: E402

# slot -> (folder, model id, label in criteria_report.PRICES)
MODELS = {
    "Gemini": ("AWARE_Gemini_Flash3.5lite_OpenRouter", "google/gemini-3.5-flash-lite",
               "gemini-3.5-flash-lite@openrouter"),
    "OpenAI": ("AWARE_GPT_5.6_Luna", "gpt-5.6-luna", "gpt-5.6-luna@openai"),
    "Qwen": ("AWARE_Qwen", "Qwen/Qwen3.5-9B", "qwen3.5-9b@together"),
    "Gemma": ("AWARE_Gemma_DeepInfra", "google/gemma-4-31B-it", "gemma-4-31b-it@deepinfra"),
    "Deepseek": ("AWARE_Deepseek", "deepseek-flash", "deepseek-flash@deepseek"),
}
STABILITY = ("4_identical_answer_rate", "4_pass_at_1_subsample", "4_full_run_acc_subsample",
             "4_pass_hat_5", "4_accuracy_spread")
ROBUST_REPS = ("5_shuffle_flip_rate_subsample", "5_noise_flip_rate_rep1_vs_rep2", "5_excess_flip_rate",
               "5_excess_flip_p_sign", "5_accuracy_change_p_sign", "5_units")


def rescored(rows):
    out = []
    for r in rows:
        pred, correct, parse_fail = core.score(r, r.get("raw_response", ""))
        out.append(dict(r, pred=pred, correct=correct, parse_fail=parse_fail))
    return out


def content(r):
    """What the row answered, independent of option order; None for a failure."""
    if r.get("parse_fail") or not r.get("pred") or not str(r.get("raw_response", "")).strip():
        return None
    choices = r.get("choices")
    return str(choices.get(r["pred"], r["pred"]) if choices else r["pred"]).strip().lower()


def same(*rows):
    """True only when every row has an answer and they are all the same answer."""
    answers = [content(r) for r in rows]
    return None not in answers and len(set(answers)) == 1


def sign_test(diffs):
    """Two-sided exact sign test over per-cluster differences, zeros dropped."""
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(pos, neg) + 1)) / 2 ** n)


def shuffle_units(rows):
    """(question, ordering >= 1) units, each against ordering 0 of the same question. An ordering
    identical to ordering 0 (the dataset ships some twice) is no perturbation and is skipped."""
    groups = {}
    for r in rows:
        if r["task"] in core.PERMUTED:
            groups.setdefault((r["task"], r["question_id"]), []).append(r)
    units = []
    for key, g in groups.items():
        g = sorted(g, key=lambda r: r["perm_id"])
        ref = g[0]
        for r in g[1:]:
            if list(r["choices"].items()) != list(ref["choices"].items()):
                units.append((key, ref, r))
    return units


def flat(rows_by_task):
    return [r for task in core.TASKS for r in rows_by_task.get(task, [])]


def load_rep(model_dir, model, k, full_configs, subsample):
    """The repeat's rows by uid, or (None, reason) when it cannot be used."""
    by_task, problems = core.collect(model_dir, model, rep=k)
    rows = {r["uid"]: r for r in rescored(flat(by_task))}
    if problems:
        return None, f"merge problem: {problems[0]}"
    wanted = {u for t in core.TASKS for u in subsample[t]}
    missing = [u for u in wanted if not str(rows.get(u, {}).get("raw_response", "")).strip()]
    if missing:
        return None, f"{len(missing)} of {len(wanted)} subsample rows missing or empty"
    if set(rows) - wanted:
        return None, f"{len(set(rows) - wanted)} rows outside the subsample"
    if any(r.get("rep") != k for r in rows.values()):
        return None, f"rows not written by repeat {k} (their rep field says otherwise)"
    if any(json.dumps(r["config"], sort_keys=True) not in full_configs for r in rows.values()):
        return None, "config differs from the full run's"
    return rows, None


def criteria(slot):
    folder, model, price_label = MODELS[slot]
    model_dir = os.path.join(HERE, folder)
    main_by_task, problems = core.collect(model_dir, model)
    rows = rescored(flat(main_by_task))
    if problems or len(rows) != sum(v[0] for v in core.EXPECTED.values()):
        raise SystemExit(f"{slot}: the full run is not whole ({len(rows)} rows; {problems})")
    configs = {json.dumps(r["config"], sort_keys=True) for r in rows}
    out = {"benchmark": "AwareBench", "slot": slot, "model": model, "rows": len(rows),
           "scorer": core.SCORER_VERSION}
    # accuracy beside the criteria, from the file the workbooks read
    overall = os.path.join(core.results_dir(model_dir), f"{core.model_slug(model)}_awareness_overall.csv")
    with open(overall) as fh:
        metrics = {row["metric"]: row["score"] for row in csv.DictReader(fh)}
    if metrics.get("COMPLETE") != "1" or metrics.get("SCORER") != core.SCORER_VERSION:
        raise SystemExit(f"{slot}: {overall} is not COMPLETE 1 under {core.SCORER_VERSION}; rescore first")
    out["0_accuracy_overall"] = float(metrics["AWARENESS_OVERALL"])

    # 1 cost · 2 token efficiency · 3 latency · 6 reliability, as on every benchmark
    out.update(criteria_report.usage_criteria(
        [dict(r, raw=r["raw_response"], parse_ok=not r["parse_fail"]) for r in rows], price_label))
    out["6_one_config"] = int(len(configs) == 1)

    # 5 · robustness, the full run's part — the same unit as the subsample column below
    units = shuffle_units(rows)
    out["5_shuffle_flip_rate_full_run"] = round(sum(not same(a, b) for _, a, b in units) / len(units), 4)

    # the repeats: used only when whole, written by that repeat, and at the full run's config
    subsample = core.load_subsample()
    reps, excluded = {}, []
    for k in range(1, core.N_REPS + 1):
        if not os.path.isdir(core.results_dir(model_dir, rep=k)):
            continue
        rep_rows, reason = load_rep(model_dir, model, k, configs, subsample)
        if reason:
            excluded.append(f"rep{k}: {reason}")
            print(f"  {slot}: repeat {k} EXCLUDED — {reason}", flush=True)
        else:
            reps[k] = rep_rows
    out["reps_whole"] = len(reps)
    out["reps_excluded"] = "; ".join(excluded)
    status = (f"excluded — {excluded[0]}" if excluded
              else f"pending — {len(reps)}/{core.N_REPS} repeats")

    # 4 · stability — needs all five. pass@1 is the mean accuracy over the repeats on the 300-row
    # subsample, every task weighted alike; it is NOT comparable with the overall score, so the full
    # run's accuracy on the same rows sits beside it.
    by_uid = {r["uid"]: r for r in rows}
    sub_uids = sorted(u for t in core.TASKS for u in subsample[t])
    if len(reps) == core.N_REPS:
        accs = [sum(reps[k][u]["correct"] for u in sub_uids) / len(sub_uids) for k in reps]
        out.update({"4_identical_answer_rate": round(sum(same(*(reps[k][u] for k in reps)) for u in sub_uids) / len(sub_uids), 4),
                    "4_pass_at_1_subsample": round(statistics.mean(accs), 4),
                    "4_full_run_acc_subsample": round(sum(by_uid[u]["correct"] for u in sub_uids) / len(sub_uids), 4),
                    "4_pass_hat_5": round(sum(all(reps[k][u]["correct"] for k in reps) for u in sub_uids) / len(sub_uids), 4),
                    "4_accuracy_spread": round(max(accs) - min(accs), 4)})
    else:
        out.update({key: status for key in STABILITY})

    # 5 · robustness, corrected by the noise floor — needs rep1 and rep2. Paired per (question,
    # ordering >= 1): shuffle flip = differs from ordering 0 in rep1; noise flip = differs from
    # itself in rep2. Orderings of one question share ordering 0, so the tests are sign tests over
    # questions, not over units.
    if 1 in reps and 2 in reps:
        r1, r2 = reps[1], reps[2]
        units = shuffle_units(list(r1.values()))
        shuf = [not same(a, b) for _, a, b in units]
        noise = [not same(b, r2[b["uid"]]) for _, a, b in units]
        per_q_flip, per_q_acc = {}, {}
        for (key, a, b), s, z in zip(units, shuf, noise):
            per_q_flip[key] = per_q_flip.get(key, 0) + int(s) - int(z)
            per_q_acc[key] = per_q_acc.get(key, 0) + b["correct"] - a["correct"]
        out.update({"5_shuffle_flip_rate_subsample": round(sum(shuf) / len(units), 4),
                    "5_noise_flip_rate_rep1_vs_rep2": round(sum(noise) / len(units), 4),
                    "5_excess_flip_rate": round((sum(shuf) - sum(noise)) / len(units), 4),
                    "5_excess_flip_p_sign": round(sign_test(per_q_flip.values()), 4),
                    "5_accuracy_change_p_sign": round(sign_test(per_q_acc.values()), 4),
                    "5_units": f"{len(units)} orderings in {len(per_q_flip)} questions"})
    else:
        out.update({key: (status if excluded else "pending — needs rep1 and rep2") for key in ROBUST_REPS})
    out["5_paraphrase"] = "not defined — no instruction text of ours"
    return out


if __name__ == "__main__":
    records = [criteria(slot) for slot in MODELS]
    fields = list(records[0])
    path = os.path.join(HERE, "criteria_awareness.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    for rec in records:
        print({k: v for k, v in rec.items() if not str(v).startswith(("pending", "not "))})
    print(f"wrote {path}")
