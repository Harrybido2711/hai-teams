#!/usr/bin/env python3
"""The six evaluation criteria for one model folder of NegotiationToM or EmoBench.

    python3 .claude/scripts/criteria_report.py neg   Interpersonal_processes_benchmarks/NegotiationToM/NEG_GPT_5.6_Luna
    python3 .claude/scripts/criteria_report.py emo   Interpersonal_processes_benchmarks/EmoBench/EMO_GPT_5.6_Luna

`neg` imports neg_eval_core, so it needs that core's dependencies (pandas, scikit-learn).

Reads only. 1 cost, 2 token efficiency, 3 latency and 6 reliability come from the main run — the
rows behind the workbook's number — and only when every row carries the per-row fields
(evaluation-criteria.md); a run made before its runner wrote them gets no estimate. 4 stability and
5 robustness come from the seven tagged runs, summarised only when all seven exist. Every rule for
those is in .claude/references/criteria-runs.md § Analysing it: content not letters, a failed row is
never an answer, all seven tags must hold the same items, and robustness is flips in excess of the
rep1-vs-rep2 baseline. Writes <folder>/criteria_summary.json beside the runs it summarised, and
prints the same numbers.

usage_criteria() and PRICES are also what AwareBench's aware_criteria.py computes 1, 2, 3 and 6
with, so those four are one algorithm and one price table on every benchmark.
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys
from datetime import datetime
from math import comb

REPS = ["rep1", "rep2", "rep3", "rep4", "rep5"]
PERTURBS = ["perturb-shuffle", "perturb-paraphrase"]
BOOTSTRAP = 2000


# ── criteria 1, 2, 3, 6: prices and the computation, shared with aware_criteria.py ───────────────

PRICE_READ = "2026-10-12"


def deepseek_price(ts):
    """DeepSeek's list price halves off-peak (peak: 01-04 and 06-10 UTC, Mon-Fri). Rows carry
    Quest-local timestamps (America/Chicago)."""
    from zoneinfo import ZoneInfo
    t = datetime.fromisoformat(ts).replace(tzinfo=ZoneInfo("America/Chicago")).astimezone(ZoneInfo("UTC"))
    peak = t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)
    return (0.30, 1.20) if peak else (0.15, 0.60)


# label -> (($/M in, $/M out) or a function of the row's timestamp, where it was read). A label is
# model@route because the same model is priced differently by another provider.
PRICES = {
    "gemini-3.5-flash-lite@openrouter": ((0.30, 2.50), "OpenRouter /models"),
    "gpt-5.6-luna@openai": ((0.20, 1.20), "model-parameters.md; OpenRouter agrees"),
    "qwen3.5-9b@together": ((0.17, 0.25), "Together /v1/models"),
    "gemma-4-31b-it@deepinfra": ((0.20, 0.40), "DeepInfra /models/google/gemma-4-31B-it"),
    "deepseek-flash@deepseek": (deepseek_price, "api-docs.deepseek.com pricing: 0.30/1.20 peak, half off-peak"),
}

# model folder -> price label, for the folders this script reads. An unlisted folder is refused
# rather than priced by guess.
FOLDER_PRICE = {
    "NEG_GPT_5.6_Luna": "gpt-5.6-luna@openai",
    "NEG_Gemini_Flash3.5lite_OpenRouter": "gemini-3.5-flash-lite@openrouter",
    "EMO_GPT_5.6_Luna": "gpt-5.6-luna@openai",
    "EMO_Gemini_Flash3.5lite_OpenRouter": "gemini-3.5-flash-lite@openrouter",
}

# Without these on every row, criteria 1-3 and 6 cannot be computed (evaluation-criteria.md).
ROW_FIELDS = ("prompt_tokens", "completion_tokens", "latency_s", "finish_reason", "n_attempts", "parse_ok")


class NotAvailable(Exception):
    """The main run cannot give criteria 1, 2, 3, 6; the message says why, and goes in the summary."""


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


def usage_problems(rows, price_label):
    """Why these rows cannot be priced and timed, or [] when they can. A key that is present but
    null is as unusable as a missing one: the runners write null for a call that answered but whose
    usage was not read, and pricing it as 0 would undercount with nothing to show for it."""
    problems = []
    lacking = {f: sum(f not in r for r in rows) for f in ROW_FIELDS}
    lacking = {f: n for f, n in lacking.items() if n}
    if lacking:
        problems.append(f"rows lack the per-row fields {lacking}; the runner did not write them")
    answered = [r for r in rows if str(r.get("raw") or "").strip()]
    unpriced = sum(not (isinstance(r.get("prompt_tokens"), int) and isinstance(r.get("completion_tokens"), int))
                   for r in answered)
    untimed = sum(not isinstance(r.get("latency_s"), (int, float)) for r in answered)
    if unpriced:
        problems.append(f"{unpriced} answered rows carry no integer token counts — billed but unpriceable")
    if untimed:
        problems.append(f"{untimed} answered rows carry no latency_s")
    if callable(PRICES[price_label][0]) and any("ts" not in r for r in rows):
        problems.append(f"{price_label} is priced by the hour and the rows carry no ts")
    if not answered:
        problems.append("no row has a response")
    return problems


def ratio(numerator, denominator, digits):
    return round(numerator / denominator, digits) if denominator else None


def wrong_over_right(rows):
    """Mean completion tokens on wrong rows over right ones; None when either side is empty."""
    right = [r["completion_tokens"] for r in rows if r["correct"] is not None and r["correct"]
             and isinstance(r.get("completion_tokens"), int)]
    wrong = [r["completion_tokens"] for r in rows if r["correct"] is not None and not r["correct"]
             and isinstance(r.get("completion_tokens"), int)]
    if not right or not wrong:
        return None
    return round(statistics.mean(wrong) / statistics.mean(right), 2)


def usage_criteria(rows, price_label):
    """Criteria 1, 2, 3 and 6 over one run's rows. Each row carries the per-row fields, the
    response text as `raw`, `parse_ok`, `correct` (0/1, or None for a row the benchmark does not
    score — billed, but neither right nor wrong) and `ts` where the price depends on the hour.
    Refuses, loudly, rows that usage_problems() rejects."""
    problems = usage_problems(rows, price_label)
    if problems:
        raise NotAvailable("; ".join(problems))
    price, source = PRICES[price_label]

    # 1 · cost — list price on the tokens each row recorded, cache discounts ignored. Calls that
    # left no row (timed-out attempts, the negotiate probe) are not in it.
    cost = 0.0
    for r in rows:
        p_in, p_out = price(r["ts"]) if callable(price) else price
        cost += ((r.get("prompt_tokens") or 0) * p_in + (r.get("completion_tokens") or 0) * p_out) / 1e6
    n_correct = sum(r["correct"] or 0 for r in rows)
    out = {"1_cost_usd_full_run": round(cost, 4),
           "1_usd_per_1k_correct": ratio(1000 * cost, n_correct, 4),
           "1_price_source": f"{source}, read {PRICE_READ}"}

    # 2 · token efficiency — completion tokens include reasoning; never counted from the text
    ct = [r["completion_tokens"] for r in rows if isinstance(r.get("completion_tokens"), int)]
    rt = [r["reasoning_tokens"] for r in rows if isinstance(r.get("reasoning_tokens"), int)]
    out.update({"2_output_tokens_mean": round(statistics.mean(ct), 1),
                "2_output_tokens_p95": pct(ct, 95),
                "2_reasoning_tokens_mean": round(statistics.mean(rt), 1) if rt else "not reported",
                "2_wrong_over_right_tokens": wrong_over_right(rows)})

    # 3 · latency — the successful call alone (SDK retries are off in every runner that writes it)
    lat = [r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float))]
    out.update({"3_latency_p50_s": pct(lat, 50), "3_latency_p95_s": pct(lat, 95)})

    # 6 · reliability
    n = len(rows)
    out.update({"6_parse_ok_rate": round(1 - sum(not r["parse_ok"] for r in rows) / n, 4),
                "6_empty_rate": round(sum(1 for r in rows if not str(r.get("raw") or "").strip()) / n, 4),
                "6_truncation_rate": round(sum(r.get("finish_reason") == "length" for r in rows) / n, 4),
                "6_retried_row_rate": round(sum((r.get("n_attempts") or 1) > 1 for r in rows) / n, 4)})
    return out


# ── correctness, one definition per benchmark for the main run and the tagged runs alike ─────────

def neg_core(folder):
    sys.path.insert(0, os.path.dirname(os.path.abspath(folder)))
    import neg_eval_core as core
    return core


def neg_parsed(row):
    """Whether the parser produced a prediction. A failed intention parse is stored as
    pred_intents == [] (bitmask 0), not None, so only parse_ok can tell it from a real empty set;
    rows from runners that do not write parse_ok fall back to the old signal."""
    if "parse_ok" in row:
        return bool(row["parse_ok"])
    return row.get("pred_intents") is not None or row.get("pred") is not None


def neg_correct(task, row):
    if task == "intention":
        # The reported metric is F1 over the set, so a row has no "correct" of its own: exact
        # bitmask match, the user's definition (criteria-runs.md § Analysing it). A failed parse
        # is wrong whatever its bitmask defaults to.
        return int(neg_parsed(row) and row["gold_bitmask"] == row["pred_bitmask"])
    return int(bool(row[f"{task}_em"]))


def emo_correct(task, row):
    if task == "EA":
        return int(str(row["label"]).strip().upper() == str(row["answer"]).strip().upper())
    return int(str(row["emo_label"]).strip().upper() == str(row["emo_answer"]).strip().upper()
               and str(row["cause_label"]).strip().upper() == str(row["cause_answer"]).strip().upper())


# ── the main run: (rows for usage_criteria, source files, set of configs seen) ────────────────────

EMO_ROWS = 200     # English items per task — emobench.md; the data files hold both languages


def one_file(pattern):
    files = glob.glob(pattern)
    if len(files) != 1:
        raise NotAvailable(f"{pattern}: expected one file, found {len(files)}")
    return files[0]


def main_neg(folder):
    core = neg_core(folder)
    rows, sources = [], []
    for task in ("desire", "belief", "intention"):
        path = one_file(os.path.join(folder, "results", task, "*_all.jsonl"))
        task_rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if len({r["uid"] for r in task_rows}) != len(task_rows):
            raise NotAvailable(f"{path}: duplicate uids — the merge is not clean")
        scored = {r["uid"] for r in core.scorable(task, task_rows)[0]}
        rows += [dict(r, task=task, raw=r.get("raw_response"),
                      correct=neg_correct(task, r) if r["uid"] in scored else None) for r in task_rows]
        sources.append(os.path.relpath(path, folder))
    # One config means every shard's saved params agree AND no shard was restarted under another:
    # an untagged shard rewrites its params file on each start, so the logs keep the earlier ones.
    params = sorted(glob.glob(os.path.join(folder, "results", "negotiated_params_shard*of*.json")))
    accepted = {line.split("accepted:", 1)[1].strip()
                for log in glob.glob(os.path.join(folder, "log_shard*.txt"))
                for line in open(log, encoding="utf-8", errors="replace") if "accepted:" in line}
    one_config = bool(params) and len({json.dumps(json.load(open(p)), sort_keys=True) for p in params}) == 1 \
        and len(accepted) <= 1
    return rows, sources, one_config


def main_emo(folder):
    rows, sources = [], []
    for task in ("EU", "EA"):
        path = one_file(os.path.join(folder, "results", task, "*_en.jsonl"))
        task_rows = [json.loads(line) for line in open(path, encoding="utf-8")]
        if len({str(r["qid"]) for r in task_rows}) != len(task_rows):
            raise NotAvailable(f"{path}: duplicate qids")
        if len(task_rows) != EMO_ROWS:
            raise NotAvailable(f"{path}: {len(task_rows)} rows, not {EMO_ROWS} — the run is not whole")
        rows += [dict(r, task=task, raw=r.get("model_response"), correct=emo_correct(task, r)) for r in task_rows]
        sources.append(os.path.relpath(path, folder))
    configs = {r.get("params") for r in rows}
    return rows, sources, len(configs) == 1 and None not in configs


# ── the tagged runs: one dict per task, {item key: (content or None, correct 0/1, cluster)} ──────

def tag_dir(benchmark, folder, tag):
    return os.path.join(folder, "results", tag) if benchmark == "neg" else os.path.join(folder, f"results_{tag}")


def load_neg(folder):
    core = neg_core(folder)

    def content(task, row):
        if not (row.get("raw_response") or "").strip() or not neg_parsed(row):
            return None
        if task == "intention":
            return row["pred_bitmask"] if row.get("pred_intents") is not None else None
        if row.get("pred") is None:
            return None
        return tuple(core.pred_item(row["pred"], k) for k in ("high", "medium", "low"))

    runs = {}
    for tag in REPS + PERTURBS:
        runs[tag] = {}
        for task in ("desire", "belief", "intention"):
            files = glob.glob(os.path.join(folder, "results", tag, task, "*.jsonl"))
            if len(files) != 1:
                raise SystemExit(f"{tag}/{task}: expected one jsonl, found {len(files)}")
            rows = [json.loads(line) for line in open(files[0], encoding="utf-8")]
            scored, _ = core.scorable(task, rows)
            runs[tag][task] = {r["uid"]: (content(task, r), neg_correct(task, r),
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
                else:
                    pair = (r.get("emo_answer_text"), r.get("cause_answer_text"))
                    text = None if None in pair else pair
                out[str(r["qid"])] = (text, emo_correct(task, r), str(r["qid"]))     # each item its own cluster
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
    ap.add_argument("folder", help="the model folder: its main run, and its seven tagged runs if they exist")
    args = ap.parse_args()
    name = os.path.basename(os.path.normpath(args.folder))

    # 1, 2, 3, 6 — a main run that cannot give them says why; it never stops 4 and 5 below
    try:
        if name not in FOLDER_PRICE:
            raise NotAvailable(f"{name} has no price label — add it to FOLDER_PRICE with where it was read")
        rows, sources, one_config = (main_neg if args.benchmark == "neg" else main_emo)(args.folder)
        main_run = {"rows": len(rows), "scored_rows": sum(r["correct"] is not None for r in rows),
                    "sources": sources, **usage_criteria(rows, FOLDER_PRICE[name]),
                    "6_one_config": int(one_config)}
        # Pooled over tasks, a ratio is dominated by whichever task has most wrong rows; kept per
        # task beside it until the user decides which one is the criterion.
        main_run["2_wrong_over_right_tokens_by_task"] = {
            t: wrong_over_right([r for r in rows if r["task"] == t]) for t in dict.fromkeys(r["task"] for r in rows)}
    except NotAvailable as why:
        main_run = {"status": f"not available — {why}; no estimate is made"}

    present = [t for t in REPS + PERTURBS if os.path.isdir(tag_dir(args.benchmark, args.folder, t))]
    summary = {"folder": args.folder,
               "definitions": "evaluation-criteria.md (1, 2, 3, 6 from the main run); "
                              "criteria-runs.md § Analysing it (4, 5; seed pinned in every rep)",
               "main_run": main_run}
    if len(present) == len(REPS + PERTURBS):
        runs = (load_neg if args.benchmark == "neg" else load_emo)(args.folder)
        summary["tasks"] = {t: summarise({tag: runs[tag][t] for tag in runs}) for t in runs["rep1"]}
    else:
        missing = [t for t in REPS + PERTURBS if t not in present]
        summary["tasks"] = f"pending — {len(present)}/7 tagged runs; missing {missing}"

    path = os.path.join(args.folder, "criteria_summary.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"== main run: {json.dumps(main_run, indent=1)}")
    if isinstance(summary["tasks"], str):
        print(f"\n== 4 and 5: {summary['tasks']}")
    else:
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
