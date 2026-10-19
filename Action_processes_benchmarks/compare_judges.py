"""Compare the four candidate judges on what they produced in the judge pilot. Offline: reads the
judge rows on disk, calls nothing, rerunnable at will.

The advisor's rule (2026-10-19): if the judges' results are about the same, use the cheap one; if
they differ a lot and the expensive one is better, use the expensive one. So per judge this puts
side by side the inputs that rule needs — it does not make the decision:
  1. reliability   refusals, parse failures, failed calls (a judge that cannot answer is out)
  2. cost          OpenRouter's billed `cost`, summed over every call ever made; latency
  3. stability     flip rate across the replicates
  4. GPT's score   under each judge — the number the choice would change
  5. agreement     between judges, pairwise, on the same units (chance-corrected)
  6. vs humans     Wonderbread's anchor sets: agreement with HUMAN labels — the only place "better"
                   is measured rather than inferred. MultiChallenge has no human labels on disk.

How labels and headlines are built, so no judge is flattered by what it failed on:
- a unit's label needs a STRICT majority of its replicates (2 of 3); otherwise it is unlabelled;
- every headline is computed on the COMMON units — those labelled by every judge that labelled at
  least half of them — and shows n. A judge under half is listed as "insufficient coverage";
- a judge row whose `judged_sha` does not match the text on disk now (an answer regenerated after
  it was judged) is dropped and counted, never matched to the new text.

    python compare_judges.py            # results/ -> judge_comparison/
    python compare_judges.py --smoke    # smoke/  -> judge_comparison/smoke/
"""

import argparse
import collections
import itertools
import json
import math
import os
import statistics
import sys

GROUP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GROUP_ROOT)
sys.path.insert(0, os.path.join(GROUP_ROOT, "Multi-challenge_bench"))
sys.path.insert(0, os.path.join(GROUP_ROOT, "Wonderbread_bench"))
import judge_core as jc  # noqa: E402
import mc_eval_core as mc  # noqa: E402
import wb_eval_core as wb  # noqa: E402

MC_ROOT = os.path.join(GROUP_ROOT, "Multi-challenge_bench")
WB_ROOT = os.path.join(GROUP_ROOT, "Wonderbread_bench")
MIN_COVERAGE = 0.5


# ---------------------------------------------------------------- statistics, stdlib only


def cohen_kappa(a, b, weights=None, categories=None):
    """Cohen's κ for paired labels; `weights="quadratic"` for an ordinal scale. None if undefined."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    cats = sorted(categories or {v for p in pairs for v in p})
    if len(cats) < 2:
        return None
    k, n = len(cats), len(pairs)
    idx = {c: i for i, c in enumerate(cats)}

    def w(i, j):
        if weights == "quadratic":
            return ((i - j) / (k - 1)) ** 2
        return 0.0 if i == j else 1.0
    obs = collections.Counter((idx[x], idx[y]) for x, y in pairs)
    ra = collections.Counter(idx[x] for x, _ in pairs)
    rb = collections.Counter(idx[y] for _, y in pairs)
    observed = sum(w(i, j) * c for (i, j), c in obs.items()) / n
    expected = sum(w(i, j) * ra[i] * rb[j] for i in range(k) for j in range(k)) / (n * n)
    return None if expected == 0 else 1 - observed / expected


def agreement(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return (sum(x == y for x, y in pairs) / len(pairs)) if pairs else None


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if sx == 0 or sy == 0 else sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return pearson(_ranks(list(xs)), _ranks(list(ys)))


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_p(b, c):
    """Exact two-sided McNemar p from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean_or_none(values):
    v = [x for x in values if x is not None]
    return statistics.mean(v) if v else None


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


# ---------------------------------------------------------------- labels


def qa_label(values):
    """A QA score from its replicates (one entry each, None where unusable): the median when every
    replicate answered; with some missing, the value a strict majority agrees on; else None. The
    lower median of [1, 3] would lean towards "best" — it is never taken."""
    got = [v for v in values if v is not None]
    if got and len(got) == len(values):
        return statistics.median_low(got)
    return mc.majority(values)


def per_rep(rows, unit, value, n_reps):
    """{unit: [value of rep 0, rep 1, ...]} with None for a rep that is missing or not ok."""
    out = collections.defaultdict(lambda: [None] * n_reps)
    for r in rows:
        if r["status"] == "ok" and r["rep"] < n_reps:
            out[unit(r)][r["rep"]] = value(r)
    return out


def flip_rate(reps):
    multi = [[v for v in vals if v is not None] for vals in reps.values()]
    multi = [v for v in multi if len(v) >= 2]
    return (sum(len(set(v)) > 1 for v in multi) / len(multi)) if multi else None


def common_units(labels, universe):
    """Units labelled by every judge covering at least MIN_COVERAGE of `universe`; and who covers."""
    covering = [j for j, lab in labels.items()
                if universe and sum(lab.get(u) is not None for u in universe) / len(universe) >= MIN_COVERAGE]
    common = [u for u in universe if covering and all(labels[j].get(u) is not None for j in covering)]
    return common, covering


# ---------------------------------------------------------------- loading


def load_all(path):
    """Every row ever written, superseded attempts included — for cost."""
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    return rows


def load_current(path, expected_sha, key):
    """Last row per uid, keeping only rows whose judged text matches the text on disk now."""
    rows = jc.read_rows(path)
    kept = [r for r in rows if expected_sha.get(key(r)) is not None
            and r.get("judged_sha") == expected_sha[key(r)]]
    return kept, len(rows) - len(kept)


def usage_row(judge, label, path, units, stale):
    every = load_all(path)
    current = jc.read_rows(path)
    called = [r for r in current if r["status"] != "skipped"]
    st = collections.Counter(r["status"] for r in current)
    cost = sum(r.get("cost") or 0 for r in every)
    lat = sorted(r["latency_s"] for r in current if r.get("latency_s") is not None)
    backends = collections.Counter(r.get("backend") for r in called if r.get("backend"))
    return {"judge": judge, "set": label, "rows": len(current), "ok": st.get("ok", 0),
            "refused": st.get("refused", 0), "parse_fail": st.get("parse_fail", 0),
            "failed": st.get("failed", 0), "skipped": st.get("skipped", 0), "stale": stale,
            "calls": sum(r.get("n_attempts") or 0 for r in every),
            "cost_usd": cost, "cost_per_unit": (cost / units) if units else None,
            "latency_p50": lat[len(lat) // 2] if lat else None,
            "latency_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
            "backends": dict(backends)}


def gpt_rows(sub, *parts):
    return {r["uid"]: r for r in jc.read_rows(os.path.join(*parts[:-1], sub, parts[-1]))
            if r["status"] == "ok"}


# ---------------------------------------------------------------- MultiChallenge


def mc_section(sub, judges):
    responses = gpt_rows(sub, MC_ROOT, "MC_GPT_5.6_Luna", f"{jc.GEN_MODEL}.jsonl")
    sha = {q: jc.text_sha(r["response"]) for q, r in responses.items()}
    labels, flips, usage, per_rep_macro = {}, {}, [], {}
    for j in judges:
        path = os.path.join(MC_ROOT, "MC_Judges", sub, j, f"{jc.GEN_MODEL}.jsonl")
        rows, stale = load_current(path, sha, lambda r: r["qid"])
        if not rows:
            continue
        reps = per_rep(rows, lambda r: r["qid"], lambda r: r["passed"], jc.N_REPS)
        labels[j] = {q: mc.majority(v) for q, v in reps.items()}
        flips[j] = flip_rate(reps)
        per_rep_macro[j] = [mc.score({q: v[k] for q, v in reps.items()})["overall_macro"]
                            for k in range(jc.N_REPS)]
        usage.append(usage_row(j, "mc/sample", path, len(responses), stale))
    universe = sorted(responses)
    common, covering = common_units(labels, universe)
    out = []
    for j in labels:
        # every sampled item goes in, the non-common ones as None: an axis left with no common item
        # then shows up in axes_missing and voids the macro, instead of vanishing from its input
        s = mc.score({q: (labels[j][q] if q in common else None) for q in universe}) if j in covering else None
        own = mc.score({q: labels[j].get(q) for q in universe})
        k = sum(1 for q in common if labels[j].get(q)) if s else 0
        lo, hi = wilson(k, len(common)) if s else (None, None)
        out.append({"judge": j, "covering": j in covering, "n_common": len(common),
                    "n_own": own["n_judged"], "macro": s and s["overall_macro"],
                    "micro": s and s["overall_micro"], "micro_ci": (lo, hi),
                    "axes": s and s["axis_scores"], "axes_missing": s and s["axes_missing"],
                    "per_rep_macro": per_rep_macro[j], "flip": flips[j]})
    pairs = []
    for a, b in itertools.combinations(sorted(labels), 2):
        both = [q for q in universe if labels[a].get(q) is not None and labels[b].get(q) is not None]
        x = [labels[a][q] for q in both]
        y = [labels[b][q] for q in both]
        only_a = sum(1 for p, q in zip(x, y) if p and not q)
        only_b = sum(1 for p, q in zip(x, y) if q and not p)
        pairs.append({"a": a, "b": b, "n": len(both), "agree": agreement(x, y),
                      "kappa": cohen_kappa(x, y, categories=[False, True]),
                      "a_only_pass": only_a, "b_only_pass": only_b, "mcnemar_p": mcnemar_p(only_a, only_b)})
    return out, pairs, usage


# ---------------------------------------------------------------- Wonderbread QA


def qa_inputs(sub, producer):
    if producer == wb.ANCHOR_PRODUCER:
        a = wb.load_anchor_qa()
        return {u: jc.text_sha(x["question"], x["human_label"], x["response"]) for u, x in a.items()}
    g = gpt_rows(sub, WB_ROOT, "WB_GPT_5.6_Luna", f"qa/{jc.GEN_MODEL}.jsonl")
    return {u: jc.text_sha(r["question"], r["human_label"], r["response"]) for u, r in g.items()}


def qa_section(sub, judges, producer):
    sha = qa_inputs(sub, producer)
    labels, flips, usage = {}, {}, []
    for j in judges:
        path = os.path.join(WB_ROOT, "WB_Judges", sub, j, "qa", f"{producer}.jsonl")
        rows, stale = load_current(path, sha, lambda r: r["item"])
        if not rows:
            continue
        reps = per_rep(rows, lambda r: (r["item"], r["criterion"]), lambda r: r["score"], jc.N_REPS)
        labels[j] = {k: qa_label(v) for k, v in reps.items()}
        flips[j] = flip_rate(reps)
        usage.append(usage_row(j, f"qa/{producer}", path, len(sha), stale))
    out = {j: {"judge": j, "flip": flips[j], "means": {}, "n": {}, "covering": {}} for j in labels}
    for c in wb.QA_CRITERIA:
        universe = [(u, c) for u in sorted(sha)]
        common, covering = common_units(labels, universe)
        for j in labels:
            out[j]["covering"][c] = j in covering
            out[j]["n"][c] = len(common)
            out[j]["means"][c] = mean_or_none(labels[j][k] for k in common) if j in covering else None
    pairs = []
    for a, b in itertools.combinations(sorted(labels), 2):
        for c in wb.QA_CRITERIA:
            keys = [(u, c) for u in sorted(sha) if labels[a].get((u, c)) is not None
                    and labels[b].get((u, c)) is not None]
            x = [labels[a][k] for k in keys]
            y = [labels[b][k] for k in keys]
            pairs.append({"a": a, "b": b, "criterion": c, "n": len(keys), "agree": agreement(x, y),
                          "qwk": cohen_kappa(x, y, "quadratic", [1, 2, 3])})
    return list(out.values()), pairs, usage, labels


def qa_vs_human(labels):
    anchor = wb.load_anchor_qa()
    rows = []
    for c in wb.QA_CRITERIA:
        universe = [(u, c) for u in sorted(anchor)]
        common, covering = common_units(labels, universe)
        sources = {"upstream gpt-4 (reference)": {(u, c): int(anchor[u]["upstream_gpt4"][c]) for u in anchor}}
        sources.update({j: labels[j] for j in covering})
        for name, lab in sources.items():
            h = [int(anchor[u]["human"][c]) for u, _ in common]
            p = [lab[k] for k in common]
            rows.append({"judge": name, "criterion": c, "n": len(common), "exact": agreement(h, p),
                         "qwk": cohen_kappa(h, p, "quadratic", [1, 2, 3]), "spearman": spearman(h, p),
                         "human_mean": mean_or_none(h), "judge_mean": mean_or_none(p)})
        for j in labels:
            if j not in covering:
                rows.append({"judge": j, "criterion": c, "n": 0, "exact": None, "qwk": None,
                             "spearman": None, "human_mean": None, "judge_mean": None,
                             "note": "insufficient coverage"})
    return rows


# ---------------------------------------------------------------- Wonderbread SOP


def sop_inputs(sub, producer):
    """{pair uid: (pred lines, gold lines, sha)}"""
    if producer == wb.ANCHOR_PRODUCER:
        src = {u: (a["pred_sop"], a["gold_sop"]) for u, a in wb.load_anchor_sop().items()}
    else:
        g = gpt_rows(sub, WB_ROOT, "WB_GPT_5.6_Luna", f"sop/{jc.GEN_MODEL}.jsonl")
        src = {u: (r["pred_sop"], r["gold_sop"]) for u, r in g.items()}
    return {u: (wb.sop_lines(p), wb.sop_lines(g), jc.text_sha(p, g)) for u, (p, g) in src.items()}


def sop_pair_metrics(u, n_pred, n_gold, match, index):
    """Precision, recall, F1 from the majority MATCH decision of each line — replicates that agree a
    line matched but name different lines still agree on what precision and recall count. Ordering
    needs the matched line's index, so it comes from the majority INDEX, and is None for the pair
    when any line lacks one. Both through upstream's scorer (wb.sop_metrics)."""
    pm = [match.get((u, "pred", i)) for i in range(n_pred)]
    gm = [match.get((u, "gold", k)) for k in range(n_gold)]
    if None in pm or None in gm or not pm or not gm:
        return None
    m = wb.sop_metrics([0 if x else -1 for x in pm], [0 if x else -1 for x in gm])
    pi = [index.get((u, "pred", i)) for i in range(n_pred)]
    gi = [index.get((u, "gold", k)) for k in range(n_gold)]
    full = wb.sop_metrics(pi, gi)
    m["ordering"] = full["ordering"] if full else None
    return m


def sop_section(sub, judges, producer, group_of):
    units = sop_inputs(sub, producer)
    sha = {u: v[2] for u, v in units.items()}
    metrics, line_labels, flips, usage = {}, {}, {}, []
    for j in judges:
        path = os.path.join(WB_ROOT, "WB_Judges", sub, j, "sop", f"{producer}.jsonl")
        rows, stale = load_current(path, sha, lambda r: r["pair"])
        if not rows:
            continue
        reps = per_rep(rows, lambda r: (r["pair"], r["direction"], r["line"]), lambda r: r["index"], jc.N_REPS)
        matched = {k: [None if x is None else x != -1 for x in v] for k, v in reps.items()}
        lab = {k: mc.majority(v) for k, v in matched.items()}     # strict-majority match per line
        index = {k: mc.majority(v) for k, v in reps.items()}       # strict-majority index per line
        line_labels[j] = lab
        flips[j] = flip_rate(matched)
        metrics[j] = {u: sop_pair_metrics(u, len(p), len(g), lab, index) for u, (p, g, _) in units.items()}
        usage.append(usage_row(j, f"sop/{producer}", path, len(units), stale))
    universe = sorted(units)
    common, covering = common_units(metrics, universe)
    out = []
    for j in metrics:
        groups = collections.defaultdict(list)
        for u in common:
            groups[group_of(u)].append(metrics[j][u])
        summary = {g: {k: mean_or_none(m[k] for m in ms) for k in ("precision", "recall", "f1", "ordering")}
                   | {"pairs": len(ms)} for g, ms in sorted(groups.items())} if j in covering else {}
        out.append({"judge": j, "covering": j in covering, "groups": summary, "n_common": len(common),
                    "n_own": sum(1 for u in universe if metrics[j].get(u)), "flip": flips[j]})
    agree = []
    for a, b in itertools.combinations(sorted(line_labels), 2):
        keys = [k for k in line_labels[a] if line_labels[a][k] is not None
                and line_labels[b].get(k) is not None]
        x = [line_labels[a][k] != -1 for k in keys]
        y = [line_labels[b][k] != -1 for k in keys]
        agree.append({"a": a, "b": b, "n_lines": len(keys), "agree": agreement(x, y),
                      "kappa": cohen_kappa(x, y, categories=[False, True])})
    return out, agree, usage, metrics, common, covering


def sop_vs_human(metrics, common, covering):
    anchor = wb.load_anchor_sop()
    rows = []
    sources = {"upstream gpt-4 (reference)": {u: a["upstream_gpt4"] for u, a in anchor.items()}}
    sources.update({j: metrics[j] for j in covering})
    for name, m in sources.items():
        for key in ("precision", "recall"):
            h = [float(anchor[u]["human"][key]) for u in common]
            p = [float(m[u][key]) for u in common]
            rows.append({"judge": name, "metric": key, "n": len(common),
                         "mae": mean_or_none(abs(x - y) for x, y in zip(h, p)),
                         "pearson": pearson(h, p), "spearman": spearman(h, p),
                         "human_mean": mean_or_none(h), "judge_mean": mean_or_none(p)})
    return rows


# ---------------------------------------------------------------- report


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="read smoke/ instead of results/")
    args = ap.parse_args()
    sub = "smoke" if args.smoke else "results"
    judges = sorted(jc.JUDGES)
    out_dir = os.path.join(GROUP_ROOT, "judge_comparison", *(["smoke"] if args.smoke else []))
    os.makedirs(out_dir, exist_ok=True)
    md, usage, data = [], [], {}
    gap = "insufficient coverage"

    md.append(f"# Judge pilot — four candidate judges compared\n\nSource: `{sub}/` rows. Judged "
              f"model: `{jc.GEN_MODEL}`. Judge settings: `judge_core.JUDGES` (prices read "
              f"{jc.PRICES_READ}). {jc.N_REPS} replicates; a unit's label needs a strict majority of "
              f"them (QA: the median when all answered). **Every headline is on the common units** — "
              f"labelled by every judge covering at least {MIN_COVERAGE:.0%} of them — and shows n.\n")

    mc_rows, mc_pairs, u = mc_section(sub, judges)
    usage += u
    data["multichallenge"] = {"scores": mc_rows, "pairs": mc_pairs}
    md.append("## MultiChallenge — GPT's pass rate under each judge\n\nUpstream headline = macro "
              "mean over the four axes (blank if an axis has no common item). Wilson 95% interval on "
              "the micro rate.\n")
    md.append(table(["judge", "n common", "macro", "micro", "micro 95% CI", "per-rep macro (own items)",
                     "flip rate", "labelled (own)"],
                    [[r["judge"], r["n_common"] if r["covering"] else gap, num(r["macro"], 1),
                      num(r["micro"], 1), f"{pct(r['micro_ci'][0], 0)}–{pct(r['micro_ci'][1], 0)}",
                      ", ".join(num(x, 1) for x in r["per_rep_macro"]), pct(r["flip"]), r["n_own"]]
                     for r in mc_rows]))
    md.append("\nPairwise agreement on the majority verdict, on the items both labelled (McNemar "
              "exact p on the discordant items):\n")
    md.append(table(["judge A", "judge B", "n", "agree", "κ", "only A passes", "only B passes", "p"],
                    [[p["a"], p["b"], p["n"], pct(p["agree"]), num(p["kappa"], 2), p["a_only_pass"],
                      p["b_only_pass"], num(p["mcnemar_p"], 3)] for p in mc_pairs]))

    for producer, title in ((jc.GEN_MODEL, "GPT's answers"), (wb.ANCHOR_PRODUCER, "anchor set")):
        rows, pairs, u, labels = qa_section(sub, judges, producer)
        usage += u
        data[f"wonderbread_qa/{producer}"] = {"scores": rows, "pairs": pairs}
        md.append(f"\n## Wonderbread QA — {title}: mean score per criterion (1 = best, 3 = worst)\n")
        md.append(table(["judge"] + [f"{c} (n)" for c in wb.QA_CRITERIA] + ["flip rate"],
                        [[r["judge"]] + [f"{num(r['means'][c], 2)} ({r['n'][c]})" if r["covering"][c] else gap
                                         for c in wb.QA_CRITERIA] + [pct(r["flip"])] for r in rows]))
        if producer == wb.ANCHOR_PRODUCER and labels:
            hv = qa_vs_human(labels)
            data["wonderbread_qa/vs_human"] = hv
            md.append("\n**Against the human labels** (quadratic-weighted κ; upstream's GPT-4 judge "
                      "as the reference row, on the same items):\n")
            md.append(table(["judge", "criterion", "n", "exact", "QWK", "Spearman", "human mean", "judge mean"],
                            [[r["judge"], r["criterion"], r["n"], pct(r["exact"]), num(r["qwk"], 2),
                              num(r["spearman"], 2), num(r["human_mean"], 2),
                              r.get("note") or num(r["judge_mean"], 2)] for r in hv]))
        md.append("\nPairwise agreement per criterion:\n")
        md.append(table(["judge A", "judge B", "criterion", "n", "exact", "QWK"],
                        [[p["a"], p["b"], p["criterion"], p["n"], pct(p["agree"]), num(p["qwk"], 2)]
                         for p in pairs]))

    for producer, group_of, title in ((jc.GEN_MODEL, lambda u: u.split("|")[1], "GPT's SOPs, by ablation"),
                                      (wb.ANCHOR_PRODUCER, lambda u: "all", "anchor set")):
        rows, agree, u, metrics, common, covering = sop_section(sub, judges, producer, group_of)
        usage += u
        data[f"wonderbread_sop/{producer}"] = {"scores": rows, "pairs": agree}
        md.append(f"\n## Wonderbread SOP Generation — {title}\n")
        md.append(table(["judge", "group", "pairs", "precision", "recall", "F1", "ordering", "flip rate",
                         "scored (own)"],
                        [[r["judge"], g, m["pairs"], num(m["precision"]), num(m["recall"]), num(m["f1"]),
                          num(m["ordering"]), pct(r["flip"]), r["n_own"]]
                         for r in rows for g, m in r["groups"].items()]
                        + [[r["judge"], gap, "—", "—", "—", "—", "—", pct(r["flip"]), r["n_own"]]
                           for r in rows if not r["covering"]]))
        md.append("\nPairwise agreement on line-match decisions (lines both labelled):\n")
        md.append(table(["judge A", "judge B", "lines", "agree", "κ"],
                        [[p["a"], p["b"], p["n_lines"], pct(p["agree"]), num(p["kappa"], 2)] for p in agree]))
        if producer == wb.ANCHOR_PRODUCER and metrics:
            hv = sop_vs_human(metrics, common, covering)
            data["wonderbread_sop/vs_human"] = hv
            md.append("\n**Against the human precision / recall** (common pairs):\n")
            md.append(table(["judge", "metric", "n", "MAE", "Pearson", "Spearman", "human mean", "judge mean"],
                            [[r["judge"], r["metric"], r["n"], num(r["mae"]), num(r["pearson"], 2),
                              num(r["spearman"], 2), num(r["human_mean"]), num(r["judge_mean"])] for r in hv]))

    md.append("\n## Reliability and cost — every judge, every set\n\nCost is OpenRouter's billed "
              "`cost`, summed over every call ever made (re-asks and superseded rows included). A "
              "refusal is the model's content filter answering instead of the judge; it is never "
              "scored. `stale` = rows dropped because the text they judged has since changed.\n")
    md.append(table(["judge", "set", "rows", "ok", "refused", "parse fail", "failed", "skipped", "stale",
                     "calls", "cost $", "$ / unit", "p50 s", "p95 s", "backends"],
                    [[r["judge"], r["set"], r["rows"], r["ok"], r["refused"], r["parse_fail"], r["failed"],
                      r["skipped"], r["stale"], r["calls"], num(r["cost_usd"], 4), num(r["cost_per_unit"], 5),
                      num(r["latency_p50"], 1), num(r["latency_p95"], 1),
                      ", ".join(f"{k} {v}" for k, v in r["backends"].items())] for r in usage]))
    totals = collections.defaultdict(float)
    for r in usage:
        totals[r["judge"]] += r["cost_usd"]
    md.append("\n" + table(["judge", "backend", "total cost $", "list price in / out per M"],
                           [[j, jc.JUDGES[j]["provider"], num(totals.get(j, 0.0), 4),
                             f"{jc.JUDGES[j]['price_in']} / {jc.JUDGES[j]['price_out']}"] for j in judges]))
    data["usage"] = usage

    with open(os.path.join(out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(data, fh, indent=1, default=str)
    print("\n".join(md))
    print(f"\nwritten: {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
