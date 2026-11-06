"""Router headroom on the basis Results/results.db reports. Reads stored rows and the database only; calls no model.

Usage: /opt/miniconda3/bin/python3 -I Router_research/router_reported.py [repo root]
Writes reported.json beside this file. Method and results: ROUTER_RESEARCH.md.

router_headroom.py weights every row alike. This script weights each row so that a model's weighted
mean equals its reported score, and stops if it does not:
  bbh, MMLU        equal weight per task (macro)
  EmoBench         equal weight EU, EA
  AwareBench       capability, mission (explicit, implicit), emotion, culture, 2nd-order perspective equal;
                   permutations collapsed per question after exact dedup; the control tasks weigh 0
  NegotiationToM   equal weight desire, belief, intention. Desire and belief equal the reported columns.
                   Intention is exact-set match per row: F1 and All_EM cannot be scored per row, so this
                   one term is not a database number.
"""
import collections, json, os, sqlite3, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True
import router_headroom as rh

ROOT = rh.ROOT
M, BENCHES = rh.M, ["bbh", "mmlu", "emo", "neg", "aware"]
COL = dict(gemini="Gemini", luna="OpenAI", qwen="Qwen", gemma="Gemma", deepseek="Deepseek")
DBNAME = dict(bbh="BBH", mmlu="MMLU", emo="EmoBench", neg="NegotiationToM", aware="AwareBench", docvqa="DocVQA")
OVERALL = {"bbh": "Overall Score (macro-avg of 20 tasks)", "mmlu": "Overall Score (macro-avg of 13 tasks)",
           "emo": "Overall Score", "aware": "Overall Score", "docvqa": "Overall accuracy"}
AWARE_DIM = {"capability": .2, "mission_explicit": .1, "mission_implicit": .1, "emotion": .2, "culture": .2,
             "perspective_story_2nd": .2}
db = sqlite3.connect(os.path.join(ROOT, "Results", "results.db"))


def score(bench, sub, model):
    row = db.execute("SELECT value FROM final_scores WHERE benchmark=? AND subbench=? AND model_column=?",
                     (DBNAME[bench], sub, COL[model])).fetchone()
    return row[0] if row else None


def aware_weights():
    """uid -> weight. Same dedup and permutation collapse as aware_eval_core's write-up."""
    path = rh.P("Transition_processes_benchmarks/Awareness_in_LLM/AWARE_GPT_5.6_Luna/results/%s/gpt-5_6-luna.jsonl")
    weights = {}
    for task, dim in AWARE_DIM.items():
        seen, by_q = set(), collections.defaultdict(list)
        for r in rh.jl(path % task):
            key = (r["question_id"], json.dumps(r.get("choices"), sort_keys=False), r["gold"], r.get("story"))
            if key not in seen:
                seen.add(key)
                by_q[r["question_id"]].append(r["uid"])
        for uids in by_q.values():
            for uid in uids:
                weights[uid] = dim / (len(by_q) * len(uids))
    return weights


def weights_for(bench, items):
    if bench == "aware":
        table = aware_weights()
        return np.array([table.get(it["key"][1], 0.0) for it in items])
    counts = collections.Counter(it["sub"] for it in items)
    return np.array([1 / (len(counts) * counts[it["sub"]]) for it in items])


def wmean(values, w):
    return float((values * w).sum() / w.sum())


def main():
    data = {"bbh": rh.load_bbh(), "mmlu": rh.load_mmlu(), "emo": rh.load_emo(), "neg": rh.load_neg(),
            "aware": rh.load_aware(), "docvqa": rh.load_docvqa()}
    out, parts = {"benchmarks": {}}, []
    for b, rows in data.items():
        ms = M if b != "docvqa" else M[:4]
        items = rh.complete(rows, ms)
        w = weights_for(b, items)
        S = np.array([[it["s"][m] for m in ms] for it in items])
        means = {m: wmean(S[:, j], w) for j, m in enumerate(ms)}
        if b in OVERALL:  # the guarantee: weighted rows reproduce the database, or nothing is reported
            for m in ms:
                gap = abs(means[m] - score(b, OVERALL[b], m))
                assert gap < 6e-4, (b, m, means[m], score(b, OVERALL[b], m))
        else:
            for m, j in zip(ms, range(len(ms))):
                for sub, col in (("neg_desire", "Desire_EM"), ("neg_belief", "Belief_EM")):
                    mask = np.array([it["sub"] == sub for it in items])
                    assert abs(S[mask, j].mean() - score(b, col, m)) < 6e-4, (m, col)
        live = w > 0
        contested = (S.min(1) != S.max(1))
        out["benchmarks"][b] = dict(
            rows=len(items), rows_weighted=int(live.sum()), means=means, best=max(means, key=means.get),
            oracle=wmean(S.max(1), w), contested_reported=wmean(contested.astype(float), w),
            contested_rows=float(contested.mean()))
        print(b, {k: round(v, 4) for k, v in means.items()}, "oracle", round(out["benchmarks"][b]["oracle"], 4))
        if b != "docvqa":
            for it, wt in zip(items, w):
                it["bench"], it["w"] = b, wt
            parts += items

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    S = np.array([[it["s"][m] for m in M] for it in parts])
    B = np.array([it["bench"] for it in parts])
    SUB = np.array([it["sub"] for it in parts])
    G = np.array([it["group"] for it in parts])
    W = np.array([it["w"] for it in parts])
    X = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit_transform(
        [it["text"][:4000] for it in parts])

    def macro(hit):
        return float(np.mean([wmean(hit[B == b], W[B == b]) for b in BENCHES]))

    single = np.array([macro(S[:, j]) for j in range(5)])
    prior = 1e-6 * single
    names = ["bench_table", "subtask_table", "knn10", "knn50", "knn100", "logreg"]
    ch = {k: np.zeros(len(S), int) for k in names}
    for tr, te in GroupKFold(5).split(X, groups=G):
        for key, lab in (("bench_table", B), ("subtask_table", SUB)):
            best = {v: int(((S[tr][lab[tr] == v] * W[tr][lab[tr] == v, None]).sum(0) + prior).argmax())
                    for v in np.unique(lab[tr])}
            ch[key][te] = [best.get(v, int(single.argmax())) for v in lab[te]]
        for k in (10, 50, 100):
            res = np.zeros(len(te), int)
            for s0 in range(0, len(te), 2000):
                sims = (X[te[s0:s0 + 2000]] @ X[tr].T).toarray()
                nn = np.argpartition(-sims, k, axis=1)[:, :k]
                res[s0:s0 + 2000] = (S[tr][nn].mean(1) + prior).argmax(1)
            ch[f"knn{k}"][te] = res
        prob = np.zeros((len(te), 5))
        for j in range(5):
            prob[:, j] = LogisticRegression(C=2.0, max_iter=2000).fit(X[tr], S[tr, j]).predict_proba(X[te])[:, 1]
        ch["logreg"][te] = (prob + prior).argmax(1)

    hits = {f"always_{m}": S[:, j] for j, m in enumerate(M)}
    hits.update({k: S[np.arange(len(S)), c] for k, c in ch.items()})
    hits["oracle"] = S.max(1)
    out["pooled"] = {"rows": len(S), "scores": {k: macro(v) for k, v in hits.items()},
                     "per_benchmark": {k: {b: wmean(v[B == b], W[B == b]) for b in BENCHES} for k, v in hits.items()},
                     "share": {k: dict(zip(M, (np.bincount(c, weights=W, minlength=5) / W.sum()).round(3).tolist()))
                               for k, c in ch.items()}}
    # paired bootstrap: resample dialogues/questions inside each benchmark, every router on the same draw
    rng = np.random.default_rng(42)
    idx = {b: np.where(B == b)[0] for b in BENCHES}
    codes = {b: np.unique(G[idx[b]], return_inverse=True) for b in BENCHES}
    n_boot, draws = 2000, {k: np.zeros(2000) for k in hits}
    for i in range(n_boot):
        for b in BENCHES:
            n_groups = len(codes[b][0])
            mult = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[codes[b][1]] * W[idx[b]]
            for k, v in hits.items():
                draws[k][i] += (v[idx[b]] * mult).sum() / mult.sum() / len(BENCHES)
    contrasts = [("bench_table", "always_gemma"), ("bench_table", "always_deepseek"), ("always_deepseek", "always_gemma"),
                 ("bench_table", "subtask_table"), ("bench_table", "knn100"), ("bench_table", "logreg"),
                 ("logreg", "always_deepseek"), ("knn100", "always_deepseek")]
    out["contrasts"] = {f"{a} - {b}": dict(diff=macro(hits[a]) - macro(hits[b]),
                                           ci95=np.percentile(draws[a] - draws[b], [2.5, 97.5]).round(4).tolist())
                        for a, b in contrasts}
    # leave one benchmark out: kNN50 on a task type it never saw, against the best single chosen without it
    out["lobo"] = {}
    for b in BENCHES:
        tr, te = np.where(B != b)[0], np.where(B == b)[0]
        res = np.zeros(len(te), int)
        for s0 in range(0, len(te), 2000):
            sims = (X[te[s0:s0 + 2000]] @ X[tr].T).toarray()
            nn = np.argpartition(-sims, 50, axis=1)[:, :50]
            res[s0:s0 + 2000] = (S[tr][nn].mean(1) + prior).argmax(1)
        others = np.array([np.mean([wmean(S[B == x, j], W[B == x]) for x in BENCHES if x != b]) for j in range(5)])
        pick = int(others.argmax())
        out["lobo"][b] = dict(knn50=wmean(S[te, res], W[te]), best_of_others=M[pick],
                              best_of_others_score=wmean(S[te, pick], W[te]), true_best=max(wmean(S[te, j], W[te]) for j in range(5)))
    json.dump(out, open(os.path.join(HERE, "reported.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("contrasts", "lobo")}, indent=1))
    print({k: round(v, 4) for k, v in out["pooled"]["scores"].items()})
    print(json.dumps(out["pooled"]["per_benchmark"]["bench_table"]))
    print(json.dumps(out["pooled"]["share"]))


if __name__ == "__main__":
    main()
