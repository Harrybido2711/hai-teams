"""Offline router headroom on stored per-query rows. Reads result files only; calls no model.

Usage: /opt/miniconda3/bin/python3 -I Router_research/router_headroom.py [repo root]
Writes headroom.json beside this file. Method and results: ROUTER_RESEARCH.md.
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = ["gemini", "luna", "qwen", "gemma", "deepseek"]
P = lambda *a: os.path.join(ROOT, *a)
csv.field_size_limit(10**9)


def jl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def put(rows, key, model, score, text, sub, group):
    r = rows.setdefault(key, {"text": text, "sub": sub, "group": group, "s": {}})
    r["s"][model] = float(score)
    if not r["text"] and text:
        r["text"] = text


# ---------- bbh ----------
def load_bbh():
    rows = {}
    pat = {"gemini": "BBH_Gemini_Flash3.5lite_OpenRouter/results/*/google-gemini-3_5-flash-lite.jsonl",
           "luna": "BBH_GPT_5.6_Luna/results/*/gpt-5_6-luna.jsonl",
           "qwen": "BBH_Qwen/results/*/Qwen-Qwen3_5-9B.jsonl",
           "gemma": "BBH_Gemma/results/*/google-gemma-4-31B-it.jsonl",
           "deepseek": "BBH_Deepseek/results/*/deepseek-reasoner.jsonl"}
    for m, g in pat.items():
        for f in glob.glob(P("Tasks_benchmarks/bbh", g)):
            task = os.path.basename(os.path.dirname(f))
            for r in jl(f):
                put(rows, ("bbh", task, r["idx"]), m, int(str(r["score"])), r["question"], task, f"bbh|{task}|{r['idx']}")
    return rows


# ---------- mmlu ----------
def load_mmlu():
    rows = {}
    for m, d, fn in [("gemini", "MMLU_Gemini_Flash3.5lite_Google", "google-gemini-3_5-flash-lite.jsonl"),
                     ("luna", "MMLU_GPT_5.6_Luna", "gpt-5_6-luna.jsonl")]:
        for f in glob.glob(P("Tasks_benchmarks/mmlu", d, "results/*", fn)):
            subj = os.path.basename(os.path.dirname(f))
            rs = sorted(jl(f), key=lambda r: int(r["idx"]))
            for pos, r in enumerate(rs):
                put(rows, ("mmlu", subj, pos), m, int(str(r["score"])), r["question"], subj, f"mmlu|{subj}|{pos}")
    for m, d, pre in [("qwen", "MMLU_Qwen", "qwen_"), ("gemma", "MMLU_Gemma", "gemma_"), ("deepseek", "MMLU_Deepseek", "deepseek_")]:
        for f in glob.glob(P("Tasks_benchmarks/mmlu", d, pre + "[A-Z]*.csv")):
            subj = os.path.basename(f)[len(pre):-4]
            with open(f, newline="") as fh:
                for pos, r in enumerate(csv.DictReader(fh)):
                    put(rows, ("mmlu", subj, pos), m, float(r["score"] or 0), r["question"], subj, f"mmlu|{subj}|{pos}")
    return rows


# ---------- docvqa (4 models) ----------
def load_docvqa():
    rows = {}
    for m, f in [("gemini", "DOCVQA_Gemini_Flash3.5lite_Google/results/gemini-3_5-flash-lite.jsonl"),
                 ("luna", "DOCVQA_GPT_5.6_Luna/results/gpt-5_6-luna.jsonl")]:
        for r in jl(P("Tasks_benchmarks/DocVQA", f)):
            put(rows, ("docvqa", r["questionId"]), m, int(str(r["score"])), r["question"], "docvqa", f"d|{r['questionId']}")
    for m, fs in [("qwen", ["qwen_DocVQA/qwen_docvqa_results.csv"]),
                  ("gemma", ["gemma_DocVQA/gemma_docvqa_results_half1.csv", "gemma_DocVQA/gemma_docvqa_results_half2.csv"])]:
        for f in fs:
            with open(P("Tasks_benchmarks/DocVQA", f), newline="") as fh:
                for r in csv.DictReader(fh):
                    put(rows, ("docvqa", str(r["questionId"])), m, float(r["score"] or 0), r["question"], "docvqa", f"d|{r['questionId']}")
    # normalise questionId type across sources
    out = {}
    for k, v in rows.items():
        kk = ("docvqa", str(k[1]))
        if kk in out:
            out[kk]["s"].update(v["s"])
        else:
            out[kk] = v
    return out


# ---------- emobench ----------
def load_emo():
    rows = {}
    base = P("Interpersonal_processes_benchmarks/EmoBench")
    pat = {"gemini": "EMO_Gemini_Flash3.5lite_OpenRouter/results/{t}/google-gemini-3_5-flash-lite_en.jsonl",
           "luna": "EMO_GPT_5.6_Luna/results/{t}/gpt-5_6-luna_en.jsonl",
           "qwen": "EMO_Qwen/results/{t}/Qwen-Qwen3_5-9B_en.jsonl",
           "gemma": "EMO_Gemma/results/{t}/google-gemma-4-31B-it_en.jsonl",
           "deepseek": "EMO_Deepseek/results/{t}/deepseek-reasoner_en_shard*of4.jsonl"}
    for m, g in pat.items():
        for t in ["EA", "EU"]:
            for f in glob.glob(os.path.join(base, g.format(t=t))):
                for r in jl(f):
                    if t == "EA":
                        sc = int(r.get("answer") == r.get("label"))
                    else:
                        sc = int(r.get("emo_answer") == r.get("emo_label") and r.get("cause_answer") == r.get("cause_label"))
                    put(rows, ("emo", t, str(r["qid"])), m, sc, f"{r.get('scenario','')} {r.get('subject','')} {r.get('question_type','')}",
                        f"emo_{t}", f"emo|{t}|{r['qid']}")
    return rows


# ---------- negotiationtom ----------
def load_neg():
    rows = {}
    base = P("Interpersonal_processes_benchmarks/NegotiationToM")
    dlg = {d["dialogue_id"]: " ".join(d["dialogue"]) for d in json.load(open(os.path.join(base, "NegotiationToM.json")))}
    pat = {"gemini": "NEG_Gemini_Flash3.5lite_OpenRouter/results/{t}/gemini-3_5-flash-lite_all.jsonl",
           "luna": "NEG_GPT_5.6_Luna/results/{t}/gpt-5_6-luna_all.jsonl",
           "qwen": "NEG_Qwen/results/{t}/Qwen3_5-9B_all.jsonl",
           "gemma": "NEG_Gemma/results/{t}/gemma-4-31B-it_all.jsonl",
           "deepseek": "NEG_Deepseek/results/{t}/deepseek-reasoner_all.jsonl"}
    skipped = 0
    for m, g in pat.items():
        for t in ["desire", "belief", "intention"]:
            for r in jl(os.path.join(base, g.format(t=t))):
                if t == "intention":
                    sc = int(r["gold_bitmask"] == r.get("pred_bitmask"))
                    text = r.get("target_utterance", "") + " || " + dlg.get(r["dialogue_id"], "")
                else:
                    gold = r["gold_desire"] if t == "desire" else {k: r[k] for k in ("gold_high", "gold_med", "gold_low")}
                    if "None" in gold.values():  # sentinel "None" = unannotated; excluded by the official scorer
                        skipped += 1
                        continue
                    sc = int(r[f"{t}_em"])
                    text = f"{t} {r.get('agent','')} || " + dlg.get(r["dialogue_id"], "")
                put(rows, ("neg", r["uid"]), m, sc, text, f"neg_{t}", "neg|" + r["dialogue_id"].split("-")[0])
    print(f"  neg: skipped {skipped} sentinel rows (all models)")
    return rows


# ---------- awarebench ----------
def load_aware():
    rows = {}
    base = P("Transition_processes_benchmarks/Awareness_in_LLM")
    pat = {"gemini": "AWARE_Gemini_Flash3.5lite_OpenRouter/results/*/google-gemini-3_5-flash-lite.jsonl",
           "luna": "AWARE_GPT_5.6_Luna/results/*/gpt-5_6-luna.jsonl",
           "qwen": "AWARE_Qwen/results/*/Qwen-Qwen3_5-9B.jsonl",
           "gemma": "AWARE_Gemma_DeepInfra/results/*/google-gemma-4-31B-it.jsonl",
           "deepseek": "AWARE_Deepseek/results/*/deepseek-flash.jsonl"}
    for m, g in pat.items():
        for f in glob.glob(os.path.join(base, g)):
            task = os.path.basename(os.path.dirname(f))
            for r in jl(f):
                put(rows, ("aware", r["uid"]), m, int(str(r["correct"])), str(r.get("prompt") or r.get("question") or ""),
                    f"aware_{task}", f"aware|{task}|{r.get('question_id')}")
    return rows


def complete(rows, models):
    return [dict(key=k, **v) for k, v in rows.items() if all(mm in v["s"] for mm in models)]


def summarize(name, items, models):
    S = np.array([[it["s"][m] for m in models] for it in items])
    means = S.mean(0)
    best = means.argmax()
    orc = S.max(1).mean()
    allc = (S.min(1) == 1).mean(); allw = (S.max(1) == 0).mean()
    # sub-task-level router: best model per sub-task (what a performance table supports)
    subs = defaultdict(list)
    for i, it in enumerate(items):
        subs[it["sub"]].append(i)
    st = sum(S[idx].mean(0).max() * len(idx) for idx in subs.values()) / len(items)
    print(f"{name:10s} n={len(items):6d} " + " ".join(f"{m}={x:.3f}" for m, x in zip(models, means)) +
          f" | best={models[best]} {means[best]:.3f} subtask-table={st:.3f} oracle={orc:.3f}"
          f" | all-right={allc:.1%} all-wrong={allw:.1%} contested={1-allc-allw:.1%}")
    return dict(n=len(items), means=dict(zip(models, means.round(4).tolist())), best=models[best], best_acc=float(means[best]),
                subtask_table=float(st), oracle=float(orc), all_right=float(allc), all_wrong=float(allw))


def main():
    print("loading…")
    data = {"bbh": load_bbh(), "mmlu": load_mmlu(), "emo": load_emo(), "neg": load_neg(), "aware": load_aware(), "docvqa": load_docvqa()}
    res = {}
    print("\n== per benchmark (rows scored for every model) ==")
    pooled = []
    for b, rows in data.items():
        ms = M if b != "docvqa" else ["gemini", "luna", "qwen", "gemma"]
        items = complete(rows, ms)
        res[b] = summarize(b, items, ms)
        if b != "docvqa":
            for it in items:
                it["bench"] = b
            pooled += items
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "headroom.json"), "w"), indent=1)

    # ---------- pooled router experiment (5 text benchmarks, 5 models) ----------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression

    benches = ["bbh", "mmlu", "emo", "neg", "aware"]
    S = np.array([[it["s"][m] for m in M] for it in pooled])
    B = np.array([it["bench"] for it in pooled])
    SUB = np.array([it["sub"] for it in pooled])
    G = np.array([it["group"] for it in pooled])
    texts = [it["text"][:4000] for it in pooled]
    X = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit_transform(texts)

    def macro(choice):  # equal weight per benchmark
        acc = S[np.arange(len(S)), choice]
        return np.mean([acc[B == b].mean() for b in benches]), {b: round(float(acc[B == b].mean()), 4) for b in benches}

    print(f"\n== pooled: {len(S)} queries, 5 benchmarks, macro = mean over benchmarks ==")
    macro_means = np.array([np.mean([S[B == b, j].mean() for b in benches]) for j in range(5)])
    for j, m in enumerate(M):
        print(f"  always {m:9s} macro={macro_means[j]:.4f}")
    gbest = int(macro_means.argmax())
    print(f"  best single = {M[gbest]}")
    orc_choice = np.where(S[:, gbest] == 1, gbest, S.argmax(1))
    print(f"  query oracle          macro={macro(orc_choice)[0]:.4f}")

    # table routers: best model per benchmark / per sub-task, chosen on the training fold
    gkf = GroupKFold(n_splits=5)
    ch = {k: np.zeros(len(S), int) for k in ["bench_table", "subtask_table", "knn10", "knn50", "knn100", "logreg"]}
    for tr, te in gkf.split(X, groups=G):
        for key, lab in [("bench_table", B), ("subtask_table", SUB)]:
            best = {}
            for v in np.unique(lab[tr]):
                idx = tr[lab[tr] == v]
                best[v] = int(S[idx].mean(0).argmax())
            ch[key][te] = [best.get(v, gbest) for v in lab[te]]
        # kNN (cosine on L2-normalised tf-idf = dot product)
        sims = (X[te] @ X[tr].T).toarray() if len(te) * len(tr) < 4e8 else None
        for k in [10, 50, 100]:
            out = np.zeros(len(te), int)
            for s0 in range(0, len(te), 2000):
                blk = (X[te[s0:s0+2000]] @ X[tr].T).toarray()
                nn = np.argpartition(-blk, k, axis=1)[:, :k]
                pred = S[tr][nn].mean(1) + 1e-6 * macro_means  # tie-break toward best single
                out[s0:s0+2000] = pred.argmax(1)
            ch[f"knn{k}"][te] = out
        # one logistic regression per model on tf-idf (multi-label correctness classifier)
        prob = np.zeros((len(te), 5))
        for j in range(5):
            y = S[tr, j]
            if y.min() == y.max():
                prob[:, j] = y[0]
                continue
            clf = LogisticRegression(C=2.0, max_iter=2000).fit(X[tr], y)
            prob[:, j] = clf.predict_proba(X[te])[:, 1]
        ch["logreg"][te] = (prob + 1e-6 * macro_means).argmax(1)
    print("  -- 5-fold GroupKFold (groups = item / dialogue / question) --")
    for k, c in ch.items():
        mac, per = macro(c)
        share = np.bincount(c, minlength=5) / len(c)
        print(f"  {k:14s} macro={mac:.4f}  per-bench={per}  share=" + ",".join(f"{m}:{s:.0%}" for m, s in zip(M, share)))

    # leave-one-benchmark-out: router has never seen the task type
    print("  -- leave-one-benchmark-out kNN50 (unseen task type) vs best single on that benchmark from the other four --")
    for b in benches:
        tr = np.where(B != b)[0]; te = np.where(B == b)[0]
        out = np.zeros(len(te), int)
        for s0 in range(0, len(te), 2000):
            blk = (X[te[s0:s0+2000]] @ X[tr].T).toarray()
            nn = np.argpartition(-blk, 50, axis=1)[:, :50]
            out[s0:s0+2000] = (S[tr][nn].mean(1) + 1e-6 * macro_means).argmax(1)
        tr_means = np.array([np.mean([S[B == bb, j].mean() for bb in benches if bb != b]) for j in range(5)])
        bs = int(tr_means.argmax())
        print(f"  held out {b:6s}: kNN50={S[te, :][np.arange(len(te)), out].mean():.4f}  "
              f"best-of-others({M[bs]})={S[te, bs].mean():.4f}  true best={S[te].mean(0).max():.4f}  oracle={S[te].max(1).mean():.4f}")


if __name__ == "__main__":
    main()
