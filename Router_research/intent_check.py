"""How well does a cheap TF-IDF classifier recognise task type? Offline, stored rows only.

Usage: /opt/miniconda3/bin/python3 -I Router_research/intent_check.py [repo root]
"""
import sys, os, numpy as np; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import router_headroom as rh
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
data = {"bbh": rh.load_bbh(), "mmlu": rh.load_mmlu(), "emo": rh.load_emo(), "neg": rh.load_neg(), "aware": rh.load_aware()}
items = []
for b, rows in data.items():
    for it in rh.complete(rows, rh.M):
        it["bench"] = b; items.append(it)
texts = [it["text"][:4000] for it in items]
B = np.array([it["bench"] for it in items]); SUB = np.array([it["sub"] for it in items]); G = np.array([it["group"] for it in items])
X = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit_transform(texts)
for name, y in [("benchmark (5 classes)", B), ("sub-task (%d classes)" % len(set(SUB)), SUB)]:
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(5).split(X, groups=G):
        pred[te] = LogisticRegression(C=4.0, max_iter=3000).fit(X[tr], y[tr]).predict(X[te])
    acc = (pred == y).mean()
    print(f"{name}: held-out accuracy {acc:.3f}")
    if "sub-task" in name:
        import collections
        wrong = collections.Counter((a, b) for a, b in zip(y, pred) if a != b)
        print("  most confused:", wrong.most_common(6))
# cost of a wrong category: accuracy gap between best and each other model, per benchmark
print("\nper benchmark: best model, and accuracy lost by sending to the 2nd-best / worst model")
for b in data:
    S = np.array([[it["s"][m] for m in rh.M] for it in items if it["bench"] == b]).mean(0)
    o = np.argsort(-S)
    print(f"  {b:6s} best={rh.M[o[0]]} {S[o[0]]:.3f}  2nd={rh.M[o[1]]} -{S[o[0]]-S[o[1]]:.3f}  worst={rh.M[o[-1]]} -{S[o[0]]-S[o[-1]]:.3f}")
