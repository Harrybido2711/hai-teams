#!/usr/bin/env python3
"""Fill the number tables of RESULTS_SUMMARY.md, the summary written to be handed to other people.

    python3 Results/summary_tables.py           # rewrite the generated tables
    python3 Results/summary_tables.py --check   # exit 1 if they differ from the database

The summary's prose is written by hand and this script never touches it. It replaces only what sits
between `<!-- generated:NAME -->` and `<!-- /generated:NAME -->`:

- `results`  — each benchmark's result row for the five selected models (`final_results`), the best
               model in bold, and the score when the best model is taken on every sub-benchmark;
- `averages` — a model's mean over benchmarks, Oracle 1 and Oracle 2, on each basis.

A printed number nobody rechecks is how a summary goes stale, so `build_results_db.py --check` calls
`stale()` and fails until this has been rerun. Definitions: `.claude/references/results-db.md`
§ Averaging across benchmarks. **The summary is for readers outside the project: it names the
workbook and nothing else that is local — no path, script, database, cluster or job (the user's
rule, 2026-11-14). Keep it that way when editing its prose.**
"""
import argparse
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "results.db")
DOC = os.path.join(HERE, "RESULTS_SUMMARY.md")
MODELS = ["Gemini", "OpenAI", "Qwen", "Gemma", "Deepseek"]
ORDER = ["BBH", "MMLU", "DocVQA", "AwareBench", "PlanBench", "Multi-party Goal Tracking", "Wonderbread",
         "MultiChallenge", "EmoBench", "NegotiationToM"]
SHOWN = {"BBH": "BIG-Bench Hard"}
PROCESS = {"BBH": "general task ability", "MMLU": "general task ability", "DocVQA": "general task ability",
           "AwareBench": "transition — mission analysis", "PlanBench": "transition — strategy formulation",
           "Multi-party Goal Tracking": "transition — goal specification",
           "Wonderbread": "action — monitoring progress", "MultiChallenge": "action — coordination",
           "EmoBench": "interpersonal — affect management", "NegotiationToM": "interpersonal — conflict management"}

# How a result row combines its sub rows where it is not their plain mean. Every run checks that
# these reproduce the stored result row, so a benchmark re-recorded on other counts stops the script.
EMO_EU = {"complex_emotions": 49, "emotional_cues": 28, "personal_beliefs_and_experiences": 56,
          "perspective_taking": 67}                                   # items; EU and EA weigh a half each
EMO_EA = ["Personal-Others", "Personal-Self", "Social-Others", "Social-Self"]
AWARE = {"capability": .2, "mission_explicit": .1, "mission_implicit": .1, "emotion": .2, "culture": .2,
         "perspective (2nd-order)": .2}
PLAN_ROWS = [199, 149, 149, 10, 149, 112, 149, 149, 149, 149, 149]    # the pilot's rows per task, sheet order
TOLERANCE = 6e-4                                                      # the sheets hold four decimals


def _same(x):
    return x


def _qa(x):
    return (3 - x) / 2          # Wonderbread QA is judged 1-3 where 1 is best


def load(path):
    db = sqlite3.connect(path)
    meta = {b: (sheet, row) for b, sheet, row in db.execute(
        "SELECT benchmark, workbook_sheet, result_row FROM benchmarks WHERE result_row IS NOT NULL")}
    rows = {}
    for b, sub, level, model, value in db.execute(
            "SELECT s.benchmark, s.subbench, s.level, s.model_column, s.value FROM scores s "
            "JOIN runs r USING (benchmark, model_column) WHERE r.selected = 1 ORDER BY s.position"):
        rows.setdefault(b, {}).setdefault(sub, {"level": level})[model] = value
    db.close()
    assert sorted(meta) == sorted(ORDER), f"benchmarks with a result row {sorted(meta)} are not {sorted(ORDER)}"
    return meta, rows


def units(name, rows, result_row):
    """[(sub row, weight, transform)] — the pieces an oracle chooses a model for."""
    subs = [k for k, d in rows.items() if d["level"] == "sub"]
    if name == "EmoBench":
        return [(k, n / 400, _same) for k, n in EMO_EU.items()] + [(k, 1 / 8, _same) for k in EMO_EA]
    if name == "AwareBench":
        return [(k, w, _same) for k, w in AWARE.items()]
    if name == "PlanBench":
        assert len(subs) == len(PLAN_ROWS), f"PlanBench has {len(subs)} task rows, PLAN_ROWS {len(PLAN_ROWS)}"
        return list(zip(subs, PLAN_ROWS, [_same] * len(subs)))
    if name == "Wonderbread":
        return [(k, 1, _qa if k.startswith("qa_") else _same) for k in subs]
    return [(k, 1, _same) for k in subs] or [(result_row, 1, _same)]


def compute(path):
    """{benchmark: {"scores": {model: value|None}, "best": (value, model), "sub": value, "pilot": bool}}"""
    meta, rows = load(path)
    out = {}
    for name in ORDER:
        sheet, result_row = meta[name]
        U = units(name, rows[name], result_row)
        total = sum(w for _, w, _ in U)
        stored = {m: rows[name][result_row].get(m) for m in MODELS}
        for m in MODELS:                               # the weights must reproduce the stored result row
            cells = [rows[name][k].get(m) for k, _, _ in U]
            mine = None if any(c is None for c in cells) else sum(w * t(c) for c, (_, w, t) in zip(cells, U)) / total
            if (mine is None) != (stored[m] is None) or (mine is not None and abs(mine - stored[m]) > TOLERANCE):
                raise SystemExit(f"{name} › {m}: the weights in summary_tables.py give {mine}, the result row "
                                 f"{result_row!r} holds {stored[m]} — update the weights to the new run")
        sub = sum(w * max(t(rows[name][k][m]) for m in MODELS if rows[name][k].get(m) is not None)
                  for k, w, t in U) / total
        best = max((v, m) for m, v in stored.items() if v is not None)
        out[name] = {"scores": stored, "best": best, "sub": sub, "pilot": "(pilot)" in sheet}
    return out


def shown(name, r):
    return SHOWN.get(name, name) + (" (pilot)" if r["pilot"] else "")


def results_table(R):
    lines = ["| Benchmark | Team process | " + " | ".join(MODELS) + " | Best of each sub-benchmark |",
             "|---|---|" + "--:|" * (len(MODELS) + 1)]
    for name in ORDER:
        r = R[name]
        cells = ["N/A" if v is None else (f"**{v:.4f}**" if m == r["best"][1] else f"{v:.4f}")
                 for m, v in r["scores"].items()]
        lines.append(f"| {shown(name, r)} | {PROCESS[name]} | " + " | ".join(cells) + f" | {r['sub']:.4f} |")
    return "\n".join(lines)


def averages_table(R):
    def row(label, names):
        means = {m: (None if any(R[b]["scores"][m] is None for b in names)
                     else sum(R[b]["scores"][m] for b in names) / len(names)) for m in MODELS}
        top = max(v for v in means.values() if v is not None)
        cells = ["—" if v is None else (f"**{v:.4f}**" if v == top else f"{v:.4f}") for v in means.values()]
        o1 = sum(R[b]["best"][0] for b in names) / len(names)
        o2 = sum(R[b]["sub"] for b in names) / len(names)
        return f"| {label} | {len(names)} | " + " | ".join(cells) + f" | {o1:.4f} | {o2:.4f} |"

    def shared(names):
        return [b for b in names if all(v is not None for v in R[b]["scores"].values())]

    full = [b for b in ORDER if not R[b]["pilot"]]
    lines = ["| Benchmarks counted | How many | " + " | ".join(MODELS) + " | Oracle 1 | Oracle 2 |",
             "|---|--:|" + "--:|" * (len(MODELS) + 2)]
    if len(full) < len(ORDER):
        lines += [row("All, the pilot included", ORDER), row("Those every model has a score on, the pilot included", shared(ORDER)),
                  row("Complete runs only", full), row("Complete runs every model has a score on", shared(full))]
    else:
        lines += [row("All", ORDER), row("Those every model has a score on", shared(ORDER))]
    return "\n".join(lines)


def blocks(path=DB):
    R = compute(path)
    return {"results": results_table(R), "averages": averages_table(R)}


def _pattern(name):
    return re.compile(rf"(<!-- generated:{name} -->\n)(.*?)(<!-- /generated:{name} -->)", re.S)


def stale(path=DB):
    """Names of the generated tables in the summary that differ from what the database gives now."""
    with open(DOC) as fh:
        text = fh.read()
    out = []
    for name, table in blocks(path).items():
        found = _pattern(name).search(text)
        if not found:
            out.append(f"{name}: no <!-- generated:{name} --> block")
        elif found.group(2) != table + "\n":
            out.append(f"{name}: differs from the database")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        bad = stale()
        for line in bad:
            print(f"stale: {os.path.basename(DOC)} › {line}")
        print(f"{os.path.basename(DOC)}: {'rerun python3 Results/summary_tables.py' if bad else 'tables match the database'}")
        sys.exit(1 if bad else 0)
    with open(DOC) as fh:
        text = fh.read()
    for name, table in blocks().items():
        assert _pattern(name).search(text), f"{DOC} has no <!-- generated:{name} --> block"
        text = _pattern(name).sub(lambda m, t=table: m.group(1) + t + "\n" + m.group(3), text)
    with open(DOC, "w") as fh:
        fh.write(text)
    print(f"written: {DOC}")


if __name__ == "__main__":
    main()
