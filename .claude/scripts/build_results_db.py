#!/usr/bin/env python3
"""Build Results/results.db: every sub-benchmark score and the six criteria, queryable with sqlite3.

    python3 .claude/scripts/build_results_db.py            # rebuild
    python3 .claude/scripts/build_results_db.py --check    # exit 1 if a source changed since the build

The database is derived and never edited by hand. Sources, never another summary of them:
- scores: Results/Tempo_results.xlsx and Results/Final_result.xlsx. `selected` means the column
  exists in Final_result. A Final_result cell that differs from Tempo_results stops the build.
- criteria: the same files and the same reading code as build_criteria_workbook.py, so the database
  and Criteria_results.xlsx cannot compute a criterion differently.

A blank stays NULL, with its reason beside it where one is known; it is never written as 0.
Schema and queries: .claude/references/results-db.md.
"""
import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tempfile

from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_criteria_workbook as cw  # noqa: E402

ROOT = cw.ROOT
OUT = os.path.join(ROOT, "Results", "results.db")

# criteria name, process group and workbook sheet per benchmark; a None sheet has no reported result
BENCHMARKS = [
    ("BBH", "tasks", "Big Bench Hard"), ("MMLU", "tasks", "MMLU"), ("DocVQA", "tasks", "DocVQA"),
    ("EmoBench", "interpersonal", "Emo"), ("NegotiationToM", "interpersonal", "NegotiationToM"),
    ("AwareBench", "transition", "Awareness"), ("PlanBench", "transition", None),
    ("Multi-party Goal Tracking", "transition", "Multi-party Goal Tracking"),
    ("Wonderbread", "action", "Wonderbread"), ("MultiChallenge", "action", "MultiChallenge"),
]
# a section header row (no values) -> the group row its sub rows belong to; mpgt's labels are
# written by Results/record_mpgt.py
SECTIONS = {"Emotional Understanding (EU)": "eu_overall", "Emotional Application (EA)": "ea_overall",
            "Dialogue state tracking (DST), zero-shot": "dst (mean of 6 styles)",
            "Goal tracking, zero-shot": "goal (mean of 6 styles)"}
GROUP_ROWS = {"eu_overall", "ea_overall", "mission", "dst (mean of 6 styles)", "goal (mean of 6 styles)"}
OVERALL_ROWS = {"ANLS", "Overall accuracy"}
# separate metrics, never averaged across benchmarks: All_EM is not NegotiationToM's total, and
# MultiChallenge's micro is the same items counted another way — beside its macro, not part of it
STANDALONE_ROWS = {"All_EM", "Overall (micro)"}
CRITERIA = {1: "Cost", 2: "Token efficiency", 3: "Latency", 4: "Stability", 5: "Robustness", 6: "Reliability"}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sources (path TEXT PRIMARY KEY, md5 TEXT NOT NULL);
CREATE TABLE benchmarks (
    benchmark TEXT PRIMARY KEY, process_group TEXT NOT NULL, workbook_sheet TEXT);
CREATE TABLE runs (
    benchmark TEXT NOT NULL REFERENCES benchmarks, model_column TEXT NOT NULL,
    model_config TEXT, selected INTEGER NOT NULL, PRIMARY KEY (benchmark, model_column));
CREATE TABLE scores (
    benchmark TEXT NOT NULL, subbench TEXT NOT NULL, parent TEXT,
    level TEXT NOT NULL CHECK (level IN ('sub', 'group', 'overall', 'standalone')),
    model_column TEXT NOT NULL, value REAL, position INTEGER NOT NULL,
    PRIMARY KEY (benchmark, subbench, model_column),
    FOREIGN KEY (benchmark, model_column) REFERENCES runs);
CREATE TABLE criteria (
    benchmark TEXT NOT NULL, model_column TEXT NOT NULL,
    task TEXT NOT NULL,              -- ALL = the benchmark, pooled item-weighted; else one task
    criterion_no INTEGER NOT NULL CHECK (criterion_no BETWEEN 1 AND 6),
    criterion TEXT NOT NULL, metric TEXT NOT NULL,
    value REAL, value_text TEXT, reason TEXT, source TEXT,
    PRIMARY KEY (benchmark, model_column, task, metric),
    FOREIGN KEY (benchmark, model_column) REFERENCES runs);
CREATE TABLE criteria_inputs (
    benchmark TEXT NOT NULL, model_column TEXT NOT NULL, source TEXT NOT NULL,
    rows_total INTEGER, rows_scored INTEGER, scorer TEXT, price_source TEXT,
    PRIMARY KEY (benchmark, model_column));
CREATE TABLE provenance (
    workbook TEXT NOT NULL, sheet TEXT NOT NULL, column_label TEXT, source TEXT, note TEXT);
CREATE VIEW final_scores AS
    SELECT s.benchmark, s.subbench, s.parent, s.level, s.model_column, r.model_config, s.value
    FROM scores s JOIN runs r USING (benchmark, model_column) WHERE r.selected = 1;
CREATE VIEW gaps AS
    SELECT benchmark, model_column, criterion_no, criterion, COUNT(*) AS metrics, MIN(reason) AS reason
    FROM criteria WHERE task = 'ALL' AND value IS NULL AND value_text IS NULL
    GROUP BY benchmark, model_column, criterion_no;
"""


def rel(path):
    return os.path.relpath(path, ROOT)


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def source_files():
    paths = [cw.TEMPO, cw.FINAL, cw.AWARE_CSV]
    for slots in cw.FOLDERS.values():
        for folder, *_ in slots.values():
            path = folder and os.path.join(folder, "criteria_summary.json")
            if path and os.path.exists(path):
                paths.append(path)
    return sorted(paths)


def read_sheet(path, sheet):
    ws = load_workbook(path, data_only=True)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    return list(rows[0][1:]), dict(zip(rows[0][1:], rows[1][1:])), [r for r in rows[2:] if r[0]]


def classify(label, section):
    """(level, parent) of one workbook row. Labels are matched exactly, never by substring."""
    if label == "Overall Score" or label.startswith("Overall Score (macro-avg") or label in OVERALL_ROWS:
        return "overall", None
    if label in STANDALONE_ROWS:
        return "standalone", None
    if label in GROUP_ROWS:
        return "group", None
    if label.startswith("mission_"):
        return "sub", "mission"
    return "sub", section


def load_scores(db, name, sheet):
    columns, config, rows = read_sheet(cw.TEMPO, sheet)
    final_cols, final_model, final_rows = read_sheet(cw.FINAL, sheet)
    final = {r[0]: dict(zip(final_cols, r[1:])) for r in final_rows}
    db.executemany("INSERT INTO runs VALUES (?,?,?,?)",
                   [(name, c, config.get(c), int(c in final_cols)) for c in columns])
    section, position = None, 0
    for row in rows:
        label, values = row[0], dict(zip(columns, row[1:]))
        if label in SECTIONS:
            section = SECTIONS[label]
            continue
        position += 1
        level, parent = classify(label, section)
        for column in columns:
            value = values.get(column)
            if column in final_cols and final[label].get(column) != value:
                raise SystemExit(f"{sheet} › {label} › {column}: Final_result has "
                                 f"{final[label].get(column)!r}, Tempo_results has {value!r}")
            db.execute("INSERT INTO scores VALUES (?,?,?,?,?,?,?)",
                       (name, label, parent, level, column, value, position))


def put_criterion(db, name, slot, task, no, metric, value, reason, source):
    number, text = (value, None) if isinstance(value, (int, float)) else (None, value)
    db.execute("INSERT INTO criteria VALUES (?,?,?,?,?,?,?,?,?,?)",
               (name, slot, task, no, CRITERIA[no], metric, number, text, None if value is not None else reason,
                source))


def load_task_rows(db, name, slot, task, t, ratio, source):
    stability, robustness = t["stability"], t["robustness"]
    rows = [(2, "tokens wrong ÷ right", ratio), (4, "items", t["items"]),
            (4, "identical in all 5", stability["identical_all_5"]), (4, "pass@1", stability["pass@1"]),
            (4, "pass^5", stability["pass^5"]), (4, "accuracy min", stability["accuracy_min"]),
            (4, "accuracy max", stability["accuracy_max"]),
            (5, "noise flip (rep1 vs rep2)", robustness["baseline_flip_rep1_vs_rep2"])]
    for key, label in (("perturb-shuffle", "shuffle"), ("perturb-paraphrase", "paraphrase")):
        r = robustness[key]
        low, high = r["excess_flip_95ci_cluster"]
        rows += [(5, f"{label} flip vs rep1", r["flip_vs_rep1"]), (5, f"{label} excess flip", r["excess_flip"]),
                 (5, f"{label} excess 95% CI low", low), (5, f"{label} excess 95% CI high", high),
                 (5, f"{label} p", r["excess_mcnemar"]["p"]),
                 (5, f"{label} accuracy", r["accuracy_perturbed"]), (5, f"{label} accuracy p", r["accuracy_mcnemar"]["p"])]
    for no, metric, value in rows:
        put_criterion(db, name, slot, task, no, metric, value, None, source)


def load_criteria(db, name, sheet, aware):
    final = cw.sheet_values(cw.FINAL, sheet)[0]
    for slot in cw.SLOTS:
        _, usage, stab, task_rows, notes, _ = cw.model_row(name, slot, [], final, aware)
        na = cw.NOT_APPLICABLE.get((name, slot))
        why_usage = na or next((n.split(": ", 1)[1] for n in notes if n.startswith("1·2·3·6:")), None)
        why_stab = na or next((n.split(": ", 1)[1] for n in notes if n.startswith("4·5:")), None)
        source, main = None, None
        if name == "AwareBench":
            source, main = rel(cw.AWARE_CSV), aware[slot]
            db.execute("INSERT INTO criteria_inputs VALUES (?,?,?,?,?,?,?)",
                       (name, slot, source, int(main["rows"]), None, main["scorer"], main["1_price_source"]))
            put_criterion(db, name, slot, "ALL", 5, "shuffle flip, full run (not noise-corrected)",
                          float(main["5_shuffle_flip_rate_full_run"]), None, source)
        elif name in cw.FOLDERS and cw.FOLDERS[name][slot][0]:
            source = rel(os.path.join(cw.FOLDERS[name][slot][0], "criteria_summary.json"))
            main = cw.summary(cw.FOLDERS[name][slot][0])["main_run"]
            db.execute("INSERT INTO criteria_inputs VALUES (?,?,?,?,?,?,?)",
                       (name, slot, source, main.get("rows"), main.get("scored_rows"), None,
                        main.get("1_price_source")))
        for _, metric, key, _ in cw.USAGE:
            put_criterion(db, name, slot, "ALL", int(key[0]), metric, usage.get(key) if usage else None,
                          why_usage, source if usage else None)
        for _, metric, key, _ in cw.RELIAB:
            put_criterion(db, name, slot, "ALL", int(key[0]), metric, usage.get(key) if usage else None,
                          why_usage, source if usage else None)
        for group, metric, key, _ in cw.STAB:
            put_criterion(db, name, slot, "ALL", int(group[0]), metric, stab[key] if stab else None,
                          why_stab, source if stab else None)
        for task, t, ratio in task_rows:
            load_task_rows(db, name, slot, task, t, ratio, source)


def load_provenance(db, label, path):
    ws = load_workbook(path, data_only=True)["Provenance"]
    db.execute("INSERT INTO meta VALUES (?,?)", (f"{label} title", ws.cell(1, 1).value))
    for row in ws.iter_rows(min_row=4, values_only=True):
        if row[0]:
            db.execute("INSERT INTO provenance VALUES (?,?,?,?,?)", (label, row[0], row[1], row[2], row[3]))


def build(path):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO sources VALUES (?,?)", [(rel(p), md5(p)) for p in source_files()])
    db.execute("INSERT INTO meta VALUES ('built_by', '.claude/scripts/build_results_db.py')")
    db.executemany("INSERT INTO benchmarks VALUES (?,?,?)", BENCHMARKS)
    with open(cw.AWARE_CSV) as fh:
        aware = {r["slot"]: r for r in csv.DictReader(fh)}
    for name, _, sheet in BENCHMARKS:
        if sheet:
            load_scores(db, name, sheet)
            load_criteria(db, name, sheet, aware)
    load_provenance(db, "Tempo_results", cw.TEMPO)
    load_provenance(db, "Final_result", cw.FINAL)
    db.commit()
    broken = db.execute("PRAGMA foreign_key_check").fetchall()
    if broken:
        raise SystemExit(f"foreign key violations: {broken[:5]}")
    db.close()


def check(path):
    """Compare the sources recorded at build time with the files on disk."""
    if not os.path.exists(path):
        print(f"missing: {rel(path)}")
        return 1
    db = sqlite3.connect(path)
    recorded = dict(db.execute("SELECT path, md5 FROM sources"))
    db.close()
    current = {rel(p): md5(p) for p in source_files()}
    stale = sorted(p for p in recorded.keys() | current.keys() if recorded.get(p) != current.get(p))
    for p in stale:
        print(f"stale: {p}")
    print(f"{rel(path)}: {'out of date, rebuild it' if stale else 'up to date'} ({len(current)} sources)")
    return 1 if stale else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.out))
    fd, tmp = tempfile.mkstemp(suffix=".db", dir=os.path.dirname(os.path.abspath(args.out)))
    os.close(fd)
    os.unlink(tmp)
    try:
        build(tmp)
        os.replace(tmp, args.out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
