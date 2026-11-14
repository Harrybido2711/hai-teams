#!/usr/bin/env python3
"""The result row of every benchmark sheet: which row it is, the three computed here, its highlight.

    python3 Results/result_rows.py            # write Tempo_results.xlsx and Final_result.xlsx
    python3 Results/result_rows.py --dry-run  # compute and diff, write nothing

The user's instruction, 2026-11-13: every benchmark has one row that is its result, and that row is
highlighted. On a sheet recorded with such a row this script only highlights it — the row labelled
`Overall Score…`, and `Overall accuracy` on DocVQA. Where upstream defines no composite the row is
ours (`DERIVED`): computed here from the sheet's own rows, written into the sheet under a label that
states its formula, and recomputed on every run. `Final_result.xlsx` also gets a `Summary` sheet,
the result rows copied one line a benchmark.

**Run it after any recorder** — `record_*.py` recreate their sheet without the highlight, and a
derived row goes stale when the rows above it change. `build_results_db.py` reads the highlighted
row as `benchmarks.result_row` and stops unless each sheet has exactly one and `Summary` agrees with
the sheets. Before anything is written, every cell that is not a result row's is compared with the
file it came from; a difference stops the run.
"""
import argparse
import copy
import os
import shutil
import sys
import tempfile

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), ".claude", "scripts"))
import build_results_db as rdb  # noqa: E402  — BENCHMARKS (name, group, sheet), RESULT_FILL, SUMMARY

TEMPO = os.path.join(HERE, "Tempo_results.xlsx")
FINAL = os.path.join(HERE, "Final_result.xlsx")
DATE = "2026-11-13"
NUM = "0.0000"
MARK = "result row — "            # column B of the Provenance rows this script owns
BESIDE = {"All_EM", "Overall (micro)", "ANLS"}   # metrics beside a result row; never bold like one


def _mean(values):
    return round(sum(values) / len(values), 4) if all(v is not None for v in values) else None


def _wonderbread(v):
    """QA is judged 1-3 where 1 is best: (3 - x) / 2 puts it on 0-1, higher better, like the SOP rows."""
    return _mean([(3 - x) / 2 if x is not None and label.startswith("qa_") else x
                  for label, x in v.items() if label.startswith(("qa_", "sop_"))])


# sheet -> the row this script computes: its label, the row it goes under, f({label: value}) for one column
DERIVED = {
    "NegotiationToM": dict(
        label="Overall Score (mean of the 4 rows above)", after="Intent_Macro_F1",
        f=lambda v: _mean([v[k] for k in ("Desire_EM", "Belief_EM", "Intent_Micro_F1", "Intent_Macro_F1")])),
    "Multi-party Goal Tracking": dict(
        label="Overall Score (mean of dst and goal)", after="goal (mean of 6 styles)",
        f=lambda v: _mean([v["dst (mean of 6 styles)"], v["goal (mean of 6 styles)"]])),
    "Wonderbread": dict(
        label="Overall Score (mean of 8 rows, QA as (3-x)/2)", after="sop_ordering", f=_wonderbread),
}
# what a result row is, in the words the Summary sheet shows beside it
READ_WITH = {
    "Big Bench Hard": "macro mean of the 20 tasks",
    "MMLU": "macro mean of the 13 subjects",
    "DocVQA": "binary accuracy; ANLS, upstream's graded metric, is beside it on the sheet",
    "Emo": "mean of the two groups, EU and EA",
    "NegotiationToM": "OURS, upstream has no composite: mean of desire EM, belief EM, intent micro F1 and "
                      "macro F1. All_EM is a separate metric and is not in it",
    "Awareness": "mean of the five dimensions",
    "PlanBench (pilot)": "PILOT, not the full run: every 10th instance. Correct ÷ rows over all tasks",
    "Multi-party Goal Tracking": "OURS, upstream has no composite: mean of the two tasks, dst and goal",
    "MultiChallenge": "macro mean of the 4 axes; every cell is an LLM judge's grade (Haiku 5.5)",
    "Wonderbread": "OURS, upstream has no composite: mean of the 8 rows, QA (1-3, 1 best) first put on 0-1. "
                   "An LLM judge's grades (Haiku 5.5). Deepseek takes no images",
}

HEAD = dict(font=Font(bold=True), fill=PatternFill("solid", fgColor="DDE5F0"),
            alignment=Alignment(horizontal="center", vertical="center", wrap_text=True))
CONF = dict(font=Font(size=9), fill=PatternFill("solid", fgColor="FFF4E0"),
            alignment=Alignment(horizontal="center", vertical="center", wrap_text=True))
WRAP = Alignment(wrap_text=True, vertical="top")
RESULT = PatternFill("solid", fgColor=rdb.RESULT_FILL)


def style(cell, spec):
    cell.font, cell.fill, cell.alignment = copy.copy(spec["font"]), copy.copy(spec["fill"]), copy.copy(spec["alignment"])


def sheets(wb):
    """The benchmark sheets of this workbook, in the workbook's own order."""
    known = {sheet for _, _, sheet in rdb.BENCHMARKS if sheet}
    missing = known - set(wb.sheetnames)
    assert not missing, f"sheets build_results_db.py expects and the workbook lacks: {sorted(missing)}"
    return [s for s in wb.sheetnames if s in known]


def label_rows(ws):
    out = {}
    for r in range(3, ws.max_row + 1):
        label = ws.cell(r, 1).value
        if label:
            assert label not in out, f"{ws.title}: row label {label!r} twice"
            out[label] = r
    return out


def columns(ws):
    """{header: column index} of the model columns — row 1, from column B."""
    return {ws.cell(1, c).value: c for c in range(2, ws.max_column + 1) if ws.cell(1, c).value}


def result_label(ws):
    """The one row that is this sheet's result. Exact labels, never a substring."""
    labels = label_rows(ws)
    if ws.title == "DocVQA":
        found = [k for k in labels if k == "Overall accuracy"]
    else:
        found = [k for k in labels if k == "Overall Score" or k.startswith("Overall Score (")]
    assert len(found) == 1, f"{ws.title}: {len(found)} candidate result rows {found}, expected 1"
    return found[0]


def apply(wb):
    """Write the derived rows, then highlight each sheet's result row and nothing else. Returns notes."""
    notes = []
    for name in sheets(wb):
        ws = wb[name]
        cols = columns(ws)
        if name in DERIVED:
            d = DERIVED[name]
            rows = label_rows(ws)
            added = d["label"] not in rows
            if added:
                ws.insert_rows(rows[d["after"]] + 1)
                ws.cell(rows[d["after"]] + 1, 1, d["label"])
                rows = label_rows(ws)
            for c in cols.values():
                cell = ws.cell(rows[d["label"]], c)
                cell.value = d["f"]({k: ws.cell(r, c).value for k, r in rows.items() if k != d["label"]})
                cell.number_format = NUM
            notes.append(f"{name}: {'added' if added else 'recomputed'} {d['label']!r}")
        result = result_label(ws)
        for label, r in label_rows(ws).items():
            for c in [1, *cols.values()]:
                cell = ws.cell(r, c)
                if label == result:
                    cell.font, cell.fill = Font(bold=True), copy.copy(RESULT)
                elif rdb.highlighted(cell):
                    cell.fill = PatternFill(fill_type=None)
                if label in BESIDE:
                    cell.font = Font(bold=False)
        notes.append(f"{name}: result row {result!r}")
    return notes


def write_summary(wb):
    """`Summary`, first sheet: the highlighted row of every benchmark sheet, one line a benchmark."""
    if rdb.SUMMARY in wb.sheetnames:
        del wb[rdb.SUMMARY]
    names = sheets(wb)
    ws = wb.create_sheet(rdb.SUMMARY, 0)
    first = wb[names[0]]
    models = list(columns(first))
    group = {sheet: g for _, g, sheet in rdb.BENCHMARKS}
    head = ["Benchmark (sheet)", "Process group", "Result row — highlighted on its sheet", *models, "Read with"]
    for c, text in enumerate(head, 1):
        style(ws.cell(1, c, text), HEAD)
        style(ws.cell(2, c), CONF)
    ws.cell(2, 1, "Model")
    for model in models:
        ws.cell(2, 4 + models.index(model), first.cell(2, columns(first)[model]).value)
    for i, name in enumerate(names, 3):
        src = wb[name]
        assert list(columns(src)) == models, f"{name}: columns {list(columns(src))} differ from {models}"
        label = result_label(src)
        row = label_rows(src)[label]
        ws.cell(i, 1, name).font = Font(bold=True)
        ws.cell(i, 2, group[name])
        ws.cell(i, 3, label)
        for c, model in enumerate(models, 4):
            value = src.cell(row, columns(src)[model]).value
            cell = ws.cell(i, c, "N/A" if value is None else value)
            cell.number_format, cell.font, cell.fill = NUM, Font(bold=True), copy.copy(RESULT)
            cell.alignment = Alignment(horizontal="right")
        ws.cell(i, len(head), READ_WITH[name])
        for c in (1, 2, 3, len(head)):
            ws.cell(i, c).alignment = copy.copy(WRAP)
    for letter, width in zip("ABC", (27, 14, 42)):
        ws.column_dimensions[letter].width = width
    for c in range(4, 4 + len(models)):
        ws.column_dimensions[ws.cell(1, c).column_letter].width = 13
    ws.column_dimensions[ws.cell(1, len(head)).column_letter].width = 78
    ws.row_dimensions[2].height = 40
    ws.freeze_panes = "D3"
    wb.active = 0
    for other in wb.worksheets:                       # two selected tabs open as a group in Excel
        other.sheet_view.tabSelected = other.title == rdb.SUMMARY


def provenance_rows(final):
    rows = [
        ("READ FIRST", MARK + "the highlighted row of every sheet", "one a sheet — Results/result_rows.py",
         f"The user's instruction, {DATE}: every benchmark sheet has exactly one highlighted row, and it is "
         "that benchmark's result — the one number to take when a benchmark is compared or averaged with the "
         "others. Seven sheets were recorded with it: their 'Overall Score…' row, and on DocVQA 'Overall "
         "accuracy'. Three have no composite upstream, so the row is OURS, computed by result_rows.py from the "
         "rows above it and labelled with its formula (next three rows). Rows that sit beside a result row and "
         "are not it: All_EM, Overall (micro), ANLS. results.db: benchmarks.result_row, view final_results."),
        ("READ FIRST", MARK + "NegotiationToM", "mean of Desire_EM, Belief_EM, Intent_Micro_F1, Intent_Macro_F1",
         "Ours; upstream reports the metrics separately. All_EM is a fifth metric — a dialogue counts only if "
         "every one of its rows is right — and is neither in the mean nor the benchmark's total (the user, "
         "2026-11-06 and 2026-11-13). Micro and macro F1 are two readings of the same intention answers, so "
         "intention weighs twice. The planner's formula, the one behind the cross-benchmark means shown to the "
         "user on 2026-11-13; not separately confirmed."),
        ("READ FIRST", MARK + "Wonderbread", "mean of the eight rows, each QA row first converted (3 - x) / 2",
         "QA is judged 1-3 where 1 IS BEST, so 1 -> 1.0, 2 -> 0.5, 3 -> 0; the SOP rows are 0-1 already. The "
         "result row is therefore higher-is-better like every other, while the four QA rows above it stay as "
         "judged. Ours; upstream defines no composite. The user had this formula recorded on 2026-11-13."),
        ("READ FIRST", MARK + "Multi-party Goal Tracking", "mean of 'dst (mean of 6 styles)' and 'goal (mean of 6 styles)'",
         "Each task is the plain mean of its six prompt styles; the result is the mean of the two tasks. Ours; "
         "upstream defines no composite. The planner's formula, not separately confirmed."),
    ]
    if final:
        rows.append(("READ FIRST", MARK + "the Summary sheet", "the highlighted rows, copied by result_rows.py",
                     "One line a benchmark. Never edit it: change the benchmark's sheet and rerun "
                     "result_rows.py; build_results_db.py stops when a Summary cell differs from its sheet. "
                     "N/A = the model cannot take that benchmark's input (Deepseek is text-only). PlanBench's "
                     "line is its pilot until the full run is recorded."))
    return rows


def write_provenance(wb, final):
    ws = wb["Provenance"]
    for r in range(ws.max_row, 3, -1):
        if ws.cell(r, 1).value == "READ FIRST" and str(ws.cell(r, 2).value or "").startswith(MARK):
            ws.delete_rows(r)
    head = [r for r in range(4, ws.max_row + 1) if ws.cell(r, 1).value == "READ FIRST"]
    assert head and head == list(range(4, 4 + len(head))), f"READ FIRST rows are not the first rows: {head}"
    rows = provenance_rows(final)
    ws.insert_rows(head[-1] + 1, len(rows))
    for r, row in enumerate(rows, head[-1] + 1):
        for c, v in enumerate(row, 1):
            ws.cell(r, c, v).alignment = copy.copy(WRAP)
    title = ws.cell(1, 1).value
    tag = f"; a result row highlighted on every sheet {DATE}"
    if tag not in title:
        ws.cell(1, 1).value = title.rstrip(".") + tag + "."


def _key(cell):
    # a cell's style proxies do not compare by value; the copies do
    return (cell.value, cell.number_format, copy.copy(cell.font), copy.copy(cell.fill), copy.copy(cell.alignment))


def diff(before_path, after_wb):
    """Every cell that is not a result row's must be as it was — value and style, matched by row label
    and column header, since a derived row shifts the rows under it. Returns notes."""
    before = load_workbook(before_path)
    assert [s for s in after_wb.sheetnames if s != rdb.SUMMARY] == [s for s in before.sheetnames if s != rdb.SUMMARY], \
        f"sheet order changed: {before.sheetnames} -> {after_wb.sheetnames}"
    notes = []
    for name in sheets(before):
        a, b = before[name], after_wb[name]
        ours = {result_label(b)} | BESIDE | ({DERIVED[name]["label"]} if name in DERIVED else set())
        rows_a, rows_b = label_rows(a), label_rows(b)
        gone = set(rows_a) - set(rows_b)
        assert not gone, f"{name}: rows lost: {sorted(gone)}"
        assert [k for k in rows_b if k in rows_a] == list(rows_a), f"{name}: row order changed"
        assert columns(a) == columns(b), f"{name}: columns changed"
        checked = 0
        for r in (1, 2):
            for c in range(1, a.max_column + 1):
                if _key(a.cell(r, c)) != _key(b.cell(r, c)):
                    raise SystemExit(f"{before_path} › {name} › R{r}C{c} changed")
        for label, r in rows_a.items():
            for c in range(1, a.max_column + 1):
                x, y = a.cell(r, c), b.cell(rows_b[label], c)
                if label in ours:
                    if x.value != y.value and not (name in DERIVED and label == DERIVED[name]["label"]):
                        raise SystemExit(f"{before_path} › {name} › {label} › C{c}: {x.value!r} -> {y.value!r}")
                elif _key(x) != _key(y):
                    raise SystemExit(f"{before_path} › {name} › {label} › C{c}: {x.value!r} -> {y.value!r}")
                checked += 1
        new = [k for k in rows_b if k not in rows_a]
        notes.append(f"{name}: {checked} cells checked, values unchanged" + (f"; new row {new}" if new else ""))
    a, b = before["Provenance"], after_wb["Provenance"]

    def kept(ws):
        return [[_key(ws.cell(r, c)) for c in range(1, a.max_column + 1)] for r in range(2, ws.max_row + 1)
                if not str(ws.cell(r, 2).value or "").startswith(MARK)]
    if kept(a) != kept(b):
        raise SystemExit(f"{before_path} › Provenance: a row that is not this script's changed or moved")
    notes.append(f"Provenance: {len(kept(a))} rows that are not this script's, unchanged and in order")
    return notes


def save(wb, path):
    fd, tmp = tempfile.mkstemp(suffix=".xlsx", dir=HERE)
    os.close(fd)
    try:
        wb.save(tmp)
        shutil.move(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"written: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    built = {}
    for path in (TEMPO, FINAL):
        wb = load_workbook(path)
        notes = apply(wb)
        write_provenance(wb, final=path == FINAL)
        if path == FINAL:
            write_summary(wb)
        for note in notes + diff(path, wb):
            print(f"{os.path.basename(path)} › {note}")
        built[path] = wb
    # a Final cell that differs from Tempo's stops build_results_db.py; stop here instead
    for name in sheets(built[FINAL]):
        t, f = built[TEMPO][name], built[FINAL][name]
        assert result_label(t) == result_label(f), name
        for col, c in columns(f).items():
            a, b = t.cell(label_rows(t)[result_label(t)], columns(t)[col]).value, f.cell(label_rows(f)[result_label(f)], c).value
            assert a == b, (name, col, a, b)
    if args.dry_run:
        print("dry run: nothing written")
        return
    for path, wb in built.items():
        save(wb, path)


if __name__ == "__main__":
    main()
