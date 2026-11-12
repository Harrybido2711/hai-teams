#!/usr/bin/env python3
"""Record the Haiku 5.5-judged MultiChallenge and Wonderbread scores in both workbooks.

    python3 Results/record_haiku_judge.py            # write Tempo_results.xlsx and Final_result.xlsx
    python3 Results/record_haiku_judge.py --dry-run  # compute and diff, write nothing

Layer 4 of the sync pass (`.claude/references/sync-and-consistency.md`), run only after the user has
confirmed the runs. It adds two sheets, `MultiChallenge` and `Wonderbread`, after `Awareness`, and
their rows at the end of `Provenance`; **every other cell is left alone**, and the script proves it
before it writes — each pre-existing cell is compared with the file it came from, and a difference
stops the run. Re-running replaces those two sheets and those Provenance rows and nothing else.

The numbers are `build_judge_summary.collect()`, the computation behind `Haiku_judge_summary.md`:
rebuilt from the judge rows on disk, never from another summary. MultiChallenge values are fractions
(0.4 = 40 %) like every other sheet. Wonderbread QA is judged 1-3 where **1 is best**, so the row
labels carry the direction; SOP is higher-is-better. Wonderbread has no composite upstream, so it has
no overall row. Then `python3 .claude/scripts/build_results_db.py` rebuilds `results.db` from these.
"""
import argparse
import collections
import copy
import os
import shutil
import sys
import tempfile

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_judge_summary as bj  # noqa: E402

TEMPO = os.path.join(HERE, "Tempo_results.xlsx")
FINAL = os.path.join(HERE, "Final_result.xlsx")
SHEETS = ("MultiChallenge", "Wonderbread")
DATE = "2026-11-11"
SLOT = {"gemini-3.5-flash-lite": "Gemini", "gpt-5.6-luna": "OpenAI", "Qwen3.5-9B": "Qwen",
        "gemma-4-31B-it": "Gemma", "deepseek-flash": "Deepseek"}
TEMPO_COLS = ["Gemini", "OpenAI", "XAI", "Qwen", "Gemma", "Deepseek"]
FINAL_COLS = ["Gemini", "OpenAI", "Qwen", "Gemma", "Deepseek"]
FINAL_MODEL = {"Gemini": "gemini-3.5-flash-lite", "OpenAI": "gpt-5.6-luna", "Qwen": "Qwen/Qwen3.5-9B",
               "Gemma": "google/gemma-4-31B-it", "Deepseek": "deepseek-reasoner"}
# Tempo row 2: the model and config that produced the column on THIS sheet (Qwen's route differs by sheet)
TEMPO_MODEL = {
    "MultiChallenge": {"Gemini": "gemini-3.5-flash-lite (OpenRouter, minimal)", "OpenAI": "gpt-5.6-luna (effort=low)",
                       "XAI": "grok-3-mini", "Qwen": "Qwen/Qwen3.5-9B (Together, reasoning off)",
                       "Gemma": "google/gemma-4-31B-it (DeepInfra)", "Deepseek": "deepseek-flash (prompt ceiling)"},
    "Wonderbread": {"Gemini": "gemini-3.5-flash-lite (OpenRouter, minimal)", "OpenAI": "gpt-5.6-luna (effort=low)",
                    "XAI": "grok-3-mini", "Qwen": "Qwen/Qwen3.5-9B (DeepInfra, reasoning off, 30-image cap)",
                    "Gemma": "google/gemma-4-31B-it (DeepInfra, 32-image cap)",
                    "Deepseek": "deepseek-flash (text-only: not run)"},
}
MC_ROWS = list(bj.AXES) + ["Overall Score (macro-avg of 4 axes)", "Overall (micro)"]
QA_ROWS = {c: f"qa_{c} (1-3, 1 = best)" for c in bj.wb.QA_CRITERIA}
SOP_ROWS = {k: f"sop_{k}" for k in ("precision", "recall", "f1", "ordering")}
WB_ROWS = list(QA_ROWS.values()) + list(SOP_ROWS.values())

HEAD = dict(font=Font(bold=True), fill=PatternFill("solid", fgColor="DDE5F0"),
            alignment=Alignment(horizontal="center", vertical="center", wrap_text=True))
CONF = dict(font=Font(size=9), fill=PatternFill("solid", fgColor="FFF4E0"),
            alignment=Alignment(horizontal="center", vertical="center", wrap_text=True))
WRAP = Alignment(wrap_text=True, vertical="top")


def values(T):
    """{sheet: {row label: {slot: value or None}}} — fractions for MultiChallenge, scale units for QA."""
    mc, wb = {r: {} for r in MC_ROWS}, {r: {} for r in WB_ROWS}
    for t in T:
        slot = SLOT[t["model"]]
        m = t["mc"]
        for a in bj.AXES:
            mc[a][slot] = round(m["axes"][a] / 100, 4)
        mc["Overall Score (macro-avg of 4 axes)"][slot] = round(m["macro"] / 100, 4)
        mc["Overall (micro)"][slot] = round(m["micro"] / 100, 4)
        for c, label in QA_ROWS.items():
            wb[label][slot] = round(t["qa"][c], 4) if t["qa"] else None
        for k, label in SOP_ROWS.items():
            wb[label][slot] = round(t["sop"][k], 4) if t["sop"] else None
    return {"MultiChallenge": mc, "Wonderbread": wb}


def judge_counts(kind, stem, task=None):
    path = os.path.join(bj.AP, "Multi-challenge_bench" if kind == "mc" else "Wonderbread_bench",
                        *(["MC_Judges"] if kind == "mc" else ["WB_Judges"]), "results", "full", bj.JUDGE,
                        *([task] if task else []), stem + ".jsonl")
    rows = bj.rows(path)
    return len(rows), dict(collections.Counter(r["status"] for r in rows.values()))


def provenance(T):
    """[(sheet, column, source, note)] — the rows appended to Provenance in both workbooks."""
    by = {SLOT[t["model"]]: t for t in T}
    n_items = by["Gemini"]["mc"]["n"]
    mc_dir = "Action_processes_benchmarks/Multi-challenge_bench"
    wb_dir = "Action_processes_benchmarks/Wonderbread_bench"
    out = [
        ("MultiChallenge", "every column", f"{mc_dir}/MC_Judges/results/full/haiku-5.5/<slug>.jsonl",
         f"Added {DATE} at the user's request, recomputed from the judge rows on disk by Results/"
         "build_judge_summary.py (collect) and written by Results/record_haiku_judge.py. EVERY CELL IS AN "
         "LLM JUDGE'S GRADE, not an exact match: Claude Haiku 5.5 (anthropic/claude-haiku-5.5 through "
         "OpenRouter), upstream's JUDGE_PROMPT, three replicates per item, an item's label the strict "
         "majority, a tie or a failed call unjudged and never NO. An axis is the pass rate over its items; "
         "'Overall Score' is the macro mean of the four axes (upstream's headline) and 'Overall (micro)' "
         "counts the same items as one pool. Values are fractions. Haiku 5.5's error rate against human "
         "labels is UNMEASURED (LLM_as_judge/JUDGE_SELECTION.md §8), so read these as the judge's grades."),
        ("MultiChallenge", f"every column — {n_items} of 273 items",
         "item 6765fc5e0af93a24e8fbe4ee left out for every model",
         "Gemini's answer to it was blocked by Google's RECITATION filter on 5 of 5 attempts, so it has no "
         "response to grade; the user decided on 2026-11-07 that it is not scored, and the five columns are "
         "scored on the same 272 items so they stay comparable. Deepseek's answer to that item hit the "
         "output cap (7,849 of 8,192 tokens spent reasoning), so it is also the one item that cap cost "
         "Deepseek. Judge files for the other four hold that item's rows; the score excludes them."),
    ]
    mc_notes = {
        "Gemini": "MC_Gemini_Flash3.5lite_OpenRouter, generation job 6252751. OpenRouter, thinking effort "
                  "minimal, seed 42, max_tokens 8192, temperature unset; the answering backend is on every row.",
        "OpenAI": "MC_GPT_5.6_Luna, generation job 6252750, 273 of 273 ok. reasoning_effort=low, seed 42, "
                  "max_completion_tokens 8192, temperature not settable.",
        "Qwen": "MC_Qwen (Together), generation job 6252835, 273 of 273 generated; one answer (6765f489…) "
                "stopped at the 8,192-token limit in a repetition loop and was judged as given. Together, "
                "reasoning off, temperature 0, seed 42, max_tokens 8192. NOT the DeepInfra route "
                "Wonderbread's Qwen column uses.",
        "Gemma": "MC_Gemma_DeepInfra, generation job 6252837, 273 of 273 ok. DeepInfra, thinking already off "
                 "(no reasoning parameter is passed), temperature 0, seed 42, max_tokens 8192.",
        "Deepseek": "MC_Deepseek, generation job 6252838, called by its real id deepseek-flash, which "
                    "deepseek-reasoner (row 2 of Final_result) aliases. It has no thinking knob, so a system "
                    "message asks it to reason briefly — a request, not a limit, and a departure from "
                    "upstream's no-system-prompt conversation (the user's decision, 2026-10-31).",
    }
    for slot in TEMPO_COLS:
        if slot == "XAI":
            out.append(("MultiChallenge", "XAI = grok-3-mini", "blank — not run on this benchmark",
                        "Tempo_results only; XAI was never run on MultiChallenge or Wonderbread."))
            continue
        t = by[slot]
        total, status = judge_counts("mc", t["stem"])
        out.append(("MultiChallenge", f"{slot} = {FINAL_MODEL[slot] if slot != 'Deepseek' else 'deepseek-flash'}",
                    f"{total} judge rows, {status}",
                    mc_notes[slot] + f" Scored on {t['mc']['n']} items."))
    out += [
        ("Wonderbread", "every column", f"{wb_dir}/WB_Judges/results/full/haiku-5.5/{{qa,sop}}/<slug>.jsonl",
         f"Added {DATE} at the user's request, recomputed from the judge rows on disk (build_judge_summary.py, "
         "record_haiku_judge.py). EVERY CELL IS AN LLM JUDGE'S GRADE: Claude Haiku 5.5, three replicates. "
         "QA — four separate calls per answer, integer 1-3 where 1 IS BEST, so lower is better, the opposite "
         "of every other row in this workbook (the labels say so); each row is the mean over answers and "
         "replicates. SOP generation — one call per line in both directions; precision, recall, F1 and "
         "ordering from wb_eval_core.sop_metrics, higher is better, mean over SOPs and replicates. Upstream "
         "defines no composite, so there is no overall row. Haiku 5.5's error rate against human labels is "
         "UNMEASURED (JUDGE_SELECTION.md §8). SOP Improvement is out of scope (§7.5)."),
        ("Wonderbread", "coverage — not the same units for every model",
         "Gemini, OpenAI 120 QA / 486 SOP; Qwen 113 / 482; Gemma 115 / 482",
         "A unit whose screenshots exceed the producer's per-request image cap is recorded as "
         "over_image_limit — never sent, never judged, not a wrong answer. So the Qwen and Gemma means are "
         "over fewer units than Gemini's and OpenAI's, and not over a subset the others also share."),
    ]
    wb_notes = {
        "Gemini": "WB_Gemini_Flash3.5lite_OpenRouter, generation job 6648995. OpenRouter, thinking effort "
                  "minimal, seed 42, max_tokens 8192; no unit over a cap.",
        "OpenAI": "WB_GPT_5.6_Luna, generation job 6648994. reasoning_effort=low, seed 42, "
                  "max_completion_tokens 8192; no unit over a cap.",
        "Qwen": "WB_Qwen_DeepInfra, the RERUN — generation job 8098450, DeepInfra, thinking off, 30-image cap, "
                "7 of 120 QA and 4 of 486 SOP units over it. The Together run (10-image cap, 84 QA / 384 SOP "
                "scored) is archived under WB_Judges/archive/qwen_together_2026-11/ and is NOT in this "
                "sheet. Six of the 482 SOP answers stopped at the 8,192-token limit and are judged as the "
                "complete answers they are, as the Together run's were. The SOP judge ran as five shards "
                "(jobs 8167758 + merge 8167759, after a cancelled single-stream job 8127980).",
        "Gemma": "WB_Gemma_DeepInfra, generation job 6648997. DeepInfra, thinking already off, 32-image cap, "
                 "5 of 120 QA and 4 of 486 SOP units over it.",
    }
    for slot in TEMPO_COLS:
        if slot == "XAI":
            out.append(("Wonderbread", "XAI = grok-3-mini", "blank — not run on this benchmark",
                        "Tempo_results only; XAI was never run on MultiChallenge or Wonderbread."))
        elif slot == "Deepseek":
            out.append(("Wonderbread", "Deepseek", "blank — text-only model, not an unrun benchmark",
                        "deepseek-flash takes no images and every Wonderbread request carries screenshots, so "
                        "this is a ceiling and not an outstanding run, the way DocVQA's Deepseek blank is. "
                        "Shown as N/A in Results/Haiku_judge_summary.md."))
        else:
            t = by[slot]
            qa_total, qa_status = judge_counts("wb", t["stem"], "qa")
            sop_total, sop_status = judge_counts("wb", t["stem"], "sop")
            out.append(("Wonderbread", f"{slot} = {FINAL_MODEL[slot]}",
                        f"QA {qa_total} judge rows {qa_status}; SOP {sop_total} rows {sop_status}",
                        wb_notes[slot] + f" Scored on {t['qa']['n_answers']} QA answers and "
                                         f"{t['sop']['n_sops']} SOPs."))
    return out


def style(cell, spec):
    cell.font, cell.fill, cell.alignment = copy.copy(spec["font"]), copy.copy(spec["fill"]), copy.copy(spec["alignment"])


def write_sheet(wb, name, columns, config, rows, data):
    if name in wb.sheetnames:
        del wb[name]
    ws = wb.create_sheet(name, index=wb.sheetnames.index("Provenance"))
    ws.cell(1, 1, "Splits")
    ws.cell(2, 1, "Model / config" if len(columns) == len(TEMPO_COLS) else "Model")
    for j, col in enumerate(columns, 2):
        ws.cell(1, j, col)
        ws.cell(2, j, config[col])
    for c in range(1, len(columns) + 2):
        style(ws.cell(1, c), HEAD)
        style(ws.cell(2, c), CONF)
    for i, label in enumerate(rows, 3):
        bold = label.startswith("Overall Score")
        ws.cell(i, 1, label).font = Font(bold=bold)
        for j, col in enumerate(columns, 2):
            cell = ws.cell(i, j, data[label].get(col))
            cell.number_format = "0.0000"
            cell.font = Font(bold=bold)
    ws.column_dimensions["A"].width = 38
    for j in range(2, len(columns) + 2):
        ws.column_dimensions[ws.cell(1, j).column_letter].width = 13
    ws.row_dimensions[2].height = 52
    ws.freeze_panes = "B3"


def write_provenance(wb, label, rows):
    ws = wb["Provenance"]
    first = next((r for r in range(4, ws.max_row + 1) if ws.cell(r, 1).value in SHEETS), None)
    if first:
        tail = {ws.cell(r, 1).value for r in range(first, ws.max_row + 1)}
        assert tail <= set(SHEETS), f"Provenance rows after row {first} are not all ours: {tail}"
        ws.delete_rows(first, ws.max_row - first + 1)
    title = ws.cell(1, 1).value
    tag = f"; MultiChallenge and Wonderbread (Haiku 5.5 judged) added {DATE}"
    if tag not in title:
        ws.cell(1, 1).value = title.rstrip(".") + tag + "."
    for r, row in enumerate(rows, ws.max_row + 1):
        for c, v in enumerate(row, 1):
            cell = ws.cell(r, c, v)
            cell.alignment = copy.copy(WRAP)


def diff(before_path, after_wb):
    """Every cell of `before_path` that is not ours must be unchanged — value and style. Returns notes."""
    before = load_workbook(before_path)
    notes = []
    kept = [s for s in after_wb.sheetnames if s not in SHEETS]
    assert [s for s in before.sheetnames] == kept, f"sheet order changed: {before.sheetnames} -> {kept}"
    for name in before.sheetnames:
        a, b = before[name], after_wb[name]
        rows = a.max_row if name != "Provenance" else a.max_row
        changed = 0
        for r in range(1, rows + 1):
            for c in range(1, a.max_column + 1):
                x, y = a.cell(r, c), b.cell(r, c)
                # a cell's style proxies do not compare by value; the copies do
                if (x.value != y.value or x.number_format != y.number_format
                        or copy.copy(x.font) != copy.copy(y.font) or copy.copy(x.fill) != copy.copy(y.fill)
                        or copy.copy(x.alignment) != copy.copy(y.alignment)):
                    if name == "Provenance" and (r, c) == (1, 1):
                        changed += 1
                        continue
                    if name == "Provenance" and a.cell(r, 1).value in SHEETS:
                        continue                     # a previous run's rows, replaced by this one
                    raise SystemExit(f"{before_path} › {name} › R{r}C{c}: {x.value!r} -> {y.value!r}")
        notes.append(f"{name}: {rows} rows checked, unchanged" + (" (title line extended)" if changed else ""))
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    T = bj.collect({bj.SKIPPED_QID})
    data = values(T)
    prov = provenance(T)
    rows_of = {"MultiChallenge": MC_ROWS, "Wonderbread": WB_ROWS}
    built = {}
    for path, columns, model_row in ((TEMPO, TEMPO_COLS, None), (FINAL, FINAL_COLS, FINAL_MODEL)):
        wb = load_workbook(path)
        for name in SHEETS:
            config = TEMPO_MODEL[name] if model_row is None else model_row
            write_sheet(wb, name, columns, config, rows_of[name], data[name])
        write_provenance(wb, os.path.basename(path), prov if model_row is None else
                         [p for p in prov if not p[1].startswith("XAI")])
        for note in diff(path, wb):
            print(f"{os.path.basename(path)} › {note}")
        built[path] = wb
    # a Final cell that differs from Tempo's stops build_results_db.py; stop here instead
    for name in SHEETS:
        for label in rows_of[name]:
            for col in FINAL_COLS:
                a = built[TEMPO][name].cell(rows_of[name].index(label) + 3, TEMPO_COLS.index(col) + 2).value
                b = built[FINAL][name].cell(rows_of[name].index(label) + 3, FINAL_COLS.index(col) + 2).value
                assert a == b, (name, label, col, a, b)
    if args.dry_run:
        print("dry run: nothing written")
        return
    for path, wb in built.items():
        fd, tmp = tempfile.mkstemp(suffix=".xlsx", dir=HERE)
        os.close(fd)
        try:
            wb.save(tmp)
            shutil.move(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        print(f"written: {path}")


if __name__ == "__main__":
    main()
