#!/usr/bin/env python3
"""Record mpgt-eval (Multi-party Goal Tracking) in both workbooks.

    python3 Results/record_mpgt.py            # write Tempo_results.xlsx and Final_result.xlsx
    python3 Results/record_mpgt.py --dry-run  # compute and diff, write nothing

Layer 4 of the sync pass (`.claude/references/sync-and-consistency.md`), run only after the user has
confirmed the run (2026-11-13). It adds one sheet, `Multi-party Goal Tracking`, right after
`Awareness` (the transition benchmarks together), and its rows in `Provenance` right after
Awareness's; **every other cell is left alone**, and the script proves it before it writes — each
pre-existing cell, and each pre-existing Provenance row in order, is compared with the file it came
from, and a difference stops the run. Re-running replaces that sheet and those rows and nothing else.

The numbers are `mpgt_eval_core.summarize` on each model's rows on disk — the benchmark's one scorer,
`mpgt_lenient_v4` — never another summary. A style is slots correct / slots, an unread slot counted
wrong; `dst` and `goal` are the plain mean of the six styles, as the benchmark page reports them.
Upstream defines no composite, so there is no overall row. Then
`python3 .claude/scripts/build_results_db.py` rebuilds `results.db` from these.
"""
import argparse
import collections
import copy
import json
import os
import shutil
import sys
import tempfile

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "Transition_processes_benchmarks", "Multi-party_Goal_Tracking_bench")
sys.path.insert(0, BENCH)
import mpgt_eval_core as mp  # noqa: E402

TEMPO = os.path.join(HERE, "Tempo_results.xlsx")
FINAL = os.path.join(HERE, "Final_result.xlsx")
SHEET = "Multi-party Goal Tracking"
AFTER = "Awareness"
DATE = "2026-11-13"
FOLDER = {"Gemini": "MPGT_Gemini_Flash3.5lite_OpenRouter", "OpenAI": "MPGT_GPT_5.6_Luna",
          "Qwen": "MPGT_Qwen_DeepInfra", "Gemma": "MPGT_Gemma_DeepInfra", "Deepseek": "MPGT_Deepseek"}
TEMPO_COLS = ["Gemini", "OpenAI", "XAI", "Qwen", "Gemma", "Deepseek"]
FINAL_COLS = ["Gemini", "OpenAI", "Qwen", "Gemma", "Deepseek"]
FINAL_MODEL = {"Gemini": "gemini-3.5-flash-lite", "OpenAI": "gpt-5.6-luna", "Qwen": "Qwen/Qwen3.5-9B",
               "Gemma": "google/gemma-4-31B-it", "Deepseek": "deepseek-reasoner"}
TEMPO_MODEL = {"Gemini": "gemini-3.5-flash-lite (OpenRouter, minimal)", "OpenAI": "gpt-5.6-luna (effort=low)",
               "XAI": "grok-3-mini", "Qwen": "Qwen/Qwen3.5-9B (DeepInfra, reasoning off, forced answer past 8192)",
               "Gemma": "google/gemma-4-31B-it (DeepInfra)",
               "Deepseek": "deepseek-flash (prompt ceiling, reasoning budget 8192)"}
# build_results_db.py reads these labels: SECTIONS maps each header to its group row, GROUP_ROWS holds both
SECTION = {"dst": "Dialogue state tracking (DST), zero-shot", "goal": "Goal tracking, zero-shot"}
GROUP = {"dst": "dst (mean of 6 styles)", "goal": "goal (mean of 6 styles)"}
ROWS = [label for task in ("dst", "goal")
        for label in [SECTION[task]] + [f"{task}_{s}" for s in mp.STYLES] + [GROUP[task]]]

HEAD = dict(font=Font(bold=True), fill=PatternFill("solid", fgColor="DDE5F0"),
            alignment=Alignment(horizontal="center", vertical="center", wrap_text=True))
CONF = dict(font=Font(size=9), fill=PatternFill("solid", fgColor="FFF4E0"),
            alignment=Alignment(horizontal="center", vertical="center", wrap_text=True))
SECT_FILL = PatternFill("solid", fgColor="F2F2F2")
WRAP = Alignment(wrap_text=True, vertical="top")


def result_file(slot):
    folder = os.path.join(BENCH, FOLDER[slot], "results", "zero-shot")
    files = [f for f in os.listdir(folder) if f.endswith(".jsonl")]
    assert len(files) == 1, (slot, files)
    return os.path.join(folder, files[0])


def collect():
    """{slot: (values {row label: value}, facts)} — refuses a file that is not a whole, clean run."""
    out = {}
    for slot in FOLDER:
        path = result_file(slot)
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        uids = {r["uid"] for r in rows}
        status = collections.Counter(r.get("status") for r in rows)
        assert len(rows) == len(uids) == 348 and status == {"ok": 348}, (slot, len(rows), len(uids), status)
        assert not any(not (r.get("response") or "").strip() for r in rows), (slot, "empty response")
        configs = {json.dumps(r.get("config"), sort_keys=True) for r in rows}
        assert len(configs) == 1, (slot, "more than one config", len(configs))
        summary = mp.summarize(rows)
        values = {}
        for task in ("dst", "goal"):
            s = summary[task]
            assert s["complete"], (slot, task, "INCOMPLETE")
            for style in mp.STYLES:
                values[f"{task}_{style}"] = round(s["styles"][style]["lenient"], 4)
            values[GROUP[task]] = round(s["mean_over_styles"], 4)
        facts = {"path": os.path.relpath(path, ROOT), "status": dict(status),
                 "finish": dict(collections.Counter(r.get("finish_reason") for r in rows)),
                 "forced": sum(1 for r in rows if r.get("forced_answer"))}
        out[slot] = (values, facts)
    return out


def provenance(data):
    """[(sheet, column, source, note)] — the rows placed after Awareness's in Provenance."""
    f = {slot: facts for slot, (_, facts) in data.items()}

    def src(slot):
        s = f[slot]
        forced = f", forced {s['forced']}" if s["forced"] else ""
        return f"{s['status']['ok']} of 348 rows ok, 0 empty; finish {s['finish']}{forced}"

    return [
        (SHEET, "every column",
         "Transition_processes_benchmarks/Multi-party_Goal_Tracking_bench/MPGT_*/results/zero-shot/<slug>.jsonl",
         f"Added {DATE} after the user confirmed the run, recomputed from the rows on disk by "
         "mpgt_eval_core.summarize (Results/record_mpgt.py). Scorer mpgt_lenient_v4, ours: an automatic "
         "lenient matcher on the [MASK] slots — upstream's metric is exact match with the residue reviewed by a "
         "person, and upstream ships no scoring code. Zero-shot only; six upstream prompt styles x 29 dialogues "
         "per task, 389 DST and 225 goal slots a style. A style is slots correct / slots, an unread slot counted "
         "wrong; dst and goal are the plain mean of the six styles. No overall row: upstream defines no composite."),
        (SHEET, "off-count lists — three columns are lower bounds",
         "v4 reads a separate answer list only when it has one answer per slot",
         "A list with n != k answers is credited nothing, even where its items are right by position. OpenAI "
         "answers story and roleplay that way; Gemini and Qwen on fewer rows. Placed item i -> slot i: OpenAI "
         "dst 0.3419 / goal 0.2119 (above Deepseek on dst), Gemini 0.2601 / 0.1511, Qwen 0.1842 / 0.0637 "
         "(before its forced answers); Deepseek and Gemma unchanged. One rule for every model; keep or align is "
         "the user's call, open. DST's top three (Deepseek, Gemma, OpenAI) are tied under a paired bootstrap "
         "over dialogues."),
        (SHEET, "goal — low for every model", "the task, not the reader",
         "Models get the G / AG / CG type and the speaker right but name the goal in their own words (gold "
         "G ( LC , drink ( coffee ) ), a model G ( LC , get-info ( coffee ( place ) ) )). Dialogues 10 and 11 "
         "show no @ annotation outside the window, so outside the reasoning style no model writes the scheme "
         "for them."),
        (SHEET, "Gemini = gemini-3.5-flash-lite", src("Gemini"),
         "MPGT_Gemini_Flash3.5lite_OpenRouter, job 8428393. OpenRouter, reasoning effort minimal, max_tokens "
         "8192, seed 42. A lower bound by less than 0.02 (off-count lists)."),
        (SHEET, "OpenAI = gpt-5.6-luna", src("OpenAI"),
         "MPGT_GPT_5.6_Luna, job 8428392. reasoning_effort=low, max_completion_tokens 8192, seed 42. A lower "
         "bound: off-count lists in story and roleplay (row above)."),
        (SHEET, "XAI = grok-3-mini", "blank — not run on this benchmark",
         "Tempo_results only; XAI was out of the selected models before mpgt-eval ran."),
        (SHEET, "Qwen = Qwen/Qwen3.5-9B", src("Qwen"),
         "MPGT_Qwen_DeepInfra, job 8428397, then job 8610516 for the rows cut at the cap. DeepInfra, "
         "reasoning.enabled=False, temperature 0, max_tokens 8192, seed 42 — it deliberates in its answer "
         "anyway. The 45 rows cut at 8192 got one forced-answer turn (the cut text handed back, 'answer now', "
         "max_tokens 4096; the user, 2026-11-13): 27 answered, 18 repetition loops filled the 4096 too and are "
         "scored as stored (finish length). Before it: dst 0.1778 / goal 0.0533. dst_reasoning is 0: every "
         "intent on one ';' line, unread."),
        (SHEET, "Gemma = google/gemma-4-31B-it", src("Gemma"),
         "MPGT_Gemma_DeepInfra, job 8428394. DeepInfra, temperature 0, max_tokens 8192, seed 42."),
        (SHEET, "Deepseek = deepseek-flash", src("Deepseek"),
         "MPGT_Deepseek, job 8433789; the run at max_tokens 16384 (job 8428390) was cancelled at 23 rows and "
         "archived. Called by its real id deepseek-flash, which deepseek-reasoner (row 2 of Final_result) "
         "aliases. Prompt ceiling; reasoning budget 8192 (the user, 2026-11-12): a first call cut at 8192 gets "
         "a second with its reasoning handed back and thinking disabled, max_tokens 4096. No forced call "
         "reasoned again; 4 forced rows answered nothing readable, 2 of them refusals (about 0.005 on dst)."),
    ]


def style(cell, spec):
    cell.font, cell.fill, cell.alignment = copy.copy(spec["font"]), copy.copy(spec["fill"]), copy.copy(spec["alignment"])


def write_sheet(wb, columns, config, data):
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET, index=wb.sheetnames.index(AFTER) + 1)
    ws.cell(1, 1, "Splits")
    ws.cell(2, 1, "Model / config" if len(columns) == len(TEMPO_COLS) else "Model")
    for j, col in enumerate(columns, 2):
        ws.cell(1, j, col)
        ws.cell(2, j, config[col])
    for c in range(1, len(columns) + 2):
        style(ws.cell(1, c), HEAD)
        style(ws.cell(2, c), CONF)
    for i, label in enumerate(ROWS, 3):
        if label in SECTION.values():
            cell = ws.cell(i, 1, label)
            cell.font, cell.fill = Font(bold=True), copy.copy(SECT_FILL)
            for j in range(2, len(columns) + 2):
                ws.cell(i, j).fill = copy.copy(SECT_FILL)
            continue
        bold = label in GROUP.values()
        ws.cell(i, 1, label).font = Font(bold=bold)
        for j, col in enumerate(columns, 2):
            cell = ws.cell(i, j, data[col][0][label] if col in data else None)
            cell.number_format = "0.0000"
            cell.font = Font(bold=bold)
    ws.column_dimensions["A"].width = 38
    for j in range(2, len(columns) + 2):
        ws.column_dimensions[ws.cell(1, j).column_letter].width = 13
    ws.row_dimensions[2].height = 52
    ws.freeze_panes = "B3"


def write_provenance(wb, rows):
    ws = wb["Provenance"]
    ours = [r for r in range(4, ws.max_row + 1) if ws.cell(r, 1).value == SHEET]
    if ours:
        assert ours == list(range(ours[0], ours[-1] + 1)), f"{SHEET} Provenance rows are not contiguous: {ours}"
        ws.delete_rows(ours[0], len(ours))
    aware = [r for r in range(4, ws.max_row + 1) if ws.cell(r, 1).value == AFTER]
    assert aware, "no Awareness rows in Provenance to place these after"
    at = aware[-1] + 1
    ws.insert_rows(at, len(rows))
    for r, row in enumerate(rows, at):
        for c, v in enumerate(row, 1):
            cell = ws.cell(r, c, v)
            cell.alignment = copy.copy(WRAP)
    title = ws.cell(1, 1).value
    tag = f"; Multi-party Goal Tracking added {DATE}"
    if tag not in title:
        ws.cell(1, 1).value = title.rstrip(".") + tag + "."


def _key(cell):
    # a cell's style proxies do not compare by value; the copies do
    return (cell.value, cell.number_format, copy.copy(cell.font), copy.copy(cell.fill), copy.copy(cell.alignment))


def diff(before_path, after_wb):
    """Every cell of `before_path` that is not ours must be unchanged — value and style. Returns notes."""
    before = load_workbook(before_path)
    notes = []
    kept = [s for s in after_wb.sheetnames if s != SHEET]
    assert before.sheetnames == kept, f"sheet order changed: {before.sheetnames} -> {kept}"
    for name in before.sheetnames:
        if name == SHEET:
            continue
        a, b = before[name], after_wb[name]
        if name == "Provenance":
            width = a.max_column
            old = [[_key(a.cell(r, c)) for c in range(1, width + 1)] for r in range(2, a.max_row + 1)
                   if a.cell(r, 1).value != SHEET]
            new = [[_key(b.cell(r, c)) for c in range(1, width + 1)] for r in range(2, b.max_row + 1)
                   if b.cell(r, 1).value != SHEET]
            if old != new:
                bad = next(i for i, (x, y) in enumerate(zip(old + [None], new + [None])) if x != y)
                raise SystemExit(f"{before_path} › Provenance: pre-existing row {bad + 2} changed or moved out of order")
            title = a.cell(1, 1).value != b.cell(1, 1).value
            notes.append(f"Provenance: {len(old)} pre-existing rows below the title, unchanged and in order"
                         + (" (title line extended)" if title else ""))
            continue
        for r in range(1, a.max_row + 1):
            for c in range(1, a.max_column + 1):
                if _key(a.cell(r, c)) != _key(b.cell(r, c)):
                    raise SystemExit(f"{before_path} › {name} › R{r}C{c}: {a.cell(r, c).value!r} -> {b.cell(r, c).value!r}")
        notes.append(f"{name}: {a.max_row} rows checked, unchanged")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data = collect()
    for slot, (values, facts) in data.items():
        print(f"{slot:9} dst {values[GROUP['dst']]:.4f}  goal {values[GROUP['goal']]:.4f}  {facts['path']}")
    prov = provenance(data)
    built = {}
    for path, columns, config in ((TEMPO, TEMPO_COLS, TEMPO_MODEL), (FINAL, FINAL_COLS, FINAL_MODEL)):
        wb = load_workbook(path)
        write_sheet(wb, columns, config, data)
        write_provenance(wb, prov if path == TEMPO else [p for p in prov if not p[1].startswith("XAI")])
        for note in diff(path, wb):
            print(f"{os.path.basename(path)} › {note}")
        built[path] = wb
    # a Final cell that differs from Tempo's stops build_results_db.py; stop here instead
    for i, label in enumerate(ROWS, 3):
        for col in FINAL_COLS:
            a = built[TEMPO][SHEET].cell(i, TEMPO_COLS.index(col) + 2).value
            b = built[FINAL][SHEET].cell(i, FINAL_COLS.index(col) + 2).value
            assert a == b, (label, col, a, b)
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
