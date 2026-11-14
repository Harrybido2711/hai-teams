#!/usr/bin/env python3
"""Record PlanBench's 10% PILOT in both workbooks.

    python3 Results/record_planbench.py            # write Tempo_results.xlsx and Final_result.xlsx
    python3 Results/record_planbench.py --dry-run  # compute and diff, write nothing
    python3 Results/result_rows.py                 # then: highlight the result row, refresh Summary

The user's instruction, 2026-11-13: write PlanBench into the result workbook now, from the pilot.
**The sheet is named `PlanBench (pilot)` because it is not the full run**: every 10th instance of
each prompt file, 1,563 rows a model. When the full run is recorded, point `SOURCE` at `results/`,
rename `SHEET` (and its entry in `build_results_db.BENCHMARKS`), and rerun.

It adds that one sheet after `Multi-party Goal Tracking` and its rows in `Provenance` after that
sheet's; **every other cell is left alone**, and the script proves it before it writes. Re-running
replaces that sheet and those rows and nothing else.

The numbers are recomputed from each model's `<slug>_scored.jsonl` — the rows `plan_scorer.py`
(`pb_lenient_v1`) wrote — and must equal the scorer's own by-task table. A task row is correct ÷
rows over every domain shipping the task; `Overall Score` is correct ÷ rows over all of them, the
"N of 1,513" the benchmark page reports. The two `obfuscated_deceptive_logistics` slices score 0 for
every model by construction (upstream's parser is broken) and are in no row, as in that table.
"""
import argparse
import collections
import copy
import csv
import glob
import json
import os

from openpyxl import load_workbook
from openpyxl.styles import Font

import result_rows as rr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, "Transition_processes_benchmarks", "LLMs-Planning_bench")
TEMPO, FINAL = rr.TEMPO, rr.FINAL
SHEET = "PlanBench (pilot)"
AFTER = "Multi-party Goal Tracking"
DATE = "2026-11-13"
SAMPLE, ROWS_PER_MODEL = "pilot10", 1563
BROKEN = "obfuscated_deceptive_logistics"
# the pilot run of each selected model, at the configuration the full run keeps
SOURCE = {"Gemini": "PLAN_Gemini_Flash3.5lite_OpenRouter/results_pilot10",
          "OpenAI": "PLAN_GPT_5.6_Luna/results_pilot10",
          "Qwen": "PLAN_Qwen_DeepInfra/results_archive_budget_prompt_20261113/results_pilot10",
          "Gemma": "PLAN_Gemma_DeepInfra/results_pilot10",
          "Deepseek": "PLAN_Deepseek/results_pilot10"}
TEMPO_COLS = ["Gemini", "OpenAI", "XAI", "Qwen", "Gemma", "Deepseek"]
FINAL_COLS = ["Gemini", "OpenAI", "Qwen", "Gemma", "Deepseek"]
FINAL_MODEL = {"Gemini": "gemini-3.5-flash-lite", "OpenAI": "gpt-5.6-luna", "Qwen": "Qwen/Qwen3.5-9B",
               "Gemma": "google/gemma-4-31B-it", "Deepseek": "deepseek-reasoner"}
TEMPO_MODEL = {"Gemini": "gemini-3.5-flash-lite (OpenRouter, minimal)", "OpenAI": "gpt-5.6-luna (effort=low)",
               "XAI": "grok-3-mini",
               "Qwen": "Qwen/Qwen3.5-9B (DeepInfra, reasoning off, budget sentence; pilot, no retry)",
               "Gemma": "google/gemma-4-31B-it (DeepInfra)",
               "Deepseek": "deepseek-flash (prompt ceiling + 16,384 budget; pilot, no retry)"}
LLM_PLANS = "task_3_plan_verification_with_llm_plans"
TASKS = ["task_1_plan_generation", "task_2_plan_optimality", "task_3_plan_verification",
         LLM_PLANS, "task_4_plan_reuse", "task_5_plan_generalization",
         "task_6_replanning", "task_7_plan_execution", "task_8_1_goal_shuffling", "task_8_2_full_to_partial",
         "task_8_3_partial_to_full"]


def one(folder, pattern):
    files = glob.glob(os.path.join(BENCH, folder, pattern))
    assert len(files) == 1, (folder, pattern, files)
    return files[0]


def collect():
    """{slot: (values {task: accuracy}, facts)} — refuses anything but a whole pilot under one config."""
    out = {}
    for slot, folder in SOURCE.items():
        path = one(folder, "*_scored.jsonl")
        with open(path) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert len(rows) == len({r["uid"] for r in rows}) == ROWS_PER_MODEL, (slot, len(rows))
        assert {r["sample"] for r in rows} == {SAMPLE}, (slot, "not the pilot sample")
        assert {r["scorer"] for r in rows} == {"pb_lenient_v1"}, (slot, "more than one scorer")
        with open(one(folder, "*_health_overall.csv")) as fh:
            health = next(csv.DictReader(fh))
        assert (health["sample"], int(health["rows"]), int(health["expected"]), int(health["configs"])) == \
            (SAMPLE, ROWS_PER_MODEL, ROWS_PER_MODEL, 1), (slot, health)
        scored = [r for r in rows if r["domain"] != BROKEN]
        n, right = collections.Counter(r["task"] for r in scored), collections.Counter(
            r["task"] for r in scored if r["correct"])
        assert sorted(n) == sorted(TASKS), (slot, sorted(n))
        with open(one(folder, "*_scores_by_task.csv")) as fh:   # the scorer's own table must say the same
            table = {r["task"]: (int(r["rows"]), int(r["correct"])) for r in csv.DictReader(fh)}
        assert table == {t: (n[t], right[t]) for t in TASKS}, (slot, "scored rows and by-task table differ")
        values = {t: round(right[t] / n[t], 4) for t in TASKS}
        facts = {"path": os.path.relpath(path, ROOT), "n": dict(n), "rows": sum(n.values()),
                 "correct": sum(right.values()), "empty": int(health["empty"]),
                 "empty_scored": sum(1 for r in scored if r["empty"]),
                 "cut": int(health["finish_length"]),
                 "cut_scored": sum(1 for r in scored if r["finish_reason"] == "length"),
                 "cut_correct": sum(1 for r in scored if r["finish_reason"] == "length" and r["correct"]),
                 "null_gold": sum(1 for r in scored if r["task"] == LLM_PLANS
                                  and "AttributeError" in str(r.get("parse_error") or ""))}
        out[slot] = (values, facts)
    counts = {json.dumps([f["n"], f["null_gold"]], sort_keys=True) for _, f in out.values()}
    assert len(counts) == 1, "the models were not scored on the same rows"
    total = next(iter(out.values()))[1]["rows"]
    label = f"Overall Score (pooled, {total:,} pilot rows)"
    for values, facts in out.values():
        values[label] = round(facts["correct"] / facts["rows"], 4)
    return out, label


def provenance(data, label):
    """[(sheet, column, source, note)] — the rows placed after Multi-party Goal Tracking's."""
    f = {slot: facts for slot, (_, facts) in data.items()}
    n = f["Gemini"]["n"]

    def src(slot):
        s = f[slot]
        return (f"{s['correct']:,} of {s['rows']:,} correct; {ROWS_PER_MODEL:,} rows, {s['empty']} empty, "
                f"{s['cut']} cut at the output cap ({s['cut_scored']} of them in the scored rows)")

    base = "Transition_processes_benchmarks/LLMs-Planning_bench/"
    return [
        (SHEET, "every column — A PILOT, NOT THE FULL RUN", base + "PLAN_*/results_pilot10/<slug>_scored.jsonl",
         f"Added {DATE} on the user's instruction to write PlanBench in from its pilot. Every 10th instance of each "
         f"of the 53 prompt files: {ROWS_PER_MODEL:,} rows a model, against 15,568 in the full set, which was "
         "submitted on Quest 2026-11-13 and replaces this sheet once it is recorded. Recomputed from each "
         "model's scored rows on disk by Results/record_planbench.py and equal to the scorer's own by-task "
         "table. Scorer pb_lenient_v1 (plan_scorer.py): upstream's text-to-PDDL parsers and the VAL validator, "
         "no LLM. A task row is correct ÷ rows over every domain that ships the task; an empty answer or one "
         "cut at the output cap is scored as it stands, in the denominator."),
        (SHEET, label, "correct ÷ rows over the 11 task rows — the highlighted result row",
         "Pooled, not the mean of the task rows: the tasks hold " + " · ".join(str(n[t]) for t in TASKS) +
         " rows, in sheet order. It is the 'N of 1,513' the benchmark page reports for every pilot. Ours: "
         "upstream reports each task and domain on its own and defines no composite."),
        (SHEET, "50 rows a model are in no row", "obfuscated_deceptive_logistics — tasks 1 and 2",
         "Upstream's parser for the obfuscated domain is broken, so both slices score 0 for every model "
         "whatever it answered; the scorer's by-task table leaves them out and so does this sheet. Whether to "
         "repair the parser or drop the slices is the user's call, open."),
        (SHEET, LLM_PLANS, f"{n[LLM_PLANS]} pilot rows, {f['Gemini']['null_gold']} of them with no gold answer",
         "49 of this task's 99 gold answers are null upstream; the scorer cannot read a null and scores the row "
         f"0 for every model, so {(n[LLM_PLANS] - f['Gemini']['null_gold']) / n[LLM_PLANS]:.1f} is the most any "
         "model can reach on this row. Whether a null gold means 'the plan is valid' is not established."),
        (SHEET, "Gemini = gemini-3.5-flash-lite", src("Gemini"),
         "PLAN_Gemini_Flash3.5lite_OpenRouter, job 8157875. OpenRouter, reasoning effort minimal, max_tokens "
         "8192, seed 42."),
        (SHEET, "OpenAI = gpt-5.6-luna", src("OpenAI"),
         "PLAN_GPT_5.6_Luna, job 8157873. reasoning_effort=low, max_completion_tokens 16384, seed 42."),
        (SHEET, "XAI = grok-3-mini", "blank — not run on this benchmark",
         "Tempo_results only; XAI was out of the selected models before PlanBench ran."),
        (SHEET, "Qwen = Qwen/Qwen3.5-9B — a lower bound", src("Qwen"),
         "PLAN_Qwen_DeepInfra/results_archive_budget_prompt_20261113, job 8432297: the configuration the user "
         "kept on 2026-11-13 — DeepInfra, reasoning.enabled=False, temperature 0, max_tokens 8192, seed 42, and "
         "a second system message stating the 8,192-token budget. It deliberates in its answer and the cap "
         f"cuts it: {f['Qwen']['cut_correct']} of its {f['Qwen']['cut_scored']} cut rows in the scored tasks are "
         "correct, so THIS COLUMN IS A LOWER BOUND SET BY THE CAP. The pilot "
         "has no retry; the full run calls a cut row again under a stricter ceiling, then forces an answer. "
         "The other three Qwen pilots (cap 4096; 8192 without the budget sentence, 653 correct; answer-only, "
         "448) are archived beside it and are not in this sheet."),
        (SHEET, "Gemma = google/gemma-4-31B-it", src("Gemma"),
         "PLAN_Gemma_DeepInfra, job 8157876, shard 4 resumed as job 8215431. DeepInfra, no reasoning "
         "parameter, temperature 0, max_tokens 4096, seed 42. Its cut rows get no retry, by design."),
        (SHEET, "Deepseek = deepseek-flash", src("Deepseek"),
         "PLAN_Deepseek, job 8436559: the rerun with the 16,384-token budget stated in its prompt ceiling (the "
         "run without it, 1,377 correct, is archived). Called by its real id deepseek-flash, which "
         "deepseek-reasoner (row 2 of Final_result) aliases. temperature 0, max_tokens 16384, seed 42. Its empty "
         f"rows are all task_2: every attempt spent the whole cap reasoning; {f['Deepseek']['empty_scored']} of "
         "them are in the scored rows and count as wrong. The pilot has no retry; the full run has."),
    ]


def write_sheet(wb, columns, config, data, label):
    if SHEET in wb.sheetnames:
        del wb[SHEET]
    ws = wb.create_sheet(SHEET, index=wb.sheetnames.index(AFTER) + 1)
    ws.cell(1, 1, "Splits")
    ws.cell(2, 1, "Model / config" if len(columns) == len(TEMPO_COLS) else "Model")
    for j, col in enumerate(columns, 2):
        ws.cell(1, j, col)
        ws.cell(2, j, config[col])
    for c in range(1, len(columns) + 2):
        rr.style(ws.cell(1, c), rr.HEAD)
        rr.style(ws.cell(2, c), rr.CONF)
    for i, row in enumerate(TASKS + [label], 3):
        bold = row == label
        ws.cell(i, 1, row).font = Font(bold=bold)
        for j, col in enumerate(columns, 2):
            cell = ws.cell(i, j, data[col][0][row] if col in data else None)
            cell.number_format = rr.NUM
            cell.font = Font(bold=bold)
    ws.column_dimensions["A"].width = 40
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
    before = [r for r in range(4, ws.max_row + 1) if ws.cell(r, 1).value == AFTER]
    assert before, f"no {AFTER} rows in Provenance to place these after"
    at = before[-1] + 1
    ws.insert_rows(at, len(rows))
    for r, row in enumerate(rows, at):
        for c, v in enumerate(row, 1):
            ws.cell(r, c, v).alignment = copy.copy(rr.WRAP)
    title = ws.cell(1, 1).value
    tag = f"; PlanBench pilot added {DATE}"
    if tag not in title:
        ws.cell(1, 1).value = title.rstrip(".") + tag + "."


def diff(before_path, after_wb):
    """Every cell of `before_path` that is not ours must be unchanged — value and style. Returns notes."""
    before = load_workbook(before_path)
    notes = []
    assert [s for s in before.sheetnames if s != SHEET] == [s for s in after_wb.sheetnames if s != SHEET], \
        f"sheet order changed: {before.sheetnames} -> {after_wb.sheetnames}"
    for name in before.sheetnames:
        if name == SHEET:
            continue
        a, b = before[name], after_wb[name]
        if name == "Provenance":
            def kept(ws):
                return [[rr._key(ws.cell(r, c)) for c in range(1, a.max_column + 1)]
                        for r in range(2, ws.max_row + 1) if ws.cell(r, 1).value != SHEET]
            if kept(a) != kept(b):
                raise SystemExit(f"{before_path} › Provenance: a pre-existing row changed or moved out of order")
            notes.append(f"Provenance: {len(kept(a))} pre-existing rows below the title, unchanged and in order")
            continue
        for r in range(1, a.max_row + 1):
            for c in range(1, a.max_column + 1):
                if rr._key(a.cell(r, c)) != rr._key(b.cell(r, c)):
                    raise SystemExit(f"{before_path} › {name} › R{r}C{c}: {a.cell(r, c).value!r} -> {b.cell(r, c).value!r}")
        notes.append(f"{name}: {a.max_row} rows checked, unchanged")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data, label = collect()
    for slot, (values, facts) in data.items():
        print(f"{slot:9} {facts['correct']:5,} of {facts['rows']:,} = {values[label]:.4f}  {facts['path']}")
    prov = provenance(data, label)
    built = {}
    for path, columns, config in ((TEMPO, TEMPO_COLS, TEMPO_MODEL), (FINAL, FINAL_COLS, FINAL_MODEL)):
        wb = load_workbook(path)
        write_sheet(wb, columns, config, data, label)
        write_provenance(wb, prov if path == TEMPO else [p for p in prov if not p[1].startswith("XAI")])
        for note in diff(path, wb):
            print(f"{os.path.basename(path)} › {note}")
        built[path] = wb
    # a Final cell that differs from Tempo's stops build_results_db.py; stop here instead
    for i, row in enumerate(TASKS + [label], 3):
        for col in FINAL_COLS:
            a = built[TEMPO][SHEET].cell(i, TEMPO_COLS.index(col) + 2).value
            b = built[FINAL][SHEET].cell(i, FINAL_COLS.index(col) + 2).value
            assert a == b, (row, col, a, b)
    if args.dry_run:
        print("dry run: nothing written")
        return
    for path, wb in built.items():
        rr.save(wb, path)
    print("next: python3 Results/result_rows.py, then python3 .claude/scripts/build_results_db.py")


if __name__ == "__main__":
    main()
