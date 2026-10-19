#!/usr/bin/env python3
"""Lay the six evaluation criteria out as one `Criteria` sheet, from the files the criteria scripts
wrote — never from another summary.

    python3 .claude/scripts/build_criteria_workbook.py --out <draft.xlsx>

Reads only `Awareness_in_LLM/criteria_awareness.csv` (aware_criteria.py) and each selected model's
`criteria_summary.json` (criteria_report.py), and writes a one-sheet draft for the user to look at.
It does not touch Tempo_results.xlsx or Final_result.xlsx: a criterion reaches those only after the
user confirms the run, the same rule as accuracy (criteria-runs.md).

Two blocks, because the criteria come from different rows: 1, 2, 3 and 6 per benchmark x model from
the main run; 4 and 5 per benchmark x task x model from the tagged runs on the subsample. A blank
says why it is blank, in the Note column.
"""
import argparse
import csv
import json
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NEG = os.path.join(ROOT, "Interpersonal_processes_benchmarks", "NegotiationToM")
EMO = os.path.join(ROOT, "Interpersonal_processes_benchmarks", "EmoBench")
AWARE_CSV = os.path.join(ROOT, "Transition_processes_benchmarks", "Awareness_in_LLM", "criteria_awareness.csv")

SLOTS = ["Gemini", "OpenAI", "Qwen", "Gemma", "Deepseek"]
LEGACY = "legacy runner: its rows carry no per-row usage fields, so no estimate (evaluation-criteria.md)"
LEGACY_REPS = "not run: the legacy runner has no tag, subsample or perturbation support (phase 2)"

# benchmark -> slot -> (model folder behind the reported number, model id, why it is blank if it is)
FOLDERS = {
    "NegotiationToM": {
        "Gemini": (os.path.join(NEG, "NEG_Gemini_Flash3.5lite_OpenRouter"), "google/gemini-3.5-flash-lite", ""),
        "OpenAI": (os.path.join(NEG, "NEG_GPT_5.6_Luna"), "gpt-5.6-luna", ""),
        "Qwen": (os.path.join(NEG, "NEG_Qwen"), "Qwen/Qwen3.5-9B", LEGACY),
        "Gemma": (os.path.join(NEG, "NEG_Gemma"), "google/gemma-4-31B-it", LEGACY),
        "Deepseek": (os.path.join(NEG, "NEG_Deepseek"), "deepseek-reasoner", LEGACY),
    },
    "EmoBench": {
        "Gemini": (os.path.join(EMO, "EMO_Gemini_Flash3.5lite_OpenRouter"), "google/gemini-3.5-flash-lite",
                   "runner writes no latency_s, n_attempts or parse_ok and has no perturbation support; not run"),
        "OpenAI": (os.path.join(EMO, "EMO_GPT_5.6_Luna"), "gpt-5.6-luna", ""),
        "Qwen": (os.path.join(EMO, "EMO_Qwen"), "Qwen/Qwen3.5-9B", LEGACY),
        "Gemma": (os.path.join(EMO, "EMO_Gemma"), "google/gemma-4-31B-it", LEGACY),
        "Deepseek": (os.path.join(EMO, "EMO_Deepseek"), "deepseek-reasoner", LEGACY),
    },
}

USAGE = [  # (header, key, number format)
    ("Rows", "rows", "0"),
    ("1 · $ per 1k correct", "1_usd_per_1k_correct", "0.0000"),
    ("1 · $ whole run", "1_cost_usd_full_run", "0.0000"),
    ("2 · output tokens, mean", "2_output_tokens_mean", "0.0"),
    ("2 · output tokens, p95", "2_output_tokens_p95", "0"),
    ("2 · reasoning tokens, mean", "2_reasoning_tokens_mean", "0.0"),
    ("2 · tokens wrong ÷ right", "2_wrong_over_right_tokens", "0.00"),
    ("3 · latency p50 (s)", "3_latency_p50_s", "0.00"),
    ("3 · latency p95 (s)", "3_latency_p95_s", "0.00"),
    ("6 · parse ok", "6_parse_ok_rate", "0.0000"),
    ("6 · empty", "6_empty_rate", "0.0000"),
    ("6 · truncated", "6_truncation_rate", "0.0000"),
    ("6 · rows retried", "6_retried_row_rate", "0.0000"),
    ("6 · one config", "6_one_config", "0"),
]
STAB = ["Items", "4 · identical in all 5", "4 · pass@1", "4 · pass^5", "4 · accuracy spread",
        "5 · noise flip (rep1 vs rep2)", "5 · shuffle excess flip", "shuffle 95% CI", "shuffle p",
        "5 · paraphrase excess flip", "paraphrase 95% CI", "paraphrase p"]

BOLD, TITLE = Font(bold=True), Font(bold=True, size=13)
HEAD_FILL = PatternFill("solid", fgColor="DDE4EE")


def summary(folder):
    path = os.path.join(folder, "criteria_summary.json")
    return json.load(open(path)) if os.path.exists(path) else None


def header(ws, row, names):
    for col, name in enumerate(names, 1):
        cell = ws.cell(row, col, name)
        cell.font, cell.fill = BOLD, HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="top")


def put(ws, row, values, formats):
    for col, (value, fmt) in enumerate(zip(values, formats), 1):
        cell = ws.cell(row, col, value)
        if fmt and isinstance(value, (int, float)):
            cell.number_format = fmt


def usage_rows():
    with open(AWARE_CSV) as fh:
        aware = {r["slot"]: r for r in csv.DictReader(fh)}
    def number(key, text):
        if text == "not reported":
            return text
        return int(text) if key == "rows" else float(text)

    for slot in SLOTS:
        r = aware[slot]
        yield ["AwareBench", slot, r["model"], *[number(k, r[k]) for _, k, _ in USAGE],
               "full run; prices " + r["1_price_source"]]
    for bench, slots in FOLDERS.items():
        for slot in SLOTS:
            folder, model, why = slots[slot]
            s = summary(folder)
            main = s and s.get("main_run")
            if main and "status" not in main:
                by_task = ", ".join(f"{t} {v}" for t, v in main["2_wrong_over_right_tokens_by_task"].items())
                yield [bench, slot, model, *[main[k] for _, k, _ in USAGE],
                       f"main run; tokens wrong ÷ right by task: {by_task}; prices {main['1_price_source']}"]
            else:
                yield [bench, slot, model, *[None] * len(USAGE), (main or {}).get("status") or why]


def p_value(p):
    """A p below what four decimals can show is written as a bound, not as 0."""
    return "<0.0001" if p < 1e-4 else p


def stab_rows():
    with open(AWARE_CSV) as fh:
        aware = {r["slot"]: r for r in csv.DictReader(fh)}
    for slot in SLOTS:
        r = aware[slot]
        yield ["AwareBench", "all", slot, r["model"], *[None] * len(STAB),
               f"{r['4_identical_answer_rate']}; full-run shuffle flip {r['5_shuffle_flip_rate_full_run']} "
               "(not noise-corrected); paraphrase not defined"]
    for bench, slots in FOLDERS.items():
        for slot in SLOTS:
            folder, model, why = slots[slot]
            s = summary(folder)
            tasks = s and s.get("tasks")
            if not isinstance(tasks, dict):
                why = LEGACY_REPS if why == LEGACY else why
                yield [bench, "all", slot, model, *[None] * len(STAB), tasks or why or "not run"]
                continue
            for task, t in tasks.items():
                st, rb = t["stability"], t["robustness"]
                sh, pa = rb["perturb-shuffle"], rb["perturb-paraphrase"]
                ci = lambda x: f"{x['excess_flip_95ci_cluster'][0]:+.3f} .. {x['excess_flip_95ci_cluster'][1]:+.3f}"  # noqa: E731
                note = ("5 repeats byte-identical with the seed pinned: noise floor 0, so every "
                        "perturbation flip counts as excess" if rb["baseline_flip_rep1_vs_rep2"] == 0
                        else "repeats vary with the seed pinned")
                yield [bench, task, slot, model, t["items"], st["identical_all_5"], st["pass@1"], st["pass^5"],
                       st["accuracy_max"] - st["accuracy_min"], rb["baseline_flip_rep1_vs_rep2"],
                       sh["excess_flip"], ci(sh), p_value(sh["excess_mcnemar"]["p"]),
                       pa["excess_flip"], ci(pa), p_value(pa["excess_mcnemar"]["p"]), note]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="where to write the one-sheet draft")
    args = ap.parse_args()

    wb = Workbook()
    ws = wb.active
    ws.title = "Criteria"
    ws.cell(1, 1, "Six evaluation criteria — DRAFT, not recorded in either workbook").font = TITLE
    ws.cell(2, 1, "Definitions: .claude/references/evaluation-criteria.md and criteria-runs.md. "
                  "A blank is no usable number, never a zero; the Note says why.")

    row = 4
    ws.cell(row, 1, "Criteria 1, 2, 3, 6 — from the main run, the rows behind each accuracy number").font = BOLD
    row += 1
    header(ws, row, ["Benchmark", "Slot", "Model", *[h for h, _, _ in USAGE], "Note"])
    formats = [None, None, None, *[f for _, _, f in USAGE], None]
    for values in usage_rows():
        row += 1
        put(ws, row, values, formats)

    row += 2
    ws.cell(row, 1, "Criteria 4, 5 — five repeats and two perturbations on a fixed subsample, "
                    "seed pinned; robustness is flips in excess of the rep1-vs-rep2 noise").font = BOLD
    row += 1
    header(ws, row, ["Benchmark", "Task", "Slot", "Model", *STAB, "Note"])
    formats = [None] * 4 + ["0", "0.000", "0.000", "0.000", "0.000", "0.000", "+0.000;-0.000;0.000", None,
                            "0.0000", "+0.000;-0.000;0.000", None, "0.0000", None]
    for values in stab_rows():
        row += 1
        put(ws, row, values, formats)

    ws.freeze_panes = "D6"
    for col in range(1, ws.max_column + 1):
        letter = ws.cell(5, col).column_letter
        ws.column_dimensions[letter].width = 14
    for letter, width in (("A", 15), ("C", 26), ("R", 60)):
        ws.column_dimensions[letter].width = width
    wb.save(args.out)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
