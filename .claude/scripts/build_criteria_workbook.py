#!/usr/bin/env python3
"""Build Criteria_results.xlsx: accuracy beside the six evaluation criteria, one sheet per
benchmark, one row per selected model, from the files the criteria scripts wrote.

    python3 .claude/scripts/build_criteria_workbook.py [--out Criteria_results.xlsx]

Sources, never another summary of them:
- accuracy — Final_result.xlsx, the reported number. A selected model whose run is finished but not
  yet confirmed has no cell there; its accuracy comes from the run's own overall file and the row is
  marked unconfirmed.
- 1, 2, 3, 6 — each model folder's criteria_summary.json `main_run` (criteria_report.py), and
  Awareness_in_LLM/criteria_awareness.csv (aware_criteria.py).
- 4, 5 — criteria_summary.json `tasks`. The benchmark row pools a benchmark's tasks item-weighted,
  with McNemar on the summed counts; the per-task table below it keeps each task's own interval.

It never writes Tempo_results.xlsx or Final_result.xlsx. A blank is no usable number, never a zero;
the Status column and the Sources sheet say why.
"""
import argparse
import csv
import json
import os
from math import comb

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NEG = os.path.join(ROOT, "Interpersonal_processes_benchmarks", "NegotiationToM")
EMO = os.path.join(ROOT, "Interpersonal_processes_benchmarks", "EmoBench")
AWARE_CSV = os.path.join(ROOT, "Transition_processes_benchmarks", "Awareness_in_LLM", "criteria_awareness.csv")
FINAL = os.path.join(ROOT, "Final_result.xlsx")
TEMPO = os.path.join(ROOT, "Tempo_results.xlsx")

SLOTS = ["Gemini", "OpenAI", "Qwen", "Gemma", "Deepseek"]
LEGACY = "legacy runner: no per-row usage fields, so no estimate"
LEGACY_REPS = "repeats not run: the legacy runner has no tag, subsample or perturbation support"
TASK_RUNNER = "no runner on this benchmark writes token or latency fields per row, so no estimate"

# sheet name, workbook sheet, accuracy columns (header, row label in the workbook sheet)
BENCHMARKS = [
    ("BBH", "Big Bench Hard", [("Overall (macro, 20 tasks)", "Overall Score (macro-avg of 20 tasks)")]),
    ("MMLU", "MMLU", [("Overall (macro, 13 subjects)", "Overall Score (macro-avg of 13 tasks)")]),
    ("DocVQA", "DocVQA", [("ANLS", "ANLS"), ("Accuracy", "Overall accuracy")]),
    ("EmoBench", "Emo", [("EU", "eu_overall"), ("EA", "ea_overall"), ("Overall", "Overall Score")]),
    ("NegotiationToM", "NegotiationToM", [("Desire EM", "Desire_EM"), ("Belief EM", "Belief_EM"),
                                          ("Intent micro F1", "Intent_Micro_F1"),
                                          ("Intent macro F1", "Intent_Macro_F1"), ("All EM", "All_EM")]),
    ("AwareBench", "Awareness", [("Capability", "capability"), ("Mission", "mission"), ("Emotion", "emotion"),
                                 ("Culture", "culture"), ("Perspective", "perspective (2nd-order)"),
                                 ("Overall", "Overall Score")]),
]
NO_RUNNER = ["PlanBench", "Multi-party Goal Tracking", "Wonderbread", "MultiChallenge"]

# benchmark -> slot -> (model folder holding criteria_summary.json, or None; why 1-3/6 is blank; why 4/5 is)
FOLDERS = {
    "NegotiationToM": {
        "Gemini": (os.path.join(NEG, "NEG_Gemini_Flash3.5lite_OpenRouter"), "", ""),
        "OpenAI": (os.path.join(NEG, "NEG_GPT_5.6_Luna"), "", ""),
        "Qwen": (None, LEGACY, LEGACY_REPS), "Gemma": (None, LEGACY, LEGACY_REPS),
        "Deepseek": (None, LEGACY, LEGACY_REPS),
    },
    "EmoBench": {
        "Gemini": (None, "the runner writes no latency_s, n_attempts or parse_ok",
                   "not run: the runner has no perturbation support yet"),
        "OpenAI": (os.path.join(EMO, "EMO_GPT_5.6_Luna"), "", ""),
        "Qwen": (None, LEGACY, LEGACY_REPS), "Gemma": (None, LEGACY, LEGACY_REPS),
        "Deepseek": (None, LEGACY, LEGACY_REPS),
    },
}
# finished and audited, not yet confirmed by the user: (benchmark, slot) -> the run's own overall file
UNCONFIRMED = {
    ("NegotiationToM", "Gemini"): os.path.join(
        NEG, "NEG_Gemini_Flash3.5lite_OpenRouter", "results", "gemini-3_5-flash-lite_negotiation_overall.csv"),
}
NOT_APPLICABLE = {("DocVQA", "Deepseek"): "not applicable: an image benchmark and a text-only model"}

USAGE = [  # (group, header, criteria key, number format)
    ("1 · Cost", "$ per 1k correct", "1_usd_per_1k_correct", "0.0000"),
    ("1 · Cost", "$ whole run", "1_cost_usd_full_run", "0.0000"),
    ("2 · Token efficiency", "output tokens, mean", "2_output_tokens_mean", "0.0"),
    ("2 · Token efficiency", "output tokens, p95", "2_output_tokens_p95", "0"),
    ("2 · Token efficiency", "reasoning tokens, mean", "2_reasoning_tokens_mean", "0.0"),
    ("2 · Token efficiency", "tokens wrong ÷ right (pooled)", "2_wrong_over_right_tokens", "0.00"),
    ("3 · Latency", "p50 (s)", "3_latency_p50_s", "0.00"),
    ("3 · Latency", "p95 (s)", "3_latency_p95_s", "0.00"),
]
STAB = [  # (group, header, pooled key, number format)
    ("4 · Stability", "identical in all 5", "identical", "0.000"),
    ("4 · Stability", "pass@1", "pass1", "0.000"),
    ("4 · Stability", "pass^5", "pass5", "0.000"),
    ("5 · Robustness", "noise flip (rep1 vs rep2)", "noise", "0.000"),
    ("5 · Robustness", "shuffle excess flip", "shuffle", "+0.000;-0.000;0.000"),
    ("5 · Robustness", "shuffle p", "shuffle_p", "0.0000"),
    ("5 · Robustness", "paraphrase excess flip", "paraphrase", "+0.000;-0.000;0.000"),
    ("5 · Robustness", "paraphrase p", "paraphrase_p", "0.0000"),
]
RELIAB = [
    ("6 · Reliability", "parse ok", "6_parse_ok_rate", "0.0000"),
    ("6 · Reliability", "empty", "6_empty_rate", "0.0000"),
    ("6 · Reliability", "truncated", "6_truncation_rate", "0.0000"),
    ("6 · Reliability", "rows retried", "6_retried_row_rate", "0.0000"),
    ("6 · Reliability", "one config", "6_one_config", "0"),
]

BOLD, TITLE, SMALL = Font(bold=True), Font(bold=True, size=14), Font(italic=True, color="595959")
FILL = {"head": PatternFill("solid", fgColor="DDE4EE"), "group": PatternFill("solid", fgColor="C9D3E3"),
        "full": PatternFill("solid", fgColor="D9EAD3"), "part": PatternFill("solid", fgColor="FFF2CC"),
        "none": PatternFill("solid", fgColor="EFEFEF"), "flag": PatternFill("solid", fgColor="FCE5CD")}
WRAP = Alignment(wrap_text=True, vertical="top")


# ── reading ────────────────────────────────────────────────────────────────────────────────────

def sheet_values(path, sheet):
    """{row label: {slot: value}} and {slot: model/config text} from one workbook sheet."""
    ws = load_workbook(path, data_only=True)[sheet]
    head = [c.value for c in ws[1]]
    rows = {r[0]: dict(zip(head[1:], r[1:])) for r in ws.iter_rows(min_row=3, values_only=True) if r[0]}
    return rows, dict(zip(head[1:], [c.value for c in ws[2]][1:]))


def summary(folder):
    path = folder and os.path.join(folder, "criteria_summary.json")
    return json.load(open(path)) if path and os.path.exists(path) else None


def mcnemar_p(ab, ba):
    n = ab + ba
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(ab, ba) + 1)) / 2 ** n) if n else 1.0


def pooled(tasks):
    """4 and 5 over all of a benchmark's items: item-weighted means, McNemar on summed counts."""
    n = sum(t["items"] for t in tasks.values())
    w = lambda f: sum(f(t) * t["items"] for t in tasks.values()) / n          # noqa: E731
    out = {"identical": w(lambda t: t["stability"]["identical_all_5"]),
           "pass1": w(lambda t: t["stability"]["pass@1"]),
           "pass5": w(lambda t: t["stability"]["pass^5"]),
           "noise": w(lambda t: t["robustness"]["baseline_flip_rep1_vs_rep2"])}
    for key, tag in (("shuffle", "perturb-shuffle"), ("paraphrase", "perturb-paraphrase")):
        out[key] = w(lambda t: t["robustness"][tag]["excess_flip"])
        counts = [t["robustness"][tag]["excess_mcnemar"] for t in tasks.values()]
        out[f"{key}_p"] = mcnemar_p(sum(c["flip_only_under_perturbation"] for c in counts),
                                    sum(c["flip_only_between_reps"] for c in counts))
    return out


def p_shown(p):
    return "<0.0001" if isinstance(p, float) and p < 1e-4 else p


# ── writing ────────────────────────────────────────────────────────────────────────────────────

def put(ws, row, col, value, fmt=None, font=None, fill=None, align=None):
    cell = ws.cell(row, col, value)
    if fmt and isinstance(value, (int, float)) and not isinstance(value, bool):
        cell.number_format = fmt
    if font:
        cell.font = font
    if fill:
        cell.fill = fill
    if align:
        cell.alignment = align
    return cell


def headers(ws, row, columns):
    """Two header rows: the group, merged over its columns, then the column name."""
    col = 1
    while col <= len(columns):
        group, end = columns[col - 1][0], col
        while end < len(columns) and columns[end][0] == group:
            end += 1
        put(ws, row, col, group, font=BOLD, fill=FILL["group"], align=Alignment(horizontal="center"))
        if end > col:
            ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=end)
        col = end + 1
    for i, (_, name, _) in enumerate(columns, 1):
        put(ws, row + 1, i, name, font=BOLD, fill=FILL["head"], align=WRAP)


def model_row(name, slot, acc_cols, final, aware):
    """(accuracy list, 1-3/6 dict or None, pooled 4/5 dict or None, per-task rows, notes, flags)."""
    notes, task_rows = [], []
    acc = [final.get(label, {}).get(slot) for _, label in acc_cols]
    unconfirmed = (name, slot) in UNCONFIRMED and all(a is None for a in acc)
    if unconfirmed:
        with open(UNCONFIRMED[(name, slot)]) as fh:
            own = {r["metric"]: float(r["score"]) for r in csv.DictReader(fh)}
        acc = [own.get(label) for _, label in acc_cols]
        notes.append("UNCONFIRMED: finished and audited, awaiting your confirmation; accuracy from the run's "
                     "own overall file, not in Final_result.xlsx")
    if (name, slot) in NOT_APPLICABLE:
        return acc, None, None, [], [NOT_APPLICABLE[(name, slot)]], (False, True)

    usage, stab, why_usage, why_stab = None, None, "", ""
    if name == "AwareBench":
        r = aware[slot]
        usage = {k: (r[k] if r[k] == "not reported" else float(r[k])) for _, _, k, _ in USAGE + RELIAB}
        why_stab = (f"repeats not run (code awaits your verification); full-run shuffle flip "
                    f"{r['5_shuffle_flip_rate_full_run']}, not noise-corrected; paraphrase not defined")
    elif name in FOLDERS:
        folder, why_usage, why_stab = FOLDERS[name][slot]
        s = summary(folder)
        if s:
            by_task = {}
            if "status" in s["main_run"]:
                why_usage = s["main_run"]["status"]
            else:
                usage = s["main_run"]
                by_task = usage.get("2_wrong_over_right_tokens_by_task", {})
                notes.append("tokens wrong ÷ right by task: " + ", ".join(f"{t} {v}" for t, v in by_task.items()))
            if isinstance(s["tasks"], dict):
                stab = pooled(s["tasks"])
                task_rows = [(task, t, by_task.get(task)) for task, t in s["tasks"].items()]
                if stab["noise"] == 0:
                    notes.append("5 repeats byte-identical with the seed pinned: noise 0, so every flip is excess")
            else:
                why_stab = s["tasks"]
    else:
        why_usage, why_stab = TASK_RUNNER, "repeats not run (phase 2)"
    if why_usage:
        notes.append(f"1·2·3·6: {why_usage}")
    if why_stab:
        notes.append(f"4·5: {why_stab}")
    return acc, usage, stab, task_rows, notes, (unconfirmed, False)


def benchmark_sheet(wb, name, sheet, acc_cols, aware):
    ws = wb.create_sheet(name)
    final, _ = sheet_values(FINAL, sheet)
    _, model_text = sheet_values(TEMPO, sheet)
    put(ws, 1, 1, f"{name} — accuracy and the six criteria, selected models", font=TITLE)
    put(ws, 2, 1, f"Accuracy from Final_result.xlsx › {sheet}; criteria from the criteria scripts' files "
                  "(Sources sheet). 4 and 5 pool the tasks item-weighted; per task below where there are tasks.",
        font=SMALL)
    columns = [("Model", "Slot", None), ("Model", "Model / config", None),
               *[("Accuracy", h, "0.0000") for h, _ in acc_cols],
               *[(g, h, f) for g, h, _, f in USAGE], *[(g, h, f) for g, h, _, f in STAB],
               *[(g, h, f) for g, h, _, f in RELIAB], ("Status", "What is here, and why a blank is blank", None)]
    headers(ws, 4, columns)

    coverage, per_task = {}, []
    for i, slot in enumerate(SLOTS):
        acc, usage, stab, task_rows, notes, (unconfirmed, na) = model_row(name, slot, acc_cols, final, aware)
        values = [slot, model_text.get(slot), *acc,
                  *[usage.get(k) if usage else None for _, _, k, _ in USAGE],
                  *[p_shown(stab[k]) if stab else None for _, _, k, _ in STAB],
                  *[usage.get(k) if usage else None for _, _, k, _ in RELIAB], "; ".join(notes)]
        for col, (value, (_, _, fmt)) in enumerate(zip(values, columns), 1):
            put(ws, 6 + i, col, value, fmt=fmt, fill=FILL["flag"] if unconfirmed else None,
                align=WRAP if col == len(columns) else None)
        per_task += [(slot, model_text.get(slot), *t) for t in task_rows]
        coverage[slot] = (any(a is not None for a in acc), usage is not None, stab is not None, unconfirmed, na)

    if per_task:
        row = 6 + len(SLOTS) + 2
        put(ws, row, 1, "Per task — 2's wrong ÷ right, and 4 and 5 with each task's own 95% interval "
                        "(clustered by negotiation on NegotiationToM)", font=BOLD)
        tcols = [("Model", "Slot", None), ("Model", "Model / config", None), ("Model", "Task", None),
                 ("2 · Token efficiency", "tokens wrong ÷ right", "0.00"), ("4 · Stability", "items", "0"),
                 ("4 · Stability", "identical in all 5", "0.000"), ("4 · Stability", "pass@1", "0.000"),
                 ("4 · Stability", "pass^5", "0.000"), ("4 · Stability", "accuracy spread", "0.000"),
                 ("5 · Robustness", "noise flip", "0.000"),
                 ("5 · Robustness", "shuffle excess", "+0.000;-0.000;0.000"),
                 ("5 · Robustness", "shuffle 95% CI", None), ("5 · Robustness", "shuffle p", "0.0000"),
                 ("5 · Robustness", "paraphrase excess", "+0.000;-0.000;0.000"),
                 ("5 · Robustness", "paraphrase 95% CI", None), ("5 · Robustness", "paraphrase p", "0.0000")]
        headers(ws, row + 1, tcols)
        row += 2
        for slot, model, task, t, ratio in per_task:
            row += 1
            st, rb = t["stability"], t["robustness"]
            sh, pa = rb["perturb-shuffle"], rb["perturb-paraphrase"]
            ci = lambda x: "{:+.3f} .. {:+.3f}".format(*x["excess_flip_95ci_cluster"])   # noqa: E731
            vals = [slot, model, task, ratio, t["items"], st["identical_all_5"], st["pass@1"], st["pass^5"],
                    st["accuracy_max"] - st["accuracy_min"], rb["baseline_flip_rep1_vs_rep2"],
                    sh["excess_flip"], ci(sh), p_shown(sh["excess_mcnemar"]["p"]),
                    pa["excess_flip"], ci(pa), p_shown(pa["excess_mcnemar"]["p"])]
            for col, (value, (_, _, fmt)) in enumerate(zip(vals, tcols), 1):
                put(ws, row, col, value, fmt=fmt)

    ws.freeze_panes = "C6"
    for col in range(1, len(columns) + 1):
        ws.column_dimensions[get_column_letter(col)].width = 12
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions[get_column_letter(len(columns))].width = 90
    ws.row_dimensions[5].height = 45
    return coverage


def overview(ws, coverage):
    put(ws, 1, 1, "Accuracy and the six evaluation criteria — coverage", font=TITLE)
    put(ws, 2, 1, "One row per benchmark, one column per selected model. Green: accuracy, 1·2·3·6 and 4·5 all "
                  "present. Amber: some. Grey: none. Orange: finished, awaiting your confirmation.", font=SMALL)
    put(ws, 4, 1, "Benchmark", font=BOLD, fill=FILL["head"])
    for j, slot in enumerate(SLOTS, 2):
        put(ws, 4, j, slot, font=BOLD, fill=FILL["head"])
    row = 4
    for name, *_ in BENCHMARKS:
        row += 1
        put(ws, row, 1, name, font=BOLD)
        for j, slot in enumerate(SLOTS, 2):
            acc, usage, stab, unconfirmed, na = coverage[name][slot]
            if na:
                put(ws, row, j, "n/a — text-only model", fill=FILL["none"])
                continue
            mark = lambda ok: "✓" if ok else "–"                          # noqa: E731
            text = f"Acc {mark(acc)}   1·2·3·6 {mark(usage)}   4·5 {mark(stab)}"
            fill = FILL["full"] if acc and usage and stab else FILL["part"] if acc or usage or stab else FILL["none"]
            if unconfirmed:
                text, fill = text + "   (unconfirmed)", FILL["flag"]
            put(ws, row, j, text, fill=fill)
    for name in NO_RUNNER:
        row += 1
        put(ws, row, 1, name, font=BOLD)
        for j in range(2, 2 + len(SLOTS)):
            put(ws, row, j, "no reported results yet", fill=FILL["none"])
    ws.column_dimensions["A"].width = 26
    for j in range(2, 2 + len(SLOTS)):
        ws.column_dimensions[get_column_letter(j)].width = 36


def sources(ws):
    put(ws, 1, 1, "Sources and definitions", font=TITLE)
    lines = [
        ("Accuracy", "Final_result.xlsx, the sheet named under each benchmark sheet's title; model/config text "
                     "from Tempo_results.xlsx row 2. An UNCONFIRMED row takes its accuracy from the run's own "
                     "overall file, because Final_result.xlsx holds it only after you confirm the run."),
        ("1 · 2 · 3 · 6", "criteria_summary.json › main_run (.claude/scripts/criteria_report.py) for NegotiationToM "
                          "and EmoBench; Awareness_in_LLM/criteria_awareness.csv (aware_criteria.py) for "
                          "AwareBench. One computation and one price table for all (usage_criteria, PRICES). List "
                          "prices read 2026-10-12; cache discounts ignored."),
        ("4 · 5", "criteria_summary.json › tasks. The benchmark row pools tasks item-weighted, McNemar on summed "
                  "counts; the per-task table keeps each task's 95% interval. Robustness is flips in excess of "
                  "the rep1-vs-rep2 noise; a model that honours its seed has noise 0, so compare raw flips too."),
        ("BBH · MMLU · DocVQA", "accuracy only: no runner on these benchmarks writes token or latency fields per "
                                "row, so 1·2·3·6 cannot be computed from the rows on disk, and no repeats were run."),
        ("Open decisions", "whether criterion 2's wrong ÷ right is pooled over tasks (the benchmark row) or kept per "
                           "task (the per-task table); the gate thresholds; the weights of any composite score."),
        ("Definitions", ".claude/references/evaluation-criteria.md (the six) and criteria-runs.md (4 and 5)."),
        ("Rebuild", "python3 .claude/scripts/build_criteria_workbook.py — never edit this file by hand."),
    ]
    for i, (k, v) in enumerate(lines, 3):
        put(ws, i, 1, k, font=BOLD, align=WRAP)
        put(ws, i, 2, v, align=WRAP)
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "Criteria_results.xlsx"))
    args = ap.parse_args()
    with open(AWARE_CSV) as fh:
        aware = {r["slot"]: r for r in csv.DictReader(fh)}
    wb = Workbook()
    first = wb.active
    first.title = "Overview"
    coverage = {name: benchmark_sheet(wb, name, sheet, acc, aware) for name, sheet, acc in BENCHMARKS}
    overview(first, coverage)
    sources(wb.create_sheet("Sources"))
    wb.save(args.out)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
