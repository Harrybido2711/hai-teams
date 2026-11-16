#!/usr/bin/env python3
"""Draw the 10% sample for the sample rerun (`.claude/references/sample10-plan.md` § WP1).

One file per benchmark, at the benchmark's root — never inside `data/`, because bbh and mmlu take
their task list from `ls data/*.json` and would read a sample file there as a task:

    Tasks_benchmarks/bbh/sample10.json      strata = task files,    ids = index into the task
    Tasks_benchmarks/mmlu/sample10.json     strata = subject files, ids = index into the subject
    Tasks_benchmarks/DocVQA/sample10.json   strata = question_types[0], ids = questionId

Format: {benchmark, seed, fraction, rule, created, total_N, total_n, strata: {name: {N, n, ids}}}.

Every model reads the same file, so every model is asked the same items and comparisons are paired.
Each stratum takes ceil(fraction * N) items with its own generator, seeded by the string
"<seed>/<benchmark>/<stratum>": string seeds are deterministic across runs and Python versions, and a
per-stratum generator means two strata of equal size do not get the same positions.

Rerunning the script gives byte-identical files (`created` is a fixed date, not now). `--check`
rebuilds in memory and compares with what is on disk, writing nothing.
"""
import argparse
import glob
import json
import math
import os
import random
import sys

SEED = 20261106
FRACTION = 0.10
CREATED = "2026-11-15"
RULE = "per stratum: ceil(fraction * N) items, random.Random('<seed>/<benchmark>/<stratum>').sample, ids sorted"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = os.path.join(ROOT, "Tasks_benchmarks")


def draw(benchmark, stratum, population):
    population = list(population)
    n = math.ceil(FRACTION * len(population))
    rng = random.Random(f"{SEED}/{benchmark}/{stratum}")
    return sorted(rng.sample(population, n))


def bbh():
    strata = {}
    for path in sorted(glob.glob(os.path.join(TASKS, "bbh", "data", "*.json"))):
        task = os.path.splitext(os.path.basename(path))[0]
        examples = json.load(open(path))["examples"]
        ids = draw("bbh", task, range(len(examples)))
        strata[task] = {"N": len(examples), "n": len(ids), "ids": ids}
    return "bbh", strata


def mmlu():
    strata = {}
    for path in sorted(glob.glob(os.path.join(TASKS, "mmlu", "data", "*.json"))):
        subject = os.path.splitext(os.path.basename(path))[0]
        examples = json.load(open(path))
        ids = draw("mmlu", subject, range(len(examples)))
        strata[subject] = {"N": len(examples), "n": len(ids), "ids": ids}
    return "mmlu", strata


def docvqa():
    data = json.load(open(os.path.join(TASKS, "DocVQA", "docvqa_output", "docvqa_validation.json")))
    by_type = {}
    for row in data:
        qtype = (row.get("question_types") or ["(none)"])[0]
        by_type.setdefault(qtype, []).append(row["questionId"])
    strata = {}
    for qtype in sorted(by_type):
        qids = sorted(by_type[qtype])
        ids = draw("docvqa", qtype, qids)
        strata[qtype] = {"N": len(qids), "n": len(ids), "ids": ids}
    return "DocVQA", strata


EXPECTED = {"bbh": (4833, 484), "mmlu": (3943, 401), "DocVQA": (5349, 539)}


def render(folder, strata):
    doc = {
        "benchmark": folder,
        "seed": SEED,
        "fraction": FRACTION,
        "rule": RULE,
        "created": CREATED,
        "total_N": sum(s["N"] for s in strata.values()),
        "total_n": sum(s["n"] for s in strata.values()),
        "strata": strata,
    }
    return json.dumps(doc, indent=1) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the files on disk; write nothing")
    args = ap.parse_args()
    bad = 0
    for build in (bbh, mmlu, docvqa):
        folder, strata = build()
        text = render(folder, strata)
        doc = json.loads(text)
        want_N, want_n = EXPECTED[folder]
        if (doc["total_N"], doc["total_n"]) != (want_N, want_n):
            print(f"{folder}: N={doc['total_N']} n={doc['total_n']}, expected {want_N}/{want_n}")
            bad += 1
        out = os.path.join(TASKS, folder, "sample10.json")
        if args.check:
            same = os.path.exists(out) and open(out).read() == text
            print(f"{folder}: {'identical' if same else 'DIFFERS'} — {out}")
            bad += 0 if same else 1
        else:
            with open(out, "w") as f:
                f.write(text)
            print(f"{folder}: {doc['total_n']} of {doc['total_N']} in {len(strata)} strata -> {out}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
