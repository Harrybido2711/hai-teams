"""Shared core for every BBH runner.

Two things live here and nowhere else.

**One scorer.** `score_response` is the lenient six-branch matcher, and it is the ONLY scorer in
this benchmark. Five of the eight runners used to compare with plain `==`, which cost them between
18 and 64 points of accuracy on identical model output — deepseek 0.448 -> 0.961, kimi 0.243 ->
0.884, openai 0.308 -> 0.834, measured 2026-08-29 over all 4,833 items. A model was being scored on
whether it wrote `(B)` or `B`. Comparing a strictly-scored model with a leniently-scored one is not
a comparison, so the scorer is imported, never copied: a runner cannot opt out of it.

**One output shape.** Every sub-task writes its own `.jsonl` (one record per item) and `.csv`, plus
a per-task `_overall.csv`, into that model's own `results/<task>/` — the EmoBench convention.

A runner supplies only the two things that are actually model-specific: a client, and a
`call(prompt) -> str` function.
"""

import csv
import json
import os
import re
import time

# Paths are resolved from THIS FILE, never from the cwd. A runner lives in <bbh>/BBH_<Slot>/ while
# the data lives in <bbh>/data/; a copy that resolved 'boolean_expressions.json' against the cwd is
# what wrote 20 splits of "No such file or directory" and an empty result file.
BBH_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BBH_ROOT, "data")
ENV_PATH = os.path.join(BBH_ROOT, ".env")

# The 20 vendored tasks. Upstream BBH has 27; the other seven are not in this repo, so a mean over
# this list is not comparable to a published 27-task average.
TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

PROMPT = """You are a helpful assistant.
Question: {question}

Please show your reasoning, then end your response with:
"Final Answer: <your concise answer here>"
"""


def model_slug(model_id):
    """`google/gemma-4-31B-it` -> `google-gemma-4-31B-it`; `kimi-k2.5` -> `kimi-k2_5`.

    EmoBench's filename convention: a result file is named after the model that produced it, so a
    folder renamed to a different slot cannot silently relabel someone else's numbers.
    """
    return model_id.replace("/", "-").replace(".", "_")


def load_task(task):
    with open(os.path.join(DATA_DIR, f"{task}.json"), "r") as fh:
        return json.load(fh)["examples"]


# ---------------------------------------------------------------- scoring


def extract_final_answer(model_output):
    """Text after the `Final Answer:` marker, quotes stripped.

    Falls back to the WHOLE response when the marker is absent — which then scores 0 on anything
    longer than the answer itself. That is not a scoring bug, it is a truncation detector: 62% of
    the gemini-2.5-flash rows in this benchmark have no marker because the response was cut off
    mid-reasoning. Use `has_marker` to tell "wrong" apart from "never finished".
    """
    if not isinstance(model_output, str):
        return ""
    match = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = match.group(1).strip() if match else model_output.strip()
    return result.strip("\"'`")


def has_marker(model_output):
    return isinstance(model_output, str) and bool(
        re.search(r"Final Answer:", model_output, re.IGNORECASE)
    )


def score_response(model_response, gold_answer, question=""):
    """The one scorer. Generous about how an answer is written, strict about what it says.

    Six branches, tried in order; any hit scores 1. Branch order is load-bearing only in that
    exact match is tried first — the rest are disjoint in practice.
    """
    if not isinstance(model_response, str) or model_response.strip() == "":
        return 0
    final_answer = extract_final_answer(model_response)
    gold_answer = str(gold_answer)
    question = str(question) if question else ""

    # 1. exact, case-folded
    if final_answer.lower().strip() == gold_answer.lower().strip():
        return 1

    if re.match(r"^\([A-Z]\)$", gold_answer.strip()):
        # 2. the letter, with the parens optional: "B", "(B)", "(B) a hexagon" for gold "(B)"
        m = re.match(r"^\(?([A-Z])\)?", final_answer.strip())
        if m and f"({m.group(1)})" == gold_answer.strip():
            return 1
        # 3. the option's TEXT instead of its letter: "hexagon" for gold "(B)"
        options = dict(re.findall(r"\(([A-Z])\)\s*([^\n(]+)", question))
        gold_content = options.get(gold_answer.strip("()"), "").strip()
        if gold_content and final_answer.lower() == gold_content.lower():
            return 1

    # 4. comma-vs-space, token level: "barn, damp" for gold "barn damp"
    if final_answer.lower().replace(",", " ").split() == gold_answer.lower().split():
        return 1

    # 5. dyck_languages: gold is only the CLOSING brackets, so a model that repeats the whole
    #    sequence is matched against the question's `Input:` line spliced onto the gold
    m = re.search(r"Input:\s*(.+)", question, re.IGNORECASE)
    if m:
        full = m.group(1).strip() + " " + gold_answer.strip()
        if final_answer.lower().strip() == full.lower().strip():
            return 1

    # 6. branch 4 again, for a gold that itself carries commas
    if (
        final_answer.lower().replace(",", " ").split()
        == gold_answer.lower().replace(",", " ").split()
    ):
        return 1

    return 0


# ---------------------------------------------------------------- output


FIELDS = ["idx", "question", "gold_answer", "model_response", "final_answer", "has_marker", "score"]


def task_dir(model_dir, task):
    d = os.path.join(model_dir, "results", task)
    os.makedirs(d, exist_ok=True)
    return d


def write_task_results(model_dir, task, model_id, records):
    """One `.jsonl`, one `.csv` and one `_overall.csv` per sub-task, named after the model."""
    d = task_dir(model_dir, task)
    slug = model_slug(model_id)

    with open(os.path.join(d, f"{slug}.jsonl"), "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(d, f"{slug}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    n = len(records)
    scored = sum(r["score"] for r in records)
    missing = sum(0 if r["has_marker"] else 1 for r in records)
    blank = sum(1 for r in records if not str(r["model_response"]).strip())
    with open(os.path.join(d, f"{slug}_overall.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "n", "score_sum", "average_score", "no_marker", "empty_response", "scorer"])
        w.writerow([model_id, task, n, scored, round(scored / n, 4) if n else "", missing, blank, "lenient_v1"])

    return {
        "model": model_id, "task": task, "n": n,
        "average_score": round(scored / n, 4) if n else "",
        "no_marker": missing, "empty_response": blank,
    }


def write_overall(model_dir, model_id, summaries):
    """The model's own roll-up across every task it ran. Not a benchmark score: `n` differs per
    task, so a macro-average over this file is a mean of task means, which is what the workbook
    reports."""
    path = os.path.join(model_dir, "results", f"{model_slug(model_id)}_bbh_overall.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "n", "average_score", "no_marker", "empty_response", "scorer"])
        for s in summaries:
            w.writerow([s["model"], s["task"], s["n"], s["average_score"],
                        s["no_marker"], s["empty_response"], "lenient_v1"])
        if summaries:
            means = [s["average_score"] for s in summaries if s["average_score"] != ""]
            w.writerow([model_id, "MACRO_AVG_over_%d_tasks" % len(means), "",
                        round(sum(means) / len(means), 4) if means else "", "", "", "lenient_v1"])
    return path


# ---------------------------------------------------------------- the loop


def run_tasks(model_dir, model_id, call, tasks=None, sleep_between=0.0, verbose=True):
    """Drive `call(prompt) -> str` over the requested tasks and write results per task.

    A failed call yields `""` for that ITEM and the task continues. The old runners let a `None`
    reach `re.search`, which raised inside the per-split try and dropped every row collected so far
    without writing a file — one bad call cost a whole 250-item task.
    """
    tasks = tasks or TASKS
    summaries = []
    for task in tasks:
        examples = load_task(task)
        records = []
        for i, ex in enumerate(examples):
            q, gold = ex["input"], ex["target"]
            try:
                resp = call(PROMPT.format(question=q))
            except Exception as e:  # never let one item take the task down
                if verbose:
                    print(f"[{task}#{i}] call failed: {e}", flush=True)
                resp = ""
            resp = resp if isinstance(resp, str) else ""
            records.append({
                "idx": i, "question": q, "gold_answer": gold,
                "model_response": resp,
                "final_answer": extract_final_answer(resp),
                "has_marker": has_marker(resp),
                "score": score_response(resp, gold, q),
            })
            if sleep_between:
                time.sleep(sleep_between)
        s = write_task_results(model_dir, task, model_id, records)
        summaries.append(s)
        if verbose:
            print(f"{task}: {s['average_score']}  (n={s['n']}, no_marker={s['no_marker']}, "
                  f"empty={s['empty_response']})", flush=True)
        write_overall(model_dir, model_id, summaries)  # survive a wall-clock kill
    return summaries


def retry(fn, tries=5, base_sleep=2.0, label=""):
    """Call `fn()`, retrying on exception AND on an empty string, with exponential backoff.

    Returns "" when every attempt fails — never None. `run_tasks` records that as an item with
    `has_marker=False` and score 0 rather than losing the task, so a provider hiccup costs one row.
    The old kimi runner recursed on failure without returning the recursive call's value, so a
    retried question always came back `None` no matter how well the retry went.
    """
    for attempt in range(tries):
        try:
            out = fn()
            if isinstance(out, str) and out.strip():
                return out.strip()
        except Exception as e:
            print(f"[{label}] attempt {attempt + 1}/{tries} failed: {e}", flush=True)
        if attempt < tries - 1:
            time.sleep(base_sleep * (2 ** attempt))
    return ""
