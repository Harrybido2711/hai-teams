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

**Two run modes, and the second never touches the first's files.** Without `--sample`, a runner
passes `call(prompt) -> str` to `run_tasks` and everything is exactly what it was before the 10%
sample rerun (2026-11-15) — same files, rows and requests, proved against commit 1cc5bbd5 by
`test_bbh_sample10.py`. With `--sample <name>` a runner passes `once(messages, model, params) ->
(text, meta)` to `run_sample` (bottom of this file): only the items listed in `<bbh>/<name>.json`, every
row carrying the per-row usage fields, written to `results_<name>/` (or `smoke_<name>/` under
`--limit`) and never to or from `results/`.
"""

import argparse
import collections
import concurrent.futures
import csv
import glob
import json
import os
import random
import re
import signal
import socket
import sys
import threading
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

# The prompt is byte-for-byte what `openai_eval.py` and `gemini_eval.py` sent, indentation
# included, because six of the eight old runners sent exactly this and the rows on disk were
# produced with it. Do not "tidy" the leading whitespace: it changes the token stream, and a new
# run would then differ from the existing 4,833 rows by the prompt as well as by the model.
#
# It was NOT uniform before. gemma and qwen — via the `_finish` twins their jobs actually
# submitted — sent the same text indented eight spaces instead of four, so their existing rows
# came from a different prompt than everyone else's. Same class of problem as the two scorers,
# found the same way; unified here, and their old rows carry the difference.
PROMPT = """
    You are a helpful assistant.
    Question: {question}

    Please show your reasoning, then end your response with:
    "Final Answer: <your concise answer here>"
    """

# v2 — same instruction, plus an explicit statement of what "concise" means. Added 2026-08-29
# because gpt-5.6-luna reads "concise answer" as a short SENTENCE: it answered `8 musical
# instruments` for gold `8` and `Elanor does not tell the truth.` for gold `No`. The scorer's v3
# branches recover the first form; the second cannot be credited without parsing negation, which
# would make the matcher task-aware. Asking for the bare answer is the cleaner fix.
#
# **A prompt version is part of the run config** and goes on every row. Two prompts in one result
# set is the thing CLAUDE.md says to archive rather than resume across, and until this existed the
# resume guard could not see a prompt change at all.
PROMPT_V2 = """
    You are a helpful assistant.
    Question: {question}

    Please show your reasoning, then end your response with:
    "Final Answer: <your concise answer here>"

    The final answer must be the answer by itself — no restatement of the question, no units or
    category words, no explanation. If the question lists options, give the option label exactly
    as it is written there.
    """

PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2}


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
    # Quotes, backticks and markdown emphasis are packaging, not answer. The asterisk was added
    # 2026-08-29 after a Luna pilot returned "**bootlegging, indifferent, trainman**" for gold
    # "bootlegging indifferent trainman" and scored 0: `**bootlegging` no longer matches a word.
    # Measured over all 36,348 stored rows it gains 31 and loses 0 — it can only remove a
    # formatting penalty, never credit a wrong answer. `_` was measured too and adds nothing, so
    # it is left out: it is likelier to be part of a real token.
    return result.strip("\"'`* ")


def has_marker(model_output):
    return isinstance(model_output, str) and bool(
        re.search(r"Final Answer:", model_output, re.IGNORECASE)
    )


CLOSED_SET = {"yes", "no", "true", "false", "valid", "invalid"}
# Yes and True are the same answer to a yes/no question. Not semantics — a fixed synonym table
# over a closed set, which is why it is safe where negation parsing would not be.
CLOSED_SYNONYMS = {"yes": {"true"}, "no": {"false"}, "true": {"yes"}, "false": {"no"},
                   "valid": {"true", "yes"}, "invalid": {"false", "no"}}
_BRACKET_CHARS = set("()[]{}<> \t")


def _BRACKETS_ONLY(text):
    t = str(text)
    return t.strip() != "" and set(t) <= _BRACKET_CHARS


def _unwrap(text):
    """Strip LaTeX inline-math delimiters. Same category as quotes, backticks and `**`."""
    return re.sub(r"\\[\(\)\[\]]", "", text).strip().strip("$").strip()


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

    # ---- v3 branches: the concise answer is there, with something wrapped around it ----
    #
    # These fire ONLY when the model actually emitted the `Final Answer:` marker. Without that,
    # `final_answer` is a scrape of the whole response, and "it contains the gold letter
    # somewhere" is not an answer — a Llama row whose reasoning happened to mention `(D)` once
    # would otherwise have been credited for an answer it never gave.
    if not has_marker(model_response):
        return 0

    bare = _unwrap(final_answer)

    # 7. LaTeX inline math is packaging, like quotes and bold: `\(-50\)` for gold `-50`
    if bare.lower() == gold_answer.lower().strip():
        return 1

    # 8. a number with its unit noun attached: `8 musical instruments` for gold `8`
    if re.match(r"^-?\d+$", gold_answer.strip()):
        head = bare.split()
        if head and head[0].rstrip(".,").lstrip("(") == gold_answer.strip():
            return 1

    # 9. a closed-set answer restated: `No, Ka does not tell the truth.` for gold `No`
    if gold_answer.strip().lower() in CLOSED_SET:
        first = re.split(r"[\s,.;:]+", final_answer.strip(), 1)[0].strip(" .,:;!\"'`*")
        if first.lower() == gold_answer.strip().lower():
            return 1

    # 12. a closed-set answer given as its synonym: `True` for gold `Yes`
    g_low = gold_answer.strip().lower()
    if g_low in CLOSED_SYNONYMS:
        if final_answer.strip().strip(" .,:;!\"'`*").lower() in CLOSED_SYNONYMS[g_low]:
            return 1

    # 11. dyck_languages: the same brackets without the spaces. `})>` for gold `} ) >` is the
    #     right answer typed without separators. Restricted to answers AND golds made only of
    #     bracket characters and whitespace — with no alphanumerics on either side, collapsing
    #     whitespace cannot merge two real words into one. Found by the user in a Flash-Lite row.
    if _BRACKETS_ONLY(gold_answer) and _BRACKETS_ONLY(final_answer):
        if re.sub(r"\s+", "", final_answer) == re.sub(r"\s+", "", gold_answer):
            return 1

    # 10. the option letter given at the END rather than the start: `11/10/2019 (B)` for `(B)`.
    #     Length-capped and requiring exactly ONE distinct letter, so a response that weighs
    #     several options cannot be read as having chosen one.
    if re.match(r"^\([A-Z]\)$", gold_answer.strip()) and len(final_answer) < 100:
        letters = re.findall(r"\(([A-Z])\)", final_answer)
        if len(set(letters)) == 1 and f"({letters[0]})" == gold_answer.strip():
            return 1

    return 0


# ---------------------------------------------------------------- output


# Bumped whenever score_response changes behaviour, and written onto every result file. v2
# added markdown-emphasis stripping; v3 added the four packaging branches below; v4 added
# whitespace-insensitive bracket matching; v5 added closed-set synonyms (all 2026-08-29). A number tagged v1 cannot be compared with one tagged v3 without rescoring.
SCORER_VERSION = "lenient_v5"

FIELDS = ["idx", "question", "gold_answer", "model_response", "final_answer", "has_marker",
          "score", "config"]


def config_string(config):
    """The generation config, flattened onto every row — `model-parameters.md` rule 8.

    "Pin a seed wherever the provider offers one, **and write it on every row**." A config that
    lives only in a job script cannot be recovered from a result file six weeks later, and on
    OpenRouter a seed is not even sufficient on its own: the backend decides, and it switches
    mid-run. So the backend goes on the row too, where the caller supplies one.
    """
    if not config:
        return ""
    return ";".join(f"{k}={v}" for k, v in sorted(config.items()))


def task_dir(model_dir, task):
    d = os.path.join(model_dir, "results", task)
    os.makedirs(d, exist_ok=True)
    return d


def write_task_results(model_dir, task, model_id, records, config=None, tag=""):
    """One `.jsonl`, one `.csv` and one `_overall.csv` per sub-task, named after the model."""
    d = task_dir(model_dir, task)
    # every artefact carries the shard tag, or shard 1 overwrites shard 0's summary
    slug = model_slug(model_id) + tag

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
        w.writerow(["model", "task", "n", "score_sum", "average_score", "no_marker",
                    "empty_response", "scorer", "config"])
        w.writerow([model_id, task, n, scored, round(scored / n, 4) if n else "", missing, blank,
                    SCORER_VERSION, config_string(config)])

    return {
        "model": model_id, "task": task, "n": n,
        "average_score": round(scored / n, 4) if n else "",
        "no_marker": missing, "empty_response": blank,
    }


def write_overall(model_dir, model_id, summaries, tag=""):
    """The model's own roll-up across every task it ran. Not a benchmark score: `n` differs per
    task, so a macro-average over this file is a mean of task means, which is what the workbook
    reports."""
    path = os.path.join(model_dir, "results", f"{model_slug(model_id)}{tag}_bbh_overall.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "task", "n", "average_score", "no_marker", "empty_response", "scorer"])
        for s in summaries:
            w.writerow([s["model"], s["task"], s["n"], s["average_score"],
                        s["no_marker"], s["empty_response"], SCORER_VERSION])
        if summaries:
            means = [s["average_score"] for s in summaries if s["average_score"] != ""]
            w.writerow([model_id, "MACRO_AVG_over_%d_tasks" % len(means), "",
                        round(sum(means) / len(means), 4) if means else "", "", "", SCORER_VERSION])
    return path


# ---------------------------------------------------------------- the loop


def parse_config(text):
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def load_checkpoint(out_path, config, verbose=True):
    """Rows already done for this task, keyed by idx — or refuse if the config has changed.

    Two lessons this project paid for, both in `CLAUDE.md`:

    * **A config change means archive, not resume.** Resuming across one leaves a single result set
      holding two configurations, which no column in it can distinguish. So a stored row whose
      run-level config differs from this run's is fatal here, not a warning. Keys the run adds
      per row (the OpenRouter backend) are ignored in the comparison — they are expected to vary.
    * **An empty row is not a done row.** A plain resume adds every stored uid to the done set
      regardless of whether the response was usable, so rows that came back empty are skipped
      forever. They are dropped here and retried.
    """
    if not os.path.exists(out_path):
        return {}
    want = {k: str(v) for k, v in (config or {}).items()}
    done, empty = {}, 0
    with open(out_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            have = parse_config(r.get("config", ""))
            differs = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
            if differs:
                raise SystemExit(
                    f"{out_path}\n  refusing to resume: the stored rows were produced with a "
                    f"different config.\n  differences (wanted, stored): {differs}\n"
                    f"  Archive the results directory to a timestamped name and start clean — "
                    f"resuming here would put two configurations in one result set.")
            if str(r.get("model_response", "")).strip():
                done[r["idx"]] = r
            else:
                empty += 1
    if verbose and (done or empty):
        print(f"  resume: {len(done)} row(s) done, {empty} empty row(s) will be retried",
              flush=True)
    return done


def run_tasks(model_dir, model_id, call, tasks=None, sleep_between=0.0, verbose=True, limit=0,
              config=None, per_row_config=None, save_every=20, resume=True, workers=1,
              prompt_version="v1", shard=0, total_shards=1):
    """Drive `call(prompt) -> str` over the requested tasks and write results per task.

    A failed call yields `""` for that ITEM and the task continues. The old runners let a `None`
    reach `re.search`, which raised inside the per-split try and dropped every row collected so far
    without writing a file — one bad call cost a whole 250-item task.
    """
    tasks = tasks or TASKS
    if prompt_version not in PROMPTS:
        raise SystemExit(f"unknown prompt version {prompt_version!r}; known: {sorted(PROMPTS)}")
    template = PROMPTS[prompt_version]
    # the prompt is part of the config, so the resume guard refuses to mix two of them and every
    # row says which one produced it
    config = dict(config or {}, prompt=prompt_version)
    summaries = []
    for task in tasks:
        examples = load_task(task)
        if limit:
            # Smoke-test lever. It truncates the work list, so what it writes is a PARTIAL run at
            # the same config — never report from it, and delete it before a real run.
            examples = examples[:limit]
            if verbose:
                print(f"[{task}] --limit {limit}: smoke test, not a run", flush=True)
        tag = shard_tag(shard, total_shards)
        out_path = os.path.join(task_dir(model_dir, task),
                                f"{model_slug(model_id)}{tag}.jsonl")
        done = load_checkpoint(out_path, config, verbose) if (resume and not limit) else {}

        def flush(recs):
            """Written every `save_every` items so a killed job keeps its paid calls, and so
            progress is visible at all — the old behaviour wrote nothing until a whole 250-item
            task finished, which left no rows to judge the run by for minutes at a time."""
            with open(out_path, "w") as fh:
                for r in sorted(recs, key=lambda x: x["idx"]):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        records = list(done.values())
        lock = threading.Lock()

        def one_item(i, ex):
            q, gold = ex["input"], ex["target"]
            try:
                resp = call(template.format(question=q))
            except Exception as e:  # never let one item take the task down
                if verbose:
                    print(f"[{task}#{i}] call failed: {e}", flush=True)
                resp = ""
            resp = resp if isinstance(resp, str) else ""
            row_cfg = dict(config or {})
            if per_row_config:
                row_cfg.update(per_row_config() or {})
            return {
                "idx": i, "question": q, "gold_answer": gold,
                "model_response": resp,
                "final_answer": extract_final_answer(resp),
                "has_marker": has_marker(resp),
                "score": score_response(resp, gold, q),
                "config": config_string(row_cfg),
            }

        # slice AFTER enumerate so idx stays the index into the full task, not into the slice —
        # otherwise merged shards would all claim idx 0..n and collide
        todo = [(i, ex) for i, ex in shard_slice(list(enumerate(examples)), shard, total_shards)
                if i not in done]
        if workers > 1:
            # 5 concurrent request streams is this project's standing limit and a measured fix,
            # not a convention (`.claude/references/quest-cluster.md`). Concurrency does not raise
            # the requests-per-DAY total — that stays at one call per item — it only changes how
            # fast they are spent, so the RPD ceiling that broke DocVQA is unaffected by this.
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(one_item, i, ex) for i, ex in todo]
                for k, fut in enumerate(concurrent.futures.as_completed(futures), 1):
                    with lock:
                        records.append(fut.result())
                        if save_every and k % save_every == 0:
                            flush(records)
                            if verbose:
                                print(f"  [{task}] {len(records)}/{len(examples)}", flush=True)
            records.sort(key=lambda r: r["idx"])
            s_ = write_task_results(model_dir, task, model_id, records, config=config, tag=tag)
            summaries.append(s_)
            if verbose:
                print(f"{task}: {s_['average_score']}  (n={s_['n']}, no_marker={s_['no_marker']}, "
                      f"empty={s_['empty_response']})", flush=True)
            write_overall(model_dir, model_id, summaries, tag=tag)
            continue

        for i, ex in enumerate(examples):
            if i in done:
                continue
            q, gold = ex["input"], ex["target"]
            try:
                resp = call(template.format(question=q))
            except Exception as e:  # never let one item take the task down
                if verbose:
                    print(f"[{task}#{i}] call failed: {e}", flush=True)
                resp = ""
            resp = resp if isinstance(resp, str) else ""
            # per_row_config() lets a runner add something only known after the call — the
            # OpenRouter backend that answered, which changes mid-run and which a seed does not
            # pin down (`.claude/references/provider-gotchas.md`).
            row_cfg = dict(config or {})
            if per_row_config:
                row_cfg.update(per_row_config() or {})
            records.append({
                "idx": i, "question": q, "gold_answer": gold,
                "model_response": resp,
                "final_answer": extract_final_answer(resp),
                "has_marker": has_marker(resp),
                "score": score_response(resp, gold, q),
                "config": config_string(row_cfg),
            })
            if sleep_between:
                time.sleep(sleep_between)
            if save_every and len(records) % save_every == 0:
                flush(records)
                if verbose:
                    print(f"  [{task}] {len(records)}/{len(examples)}", flush=True)
        records.sort(key=lambda r: r["idx"])
        s = write_task_results(model_dir, task, model_id, records, config=config, tag=tag)
        summaries.append(s)
        if verbose:
            print(f"{task}: {s['average_score']}  (n={s['n']}, no_marker={s['no_marker']}, "
                  f"empty={s['empty_response']})", flush=True)
        write_overall(model_dir, model_id, summaries, tag=tag)  # survive a wall-clock kill
    return summaries


def shard_tag(shard, total_shards):
    """`_shard2of5`, or **empty when total_shards is 1**.

    Copied deliberately from `NegotiationToM/NEG_GPT/openai_neg_eval.py:264`. The empty case is the
    load-bearing half: it means an unsharded run writes the same filename it always did, so the
    result files already on disk stay valid and a merge is only needed when one actually sharded.
    Without the tag every shard overwrites the last (`.claude/INDEX.md`).
    """
    return f"_shard{shard}of{total_shards}" if total_shards > 1 else ""


def shard_slice(items, shard, total_shards):
    """Contiguous block split, same arithmetic as `openai_neg_eval.py::shard_slice`.

    Contiguous rather than round-robin so a shard's `idx` values stay in one range — which makes a
    missing shard obvious as a gap in the merged file rather than scattered holes.
    """
    if total_shards <= 1:
        return list(items)
    size = (len(items) + total_shards - 1) // total_shards
    return list(items)[shard * size: min((shard + 1) * size, len(items))]


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


# ---------------------------------------------------------------- capability negotiation

AUTH_MARKERS = ("authentication", "api key", "invalid_api_key", "insufficient_quota",
                "billing", "401", "403")

# Preference order when a provider refuses the effort VALUE but supports the parameter. Lowest
# first: the point of the cap is to spend less, so accept the cheapest offered rather than the
# first named. `none` is never here — that is removal, not a cap, and on gpt-5.6-luna it costs
# 9-11 points (`.claude/references/model-parameters.md`).
EFFORT_FALLBACKS = ("minimal", "low", "medium", "high")


def negotiate(client, model, wanted, verbose=True):
    """Find which of `wanted` this model actually accepts, in one probe call before the run.

    Ported from `EmoBench/EMO_GPT_5.6_Luna/gpt56luna_emo_eval.py`. Hardcoding the surface is what
    the model page invites and what gets a run killed on item 1; try/except per item discovers the
    same permanent rejection 4,833 times. OpenAI names the offending parameter, and often its
    replacement, so a rejection is actionable.

    Returns (params, notes). Raises on anything that is not a parameter problem — an unknown model
    or a bad key is not something to negotiate around.
    """
    params, notes = dict(wanted), []
    for _ in range(len(wanted) + 2):
        try:
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with: ok"}], **params)
            return params, notes
        except Exception as e:
            err = str(e)
            if any(m in err.lower() for m in AUTH_MARKERS):
                raise SystemExit(f"Authentication or quota failure, not negotiated around: {e}")
            offender = next((k for k in params if f"'{k}'" in err), None)
            if offender is None:
                raise

            # A refused VALUE is not a refused PARAMETER, and conflating them is how a cap gets
            # dropped instead of corrected: gpt-5.6-luna refuses reasoning_effort="minimal" while
            # supporting the parameter perfectly. Dropping it there runs the whole benchmark at the
            # model default, uncapped.
            if "Unsupported value" in err or "does not support" in err:
                m = re.search(r"[Ss]upported values are:?\s*([^.}]+)", err)
                options = re.findall(r"'([^']+)'", m.group(1)) if m else []
                pick = next((v for v in EFFORT_FALLBACKS if v in options),
                            options[0] if options else None)
                if pick is not None:
                    asked = params[offender]
                    params[offender] = pick
                    notes.append(f"{offender}={pick} (asked {asked!r}, refused)")
                    if verbose:
                        print(f"  negotiate: {offender} refused {asked!r}; supported {options}, "
                              f"using {pick!r}", flush=True)
                    continue

            value = params.pop(offender)
            rename = re.search(r"[Uu]se '([A-Za-z_]+)' instead", err)
            if rename:
                params[rename.group(1)] = value
                notes.append(f"{offender} -> {rename.group(1)}")
                if verbose:
                    print(f"  negotiate: {offender} rejected, using {rename.group(1)}", flush=True)
            else:
                notes.append(f"{offender} unsupported, DROPPED")
                if verbose:
                    print(f"  negotiate: {offender} unsupported, dropped", flush=True)
    raise SystemExit(f"no working parameter set for {model}; last tried {params}")


# ================================================================ sample runs (`--sample <name>`)
#
# Everything below runs only when a runner is given `--sample`. Nothing above changed behaviour: a
# run without it still goes through `run_tasks`, and `test_bbh_sample10.py` proves its files, rows
# and requests byte-identical to commit 1cc5bbd5.
#
# The harness — billing and quota halts, `call_with_retries`, `openai_meta`, `negotiate_pinned`,
# `CutRetry` — is PORTED from Transition_processes_benchmarks/LLMs-Planning_bench/plan_eval_core.py,
# where the five settled configs it serves were built (2026-11-09 to 11-15). Copied rather than
# imported, because each benchmark's core travels to Quest on its own. What differs from PlanBench's:
# the billed-usage fields (every attempt summed, `billed`), status codes matched as whole words, the
# ceiling placed first because bbh sends no system message, and an explicit refusal to call off the
# main thread — the SIGALRM watchdog works nowhere else, so sample calls are serial and `--workers`
# is refused for `--sample` (`require_main_thread`). The lock on the halt counters and CutRetry's
# per-thread slot are defensive leftovers of that rule, not a licence to thread.

SAMPLE_NAME = re.compile(r"^sample\d+$")


def load_sample(name):
    """`{task: [idx, ...]}` from `<bbh>/<name>.json` (`.claude/scripts/build_sample10.py`), checked
    against the data: every id indexes its task's full `examples`, so a sample row's `idx` is the
    same `idx` as that item's row in `results/` and the two can be paired."""
    if not SAMPLE_NAME.match(name or ""):
        raise SystemExit(f"--sample {name!r}: expected a name like sample10, the file <bbh>/<name>.json")
    path = os.path.join(BBH_ROOT, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"--sample {name}: {path} does not exist")
    with open(path) as fh:
        spec = json.load(fh)
    if spec.get("benchmark") != "bbh" or sorted(spec.get("strata", {})) != sorted(TASKS):
        raise SystemExit(f"{path}: not a bbh sample over the {len(TASKS)} tasks")
    out = {}
    for task, stratum in spec["strata"].items():
        ids, size = stratum["ids"], len(load_task(task))
        if (stratum["N"] != size or stratum["n"] != len(ids) or ids != sorted(set(ids))
                or any(not isinstance(i, int) or not 0 <= i < size for i in ids)):
            raise SystemExit(f"{path}: stratum {task} does not match data/{task}.json "
                             f"(N={stratum['N']} vs {size} items, n={stratum['n']}, {len(ids)} ids)")
        out[task] = list(ids)
    if sum(len(v) for v in out.values()) != spec["total_n"]:
        raise SystemExit(f"{path}: total_n {spec['total_n']} != the ids listed")
    return out


def sample_root(model_dir, sample, smoke=False):
    """`results_<sample>/`, or `smoke_<sample>/` for a `--limit` run. Never `results/`: a sample row
    there would be resumed past by a full run, and a full-run row resumed past by the sample."""
    if not SAMPLE_NAME.match(sample or ""):
        raise SystemExit(f"bad sample name {sample!r}")
    return os.path.join(model_dir, ("smoke_" if smoke else "results_") + sample)


# ---------------------------------------------------------------- refusals that must stop a run

BILLING_SIGNATURES = (
    "insufficient_quota", "insufficient balance", "insufficient credits", "out of credits",
    "monthly budget", "used all available credits", "spending limit", "credit balance",
    "payment required", "arrears",
    # OpenRouter's out-of-credit 402 (provider-gotchas.md § Classifying a refusal)
    "requires more credits", "can only afford", "openrouter_credits",
    # DeepInfra's out-of-balance 402 — what stopped PlanBench Qwen as FAILURE_HALT on 2026-11-15
    "positive balance",
)
# A backend that cut the answer off on HTTP 200: a failed attempt, never an answer. "length" is not
# here — that is our cap, a measurement the overall files count.
INTERRUPTED_FINISH = ("error", "insufficient_system_resource")
DAILY_QUOTA_SIGNATURES = ("per_model_per_day", "requests per day", "requests_per_day", "per day",
                          "daily limit", "daily quota")
DAILY_HALT_THRESHOLD_SECONDS = 600
FATAL_SIGNATURES = ("invalid_argument", "extra inputs are not permitted", "validationerror",
                    "unsupported parameter", "unsupported value", "unrecognized request argument",
                    "unexpected keyword argument")
# plan_eval_core's AUTH_MARKERS, renamed: this file's AUTH_MARKERS above belongs to the legacy
# `negotiate` and stays as it was. The status code is matched as a WORD, not a substring: "401"
# inside a hex request id of a 500 error would otherwise make a transient error fatal (reviewer of
# the DocVQA package, 2026-11-15).
FATAL_AUTH_MARKERS = ("authentication", "api key", "invalid_api_key", "incorrect api key")
FATAL_AUTH_CODES = ("401",)


def has_status_code(text, codes):
    """Whether one of `codes` appears in `text` as a whole word (`\\b401\\b`), never inside a token."""
    return bool(re.search(r"\b(?:%s)\b" % "|".join(map(re.escape, codes)), str(text)))


def is_auth_failure(text):
    low = str(text).lower()
    return any(m in low for m in FATAL_AUTH_MARKERS) or has_status_code(low, FATAL_AUTH_CODES)

_RETRY_PHRASE = re.compile(r"(?:try again|retry|available again)\s+in\s+([0-9hms.\s]+)", re.I)
_DURATION_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.I)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def retry_after_seconds(error):
    match = _RETRY_PHRASE.search(str(error))
    if not match:
        return None
    total, found = 0.0, False
    for value, unit in _DURATION_TOKEN.findall(match.group(1)):
        total += float(value) * _UNIT_SECONDS[unit.lower()]
        found = True
    return total if found else None


def is_daily_quota_failure(error):
    if not any(sig in str(error).lower() for sig in DAILY_QUOTA_SIGNATURES):
        return False
    wait = retry_after_seconds(error)
    return True if wait is None else wait >= DAILY_HALT_THRESHOLD_SECONDS


def is_billing_failure(error):
    return any(sig in str(error).lower() for sig in BILLING_SIGNATURES)


def is_fatal_request(error):
    text = f"{type(error).__name__}: {error}".lower()
    return any(sig in text for sig in FATAL_SIGNATURES) or is_auth_failure(text)


STATS = {"ok": 0, "empty": 0, "interrupted": 0, "errors": 0, "timeouts": 0, "failed_rows": 0}
_STATS_LOCK = threading.Lock()
HALT_MARKERS = ("BILLING_HALT.txt", "QUOTA_HALT.txt", "FAILURE_HALT.txt")
# Set by run_sample: `sample10_`, plus `shard2_` when sharded, so a sample run's markers never clear
# or shadow another run's. Every marker still matches *_HALT.txt.
MARKER_PREFIX = ""


def _count(key):
    with _STATS_LOCK:
        STATS[key] += 1


def _halt(kind, marker_name, detail, model, model_dir):
    """Write a marker an agent can find without reading logs, say why on both streams, exit."""
    marker = os.path.join(model_dir, MARKER_PREFIX + marker_name)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    spoiled = STATS["empty"] + STATS["errors"] + STATS["timeouts"]
    advice = ("Rows written empty are retried on resume (load_sample_checkpoint skips only non-empty "
              "rows), so a plain resubmit is safe once the cause is fixed."
              if spoiled else "No call had failed before this point.")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason={kind}\n{detail}\n\n{advice}\n")
    except OSError as error:
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! {kind} HALT [{model}] {stamp}\n{detail}\n", file=sys.stderr, flush=True)
    raise SystemExit(f"aborting: {model} stopped on {kind.lower()}. {advice}")


def halt_on_billing(error, model, model_dir):
    """Called first in every except block. Billing and an exhausted daily cap stop the run."""
    detail = f"{type(error).__name__}: {error}"
    if is_daily_quota_failure(error):
        _halt("DAILY QUOTA", "QUOTA_HALT.txt", detail, model, model_dir)
    if is_billing_failure(error):
        _halt("BILLING", "BILLING_HALT.txt", detail, model, model_dir)


def clear_stale_halt_markers(model_dir):
    for name in HALT_MARKERS:
        path = os.path.join(model_dir, MARKER_PREFIX + name)
        if os.path.exists(path):
            os.remove(path)
            print(f"[startup] cleared stale {MARKER_PREFIX + name}", flush=True)


# Consecutive failures catch a provider that stops serving; the rolling window one that fails
# intermittently — the shape that fills a result set with empties.
CONSECUTIVE_FAILURE_LIMIT = 10
FAILURE_WINDOW = 50
FAILURE_WINDOW_LIMIT = 0.5
_recent = collections.deque(maxlen=FAILURE_WINDOW)
_consecutive = [0]


def note_outcome(ok, model, model_dir):
    with _STATS_LOCK:
        _recent.append(bool(ok))
        _consecutive[0] = 0 if ok else _consecutive[0] + 1
        run, bad = _consecutive[0], _recent.count(False)
        full = len(_recent) == FAILURE_WINDOW
    if run >= CONSECUTIVE_FAILURE_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt", f"{run} consecutive rows failed every attempt",
              model, model_dir)
    if full and bad / FAILURE_WINDOW >= FAILURE_WINDOW_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt", f"{bad} of the last {FAILURE_WINDOW} rows failed",
              model, model_dir)


# ---------------------------------------------------------------- one call, guarded


class CallTimeout(BaseException):
    """BaseException on purpose: an `except Exception` below would otherwise swallow the watchdog."""


# Hard ceiling on one HTTP attempt, by SIGALRM. Set per runner through run_sample(call_timeout=...).
HARD_CALL_TIMEOUT = 300
# Seconds before retrying an empty or interrupted reply (an exception waits retry_delay instead).
EMPTY_RETRY_WAIT = 5.0


def _raise_timeout(signum, frame):
    raise CallTimeout(f"call exceeded {HARD_CALL_TIMEOUT}s hard limit")


def require_main_thread():
    """The watchdog is SIGALRM, and a signal handler can be installed — and is run — only on the main
    thread. Off it, the guard either raises at `signal.signal` or never fires, and the client's own
    timeout is no substitute: httpx's is per read, so a server that trickles bytes defeats it forever
    (shown against a blocked socket in test_bbh_sample10.py). So every sample call runs serially, on
    the main thread, and anything else is refused rather than run unguarded."""
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("no SIGALRM on this platform: the per-call watchdog cannot run")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("call_with_retries must run on the main thread: its SIGALRM watchdog "
                           "cannot be armed anywhere else")


def _alarm_ok():
    """Whether to arm the watchdog: on the main thread, with a nonzero ceiling."""
    return (hasattr(signal, "SIGALRM") and bool(HARD_CALL_TIMEOUT)
            and threading.current_thread() is threading.main_thread())


def retry_delay(error, attempt, default=5.0):
    """Provider's hint as a floor, exponential growth, jitter, capped at a quarter of the watchdog."""
    text = str(error)
    hint = None
    for pattern in (r"try again in ([\d.]+)\s*(ms|s)\b",
                    r"retry (?:starting )?(?:from|after)\s*~?([\d.]+)\s*(ms|s)\b",
                    r"retry[- ]after[\"':\s]+([\d.]+)()"):
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            hint = float(m.group(1)) / (1000 if (m.group(2) or "").lower() == "ms" else 1)
            hint = max(hint + 1, 1)
            break
    base = max(hint or 0.0, default * (2 ** max(0, attempt)))
    return min(base * random.uniform(0.5, 1.5), max(5.0, min(60.0, HARD_CALL_TIMEOUT / 4.0)))


# ---------------------------------------------------------------- billed usage, every call of a row
#
# PlanBench's call_with_retries keeps only the LAST attempt's meta, so a row's tokens describe one
# call: an attempt that came back HTTP 200 empty after 16,384 billed reasoning tokens, then a 50-token
# answer, reads 50 where 16,434 were billed (reviewer of the DocVQA package, 2026-11-15). The fields
# below are additive — `prompt_tokens` and the rest keep their meaning (the answering attempt's,
# CutRetry's stages summed) — and these say what the row cost. Names shared with MMLU and DocVQA.
#
#   billed_prompt_tokens, billed_completion_tokens, billed_reasoning_tokens — the sum over EVERY
#       HTTP call of the row that returned a usage object: failed, empty and interrupted attempts,
#       and each CutRetry stage, counted once even when a later attempt reuses a kept stage.
#       billed_reasoning_tokens is None when no call reported a reasoning count.
#   billed_cost, billed_estimated_cost — the same sum of OpenRouter's `usage.cost` and DeepInfra's
#       `usage.estimated_cost` (dollars, the provider's own figure); None when no call reported one.
#       The answering call's own are `last_call_cost` / `last_call_estimated_cost`.
#   attempts_without_usage — every HTTP call that returned no usage: a timeout, an exception, or a
#       reply without a usage object — CutRetry's stages included, so it counts calls, not attempts.
#   reasoning_chars — characters of reasoning TEXT (`reasoning_content` / `reasoning`) summed over
#       every call of the row; 0 when there was none.
#   think_tag_calls — how many of the row's calls had a `<think>` or `</think>` tag in `content`.
#   halted — None, or the stop that interrupted the row's call ("SystemExit: 143"), on the failed
#       row written when a run stops mid-call. The thinking-off check reads reasoning_chars,
#       think_tag_calls and the reasoning tokens. Names and meanings shared with mmlu_eval_core.
#
# A resumed row folds the stored row's figures in (fold_prior), so all of these are cumulative over
# every run of the row, and `prior_n_attempts` / `prior_failed_runs` say how many runs came before.
#
# The ledger is per row. A `once` that makes several calls (CutRetry) records each with
# record_usage() and marks its meta `_usage_recorded`; any other `once` is recorded here.

BILLED_KEYS = ("billed_prompt_tokens", "billed_completion_tokens", "billed_reasoning_tokens",
               "attempts_without_usage", "billed_cost", "billed_estimated_cost")
PRIOR_KEYS = ("prior_n_attempts", "prior_failed_runs", "resumed_after_failure")
COST_KEYS = ("last_call_cost", "last_call_estimated_cost")
_LEDGER = threading.local()
# What call_with_retries ended with, as (text, meta) — the answer, the failed row's meta, or the
# meta of a row a halt or SIGTERM interrupted, each with the calls already paid for. Set before
# control leaves call_with_retries on every path, so a run that ends while the row is being built
# or written still has it: run_sample_tasks writes it (`take_inflight`) instead of losing it.
_INFLIGHT = threading.local()


def _ledger():
    return getattr(_LEDGER, "book", None)


def _sum_or_none(values):
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return round(sum(values), 10) if values else None


def open_call():
    """An HTTP call of the row is about to be made: opened at every attempt and at every CutRetry
    stage. Recording its response closes it (record_usage); a failure while a call is still open
    is that call's, and counts once as a call without usage (note_call_without_usage). So an
    ordinary error counted in the inner handler is not counted again when a halt raised from that
    handler reaches the outer one, and a reply booked before its `choices` raised is not counted
    at all. The same boundary as docvqa_eval_core and mmlu_eval_core."""
    book = _ledger()
    if book is not None:
        book["open"] = True


def record_usage(meta):
    """One response, into the ledger of the row being worked on — once: `openai_meta` books every
    response as it arrives and flags the meta, so a later record_usage of the same meta does
    nothing. A meta that never went through openai_meta (a test's fake) is booked here instead.
    Its tokens, the provider's cost fields and its `reasoning_chars` are added; a response with no
    usage object is one more call without usage."""
    book = _ledger()
    if book is None or meta is None or meta.get("_usage_recorded"):
        return
    meta["_usage_recorded"] = True
    book["open"] = False                     # this call is accounted for: its response is here
    if any(isinstance(meta.get(k), int) for k in USAGE_KEYS):
        book["calls"].append({k: meta.get(k) for k in USAGE_KEYS + COST_KEYS})
    else:
        book["no_usage"] += 1
    if isinstance(meta.get("reasoning_chars"), int):
        book["reasoning_chars"] += meta["reasoning_chars"]
    book["think_tag_calls"] += int(bool(meta.pop("think_tag", 0)))


def note_call_without_usage():
    """A failure — exception, timeout, halt, SIGTERM — counts the call still OPEN as one call
    without usage, and closes it. With no call open (its response was recorded first, or the inner
    handler already counted it) there is nothing to count."""
    book = _ledger()
    if book is not None and book["open"]:
        book["no_usage"] += 1
        book["open"] = False


def billed(book):
    calls = book["calls"]
    out = {"billed_" + k: _sum_or_none(c.get(k) for c in calls) for k in USAGE_KEYS}
    for k in ("billed_prompt_tokens", "billed_completion_tokens"):
        out[k] = out[k] or 0
    out["attempts_without_usage"] = book["no_usage"]
    out["billed_cost"] = _sum_or_none(c.get("last_call_cost") for c in calls)
    out["billed_estimated_cost"] = _sum_or_none(c.get("last_call_estimated_cost") for c in calls)
    out["reasoning_chars"] = book["reasoning_chars"]
    out["think_tag_calls"] = book["think_tag_calls"]
    return out


def take_inflight():
    """What the last call_with_retries ended with, `(text, meta)` or None — and clear it."""
    result = getattr(_INFLIGHT, "result", None)
    _INFLIGHT.result = None
    return result


def call_with_retries(once, messages, model, model_dir, tries=5):
    """`once(messages) -> (text, meta)`, up to `tries` attempts. Retries an exception, a timeout AND
    an empty string — HTTP 200 with an empty body raises nothing. Returns ("", meta) when every
    attempt failed; the row is then written failed and retried on resume.

    `latency_s` is the successful attempt alone and `n_attempts` every attempt spent on the row
    (this run's); both are true only because every client is built with `max_retries=0`
    (script-skeleton.md §4). The `billed_*` fields sum every call (above). Main thread only:
    `require_main_thread`.

    **A run that ends inside this function keeps what the row paid for.** Every exit stashes its
    result first (`take_inflight`). Any BaseException — a halt's SystemExit, SIGTERM's — is caught
    on the way out: unless a result was already stashed, the row's meta and ledger are, with
    `n_attempts` so far, `latency_s` None and `error` "interrupted: …", the call still open counted
    as one without usage; then the exception goes on. run_sample_tasks writes the stash — an
    answered row as answered, anything else as failed (resume calls it again)."""
    # First, before anything that can be stopped: no earlier row's result or ledger survives into
    # this one (S1 — a stop landing at this function's entry must find nothing to write).
    _INFLIGHT.result = None
    _LEDGER.book = None
    require_main_thread()
    book = _LEDGER.book = {"calls": [], "no_usage": 0, "reasoning_chars": 0, "think_tag_calls": 0,
                           "open": False}
    meta, last_error = {}, None
    attempt = -1
    try:
        for attempt in range(tries):
            started = time.time()
            wait = EMPTY_RETRY_WAIT
            has_alarm = _alarm_ok()
            previous = signal.signal(signal.SIGALRM, _raise_timeout) if has_alarm else None
            try:
                if has_alarm:
                    signal.alarm(HARD_CALL_TIMEOUT)
                try:
                    open_call()
                    text, meta = once(messages)
                finally:
                    if has_alarm:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, previous)
                meta = dict(meta or {})
                record_usage(meta)                       # a no-op if openai_meta booked it already
                meta.pop("_usage_recorded", None)
                meta.update(latency_s=round(time.time() - started, 2), n_attempts=attempt + 1)
                text = text.strip() if isinstance(text, str) else ""
                if text and meta.get("finish_reason") in INTERRUPTED_FINISH:
                    _count("interrupted")
                    last_error = f"interrupted: finish_reason={meta.get('finish_reason')}"
                    print(f"[{model}] INTERRUPTED ({attempt + 1}/{tries}) "
                          f"finish_reason={meta.get('finish_reason')}: backend cut the answer off",
                          flush=True)
                    text = ""
                elif text:
                    _count("ok")
                    result = _INFLIGHT.result = (text, dict(meta, **billed(book)))
                    note_outcome(True, model, model_dir)
                    return result
                else:
                    last_error = f"empty response: finish_reason={meta.get('finish_reason')}"
                _count("empty")
                print(f"[{model}] empty response ({attempt + 1}/{tries}) "
                      f"finish_reason={meta.get('finish_reason')} "
                      f"completion_tokens={meta.get('completion_tokens')}", flush=True)
            except CallTimeout as error:
                note_call_without_usage()
                _count("timeouts")
                last_error = f"timeout: {error}"
                print(f"[{model}] TIMEOUT ({attempt + 1}/{tries}): {error}", flush=True)
            except Exception as error:
                note_call_without_usage()
                _count("errors")
                last_error = f"{type(error).__name__}: {str(error)[:400]}"
                print(f"[{model}] API error ({attempt + 1}/{tries}): {type(error).__name__}: {error}",
                      flush=True)
                halt_on_billing(error, model, model_dir)
                if is_fatal_request(error):
                    _halt("MALFORMED REQUEST", "FAILURE_HALT.txt",
                          f"{type(error).__name__}: {error}", model, model_dir)
                wait = retry_delay(error, attempt)
            if attempt + 1 < tries:
                time.sleep(wait)
        _count("failed_rows")
        print(f"[{model}] all {tries} attempts failed; row written failed, retried on resume", flush=True)
        result = _INFLIGHT.result = ("", dict(meta or {}, latency_s=None, n_attempts=tries,
                                              error=last_error, **billed(book)))
        note_outcome(False, model, model_dir)        # a FAILURE halt here keeps `result` stashed
        return result
    except BaseException as stop:          # a halt's SystemExit, SIGTERM's: the calls made are billed
        note_call_without_usage()          # the call still open, if any — counted once, here
        if getattr(_INFLIGHT, "result", None) is None:
            _INFLIGHT.result = ("", dict(meta or {}, latency_s=None, n_attempts=attempt + 1,
                                         error=f"interrupted: {type(stop).__name__}: {stop}"[:400],
                                         halted=f"{type(stop).__name__}: {stop}"[:300],
                                         **billed(book)))
        raise
    finally:
        _LEDGER.book = None


_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def has_think_tag(content):
    """Whether a model wrote thinking INTO its content: a `<think>` or `</think>` tag, even around
    nothing. One per call is counted into the row's `think_tag_calls`."""
    return isinstance(content, str) and bool(_THINK_TAG.search(content))


def openai_meta(response):
    """What an OpenAI-compatible response says about itself — and it BOOKS the response onto the
    row's ledger before anything else can fail, so a reply with usage but no `choices` is still
    billed (the runner then raises on `choices[0]`, and that call is not counted again).

    `reasoning_tokens` is None — empty, not 0 — where the provider reports no breakdown
    (evaluation-criteria.md § Per-row fields). `reasoning_chars` is the length of
    `reasoning_content` (or `reasoning`); a `<think>` tag in the content is flagged separately and
    counted into the row's `think_tag_calls` — Qwen and Gemma must have neither. The provider's own
    dollar figure, where it
    gives one, is `last_call_cost` (OpenRouter `usage.cost`) or `last_call_estimated_cost`
    (DeepInfra `usage.estimated_cost`)."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    choices = getattr(response, "choices", None)
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    finish = getattr(choice, "finish_reason", None)
    finish = getattr(finish, "value", finish)
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)

    def as_int(value):
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def as_number(value):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    meta = {
        "finish_reason": None if finish is None else str(finish),
        "prompt_tokens": as_int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": as_int(getattr(usage, "completion_tokens", None)),
        "reasoning_tokens": as_int(getattr(details, "reasoning_tokens", None)),
        "served_model": str(getattr(response, "model", "") or "") or None,
        "last_call_cost": as_number(getattr(usage, "cost", None)),
        "last_call_estimated_cost": as_number(getattr(usage, "estimated_cost", None)),
        "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
        "think_tag": int(has_think_tag(getattr(message, "content", None))),   # into the ledger only
    }
    provider = getattr(response, "provider", None)       # OpenRouter's answering backend
    if provider:
        meta["backend"] = str(provider)
    record_usage(meta)                                   # booked first; flags the meta
    return meta


# ---------------------------------------------------------------- parameter negotiation
# plan_eval_core.negotiate under another name, because this file's `negotiate` above is the legacy
# runners' and stays as it was. `fixed` is sent on every call and never dropped; `required`
# ({parameter: value}) stops the run if negotiation loses the parameter OR swaps its value
# (model-parameters.md rules 1 and 7).

_TRANSIENT = ("rate limit", "timed out", "timeout", "connection", "overloaded", "unavailable")
_TRANSIENT_CODES = ("429", "500", "502", "503", "504")     # whole words, as FATAL_AUTH_CODES


def negotiate_pinned(client, model, wanted, model_dir, fixed=None, required=None):
    params, notes = dict(wanted), []
    fixed = dict(fixed or {})
    transient_left = 3
    for _ in range(len(wanted) + 3 + transient_left):
        has_alarm = _alarm_ok()
        previous = signal.signal(signal.SIGALRM, _raise_timeout) if has_alarm else None
        try:
            if has_alarm:
                signal.alarm(HARD_CALL_TIMEOUT)
            try:
                client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Reply with: ok"}],
                    **fixed, **params)
            finally:
                if has_alarm:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
            break
        except CallTimeout as error:
            if transient_left <= 0:
                raise SystemExit(f"negotiation probe for {model} timed out repeatedly: {error}")
            transient_left -= 1
            print(f"  negotiate: probe timed out, retrying ({error})", flush=True)
            continue
        except Exception as error:
            err = str(error)
            print(f"  negotiate: refused: {type(error).__name__}: {err[:300]}", flush=True)
            halt_on_billing(error, model, model_dir)
            if is_auth_failure(err):
                raise SystemExit(f"authentication failure, not negotiated around: {err}")
            offender = next((k for k in params if f"'{k}'" in err or f'"{k}"' in err
                             or f"`{k}`" in err), None)
            if (offender is None and transient_left > 0
                    and (any(x in err.lower() for x in _TRANSIENT)
                         or has_status_code(err, _TRANSIENT_CODES))):
                transient_left -= 1
                time.sleep(retry_delay(error, 3 - transient_left))
                continue
            if offender is None:
                raise SystemExit(f"probe refused and no negotiable parameter is named: {err}")
            if "unsupported value" in err.lower() or "does not support" in err.lower():
                m = re.search(r"[Ss]upported values are:?\s*([^.}]+)", err)
                options = re.findall(r"'([^']+)'", m.group(1)) if m else []
                pick = next((v for v in EFFORT_FALLBACKS if v in options),
                            options[0] if options else None)
                if pick is not None:
                    notes.append(f"{offender}={pick} (asked {params[offender]!r}, refused)")
                    params[offender] = pick
                    continue
            value = params.pop(offender)
            rename = re.search(r"[Uu]se '([A-Za-z_]+)' instead", err)
            if rename:
                params[rename.group(1)] = value
                notes.append(f"{offender} -> {rename.group(1)}")
            else:
                notes.append(f"{offender} unsupported, DROPPED")
    else:
        raise SystemExit(f"no working parameter set for {model}; last tried {params}")
    lost = {k: (v, params.get(k)) for k, v in (required or {}).items() if params.get(k) != v}
    if lost:
        raise SystemExit(f"required parameter(s) not held after negotiation (wanted, got): {lost} — "
                         f"that removes or changes a cap this model must run under "
                         f"(model-parameters.md rules 1 and 7). Fix the request instead.")
    return params, notes


# ---------------------------------------------------------------- a reply cut at the output cap

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")


def rearm_call_timeout():
    """Give the next HTTP call inside one `once` its own HARD_CALL_TIMEOUT — only where
    call_with_retries' own handler is installed, which is only ever on the main thread."""
    if _alarm_ok() and signal.getsignal(signal.SIGALRM) is _raise_timeout:
        signal.alarm(HARD_CALL_TIMEOUT)


def _usage_sum(metas):
    out = {}
    for key in USAGE_KEYS:
        parts = [m.get(key) for m in metas if isinstance(m.get(key), int)]
        out[key] = sum(parts) if parts else None
    return out


def forced_reasoned(row):
    """A forced answer whose call reasoned again — the one thing a forced answer may not do."""
    return bool(row.get("forced_answer") and (row.get("forced_reasoning_chars")
                                              or row.get("forced_reasoning_tokens")))


class CutRetry:
    """plan_eval_core.CutRetry for bbh — what a row whose reply hits the output cap gets, the user's
    rule of 2026-11-13 ("让他第二次重新跑的时候，不要超出这个思考的上限…如果说他就是不受约束，那就直接回答"):

    1. **The call as configured**, with the runner's prompt ceiling as a system message. A reply
       that ends any way but `length` is the row's answer.
    2. **Cut at the cap: called again, fresh**, with `retry_ceiling` in its place. The row carries
       `cut_retry: true` and `first_*`, among them `first_response`, the cut call's text in full.
    3. **Cut again: the forced answer.** What the second attempt wrote goes back as an assistant
       turn, then `force_prompt`, in a call the runner caps at `answer_max`. The row carries
       `forced_answer: true`, `retry_*` and `forced_*`; its `finish_reason` is the forced call's.

    `call(sent, stage) -> (answer, handback, meta)` is the runner's client call, stage "first",
    "retry" or "forced". Usage on the row is every call summed, so `completion_tokens` can pass the
    cap. `latency_s` is the last call's; `first_latency_s` and the rest time each one. Every stage
    call also goes into the row's ledger as it returns (`record_usage`): a stage kept across
    attempts is billed once, a stage whose attempt later failed is still billed, a stage that
    returned no usage counts in `attempts_without_usage`, and the row's `reasoning_chars` is the sum
    over every stage. True calls on an answered row = `n_attempts` + `cut_retry` + `forced_answer`.

    Two departures from PlanBench's. **bbh sends no system message**, so the ceiling goes first,
    before the user message, rather than after a shared system message (`with_system`) — where MPGT
    puts it too. And **the row in progress is kept per thread** — defensive only, since sample calls
    run serially on the main thread (`require_main_thread`). It is kept by the identity of the
    messages list the core builds once per row, so call_with_retries' retry repeats only the stage
    that failed."""

    def __init__(self, ceiling, retry_ceiling, force_prompt, answer_max):
        self.ceiling, self.retry_ceiling = ceiling, retry_ceiling
        self.force_prompt, self.answer_max = force_prompt, answer_max
        self._local = threading.local()

    @staticmethod
    def with_system(messages, text):
        """After a leading system message if there is one, else first; the rest stays untouched."""
        head = 1 if messages and messages[0].get("role") == "system" else 0
        return messages[:head] + [{"role": "system", "content": text}] + messages[head:]

    def config(self, **forced_extra):
        """The keys this adds to the runner's extra_config — on every row's `config`."""
        return {"prompt_ceiling": self.ceiling,
                "cut_retry": {"prompt": self.retry_ceiling, "when": "the first call ends at length"},
                "forced_answer": dict({"prompt": self.force_prompt, "max_tokens": self.answer_max,
                                       "when": "the second call also ends at length, with text to "
                                               "hand back"}, **forced_extra)}

    def __call__(self, messages, call, model=""):
        kept = getattr(self._local, "kept", None)
        if kept is None or kept.get("messages") is not messages:
            kept = self._local.kept = {"messages": messages}
        if "first" not in kept:
            started = time.time()
            open_call()
            answer, handback, meta = call(self.with_system(messages, self.ceiling), "first")
            # into the row's ledger first — tokens billed once even if a later attempt reuses this
            # stage, and its reasoning_chars into the row's sum — then off the stage's own meta
            record_usage(meta)
            meta.pop("reasoning_chars", None)
            if meta.get("finish_reason") != "length":
                return answer, dict(meta, _usage_recorded=True)
            kept["first"] = (answer, meta, round(time.time() - started, 2))
            print(f"[{model}] reply reached the cap ({meta.get('completion_tokens')} tokens): "
                  f"called again with the retry ceiling", flush=True)
            rearm_call_timeout()
        first_text, first_meta, first_latency = kept["first"]
        marks = {"cut_retry": True,
                 "first_finish_reason": first_meta.get("finish_reason"),
                 "first_completion_tokens": first_meta.get("completion_tokens"),
                 "first_reasoning_tokens": first_meta.get("reasoning_tokens"),
                 "first_latency_s": first_latency,
                 "first_response": first_text}
        retry_sent = self.with_system(messages, self.retry_ceiling)
        if "retry" not in kept:
            started = time.time()
            open_call()
            answer, handback, meta = call(retry_sent, "retry")
            record_usage(meta)
            meta.pop("reasoning_chars", None)
            retry_latency = round(time.time() - started, 2)
            if meta.get("finish_reason") != "length" or not handback.strip():
                out = dict(meta, **_usage_sum([first_meta, meta]), **marks, retry_latency_s=retry_latency,
                           _usage_recorded=True)
                print(f"[{model}] retry: {meta.get('completion_tokens')} tokens, "
                      f"finish={meta.get('finish_reason')}", flush=True)
                return answer, out
            kept["retry"] = (answer, handback, meta, retry_latency)
            print(f"[{model}] retry reached the cap too ({meta.get('completion_tokens')} tokens): "
                  f"forced answer", flush=True)
            rearm_call_timeout()
        retry_text, retry_handback, retry_meta, retry_latency = kept["retry"]
        started = time.time()
        open_call()
        answer, _, meta = call(retry_sent + [{"role": "assistant", "content": retry_handback},
                                             {"role": "user", "content": self.force_prompt}], "forced")
        record_usage(meta)
        reasoning_chars = meta.pop("reasoning_chars", 0) or 0
        out = dict(meta, **_usage_sum([first_meta, retry_meta, meta]), **marks, _usage_recorded=True)
        out.update(retry_finish_reason=retry_meta.get("finish_reason"),
                   retry_completion_tokens=retry_meta.get("completion_tokens"),
                   retry_reasoning_tokens=retry_meta.get("reasoning_tokens"),
                   retry_latency_s=retry_latency,
                   retry_response=retry_text,
                   forced_latency_s=round(time.time() - started, 2),
                   forced_answer=True,
                   forced_finish_reason=meta.get("finish_reason"),
                   forced_completion_tokens=meta.get("completion_tokens"),
                   forced_reasoning_tokens=meta.get("reasoning_tokens"),
                   forced_reasoning_chars=reasoning_chars)
        print(f"[{model}] forced answer: {meta.get('completion_tokens')} tokens, "
              f"finish={meta.get('finish_reason')}, reasoning {reasoning_chars} chars", flush=True)
        return answer, out


# ---------------------------------------------------------------- sample rows and files


def parse_ok(model_output):
    """Whether the scorer extracted a prediction: `extract_final_answer` returned something non-empty
    — the text after `Final Answer:`, or, with no marker, the whole response, which the exact,
    letter, option-text, comma and dyck branches of `score_response` still match (a bare `(B)` is
    credited). So a row scored 1 always has parse_ok, and parse_ok is False only on an empty response
    or an empty marker. Whether the model FINISHED is a different question: `has_marker` /
    `no_marker` answers it, as it always has in bbh."""
    return bool(extract_final_answer(model_output).strip())


# The per-row fields of evaluation-criteria.md, present as keys on every sample row — None where a
# value does not exist (a failed row's latency, a provider that reports no reasoning breakdown).
ROW_USAGE = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s", "finish_reason",
             "n_attempts", "served_model")
ROW_KEYS = ROW_USAGE + BILLED_KEYS + ("reasoning_chars", "think_tag_calls", "halted") + COST_KEYS
SAMPLE_FIELDS = ["idx", "task", "question", "gold_answer", "model_response", "final_answer",
                 "has_marker", "parse_ok", "score", "failed", *ROW_KEYS, *PRIOR_KEYS, "backend",
                 "cut_retry", "forced_answer", "sample", "ts", "config"]


def is_unfinished(row):
    """A row of a forced-answer runner (CutRetry) that still ended at the cap with no forced answer:
    DeepSeek's second call cut with `content` but no `reasoning_content` to hand back. PlanBench
    exits 1 on it (`plan_eval_core.run_cli`); here it is not a done row either — the run exits 1
    and resume calls it again, keeping its billing (fold_prior)."""
    return bool("forced_answer" in (row.get("config") or {}) and not row.get("failed")
                and row.get("finish_reason") == "length" and not row.get("forced_answer"))


def is_thinking(row):
    """Reasoning came back, on any call of the row: reasoning text, a `<think>` tag in the content,
    or reasoning tokens billed."""
    return any((row.get(k) or 0) > 0 for k in ("reasoning_chars", "think_tag_calls",
                                                "reasoning_tokens", "billed_reasoning_tokens"))


def row_done(row):
    """A stored row resume may skip: answered, not cut off by the backend, not left unfinished."""
    return bool(str(row.get("model_response", "")).strip()
                and row.get("finish_reason") not in INTERRUPTED_FINISH and not is_unfinished(row))


def sample_row(task, idx, example, template, once, model, model_dir, config, sample):
    """One sampled item: the call through call_with_retries, scored by the one scorer. Rows are built
    here and only here, so every row carries every field.

    A row whose every attempt failed is WRITTEN, `failed: 1`, empty response, score 0 — counted
    against the model, as bbh has always counted an unusable row, and visible as a failure rather
    than vanishing from the reliability rate. Resume calls it again."""
    messages = [{"role": "user", "content": template.format(question=example["input"])}]
    text, meta = call_with_retries(once, messages, model, model_dir)
    return build_row(task, idx, example, text, meta, config, sample)


def build_row(task, idx, example, text, meta, config, sample):
    """The row for one item from what call_with_retries returned — or, for a row a halt or SIGTERM
    interrupted, from its stashed meta with an empty response (a failed row)."""
    q, gold = example["input"], example["target"]
    text = text if isinstance(text, str) else ""
    row = {"idx": idx, "task": task, "question": q, "gold_answer": gold, "model_response": text,
           "final_answer": extract_final_answer(text), "has_marker": has_marker(text),
           "parse_ok": parse_ok(text), "score": score_response(text, gold, q),
           "failed": int(not text)}
    row.update({k: meta.get(k) for k in ROW_KEYS})
    row.update(prior_n_attempts=0, prior_failed_runs=0, resumed_after_failure=False)
    row.update({k: v for k, v in meta.items() if k not in row})
    row.update(sample=sample, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), config=config)
    return row


def fold_prior(row, prior):
    """`row` replaces `prior`, a stored row that was not done (failed, interrupted or unfinished).
    What the earlier runs cost is not lost: the billed figures become cumulative over every run of
    the row, `prior_n_attempts` sums the earlier runs' attempts and `prior_failed_runs` counts them.
    `n_attempts`, `latency_s` and the token fields stay this run's."""
    for k in ("billed_prompt_tokens", "billed_completion_tokens", "attempts_without_usage"):
        row[k] = (row.get(k) or 0) + (prior.get(k) or 0)
    for k in ("billed_reasoning_tokens", "billed_cost", "billed_estimated_cost"):
        row[k] = _sum_or_none([row.get(k), prior.get(k)])
    row["prior_n_attempts"] = (prior.get("prior_n_attempts") or 0) + (prior.get("n_attempts") or 0)
    row["prior_failed_runs"] = (prior.get("prior_failed_runs") or 0) + 1
    row["resumed_after_failure"] = True
    return row


def _atomic_write(path, write):
    # host AND pid: array tasks on different nodes share one GPFS directory
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        write(fh)
    os.replace(tmp, path)


def _save_jsonl(path, rows):
    def write(fh):
        for r in sorted(rows, key=lambda x: x["idx"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _atomic_write(path, write)


class StoredThinking(SystemExit):
    """A stored row of a thinking-off runner already shows thinking. Exit code 3; run_sample turns
    it into a rewritten THINKING_HALT marker before any row is called."""

    def __init__(self, row, path):
        super().__init__(3)
        self.row, self.path = row, path


def load_sample_checkpoint(path, config, allowed, verbose=True, thinking_off=False):
    """`(done, held)` from a sample file, each keyed by idx. `done` rows are skipped. `held` rows —
    failed, interrupted, or unfinished (`is_unfinished`) — are called again, and STAY in what is
    saved until a new row replaces them (PlanBench's `held`): a resubmit killed before it reaches
    one must not drop it, and its replacement folds its billing in (`fold_prior`).

    Refuses a file written under another config (archive, never resume across one — the prompt is
    in the config), a row this shard would not call (saving would drop it), and — for a thinking-off
    runner — a stored row that came back thinking: the config is wrong and those rows are not data."""
    if not os.path.exists(path):
        return {}, {}
    done, held = {}, {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("config") != config:
                stored = r.get("config") or {}
                stored = stored if isinstance(stored, dict) else {"<not a dict>": stored}
                diff = {k: (config.get(k), stored.get(k)) for k in set(config) | set(stored)
                        if config.get(k) != stored.get(k)}
                raise SystemExit(f"{path}\n  refusing to resume: stored rows ran under a different "
                                 f"config.\n  (wanted, stored): {diff}\n  Archive the directory to a "
                                 f"timestamped name and start clean.")
            if r.get("idx") not in allowed:
                raise SystemExit(f"{path}\n  holds idx {r.get('idx')}, which this run would not call "
                                 f"(another sample or other shard boundaries). Saving would drop it; "
                                 f"archive the file and look at it first.")
            if thinking_off and is_thinking(r):
                raise StoredThinking(r, path)
            (done if row_done(r) else held)[r["idx"]] = r
    if verbose and (done or held):
        print(f"  resume: {len(done)} row(s) done, {len(held)} failed or unfinished row(s) held and "
              f"called again", flush=True)
    return done, held


def _task_counts(rows):
    n = len(rows)
    scored = sum(r["score"] for r in rows)
    return {"n": n, "score_sum": scored, "average_score": round(scored / n, 4) if n else "",
            "no_marker": sum(0 if r["has_marker"] else 1 for r in rows),
            "empty_response": sum(1 for r in rows if not str(r["model_response"]).strip()),
            "failed": sum(r.get("failed", 0) for r in rows),
            "finish_length": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "cut_retry": sum(1 for r in rows if r.get("cut_retry")),
            "forced_answer": sum(1 for r in rows if r.get("forced_answer")),
            "unfinished": sum(1 for r in rows if is_unfinished(r)),
            "resumed_after_failure": sum(1 for r in rows if r.get("resumed_after_failure")),
            "forced_odd": sum(1 for r in rows if r.get("forced_answer")
                              and (r.get("finish_reason") == "length" or forced_reasoned(r))),
            # thinking that came back: only a problem for a runner that declares it off
            "thinking_rows": sum(1 for r in rows if is_thinking(r))}


def write_sample_task(root, task, model_id, records, config, tag="", thinking_off=False):
    """The same three files per task as `write_task_results`, under the sample's root: the `.jsonl`,
    a `.csv` with the per-row fields, and a per-task `_overall.csv` with failures counted."""
    d = os.path.join(root, task)
    os.makedirs(d, exist_ok=True)
    slug = model_slug(model_id) + tag
    _save_jsonl(os.path.join(d, f"{slug}.jsonl"), records)

    def write_csv(fh):
        w = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
        w.writeheader()
        for r in sorted(records, key=lambda x: x["idx"]):
            out = {k: r.get(k, "") for k in SAMPLE_FIELDS}
            out["config"] = json.dumps(r.get("config"), sort_keys=True, ensure_ascii=False)
            w.writerow(out)
    _atomic_write(os.path.join(d, f"{slug}.csv"), write_csv)

    counts = _task_counts(records)

    def write_overall_csv(fh):
        w = csv.writer(fh)
        # thinking_rows means something only where thinking is declared off; elsewhere it would count
        # every reasoning model's ordinary rows, so the column is left out
        cols = [c for c in counts if thinking_off or c != "thinking_rows"]
        w.writerow(["model", "task", *cols, "scorer", "config"])
        w.writerow([model_id, task, *[counts[c] for c in cols], SCORER_VERSION,
                    json.dumps(config, sort_keys=True, ensure_ascii=False)])
    _atomic_write(os.path.join(d, f"{slug}_overall.csv"), write_overall_csv)
    return dict(counts, model=model_id, task=task)


def write_sample_overall(root, model_id, ids_by_task, limit=0, thinking_off=False):
    """The roll-up, rebuilt from EVERY task file on disk, shards merged — never from this process's
    own summaries. bbh's `results/*_bbh_overall.csv` went stale for two models exactly that way
    (bbh.md § traps): a run over some tasks overwrote the roll-up with those tasks alone.

    `complete` is 1 only when every task holds exactly its sampled ids, no row failed or is left
    unfinished, all rows ran under one config, and this is not a smoke run — and, for a runner that
    declares thinking off, no row came back with reasoning. A macro over a sample is an estimate of
    the full benchmark's, not a replacement for it."""
    slug = model_slug(model_id)
    recs, problems, configs = [], [], set()
    for task in TASKS:
        d = os.path.join(root, task)
        plain = os.path.join(d, f"{slug}.jsonl")
        shards = sorted(glob.glob(os.path.join(d, f"{slug}_shard*of*.jsonl")))
        if os.path.exists(plain) and shards:
            problems.append(f"{task}: both an unsharded file and shard files")
        rows = {}
        for f in ([plain] if os.path.exists(plain) else []) + shards:
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        if r["idx"] in rows:
                            problems.append(f"{task}: idx {r['idx']} in more than one file")
                        rows[r["idx"]] = r
        expected = ids_by_task[task][:limit] if limit else ids_by_task[task]
        stray = set(rows) - set(expected)
        if stray:
            problems.append(f"{task}: {len(stray)} row(s) not in the sample")
        configs |= {json.dumps(r.get("config"), sort_keys=True) for r in rows.values()}
        recs.append(dict(model=model_id, task=task, expected=len(expected),
                         **_task_counts(list(rows.values()))))
    present = [r for r in recs if r["n"]]
    means = [r["average_score"] for r in present]
    if len(configs) > 1:
        problems.append(f"rows ran under {len(configs)} different configs")
    unfinished = sum(r["unfinished"] for r in recs)
    if unfinished:
        problems.append(f"{unfinished} row(s) cut at the cap with no forced answer — unfinished; "
                        f"resume calls them again")
    thinking = sum(r["thinking_rows"] for r in recs)
    if thinking_off and thinking:
        problems.append(f"THINKING CAME BACK ON: {thinking} row(s) with reasoning_chars > 0 or "
                        f"reasoning tokens > 0, on a route where this model must answer without it")
    complete = int(not limit and not problems
                   and all(r["n"] == r["expected"] and r["failed"] == 0 for r in recs))
    path = os.path.join(root, f"{slug}_bbh_overall.csv")
    cols = ["model", "task", "n", "expected", "average_score", "no_marker", "empty_response",
            "failed", "finish_length", "cut_retry", "forced_answer", "unfinished",
            "resumed_after_failure", "thinking_rows", "scorer"]
    summed = cols[5:13]                          # no_marker .. resumed_after_failure

    def write(fh):
        w = csv.writer(fh)
        w.writerow(cols + ["complete", "problems"])
        for r in recs:
            w.writerow([SCORER_VERSION if c == "scorer" else
                        "" if c == "thinking_rows" and not thinking_off else r.get(c, "")
                        for c in cols] + ["", ""])
        w.writerow([model_id, "MACRO_AVG_over_%d_tasks" % len(means),
                    sum(r["n"] for r in recs), sum(r["expected"] for r in recs),
                    round(sum(means) / len(means), 4) if means else "",
                    *[sum(r[c] for r in recs) for c in summed],
                    thinking if thinking_off else "", SCORER_VERSION, complete, " | ".join(problems)])
    os.makedirs(root, exist_ok=True)
    _atomic_write(path, write)
    return complete, problems, thinking


# ---------------------------------------------------------------- the sample loop


def _thinking_marker(row, model, model_dir, why):
    """Write `<prefix>THINKING_HALT.txt` for `row`; returns its path and the detail line."""
    marker = os.path.join(model_dir, MARKER_PREFIX + "THINKING_HALT.txt")
    detail = (f"{row.get('task')}#{row.get('idx')}: reasoning_chars={row.get('reasoning_chars')} "
              f"think_tag_calls={row.get('think_tag_calls')} "
              f"reasoning_tokens={row.get('reasoning_tokens')} "
              f"billed_reasoning_tokens={row.get('billed_reasoning_tokens')} ({why})")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason=THINKING CAME BACK ON\n{detail}\n\nThis route "
                     f"must answer without thinking, so the call's config is wrong. Read the row, "
                     f"archive the results directory, fix the call, then delete this file: a new run "
                     f"refuses to start while it exists.\n")
    except OSError as error:
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! THINKING HALT [{model}] {stamp}\n{detail}\n", file=sys.stderr, flush=True)
    return marker, detail


def _thinking_halt(row, model, model_dir):
    """Stop at the FIRST row that came back thinking on a thinking-off route: the config is wrong,
    and finishing 484 rows on it would only spend money on rows that are not the measurement.
    Writes `<prefix>THINKING_HALT.txt`, which — unlike the other markers — a new run does not clear:
    it refuses to start until someone has read the rows and deleted it. Exit 3."""
    marker, _ = _thinking_marker(row, model, model_dir, "this run")
    print(f"\nSTOP: THINKING CAME BACK ON at {row['task']}#{row['idx']} — halted at the first such "
          f"row; marker {marker}", flush=True)
    raise SystemExit(3)


def run_sample_tasks(model_dir, model_id, once, sample, tasks=None, config=None, limit=0,
                     prompt_version="v1", shard=0, total_shards=1, sleep_between=0.0,
                     save_every=20, thinking_off=False, verbose=True):
    """Drive `once(messages) -> (text, meta)` over the sample's items of each task, one at a time on
    the main thread — the only place the SIGALRM watchdog works (`require_main_thread`).

    The sample's ids are the work list, so a row's `idx` indexes the whole task. `--limit N` takes
    the first N sampled ids of each task into `smoke_<sample>/`, with no resume. A shard slices the
    sampled ids (the legacy serial loop ignores the shard; this one does not). A held row (failed or
    unfinished on an earlier run) stays in every save until its new row replaces it."""
    require_main_thread()
    if save_every < 1:
        raise SystemExit(f"--save-every {save_every}: must be 1 or more")
    tasks = tasks or TASKS
    if prompt_version not in PROMPTS:
        raise SystemExit(f"unknown prompt version {prompt_version!r}; known: {sorted(PROMPTS)}")
    template = PROMPTS[prompt_version]
    ids_by_task = load_sample(sample)
    root = sample_root(model_dir, sample, smoke=bool(limit))
    config = json.loads(json.dumps(dict(config or {}, prompt=prompt_version), sort_keys=True))
    tag = shard_tag(shard, total_shards)
    summaries = []
    for task in tasks:
        examples = load_task(task)
        ids = ids_by_task[task][:limit] if limit else ids_by_task[task]
        mine = shard_slice(ids, shard, total_shards)
        out_path = os.path.join(root, task, f"{model_slug(model_id)}{tag}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        done, held = ({}, {}) if limit else load_sample_checkpoint(
            out_path, config, set(mine), verbose, thinking_off=thinking_off)
        records = [done[i] for i in mine if i in done]
        todo = [i for i in mine if i not in done]

        def to_save():
            # a held row stays until a new row for its idx is in `records` — then only the new one
            replaced = {r["idx"] for r in records}
            return records + [r for i, r in held.items() if i not in replaced]
        current = None
        if verbose:
            print(f"[{task}] {len(mine)} sampled item(s) here, {len(todo)} to call -> {out_path}"
                  + ("  (--limit: smoke test, not a run)" if limit else ""), flush=True)
        try:
            for k, i in enumerate(todo, 1):
                # S1: clear the slots BEFORE this row's id becomes `current`, so a stop landing
                # before its call starts cannot write the previous row's result or bill under it.
                _INFLIGHT.result = None
                _LEDGER.book = None
                current = i
                row = sample_row(task, i, examples[i], template, once, model_id, model_dir, config,
                                 sample)
                prior = held.get(i)
                if prior is not None:
                    fold_prior(row, prior)
                records.append(row)
                held.pop(i, None)
                take_inflight()                        # this row is in records: drop its stash
                current = None
                if thinking_off and is_thinking(row):
                    _save_jsonl(out_path, to_save())
                    _thinking_halt(row, model_id, model_dir)
                if k % save_every == 0:
                    _save_jsonl(out_path, to_save())
                    if verbose:
                        print(f"  [{task}] {len(records)}/{len(mine)}", flush=True)
                if sleep_between and row["model_response"]:
                    time.sleep(sleep_between)
        finally:
            # A halt or the wall clock must not lose rows paid for since the last periodic save, nor
            # a held row this run never reached — nor the row in flight: call_with_retries stashed
            # how it ended, and it is written here with its history — answered if its call had
            # returned an answer before the kill landed (mid-build, mid-write), failed otherwise.
            # S2: whether the row is already in `records` is read from `records` itself (its idx),
            # never from a flag set after the append — a stop between the two would write it twice.
            inflight = take_inflight()
            if current is not None and inflight is not None and \
                    current not in {r["idx"] for r in records}:
                text, meta = inflight
                row = build_row(task, current, examples[current], text, meta, config, sample)
                if held.get(current) is not None:
                    fold_prior(row, held[current])
                records.append(row)
                held.pop(current, None)
            s = write_sample_task(root, task, model_id, to_save(), config, tag=tag,
                                  thinking_off=thinking_off)
        summaries.append(s)
        if verbose:
            print(f"{task}: {s['average_score']}  (n={s['n']}/{len(mine)}, failed={s['failed']}, "
                  f"no_marker={s['no_marker']}, length={s['finish_length']}, "
                  f"cut_retry={s['cut_retry']}, unfinished={s['unfinished']}"
                  + (f", thinking={s['thinking_rows']}" if thinking_off else "") + ")", flush=True)
        write_sample_overall(root, model_id, ids_by_task, limit=limit, thinking_off=thinking_off)
    return summaries


def run_sample(model_dir, model_id, *, client, once, wanted, sample, tasks=None, fixed=None,
               required=None, extra_config=None, call_timeout=300, limit=0, workers=1,
               prompt_version="v1", shard=0, total_shards=1, sleep_between=0.0, save_every=20,
               thinking_off=False):
    """A whole `--sample` run for one model; returns the process exit code. `once(messages, model,
    params) -> (text, meta)`, with a client built `max_retries=0`.

    Refused before anything is called: `workers` above 1 (the watchdog cannot guard a call off the
    main thread, `require_main_thread`), `save_every` below 1, and a THINKING_HALT marker left by
    an earlier run. Exit codes, so SLURM cannot report COMPLETED 0:0 over a bad run:
    3 — `thinking_off` and a row came back with reasoning: halted at that row (`_thinking_halt`);
    1 — a row failed every attempt, or was left unfinished at the cap: resubmit, resume calls
        exactly those and keeps their history;
    2 — a forced answer filled its own cap or reasoned again: read those rows.

    Halt markers are `sample10_*_HALT.txt` for the real run and `smoke_sample10_*_HALT.txt` for a
    `--limit` run, so a smoke never clears the real run's marker."""
    global HARD_CALL_TIMEOUT, MARKER_PREFIX
    if workers != 1:
        raise SystemExit(f"--workers {workers} with --sample: refused. Sample calls run one at a time "
                         f"on the main thread, the only place the SIGALRM watchdog can end a hung "
                         f"call; a worker thread would run unguarded (bbh_eval_core.require_main_thread)")
    if save_every < 1:
        raise SystemExit(f"--save-every {save_every}: must be 1 or more")
    require_main_thread()
    load_sample(sample)                     # a bad sample file fails before the probe call is paid
    HARD_CALL_TIMEOUT = int(call_timeout)
    MARKER_PREFIX = (f"smoke_{sample}_" if limit else f"{sample}_") + (
        f"shard{shard}_" if total_shards > 1 else "")
    thinking_marker = os.path.join(model_dir, MARKER_PREFIX + "THINKING_HALT.txt")
    if os.path.exists(thinking_marker):
        print(f"{thinking_marker} exists: an earlier run came back thinking on a route where it must "
              f"not. Read it, archive the results, fix the call, and delete the marker first.",
              flush=True)
        raise SystemExit(3)
    # SLURM ends a job (wall clock, scancel) with SIGTERM, whose default skips `finally`
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    clear_stale_halt_markers(model_dir)
    print(f"negotiating parameter surface for {model_id} ...", flush=True)
    params, notes = negotiate_pinned(client, model_id, wanted, model_dir, fixed=fixed,
                                     required=required)
    print(f"  accepted: {params}" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    if "seed" in wanted and "seed" not in params:
        print("  WARNING: seed was dropped — this run is not reproducible", flush=True)
    config = json.loads(json.dumps(dict(extra_config or {}, model=model_id, params=params,
                                        fixed=fixed or {}, prompt=prompt_version), sort_keys=True))
    root = sample_root(model_dir, sample, smoke=bool(limit))
    if not limit:
        # every task's checkpoint is checked BEFORE anything is written or called, so a refused
        # resume leaves the directory exactly as it was — negotiated_params.json included
        # A stored row that already shows thinking stops the run here, before any row is called:
        # the marker is written again (someone may have deleted it without reading the rows) and the
        # job exits 3 — otherwise a resubmit would call every remaining row on a config already known
        # to be wrong. Only the negotiation probe above has been spent.
        ids_by_task = load_sample(sample)
        for task in tasks or TASKS:
            try:
                load_sample_checkpoint(
                    os.path.join(root, task,
                                 f"{model_slug(model_id)}{shard_tag(shard, total_shards)}.jsonl"),
                    config, set(shard_slice(ids_by_task[task], shard, total_shards)), verbose=False,
                    thinking_off=thinking_off)
            except StoredThinking as stored:
                marker, _ = _thinking_marker(stored.row, model_id, model_dir,
                                             f"stored in {stored.path}, found on resume")
                print(f"\nSTOP: a stored row already shows THINKING — no row called; marker "
                      f"rewritten: {marker}", flush=True)
                raise SystemExit(3)
    os.makedirs(root, exist_ok=True)
    _atomic_write(os.path.join(root, f"{model_slug(model_id)}{shard_tag(shard, total_shards)}"
                                     f"_negotiated_params.json"),
                  lambda fh: json.dump({"asked": wanted, "accepted": params, "notes": notes,
                                        "config": config}, fh, indent=2, ensure_ascii=False))
    started = time.time()
    try:
        summaries = run_sample_tasks(model_dir, model_id, lambda m: once(m, model_id, params), sample,
                                     tasks=tasks, config=config, limit=limit,
                                     prompt_version=prompt_version, shard=shard,
                                     total_shards=total_shards, sleep_between=sleep_between,
                                     save_every=save_every, thinking_off=thinking_off)
    finally:
        print(f"\ncalls: {STATS}  elapsed {time.time() - started:.0f}s", flush=True)
    failed = sum(s["failed"] for s in summaries)
    unfinished = sum(s["unfinished"] for s in summaries)
    odd = sum(s["forced_odd"] for s in summaries)
    thinking = sum(s["thinking_rows"] for s in summaries)
    print(f"done -> {root}", flush=True)
    if thinking_off and thinking:              # the halt stops a run first; this is the backstop
        print(f"\nSTOP: THINKING CAME BACK ON — {thinking} row(s).", flush=True)
        return 3
    if failed or unfinished:
        print(f"\n{failed} row(s) failed every attempt and {unfinished} were left unfinished at the "
              f"cap — resubmit: resume calls exactly those, keeping their history.", flush=True)
        return 1
    if odd:
        print(f"\nSTOP: {odd} forced answer(s) filled their own cap or reasoned again — read them "
              f"before scoring.", flush=True)
        return 2
    return 0


def parse_tasks(text):
    tasks = TASKS if text == "all" else [t.strip() for t in text.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise SystemExit("unknown task(s): %s\nknown: %s" % (unknown, TASKS))
    return tasks


def sample_main(*, description, default_model, model_dir, client, once, wanted, fixed=None,
                required=None, extra_config=None, call_timeout=300, default_sleep=0.0,
                thinking_off=False):
    """The command line of a sample-only runner (`BBH_Qwen_DeepInfra`, `BBH_Gemma_DeepInfra`,
    `BBH_Deepseek/deepseek_bbh_sample.py`). Exits with run_sample's code. There is no `--workers`:
    sample calls run one at a time on the main thread (`require_main_thread`)."""
    def at_least_one(text):
        value = int(text)
        if value < 1:
            raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
        return value

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default=default_model,
                    help="model id; it is BOTH what is called and what the result files are named after")
    ap.add_argument("--sample", required=True,
                    help="the sample file <bbh>/<name>.json, e.g. sample10; rows go to results_<name>/")
    ap.add_argument("--task", default="all", help="'all' or a comma-separated list of task names")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N sampled items of each task, into smoke_<name>/ — a smoke test")
    ap.add_argument("--prompt", default="v1", choices=sorted(PROMPTS),
                    help="prompt version; part of the config on every row")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="at 1 the filenames carry no shard tag")
    ap.add_argument("--sleep", type=float, default=default_sleep,
                    help="seconds after each answered item")
    ap.add_argument("--save-every", dest="save_every", type=at_least_one, default=20,
                    help="rewrite the task file every N rows; 1 or more")
    args = ap.parse_args()
    sys.exit(run_sample(model_dir, args.model, client=client, once=once, wanted=wanted,
                        sample=args.sample, tasks=parse_tasks(args.task), fixed=fixed,
                        required=required, extra_config=extra_config, call_timeout=call_timeout,
                        limit=args.limit, prompt_version=args.prompt, shard=args.shard,
                        total_shards=args.total_shards, sleep_between=args.sleep,
                        save_every=args.save_every, thinking_off=thinking_off))
