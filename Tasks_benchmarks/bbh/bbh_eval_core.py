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

import concurrent.futures
import csv
import json
import os
import re
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


def write_task_results(model_dir, task, model_id, records, config=None):
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
        w.writerow(["model", "task", "n", "score_sum", "average_score", "no_marker",
                    "empty_response", "scorer", "config"])
        w.writerow([model_id, task, n, scored, round(scored / n, 4) if n else "", missing, blank,
                    SCORER_VERSION, config_string(config)])

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
              prompt_version="v1"):
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
        out_path = os.path.join(task_dir(model_dir, task), f"{model_slug(model_id)}.jsonl")
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

        todo = [(i, ex) for i, ex in enumerate(examples) if i not in done]
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
            s_ = write_task_results(model_dir, task, model_id, records, config=config)
            summaries.append(s_)
            if verbose:
                print(f"{task}: {s_['average_score']}  (n={s_['n']}, no_marker={s_['no_marker']}, "
                      f"empty={s_['empty_response']})", flush=True)
            write_overall(model_dir, model_id, summaries)
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
        s = write_task_results(model_dir, task, model_id, records, config=config)
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
