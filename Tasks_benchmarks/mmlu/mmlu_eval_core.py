"""Shared core for every MMLU runner. Modelled on `Tasks_benchmarks/bbh/bbh_eval_core.py`.

**One scorer.** `score_response` is the lenient matcher and it is the ONLY scorer in this benchmark.
MMLU arrived with the same split bbh had: `<provider>_eval.py` compared the model's text to the
gold choice with `==`, while a separate `<provider>_rescore.py` — written for only three of the
seven providers — accepted a letter, an index, or letter-plus-text. A model that answered `C` was
scored 0 by one and 1 by the other. Comparing two models scored differently is not a comparison, so
the scorer is imported, never copied.

**Two prompts, and they are not interchangeable.** Variant `v1` dumps the choice list and asks for
the answer text; `v2` labels the choices `A.`–`D.` and asks for a letter. Four runners used v2 and
three used v1, so MMLU was asking different models to do a materially different task. New runs
default to **v2** — it is the majority, and a single letter is unambiguous to score. The prompt
version is part of the run config and is written onto every row.
"""

import concurrent.futures
import csv
import json
import os
import re
import threading
import time

# Paths resolve from THIS FILE, never the cwd: a runner lives in <mmlu>/MMLU_<Slot>/ while the data
# lives in <mmlu>/data/.
MMLU_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(MMLU_ROOT, "data")
ENV_PATH = os.path.join(MMLU_ROOT, ".env")

SUBJECTS = [
    "Business_ethics", "Econometrics", "Elementary_math", "Formal_logic", "Jurisprudence",
    "Logical_fallacies", "Management", "Marketing", "Miscellaneous", "Moral_disputes",
    "Moral_scenarios", "Philosophy", "Professional_accounting",
]

LABELS = ["A", "B", "C", "D"]

# Byte-for-byte what the runners that produced the rows on disk sent. Do not tidy the indentation:
# it changes the token stream, and a new run would then differ from the stored rows by the prompt
# as well as by the model.
PROMPT_V1 = """
    The following is a multiple-choice question on the subject of {subject}. 
    Question: {question}
    Choices: {choices}

    Please respond with the correct answer from the choices, and end your response with:
    "Final Answer: <your answer here>"
    """

PROMPT_V2 = """
    The following is a multiple-choice question on the subject of {subject}.
    Question: {question}
    Choices:
    {formatted_choices}

    Please show your reasoning, then end your response with:
    "Final Answer: <A, B, C, or D>"
    """

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

# Bumped whenever score_response changes behaviour, and written onto every result file. v1 is the
# five branches of the old `*_rescore.py` plus the packaging lessons bbh paid for across its own
# five versions (`.claude/references/benchmarks/tasks/bbh-scoring.md`).
SCORER_VERSION = "mmlu_lenient_v1"


def model_slug(model_id):
    return model_id.replace("/", "-").replace(".", "_")


def load_subject(subject):
    """Returns the raw list of {question, subject, choices, answer}. `answer` is an INDEX as a
    string — `"2"` means `choices[2]`, not choice number 2."""
    with open(os.path.join(DATA_DIR, f"{subject}.json")) as fh:
        return json.load(fh)


def build_prompt(example, version):
    choices = example["choices"]
    if version == "v1":
        return PROMPTS["v1"].format(subject=example["subject"], question=example["question"],
                                    choices=choices)
    formatted = "\n".join(f"{LABELS[i]}. {choices[i]}" for i in range(len(choices)))
    return PROMPTS["v2"].format(subject=example["subject"], question=example["question"],
                                formatted_choices=formatted)


# ---------------------------------------------------------------- scoring


def extract_final_answer(model_output):
    """Text after the `Final Answer:` marker; the whole response when the marker is absent.

    That fallback is a truncation detector, not a scoring bug — pair it with `has_marker` to tell
    "wrong" apart from "never finished".
    """
    if not isinstance(model_output, str):
        return ""
    m = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    result = m.group(1).strip() if m else model_output.strip()
    return result.strip("\"'`* ").strip()


def has_marker(model_output):
    return isinstance(model_output, str) and bool(
        re.search(r"Final Answer:", model_output, re.IGNORECASE))


def _unwrap(text):
    """LaTeX inline-math delimiters are packaging, like quotes and bold."""
    return re.sub(r"\\[\(\)\[\]]", "", text).strip().strip("$").strip()


def _norm(text):
    return re.sub(r"\s+", " ", str(text).strip().rstrip(".")).lower()


def score_response(model_response, gold_text, gold_letter, answer_index):
    """The one scorer. Generous about how the answer is written, strict about which choice it names.

    Every branch identifies the SAME choice a different way. None of them guesses: an answer that
    names a different choice, or names none, scores 0.
    """
    if not isinstance(model_response, str) or not model_response.strip():
        return 0
    fa = extract_final_answer(model_response)
    if not fa:
        return 0
    bare = _unwrap(fa)
    gl = str(gold_letter).upper()

    # 1. the choice text, whitespace/case/trailing-period normalised
    if _norm(bare) == _norm(gold_text):
        return 1

    # 2. the letter alone — "A", "(A)", "A.", "A)"
    m = re.match(r"^\(?([A-D])\)?[.):]?\s*$", bare, re.IGNORECASE)
    if m and m.group(1).upper() == gl:
        return 1

    # 3. the index — the model answered "2" where the gold is choices[2]
    if bare.strip() == str(answer_index):
        return 1

    # 4. letter then text — "A. Cryptocurrencies, Cheap…"
    m = re.match(r"^\(?([A-D])\)?[.):]?\s+(.+)", bare, re.IGNORECASE | re.DOTALL)
    if m and m.group(1).upper() == gl:
        return 1

    # 5. comma-vs-space inside the choice text
    if _norm(bare).replace(",", " ").split() == _norm(gold_text).replace(",", " ").split():
        return 1

    # ---- packaging branches: only for an answer the model actually declared ----
    # Without the marker `fa` is a scrape of the whole response, and "the gold letter appears
    # somewhere in the reasoning" is not an answer.
    if not has_marker(model_response):
        return 0

    # 6. the letter at the END rather than the start — "Cryptocurrencies … (A)". Length-capped and
    #    requiring exactly one distinct letter, so a response weighing several options is not read
    #    as choosing one.
    if len(fa) < 160:
        letters = re.findall(r"\(?\b([A-D])\)", fa)
        if len(set(letters)) == 1 and letters[0].upper() == gl:
            return 1

    # 7. the answer is the gold text with the option label glued on — "A) …" already covered, but
    #    also "Answer: A" and "The answer is A."
    m = re.search(r"\b(?:answer|option|choice)\b\D{0,12}\b([A-D])\b", fa, re.IGNORECASE)
    if m and m.group(1).upper() == gl:
        return 1

    return 0


# ---------------------------------------------------------------- output

FIELDS = ["idx", "subject", "question", "choices", "gold_answer", "gold_letter", "answer_index",
          "model_response", "final_answer", "has_marker", "score", "config"]


def config_string(config):
    """The generation config, flattened onto every row — `model-parameters.md` rule 8: pin a seed
    **and write it on every row**. A config that lives only in a job script cannot be recovered
    from a result file later."""
    if not config:
        return ""
    return ";".join(f"{k}={v}" for k, v in sorted(config.items()))


def parse_config(text):
    out = {}
    for part in (text or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def subject_dir(model_dir, subject):
    d = os.path.join(model_dir, "results", subject)
    os.makedirs(d, exist_ok=True)
    return d


def write_subject_results(model_dir, subject, model_id, records, config=None):
    d = subject_dir(model_dir, subject)
    slug = model_slug(model_id)
    with open(os.path.join(d, f"{slug}.jsonl"), "w") as fh:
        for r in sorted(records, key=lambda x: x["idx"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, f"{slug}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(records, key=lambda x: x["idx"]):
            w.writerow({k: r.get(k, "") for k in FIELDS})
    n = len(records)
    scored = sum(r["score"] for r in records)
    missing = sum(0 if r["has_marker"] else 1 for r in records)
    blank = sum(1 for r in records if not str(r["model_response"]).strip())
    with open(os.path.join(d, f"{slug}_overall.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "subject", "n", "score_sum", "average_score", "no_marker",
                    "empty_response", "scorer", "config"])
        w.writerow([model_id, subject, n, scored, round(scored / n, 4) if n else "", missing,
                    blank, SCORER_VERSION, config_string(config)])
    return {"model": model_id, "subject": subject, "n": n,
            "average_score": round(scored / n, 4) if n else "",
            "no_marker": missing, "empty_response": blank}


def write_overall(model_dir, model_id, summaries):
    path = os.path.join(model_dir, "results", f"{model_slug(model_id)}_mmlu_overall.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "subject", "n", "average_score", "no_marker", "empty_response",
                    "scorer"])
        for s in summaries:
            w.writerow([s["model"], s["subject"], s["n"], s["average_score"], s["no_marker"],
                        s["empty_response"], SCORER_VERSION])
        means = [s["average_score"] for s in summaries if s["average_score"] != ""]
        if means:
            # macro-average over subjects, which is what the workbook reports. Subject sizes range
            # from 100 to 895, so this is NOT the same as pooling all 3,943 items.
            w.writerow([model_id, "MACRO_AVG_over_%d_subjects" % len(means), "",
                        round(sum(means) / len(means), 4), "", "", SCORER_VERSION])
    return path


# ---------------------------------------------------------------- the loop


def retry(fn, tries=5, base_sleep=2.0, label=""):
    """Retries on exception AND on an empty string, returning "" when every attempt fails — never
    None. An empty completion at HTTP 200 is a real provider failure mode on these routes."""
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


def load_checkpoint(out_path, config, verbose=True):
    """Rows already done, or a refusal if the config changed.

    A config change means **archive, not resume** — resuming across one puts two configurations in
    one result set and no column can tell them apart. And an empty row is not a done row: a plain
    resume would skip it forever.
    """
    if not os.path.exists(out_path):
        return {}
    want = {k: str(v) for k, v in (config or {}).items()}
    done, empty = {}, 0
    for line in open(out_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        have = parse_config(r.get("config", ""))
        diff = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v}
        if diff:
            raise SystemExit(
                f"{out_path}\n  refusing to resume: stored rows have a different config.\n"
                f"  differences (wanted, stored): {diff}\n"
                f"  Archive the results directory and start clean.")
        if str(r.get("model_response", "")).strip():
            done[r["idx"]] = r
        else:
            empty += 1
    if verbose and (done or empty):
        print(f"  resume: {len(done)} done, {empty} empty row(s) will be retried", flush=True)
    return done


def run_subjects(model_dir, model_id, call, subjects=None, sleep_between=0.0, verbose=True,
                 limit=0, config=None, per_row_config=None, save_every=20, resume=True,
                 workers=1, prompt_version="v2"):
    subjects = subjects or SUBJECTS
    if prompt_version not in PROMPTS:
        raise SystemExit(f"unknown prompt version {prompt_version!r}; known: {sorted(PROMPTS)}")
    config = dict(config or {}, prompt=prompt_version)
    summaries = []
    for subject in subjects:
        examples = load_subject(subject)
        if limit:
            examples = examples[:limit]
            if verbose:
                print(f"[{subject}] --limit {limit}: smoke test, not a run", flush=True)
        out_path = os.path.join(subject_dir(model_dir, subject),
                                f"{model_slug(model_id)}.jsonl")
        done = load_checkpoint(out_path, config, verbose) if (resume and not limit) else {}
        records = list(done.values())
        lock = threading.Lock()

        def flush(recs):
            with open(out_path, "w") as fh:
                for r in sorted(recs, key=lambda x: x["idx"]):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        def one(i, ex):
            gold_text = ex["choices"][int(ex["answer"])]
            gold_letter = LABELS[int(ex["answer"])]
            try:
                resp = call(build_prompt(ex, prompt_version))
            except Exception as e:
                if verbose:
                    print(f"[{subject}#{i}] call failed: {e}", flush=True)
                resp = ""
            resp = resp if isinstance(resp, str) else ""
            row_cfg = dict(config or {})
            if per_row_config:
                row_cfg.update(per_row_config() or {})
            return {"idx": i, "subject": ex["subject"], "question": ex["question"],
                    "choices": ex["choices"], "gold_answer": gold_text,
                    "gold_letter": gold_letter, "answer_index": int(ex["answer"]),
                    "model_response": resp, "final_answer": extract_final_answer(resp),
                    "has_marker": has_marker(resp),
                    "score": score_response(resp, gold_text, gold_letter, int(ex["answer"])),
                    "config": config_string(row_cfg)}

        todo = [(i, ex) for i, ex in enumerate(examples) if i not in done]
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(one, i, ex) for i, ex in todo]
                for k, f in enumerate(concurrent.futures.as_completed(futs), 1):
                    with lock:
                        records.append(f.result())
                        if save_every and k % save_every == 0:
                            flush(records)
                            if verbose:
                                print(f"  [{subject}] {len(records)}/{len(examples)}", flush=True)
        else:
            for i, ex in todo:
                records.append(one(i, ex))
                if sleep_between:
                    time.sleep(sleep_between)
                if save_every and len(records) % save_every == 0:
                    flush(records)
                    if verbose:
                        print(f"  [{subject}] {len(records)}/{len(examples)}", flush=True)
        records.sort(key=lambda r: r["idx"])
        s = write_subject_results(model_dir, subject, model_id, records, config=config)
        summaries.append(s)
        if verbose:
            print(f"{subject}: {s['average_score']}  (n={s['n']}, no_marker={s['no_marker']}, "
                  f"empty={s['empty_response']})", flush=True)
        write_overall(model_dir, model_id, summaries)
    return summaries


# ---------------------------------------------------------------- capability negotiation

AUTH_MARKERS = ("authentication", "api key", "invalid_api_key", "insufficient_quota", "billing",
                "401", "403")
EFFORT_FALLBACKS = ("minimal", "low", "medium", "high")


def negotiate(client, model, wanted, verbose=True):
    """Find which of `wanted` the model actually accepts, in one probe call. A refused VALUE is not
    a refused PARAMETER — conflating them drops a cap instead of correcting it."""
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
                        print(f"  negotiate: {offender} refused {asked!r}; using {pick!r}",
                              flush=True)
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
