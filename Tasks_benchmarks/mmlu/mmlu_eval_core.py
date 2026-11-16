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

**The 10% sample rerun** (`.claude/references/sample10-plan.md`, WP3) is the last section of this
file, used only by a run given `--sample`. Nothing above it changed, so a run without `--sample`
writes the files, prompts and rows it always did — `test_sample10.py` proves it against the commit
before the change.
"""

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import signal
import socket
import sys
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


def write_subject_results(model_dir, subject, model_id, records, config=None, tag=""):
    d = subject_dir(model_dir, subject)
    # every artefact carries the shard tag, or shard 1 overwrites shard 0's summary
    slug = model_slug(model_id) + tag
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


def write_overall(model_dir, model_id, summaries, tag=""):
    path = os.path.join(model_dir, "results", f"{model_slug(model_id)}{tag}_mmlu_overall.csv")
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
                 workers=1, prompt_version="v2", shard=0, total_shards=1):
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
        tag = shard_tag(shard, total_shards)
        out_path = os.path.join(subject_dir(model_dir, subject),
                                f"{model_slug(model_id)}{tag}.jsonl")
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

        # slice AFTER enumerate so idx indexes the full subject, not the slice
        todo = [(i, ex) for i, ex in shard_slice(list(enumerate(examples)), shard, total_shards)
                if i not in done]
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
        s = write_subject_results(model_dir, subject, model_id, records, config=config, tag=tag)
        summaries.append(s)
        if verbose:
            print(f"{subject}: {s['average_score']}  (n={s['n']}, no_marker={s['no_marker']}, "
                  f"empty={s['empty_response']})", flush=True)
        write_overall(model_dir, model_id, summaries, tag=tag)
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


# ==================================================================== the 10% sample rerun
#
# Everything below is used ONLY by a run given `--sample` (`.claude/references/sample10-plan.md`,
# WP3). It defines new names and rebinds none above, so the legacy path — run_subjects, retry,
# load_checkpoint, write_subject_results, write_overall, negotiate — is byte-identical to the commit
# before it and a run without `--sample` behaves exactly as before (`test_sample10.py`).
#
# What a sample run is: the items listed in `sample10.json` (strata = subject files, ids = index into
# the subject, so `idx` still indexes the whole subject); `PROMPT_V2` for every model; output in
# `<model folder>/results_sample10/` — a `--limit` smoke in `smoke_sample10/` — and never `results/`,
# so neither can resume from or write into the other; and every row carries the per-row usage fields
# of `evaluation-criteria.md`. The harness — call_with_retries, openai_meta, halt_on_billing,
# negotiate_pinned, CutRetry — is ported from `LLMs-Planning_bench/plan_eval_core.py`, where each
# piece is explained by the incident that added it. Cores travel to Quest one benchmark at a time,
# so it is copied here, never imported across benchmarks.

SAMPLES = {"sample10": os.path.join(MMLU_ROOT, "sample10.json")}
# One prompt for all five models in the sample — the user, 2026-11-15 (D1 = B). Deepseek leaves v1.
SAMPLE_PROMPT = "v2"


def load_sample(name):
    """({subject: ids}, sha1 of the file) — checked against the data first. A sample whose counts or
    ids disagree with the subject files would scale the wrong items, silently."""
    if name not in SAMPLES:
        raise SystemExit(f"unknown sample {name!r}; known: {sorted(SAMPLES)}")
    with open(SAMPLES[name], "rb") as fh:
        raw = fh.read()
    spec = json.loads(raw)
    strata = spec.get("strata") or {}
    problems = []
    if spec.get("benchmark") != "mmlu":
        problems.append(f"benchmark is {spec.get('benchmark')!r}, not 'mmlu'")
    if set(strata) != set(SUBJECTS):
        problems.append(f"strata {sorted(set(strata) ^ set(SUBJECTS))} differ from SUBJECTS")
    total = 0
    for subject in SUBJECTS:
        st = strata.get(subject)
        if not st:
            continue
        n_items, ids = len(load_subject(subject)), list(st.get("ids") or [])
        total += len(ids)
        if st.get("N") != n_items:
            problems.append(f"{subject}: N={st.get('N')} but the data has {n_items}")
        if len(ids) != st.get("n") or len(set(ids)) != len(ids) or ids != sorted(ids):
            problems.append(f"{subject}: ids are not n={st.get('n')} unique sorted values")
        if any(not isinstance(i, int) or not 0 <= i < n_items for i in ids):
            problems.append(f"{subject}: an id falls outside 0..{n_items - 1}")
    if total != spec.get("total_n"):
        problems.append(f"{total} ids but total_n={spec.get('total_n')}")
    if problems:
        raise SystemExit(f"{SAMPLES[name]} does not match the data:\n  " + "\n  ".join(problems))
    return {s: list(strata[s]["ids"]) for s in SUBJECTS}, hashlib.sha1(raw).hexdigest()[:12]


def build_messages(example):
    """The messages every model is sent: PROMPT_V2 as the one user turn, nothing else — the shape the
    legacy `call(prompt)` sends. A model's own addition (a prompt ceiling) is a system message its
    runner adds through CutRetry, never a change here."""
    return [{"role": "user", "content": build_prompt(example, SAMPLE_PROMPT)}]


def named_choices(model_response, choices):
    """The choices the response names, found by asking the one scorer about each option in turn — so
    `parse_ok` cannot disagree with `score_response` about what the response said. A response the
    scorer would mark right always names at least one choice."""
    return [k for k in range(len(choices))
            if score_response(model_response, choices[k], LABELS[k], k)]


# ---------------------------------------------------------------- refusals that must stop a run
# plan_eval_core.py, unchanged in content; only wordings that unambiguously mean money.

BILLING_SIGNATURES = (
    "insufficient_quota", "insufficient balance", "insufficient credits", "out of credits",
    "monthly budget", "used all available credits", "spending limit", "credit balance",
    "payment required", "arrears",
    # OpenRouter's out-of-credit 402 (MMLU's own Gemini run died on it, 2026-08-30)
    "requires more credits", "can only afford", "openrouter_credits",
    # DeepInfra's out-of-balance 402 (PlanBench Qwen, 2026-11-15)
    "positive balance",
)
# A backend that cut the answer off: a failed attempt, never an answer. "length" is not here — that
# is our own cap, and it is measured, not retried.
INTERRUPTED_FINISH = ("error", "insufficient_system_resource")
DAILY_QUOTA_SIGNATURES = ("per_model_per_day", "requests per day", "requests_per_day", "per day",
                          "daily limit", "daily quota")
DAILY_HALT_THRESHOLD_SECONDS = 600
FATAL_SIGNATURES = ("invalid_argument", "extra inputs are not permitted", "validationerror",
                    "unsupported parameter", "unsupported value", "unrecognized request argument",
                    "unexpected keyword argument")
# Not AUTH_MARKERS: that name belongs to the legacy negotiate above, whose list differs.
SAMPLE_AUTH_MARKERS = ("authentication", "api key", "invalid_api_key", "incorrect api key", "401")

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
    if not mentions(error, DAILY_QUOTA_SIGNATURES):
        return False
    wait = retry_after_seconds(error)
    return True if wait is None else wait >= DAILY_HALT_THRESHOLD_SECONDS


def mentions(text, markers):
    """Whether `text` contains any marker. A numeric marker (a status code) must stand as a word —
    `\b401\b` — or "401" inside a hex request id of a 500 error would read as an auth failure."""
    text = str(text).lower()
    return any(re.search(r"\b" + re.escape(m) + r"\b", text) if m.isdigit() else m in text
               for m in markers)


def is_billing_failure(error):
    return mentions(error, BILLING_SIGNATURES)


def is_fatal_request(error):
    text = f"{type(error).__name__}: {error}"
    return mentions(text, FATAL_SIGNATURES) or mentions(text, SAMPLE_AUTH_MARKERS)


STATS = {"ok": 0, "empty": 0, "interrupted": 0, "errors": 0, "timeouts": 0, "failed_rows": 0}
# Sample calls run serially in the main thread; the lock stays so nothing here assumes it.
_STATS_LOCK = threading.Lock()
HALT_MARKERS = ("BILLING_HALT.txt", "QUOTA_HALT.txt", "FAILURE_HALT.txt", "THINKING_HALT.txt")
# Set by run_sample from marker_prefix(); markers are cleared by this exact prefix only.
MARKER_PREFIX = ""


def marker_prefix(sample, smoke, shard, total_shards):
    """`sample10_` / `sample10_shard2_` for a real run, `smoke_sample10_` for a --limit smoke. Each
    run clears only markers with exactly its own prefix, so a smoke never clears a real run's marker,
    nor a late shard another shard's; `ls MMLU_*/*_HALT.txt` still finds every one."""
    return (("smoke_" if smoke else "") + f"{sample}_"
            + (f"shard{shard}_" if total_shards > 1 else ""))


def _count(key):
    with _STATS_LOCK:
        STATS[key] += 1


def _halt(kind, marker_name, detail, model, model_dir):
    """Write a marker an agent can find without reading logs, say why on both streams, exit."""
    marker = os.path.join(model_dir, MARKER_PREFIX + marker_name)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    advice = ("Rows written failed (empty) are retried on resume — load_sample_checkpoint skips only "
              "answered rows — so a plain resubmit is safe once the cause is fixed.")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason={kind}\n{detail}\n\n{advice}\n")
    except OSError as error:
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! {kind} HALT [{model}] {stamp}\n{detail}\n", file=sys.stderr, flush=True)
    raise SystemExit(f"aborting: {model} stopped on {kind.lower()}. {advice}")


def halt_on_thinking(row, model, model_dir):
    """A model run with thinking off (Qwen, Gemma) answered with reasoning: the config is already
    known to be wrong, so the shard stops at this row — marker, then exit 3. The rows so far are
    saved by run_sample_subjects' `finally`; this row is among them, as evidence."""
    marker = os.path.join(model_dir, MARKER_PREFIX + "THINKING_HALT.txt")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    detail = (f"{row.get('subject')}#{row.get('idx')}: reasoning_chars={row.get('reasoning_chars')}, "
              f"reasoning_tokens={row.get('reasoning_tokens')}, "
              f"billed_reasoning_tokens={row.get('billed_reasoning_tokens')}")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason=THINKING CAME BACK ON\n{detail}\n\n"
                     f"The model is run with thinking off and answered with reasoning: the request "
                     f"or the provider's default changed. Fix the config; the rows already written "
                     f"ran under it and must be archived, not resumed.\n")
    except OSError as error:
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! THINKING CAME BACK ON — HALT [{model}] {stamp}\n{detail}\n", file=sys.stderr,
          flush=True)
    print(f"STOP: THINKING CAME BACK ON at {detail}", flush=True)
    raise SystemExit(3)


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
        run, window = _consecutive[0], list(_recent)
    if run >= CONSECUTIVE_FAILURE_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt", f"{run} consecutive rows failed every attempt",
              model, model_dir)
    if len(window) == FAILURE_WINDOW and window.count(False) / FAILURE_WINDOW >= FAILURE_WINDOW_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt",
              f"{window.count(False)} of the last {FAILURE_WINDOW} rows failed", model, model_dir)


# ---------------------------------------------------------------- one call, guarded


class CallTimeout(BaseException):
    """BaseException on purpose: an `except Exception` anywhere below would otherwise swallow the
    watchdog, and the rest of that call would run unprotected (provider-gotchas.md)."""


# Hard ceiling on one HTTP attempt, by SIGALRM; set per runner through run_sample(call_timeout=).
HARD_CALL_TIMEOUT = 300


def _raise_timeout(signum, frame):
    raise CallTimeout(f"call exceeded {HARD_CALL_TIMEOUT}s hard limit")


def _require_watchdog():
    """The watchdog is SIGALRM, which can only be installed from the main thread and only interrupts
    it. So every sample call runs serially in the main thread — run_sample refuses --workers > 1 —
    and a call from anywhere else raises here rather than running unguarded. The client's own
    `timeout=` is no substitute: a server that keeps the connection alive defeats the SDK's read
    timeout (test_sample10.py, against a trickling mock server)."""
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("no SIGALRM on this platform: a sample call would run without its watchdog")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("a sample call must run in the main thread: the SIGALRM watchdog cannot "
                           "interrupt any other")
    if not HARD_CALL_TIMEOUT:
        raise RuntimeError("HARD_CALL_TIMEOUT is unset: a sample call would run without its watchdog")


def retry_delay(error, attempt, default=5.0):
    """Provider's hint as a floor, exponential growth, jitter so shards do not retry in lockstep,
    capped at a quarter of the watchdog (plan_eval_core.retry_delay)."""
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


def _new_ledger():
    return {"calls": 0, "with_usage": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": None, "cost": None, "estimated_cost": None, "reasoning_chars": 0,
            "think_tag_calls": 0, "attempts": 0}


# The ledger of the row being called, tracking the call boundary: `create` opens a call for every
# HTTP request (every attempt, every CutRetry stage); `openai_meta` closes it when it records the
# response — its usage, its cost, its reasoning text. A call without usage is one still open (it
# timed out or raised) or one closed without a usage object: `calls - with_usage`. No handler counts
# anything, so a failure is never counted twice — not an error that a halt then re-raises from inside
# its handler, nor a response whose usage was booked before the runner raised on empty `choices`.
# The row's bill is therefore counted once per call: failed and empty attempts and every CutRetry
# stage included, and a stage CutRetry keeps across attempts never twice.
_LEDGER = threading.local()


def create(client, **kwargs):
    """Every sample HTTP call goes through here, so the ledger knows how many calls the row made —
    including the ones that never returned (timeouts, exceptions) and so never reached openai_meta."""
    ledger = getattr(_LEDGER, "current", None)
    if ledger is not None:
        ledger["calls"] += 1
    return client.chat.completions.create(**kwargs)


def _add(a, b):
    """None + None = None (nothing reported); otherwise a sum."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def _with_billing(meta, ledger):
    return dict(meta, billed_prompt_tokens=ledger["prompt_tokens"],
                billed_completion_tokens=ledger["completion_tokens"],
                billed_reasoning_tokens=ledger["reasoning_tokens"],
                billed_cost=ledger["cost"], billed_estimated_cost=ledger["estimated_cost"],
                attempts_without_usage=ledger["calls"] - ledger["with_usage"],
                reasoning_chars=ledger["reasoning_chars"],
                think_tag_calls=ledger["think_tag_calls"])


def halted_meta(stop):
    """The meta of a row whose call_with_retries was stopped by `stop` (SIGTERM's SystemExit, a
    halt): everything its calls were billed, from the ledger call_with_retries keeps on exit — or
    None when no call had started (nothing billed, nothing to write)."""
    ledger = getattr(_LEDGER, "last", None)
    if ledger is None or not ledger["attempts"]:
        return None
    return _with_billing({"halted": f"{type(stop).__name__}: {stop}"[:300], "latency_s": None,
                          "n_attempts": ledger["attempts"]}, ledger)


def call_with_retries(once, messages, model, model_dir, tries=5):
    """`once(messages) -> (text, meta)`. Retries an exception, a timeout AND an empty string — HTTP
    200 with an empty body raises nothing. Returns ("", meta) when every attempt failed; the row is
    then written as failed, and retried on resume. A SystemExit raised inside (SIGTERM, a halt)
    propagates; the ledger stays readable through `halted_meta` so the row is still written.

    `latency_s` is the answering attempt alone (back-off excluded) and `n_attempts` the attempts
    spent on the row; both are true only because every sample client is built with max_retries=0
    (script-skeleton.md §4). A CutRetry row makes more HTTP calls than attempts: the true count is
    `n_attempts + cut_retry + forced_answer`. `prompt_tokens`, `completion_tokens`,
    `reasoning_tokens` are the answering attempt's; `billed_*` sum every HTTP call of the row that
    returned usage — an empty attempt that burned 16,384 reasoning tokens is billed, and only
    `billed_*` shows it — `billed_cost` and `billed_estimated_cost` likewise (None when no call
    reported one), and `attempts_without_usage` counts the HTTP calls that returned no usage
    (timeouts, exceptions, a response without a usage object), CutRetry stages included.
    `reasoning_chars` is the reasoning text of every call of the row, summed."""
    _LEDGER.last = None
    _require_watchdog()
    ledger = _LEDGER.current = _new_ledger()
    meta = {}
    try:
        for attempt in range(tries):
            started = time.time()
            wait = 5.0
            ledger["attempts"] = attempt + 1
            previous = signal.signal(signal.SIGALRM, _raise_timeout)
            try:
                signal.alarm(HARD_CALL_TIMEOUT)
                try:
                    text, meta = once(messages)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
                meta = dict(meta or {}, latency_s=round(time.time() - started, 2),
                            n_attempts=attempt + 1)
                text = text.strip() if isinstance(text, str) else ""
                if text and meta.get("finish_reason") in INTERRUPTED_FINISH:
                    # a fragment on HTTP 200: non-empty, so it would pass every check below
                    _count("interrupted")
                    print(f"[{model}] INTERRUPTED ({attempt + 1}/{tries}) "
                          f"finish_reason={meta.get('finish_reason')}: backend cut the answer off",
                          flush=True)
                    text = ""
                elif text:
                    _count("ok")
                    return text, _with_billing(meta, ledger)
                _count("empty")
                print(f"[{model}] empty response ({attempt + 1}/{tries}) "
                      f"finish_reason={meta.get('finish_reason')} "
                      f"completion_tokens={meta.get('completion_tokens')}", flush=True)
            except CallTimeout as error:
                _count("timeouts")
                print(f"[{model}] TIMEOUT ({attempt + 1}/{tries}): {error}", flush=True)
            except Exception as error:
                _count("errors")
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
        print(f"[{model}] all {tries} attempts failed; row written as failed, retried on resume",
              flush=True)
        return "", _with_billing(dict(meta or {}, latency_s=None, n_attempts=tries), ledger)
    finally:
        # Kept, not dropped: if a SystemExit (SIGTERM, a billing/quota/malformed halt) stopped this
        # row, run_sample_subjects writes it as a failed, `halted` row from this ledger — its calls
        # were billed. note_outcome is the caller's, after the row is appended, for the same reason.
        _LEDGER.last = ledger
        _LEDGER.current = None


THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def openai_meta(response):
    """What an OpenAI-compatible response says about itself — written onto every row, because a
    truncation, a reasoning bill or a backend switch is invisible in the text. A count the provider
    did not report is None (an empty CSV cell), never 0."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    choice = response.choices[0] if getattr(response, "choices", None) else None
    finish = getattr(choice, "finish_reason", None)
    finish = getattr(finish, "value", finish)

    def as_int(value):
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    meta = {
        "finish_reason": None if finish is None else str(finish),
        "prompt_tokens": as_int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": as_int(getattr(usage, "completion_tokens", None)),
        "reasoning_tokens": as_int(getattr(details, "reasoning_tokens", None)),
        "served_model": str(getattr(response, "model", "") or "") or None,
    }
    provider = getattr(response, "provider", None)       # OpenRouter's answering backend
    if provider:
        meta["backend"] = str(provider)
    # this call's own bill — the row's whole bill is `billed_cost` / `billed_estimated_cost`
    cost = getattr(usage, "cost", None)                  # OpenRouter's
    cost = cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
    if cost is not None:
        meta["last_call_cost"] = cost
    estimated = getattr(usage, "estimated_cost", None)   # DeepInfra's
    estimated = (estimated if isinstance(estimated, (int, float)) and not isinstance(estimated, bool)
                 else None)
    if estimated is not None:
        meta["last_call_estimated_cost"] = estimated
    # reasoning text that came back with the reply (DeepInfra and DeepSeek: reasoning_content;
    # OpenRouter: reasoning). On a model run with thinking off any at all means it came back on.
    message = getattr(choice, "message", None)
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    meta["reasoning_chars"] = len(reasoning) if isinstance(reasoning, str) else 0
    # Thinking can also come back inline, as <think>…</think> in the content itself
    content = getattr(message, "content", None)
    meta["think_tag"] = bool(isinstance(content, str) and THINK_TAG.search(content))
    ledger = getattr(_LEDGER, "current", None)
    if ledger is not None:
        ledger["reasoning_chars"] += meta["reasoning_chars"]
        ledger["think_tag_calls"] += int(meta["think_tag"])
        if usage is not None:
            ledger["with_usage"] += 1
            for key in ("prompt_tokens", "completion_tokens"):
                if meta[key] is not None:
                    ledger[key] += meta[key]
            ledger["reasoning_tokens"] = _add(ledger["reasoning_tokens"], meta["reasoning_tokens"])
            ledger["cost"] = _add(ledger["cost"], cost)
            ledger["estimated_cost"] = _add(ledger["estimated_cost"], estimated)
    return meta


# ---------------------------------------------------------------- parameter negotiation
# plan_eval_core.negotiate: `fixed` is sent on every call and never dropped; `required` stops the run
# if negotiation loses a parameter OR swaps its value (model-parameters.md rules 1 and 7). Named
# apart from the legacy `negotiate` above, which the full-run path still uses.

_TRANSIENT = ("rate limit", "429", "500", "502", "503", "504", "timed out", "timeout",
              "connection", "overloaded", "unavailable")


def negotiate_pinned(client, model, wanted, model_dir, fixed=None, required=None):
    _require_watchdog()
    params, notes = dict(wanted), []
    fixed = dict(fixed or {})
    transient_left = 3
    for _ in range(len(wanted) + 3 + transient_left):
        previous = signal.signal(signal.SIGALRM, _raise_timeout)
        try:
            signal.alarm(HARD_CALL_TIMEOUT)
            try:
                client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "Reply with: ok"}],
                    **fixed, **params)
            finally:
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
            if mentions(err, SAMPLE_AUTH_MARKERS):
                raise SystemExit(f"authentication failure, not negotiated around: {err}")
            offender = next((k for k in params if f"'{k}'" in err or f'"{k}"' in err
                             or f"`{k}`" in err), None)
            if offender is None and transient_left > 0 and mentions(err, _TRANSIENT):
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
                         f"that removes or changes a setting this run is decided on "
                         f"(model-parameters.md rules 1 and 7). Fix the request instead.")
    return params, notes


# ---------------------------------------------------------------- a reply cut at the output cap

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")


def rearm_call_timeout():
    """Give the next HTTP call inside one `once` its own HARD_CALL_TIMEOUT — only while
    call_with_retries' handler is installed (always the main thread: _require_watchdog)."""
    if (threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGALRM) is _raise_timeout):
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


def is_cut(row):
    """A stored row the output cap cut off that no forced answer replaced."""
    return row.get("finish_reason") == "length" and not row.get("forced_answer")


class CutRetry:
    """plan_eval_core.CutRetry — the user's rule of 2026-11-13 for a reply that hits the output cap
    ("让他第二次重新跑的时候，不要超出这个思考的上限…如果说他就是不受约束，那就直接回答"):

    1. **The call as configured**, the runner's prompt ceiling as a system message. A reply that
       ends any way but `length` is the row's answer.
    2. **Cut at the cap: called again, fresh**, `retry_ceiling` in place of the ceiling. The row
       carries `cut_retry: true` and `first_*` (with `first_response`, the cut call's text in full).
    3. **Cut again: the forced answer.** What the second attempt wrote goes back as an assistant
       turn, then `force_prompt`; the runner makes that call with its small answer cap. The row
       carries `forced_answer: true`, `retry_*` and `forced_*`.

    One difference from PlanBench's, because of where it runs: **the ceiling is PREPENDED as the
    first message** — MMLU sends no shared system message, so PlanBench's "after the first message"
    would put it after the question. The per-row state is kept by the identity of the messages list
    run_sample_subjects builds once per row, never by text; it is also thread-local, which costs
    nothing while sample calls run serially in the main thread (_require_watchdog).

    `call(sent, stage) -> (answer, handback, meta)`, stage "first", "retry" or "forced". Usage on the
    row is every call of the attempt summed, so `completion_tokens` can pass the cap. The row's
    `reasoning_chars` is not taken from a stage: call_with_retries sets it from the ledger, summed
    over every call of the row; `forced_reasoning_chars` is the forced call's alone."""

    def __init__(self, ceiling, retry_ceiling, force_prompt, answer_max):
        self.ceiling, self.retry_ceiling = ceiling, retry_ceiling
        self.force_prompt, self.answer_max = force_prompt, answer_max
        self._local = threading.local()

    @staticmethod
    def with_system(messages, text):
        """The ceiling first, then the shared messages untouched."""
        return [{"role": "system", "content": text}] + list(messages)

    def config(self, **forced_extra):
        """The keys this adds to the runner's extra_config — on every row's `config`."""
        return {"prompt_ceiling": self.ceiling,
                "cut_retry": {"prompt": self.retry_ceiling, "when": "the first call ends at length"},
                "forced_answer": dict({"prompt": self.force_prompt, "max_tokens": self.answer_max,
                                       "when": "the second call also ends at length, with text to "
                                               "hand back"}, **forced_extra)}

    def __call__(self, messages, call, model=""):
        kept = getattr(self._local, "kept", None)
        if kept is None:
            kept = self._local.kept = {}
        if kept.get("messages") is not messages:
            kept.clear()
            kept["messages"] = messages
        if "first" not in kept:
            started = time.time()
            answer, handback, meta = call(self.with_system(messages, self.ceiling), "first")
            meta.pop("reasoning_chars", None)
            if meta.get("finish_reason") != "length":
                return answer, meta
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
            answer, handback, meta = call(retry_sent, "retry")
            meta.pop("reasoning_chars", None)
            retry_latency = round(time.time() - started, 2)
            if meta.get("finish_reason") != "length" or not handback.strip():
                out = dict(meta, **_usage_sum([first_meta, meta]), **marks,
                           retry_latency_s=retry_latency)
                print(f"[{model}] retry: {meta.get('completion_tokens')} tokens, "
                      f"finish={meta.get('finish_reason')}", flush=True)
                return answer, out
            kept["retry"] = (answer, handback, meta, retry_latency)
            print(f"[{model}] retry reached the cap too ({meta.get('completion_tokens')} tokens): "
                  f"forced answer", flush=True)
            rearm_call_timeout()
        retry_text, retry_handback, retry_meta, retry_latency = kept["retry"]
        started = time.time()
        answer, _, meta = call(retry_sent + [{"role": "assistant", "content": retry_handback},
                                             {"role": "user", "content": self.force_prompt}],
                               "forced")
        reasoning_chars = meta.pop("reasoning_chars", 0) or 0
        out = dict(meta, **_usage_sum([first_meta, retry_meta, meta]), **marks)
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


# ---------------------------------------------------------------- rows, checkpoint, files

# The per-row usage fields (evaluation-criteria.md § Per-row fields) sit beside the legacy columns.
SAMPLE_FIELDS = ["idx", "subject", "question", "choices", "gold_answer", "gold_letter",
                 "answer_index", "model_response", "final_answer", "has_marker", "pred_letter",
                 "parse_ok", "score", "failed", "prompt_tokens", "completion_tokens",
                 "reasoning_tokens", "billed_prompt_tokens", "billed_completion_tokens",
                 "billed_reasoning_tokens", "billed_cost", "billed_estimated_cost",
                 "attempts_without_usage", "latency_s", "finish_reason", "n_attempts",
                 "prior_n_attempts", "prior_failed_runs", "resumed_after_failure", "served_model",
                 "backend", "last_call_cost", "last_call_estimated_cost", "reasoning_chars",
                 "think_tag_calls", "halted", "cut_retry", "forced_answer", "sample", "ts",
                 "config"]

# Summed over every run of a row when a resubmit replaces a failed or cut row (fold_history).
CUMULATIVE_KEYS = ("billed_prompt_tokens", "billed_completion_tokens", "billed_reasoning_tokens",
                   "attempts_without_usage", "billed_cost", "billed_estimated_cost")


def sample_out_dir(model_dir, sample, smoke=False):
    """`results_sample10/`, or `smoke_sample10/` for a --limit run — never `results/`: a shared
    checkpoint would make resume skip or mix rows (sample10-plan.md § Rules)."""
    return os.path.join(model_dir, ("smoke_" if smoke else "results_") + sample)


def _atomic_write(path, write):
    # host AND pid: array tasks on different nodes share this GPFS directory
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        write(fh)
    os.replace(tmp, path)


def _save_jsonl(path, rows):
    def write(fh):
        for r in sorted(rows, key=lambda x: x["idx"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _atomic_write(path, write)


def sample_row(idx, example, text, meta, config, sample):
    """One row. A failed call is a row too — `failed: true`, empty response, `parse_ok: false`,
    scored 0 and counted — never dropped, so it stays in the reliability rate."""
    gold_index = int(example["answer"])
    gold_text, gold_letter = example["choices"][gold_index], LABELS[gold_index]
    named = named_choices(text, example["choices"])
    row = {"idx": idx, "subject": example["subject"], "question": example["question"],
           "choices": example["choices"], "gold_answer": gold_text, "gold_letter": gold_letter,
           "answer_index": gold_index, "model_response": text,
           "final_answer": extract_final_answer(text), "has_marker": has_marker(text),
           # the one choice the response names; "" with parse_ok true means it named two, which
           # happens only when one option's text is another's index
           "pred_letter": LABELS[named[0]] if len(named) == 1 else "",
           "parse_ok": bool(named),
           "score": score_response(text, gold_text, gold_letter, gold_index),
           "failed": not text}
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens") + CUMULATIVE_KEYS + (
            "latency_s", "finish_reason", "n_attempts", "served_model", "reasoning_chars",
            "think_tag_calls", "halted"):
        row[key] = meta.get(key)
    # this row's first run; fold_history fills these when a resubmit replaces an earlier run
    row.update(prior_n_attempts=0, prior_failed_runs=0, resumed_after_failure=False)
    for key, value in meta.items():          # backend, last_call_*, CutRetry's marks, fallback
        if key != "think_tag":               # the last call's; think_tag_calls is the row's
            row.setdefault(key, value)
    row.update(sample=sample, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), config=config)
    return row


def fold_history(new, old):
    """`new` replaces `old`, a stored row that failed or stayed cut at the cap. Without this a
    resubmit would erase what the earlier runs cost: a row that failed 5 times (81,920 tokens billed)
    and answered on resubmit would say 1 attempt and 50 tokens. The bill becomes cumulative over
    every run; `n_attempts` stays this run's, and `prior_n_attempts` / `prior_failed_runs` hold the
    earlier runs'."""
    new = dict(new)
    for key in CUMULATIVE_KEYS:
        new[key] = _add(new.get(key), old.get(key))
    new["prior_n_attempts"] = (old.get("prior_n_attempts") or 0) + (old.get("n_attempts") or 0)
    new["prior_failed_runs"] = (old.get("prior_failed_runs") or 0) + 1
    new["resumed_after_failure"] = True
    return new


def load_sample_checkpoint(path, config, cut_retry=False):
    """(done, held) by idx — or a refusal if the stored config differs: resuming across a change
    puts two configurations in one result set. `held` are the rows to call again: a failed row, and
    under a CutRetry config a row still cut at the cap. They stay in the file until a new row
    replaces them — a resubmit killed mid-row leaves them as they were — and the new row carries
    their history (fold_history)."""
    if not os.path.exists(path):
        return {}, {}
    done, held, failed = {}, {}, {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("config") != config:
                stored = r.get("config") or {}
                stored = stored if isinstance(stored, dict) else {"<legacy string>": stored}
                diff = {k: (config.get(k), stored.get(k)) for k in set(config) | set(stored)
                        if config.get(k) != stored.get(k)}
                raise SystemExit(f"{path}\n  refusing to resume: stored rows ran under a different "
                                 f"config.\n  (wanted, stored): {diff}\n  Archive the directory "
                                 f"to a timestamped name and start clean.")
            if not str(r.get("model_response", "")).strip():
                failed[r["idx"]] = r
            elif cut_retry and is_cut(r):
                held[r["idx"]] = r
            else:
                done[r["idx"]] = r
    if done or held or failed:
        print(f"  resume {os.path.basename(path)}: {len(done)} done, {len(failed)} failed row(s) to "
              f"retry" + (f", {len(held)} cut at the cap to call again" if held else ""), flush=True)
    return done, {**failed, **held}


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _sum_ints(rows, key):
    parts = [r.get(key) for r in rows if isinstance(r.get(key), int)]
    return sum(parts) if parts else ""


def _sum_nums(rows, key):
    parts = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))
             and not isinstance(r.get(key), bool)]
    return round(sum(parts), 8) if parts else ""


def thinking_signal(row):
    """Reasoning came back with the reply: text (`reasoning_chars`), a reasoning token count, or
    inline `<think>` tags in the content (`think_tag_calls`), on any call of the row. On a model run with thinking off (Qwen, Gemma) any at all means it came back
    on."""
    return ((row.get("reasoning_chars") or 0) > 0 or (row.get("reasoning_tokens") or 0) > 0
            or (row.get("billed_reasoning_tokens") or 0) > 0
            or (row.get("think_tag_calls") or 0) > 0
            or bool(THINK_TAG.search(str(row.get("model_response") or ""))))


def write_sample_subject(out_dir, subject, model_id, records, expected, tag=""):
    d = os.path.join(out_dir, subject)
    os.makedirs(d, exist_ok=True)
    slug = model_slug(model_id) + tag
    records = sorted(records, key=lambda r: r["idx"])
    _save_jsonl(os.path.join(d, f"{slug}.jsonl"), records)

    def write_rows(fh):
        w = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: _csv_value(r.get(k)) for k in SAMPLE_FIELDS})
    _atomic_write(os.path.join(d, f"{slug}.csv"), write_rows)
    n = len(records)
    scored = sum(r["score"] for r in records)
    lat = sorted(r["latency_s"] for r in records if isinstance(r.get("latency_s"), (int, float)))
    summary = {"model": model_id, "subject": subject, "n": n, "expected": expected,
               "score_sum": scored, "average_score": round(scored / n, 4) if n else "",
               "failed": sum(1 for r in records if r.get("failed")),
               "parse_fail": sum(1 for r in records if not r.get("parse_ok")),
               "no_marker": sum(1 for r in records if not r.get("has_marker")),
               "finish_length": sum(1 for r in records if r.get("finish_reason") == "length"),
               "cut_retry": sum(1 for r in records if r.get("cut_retry")),
               "forced_answer": sum(1 for r in records if r.get("forced_answer")),
               "retried_rows": sum(1 for r in records if (r.get("n_attempts") or 1) > 1),
               "prompt_tokens": _sum_ints(records, "prompt_tokens"),
               "completion_tokens": _sum_ints(records, "completion_tokens"),
               "reasoning_tokens": _sum_ints(records, "reasoning_tokens"),
               "billed_prompt_tokens": _sum_ints(records, "billed_prompt_tokens"),
               "billed_completion_tokens": _sum_ints(records, "billed_completion_tokens"),
               "billed_reasoning_tokens": _sum_ints(records, "billed_reasoning_tokens"),
               "billed_cost": _sum_nums(records, "billed_cost"),
               "billed_estimated_cost": _sum_nums(records, "billed_estimated_cost"),
               "attempts_without_usage": _sum_ints(records, "attempts_without_usage"),
               "resumed_after_failure": sum(1 for r in records if r.get("resumed_after_failure")),
               # rows whose calls carried reasoning: expected on Deepseek; on a model run with
               # thinking off (Qwen, Gemma) any at all means it came back on
               "rows_with_reasoning": sum(1 for r in records if thinking_signal(r)),
               "latency_s_median": lat[len(lat) // 2] if lat else "",
               "configs": len({json.dumps(r.get("config"), sort_keys=True) for r in records}),
               "scorer": SCORER_VERSION}

    def write_overall_row(fh):
        w = csv.DictWriter(fh, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    _atomic_write(os.path.join(d, f"{slug}_overall.csv"), write_overall_row)
    return summary


def write_sample_overall(out_dir, model_id, summaries, tag=""):
    path = os.path.join(out_dir, f"{model_slug(model_id)}{tag}_mmlu_overall.csv")
    os.makedirs(out_dir, exist_ok=True)
    fields = ["model", "subject", "n", "expected", "average_score", "failed", "parse_fail",
              "no_marker", "finish_length", "cut_retry", "forced_answer", "retried_rows",
              "prompt_tokens", "completion_tokens", "reasoning_tokens", "billed_prompt_tokens",
              "billed_completion_tokens", "billed_reasoning_tokens", "billed_cost",
              "billed_estimated_cost", "attempts_without_usage", "resumed_after_failure",
              "rows_with_reasoning", "scorer"]

    def write(fh):
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
        means = [s["average_score"] for s in summaries if s["average_score"] != ""]
        if means:
            # macro over subjects, as the workbook reports; NOT an estimate of the full run's
            # macro by itself — sample10_report.py does the scaling and the intervals
            w.writerow({"model": model_id, "subject": "MACRO_AVG_over_%d_subjects" % len(means),
                        "average_score": round(sum(means) / len(means), 4),
                        "scorer": SCORER_VERSION})
    _atomic_write(path, write)
    return path


# ---------------------------------------------------------------- the sample loop


def run_sample_subjects(model_dir, model_id, once, config, sample, ids_by_subject, subjects,
                        limit=0, shard=0, total_shards=1, save_every=20, sleep=0.0):
    """`once(messages) -> (text, meta)`, called serially in the main thread — the watchdog is
    SIGALRM (_require_watchdog). Returns every row this shard holds, across subjects."""
    if save_every < 1:
        raise SystemExit(f"--save-every must be at least 1, got {save_every}")
    smoke = bool(limit)
    out_dir = sample_out_dir(model_dir, sample, smoke)
    tag = shard_tag(shard, total_shards)
    cut_retry = "forced_answer" in config
    if config.get("forbid_thinking") and not smoke:
        # A stored row of this run that already shows thinking means the config is known wrong:
        # stop BEFORE calling any row — every subject's file is read first, so a thinking row in a
        # later subject cannot let earlier subjects be called. Rows under another config are not
        # this run's; load_sample_checkpoint refuses them.
        for subject in subjects:
            path = os.path.join(out_dir, subject, f"{model_slug(model_id)}{tag}.jsonl")
            if os.path.exists(path):
                with open(path) as fh:
                    for line in fh:
                        if line.strip():
                            row = json.loads(line)
                            if row.get("config") == config and thinking_signal(row):
                                print(f"[{subject}] a stored row already shows thinking: stopping "
                                      f"before any call", flush=True)
                                halt_on_thinking(row, model_id, model_dir)
    summaries, every_row = [], []
    for subject in subjects:
        data = load_subject(subject)
        ids = ids_by_subject[subject][:limit] if limit else ids_by_subject[subject]
        if limit:
            print(f"[{subject}] --limit {limit}: smoke test, not a run", flush=True)
        # idx is the id from the sample file — an index into the WHOLE subject — and the shard
        # slices the sampled items, never re-numbering them
        mine = shard_slice([(i, data[i]) for i in ids], shard, total_shards)
        sdir = os.path.join(out_dir, subject)
        os.makedirs(sdir, exist_ok=True)
        path = os.path.join(sdir, f"{model_slug(model_id)}{tag}.jsonl")
        done, held = ({}, {}) if smoke else load_sample_checkpoint(path, config, cut_retry)
        foreign = (set(done) | set(held)) - {i for i, _ in mine}
        if foreign:
            raise SystemExit(f"{path}\n  holds {len(foreign)} row(s) not in this shard (first: "
                             f"{sorted(foreign)[0]}) — written under other shard boundaries or "
                             f"another sample. Saving would drop them; archive the file first.")
        records = [done[i] for i, _ in mine if i in done]
        todo = [(i, ex) for i, ex in mine if i not in done]
        print(f"[{subject}] {len(mine)} sampled row(s) in this "
              f"{'smoke test' if smoke else 'shard'}, {len(todo)} to call -> {path}", flush=True)

        def to_save():
            # A stored failed or cut row (`held`) stays until its replacement is in `records`: a
            # resubmit killed mid-row must not drop it, nor what it cost. Skipped once replaced, so a
            # save between the append and the pop below cannot write the row twice.
            have = {r["idx"] for r in records}
            return records + [r for i, r in held.items() if i not in have]

        try:
            for k, (i, ex) in enumerate(todo, 1):
                messages = build_messages(ex)    # once per row: CutRetry keys its state on this list
                prior = held.get(i)
                text, meta = None, None
                # A stop before this row's call starts must not find the last row's ledger and write
                # its bill onto this idx; call_with_retries clears it again on entry.
                _LEDGER.last = None
                # Whether this row is already in `records`, by count — no flag, so a stop landing
                # between the append and anything after it cannot get the row appended twice.
                n_before = len(records)
                try:
                    text, meta = call_with_retries(once, messages, model_id, model_dir)
                    row = sample_row(i, ex, text, meta, config, sample)
                    if prior is not None:
                        row = fold_history(row, prior)
                    records.append(row)
                    held.pop(i, None)
                except BaseException as stop:
                    # SIGTERM, or a billing/quota/malformed halt, stopped this row. Its calls were
                    # billed, so the row is written before the stop goes on — the finally below
                    # saves it, folded into any stored row. If the call had returned (`meta`), the
                    # kill landed while its row was being written: the row is written as answered.
                    # If not, it was stopped mid-call: a failed row marked `halted`, from the ledger
                    # call_with_retries keeps on exit.
                    if len(records) == n_before:
                        recovered = meta if meta is not None else halted_meta(stop)
                        if recovered is not None:
                            row = sample_row(i, ex, text or "", recovered, config, sample)
                            records.append(fold_history(row, prior) if prior is not None else row)
                            held.pop(i, None)
                    raise
                # after the append: a FAILURE_HALT raised here still leaves this row in the file
                note_outcome(not row["failed"], model_id, model_dir)
                if config.get("forbid_thinking") and thinking_signal(row):
                    halt_on_thinking(row, model_id, model_dir)     # saved by the finally below
                if k % save_every == 0:
                    _save_jsonl(path, to_save())
                    print(f"  [{subject}] {len(records)}/{len(mine)}", flush=True)
                if text and sleep:
                    time.sleep(sleep)
        finally:
            # a halt's SystemExit, or SIGTERM at the wall clock (run_sample turns it into
            # SystemExit), must not lose rows paid for since the last periodic save
            _save_jsonl(path, to_save())
        rows = to_save()
        s = write_sample_subject(out_dir, subject, model_id, rows, len(mine), tag=tag)
        summaries.append(s)
        every_row.extend(rows)
        print(f"{subject}: {s['average_score']}  (n={s['n']}/{s['expected']}, failed={s['failed']}, "
              f"parse_fail={s['parse_fail']}, length={s['finish_length']})", flush=True)
        write_sample_overall(out_dir, model_id, summaries, tag=tag)
    return out_dir, every_row


# ---------------------------------------------------------------- CLI shared by every runner


def add_sample_args(ap, default=None):
    """`--sample` and `--save-every`. A runner with a legacy full-run path passes default=None, so
    leaving `--sample` off runs the legacy path exactly as before."""
    ap.add_argument("--sample", default=default, choices=sorted(SAMPLES),
                    help="run the named sample (sample10 = the 401 items of sample10.json) under "
                         "the settled config, writing results_<sample>/ — never results/")
    ap.add_argument("--save-every", dest="save_every", type=int, default=20,
                    help="sample runs only: rows between checkpoint saves")
    return ap


def sample_parser(description, default_model):
    """The whole parser for a runner that exists only for the sample — the legacy runners' flags,
    with --sample defaulting to sample10."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default=default_model,
                    help="model id; it is BOTH what is called and what the result files are named "
                         "after, so a copied folder cannot relabel another model's numbers")
    ap.add_argument("--subject", default="all",
                    help="'all' or a comma-separated list of subject names")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds after each answered call")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N sampled items of each subject, written to smoke_<sample>/ — a "
                         "smoke test, not a run")
    ap.add_argument("--prompt", default=SAMPLE_PROMPT, choices=[SAMPLE_PROMPT],
                    help="the sample runs PROMPT_V2 for every model (the user, 2026-11-15)")
    ap.add_argument("--shard", type=int, default=0, help="this shard's index, 0-based")
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="at 1 the output filenames carry no shard tag")
    # no --workers: sample calls run serially in the main thread, where the SIGALRM watchdog works;
    # parallelism is the sbatch array's shards
    return add_sample_args(ap, default="sample10")


def run_sample(args, *, model_dir, client, once, wanted, fixed=None, required=None,
               call_timeout=300, extra_config=None, forbid_thinking=False):
    """`once(messages, model, params) -> (text, meta)`. Returns the exit code: 0 clean; 1 if a row of
    this shard failed or is still cut at the cap (resubmit — resume retries exactly those); 2 if a
    forced answer reasoned again or filled its own cap (read them before scoring); 3 if
    `forbid_thinking` and a row's calls came back with reasoning (`thinking_signal`) — the model is
    run with thinking off, so its config did not hold. The highest applicable code wins."""
    global HARD_CALL_TIMEOUT, MARKER_PREFIX
    if getattr(args, "prompt", SAMPLE_PROMPT) != SAMPLE_PROMPT:
        raise SystemExit(f"the sample runs prompt {SAMPLE_PROMPT} for every model (the user, "
                         f"2026-11-15); got --prompt {args.prompt}")
    if getattr(args, "workers", 1) != 1:
        raise SystemExit("--workers must be 1 for a sample run: its calls run serially in the main "
                         "thread, the only place the SIGALRM watchdog can interrupt a hung call. "
                         "Parallelism is the sbatch array's shards.")
    if args.save_every < 1:
        raise SystemExit(f"--save-every must be at least 1, got {args.save_every}")
    subjects = SUBJECTS if args.subject == "all" else [s.strip() for s in args.subject.split(",")]
    unknown = [s for s in subjects if s not in SUBJECTS]
    if unknown:
        raise SystemExit(f"unknown subject(s): {unknown}\nknown: {SUBJECTS}")
    ids_by_subject, sample_sha1 = load_sample(args.sample)
    smoke = bool(args.limit)
    HARD_CALL_TIMEOUT = int(call_timeout)
    MARKER_PREFIX = marker_prefix(args.sample, smoke, args.shard, args.total_shards)
    # SLURM ends a job at its wall clock (and on scancel) with SIGTERM, whose default kills the
    # process without running `finally`; as SystemExit, the last save happens.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    clear_stale_halt_markers(model_dir)
    print(f"negotiating parameter surface for {args.model} ...", flush=True)
    params, notes = negotiate_pinned(client, args.model, wanted, model_dir, fixed=fixed,
                                     required=required)
    print(f"  accepted: {params}" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    # The run config, on every row; a resume across any change of it is refused.
    config = json.loads(json.dumps(dict(extra_config or {}, model=args.model, params=params,
                                        fixed=fixed or {}, prompt=SAMPLE_PROMPT,
                                        sample=args.sample, sample_sha1=sample_sha1,
                                        **({"forbid_thinking": True} if forbid_thinking else {})),
                                   sort_keys=True))
    out_dir = sample_out_dir(model_dir, args.sample, smoke)
    os.makedirs(out_dir, exist_ok=True)
    tag = shard_tag(args.shard, args.total_shards)
    _atomic_write(os.path.join(out_dir, f"{model_slug(args.model)}{tag}_negotiated_params.json"),
                  lambda fh: json.dump({"asked": wanted, "accepted": params, "notes": notes,
                                        "config": config}, fh, indent=2))
    print(f"{args.model}: sample={args.sample} subjects={len(subjects)} prompt={SAMPLE_PROMPT} "
          f"shard={args.shard}/{args.total_shards} -> {out_dir}", flush=True)
    started = time.time()
    rows = []
    try:
        _, rows = run_sample_subjects(
            model_dir, args.model, lambda m: once(m, args.model, params), config, args.sample,
            ids_by_subject, subjects, limit=args.limit, shard=args.shard,
            total_shards=args.total_shards, save_every=args.save_every, sleep=args.sleep)
    finally:
        print(f"\ncalls: {STATS}  elapsed {time.time() - started:.0f}s", flush=True)
    code = 0
    still = [r for r in rows if is_cut(r)] if "forced_answer" in config else []
    failed = [r for r in rows if r.get("failed")]
    if still or failed:
        print(f"\n{len(failed)} failed and {len(still)} cut row(s) in this shard — resubmit it: "
              f"resume calls exactly those. First: "
              f"{(failed or still)[0]['subject']}#{(failed or still)[0]['idx']}", flush=True)
        code = 1
    odd = [r for r in rows if r.get("forced_answer")
           and (r.get("finish_reason") == "length" or forced_reasoned(r))]
    if odd:
        print(f"\nSTOP: {len(odd)} forced answer(s) filled their own cap or reasoned again — read "
              f"them before scoring. First: {odd[0]['subject']}#{odd[0]['idx']}", flush=True)
        code = 2
    thinking = [r for r in rows if thinking_signal(r)] if forbid_thinking else []
    if thinking:
        first = thinking[0]
        print(f"\nSTOP: THINKING CAME BACK ON — {len(thinking)} row(s) came back with reasoning on a "
              f"model run with thinking off; the config did not hold. First: "
              f"{first['subject']}#{first['idx']} (reasoning_chars={first.get('reasoning_chars')}, "
              f"reasoning_tokens={first.get('reasoning_tokens')}, "
              f"billed_reasoning_tokens={first.get('billed_reasoning_tokens')})", flush=True)
        code = 3
    return code


def merge_sample(model_dir, model_id, sample, total_shards, subjects=None):
    """Shard files of one sample back into one file per subject, under the plain name. A missing
    shard is reported and the rest merged (script-skeleton.md 7b); a subject whose merged ids are not
    exactly the sample's, or whose rows ran under more than one config, is reported as a problem."""
    ids_by_subject, _ = load_sample(sample)
    out_dir = sample_out_dir(model_dir, sample)
    slug = model_slug(model_id)
    summaries, problems = [], []
    for subject in subjects or SUBJECTS:
        d = os.path.join(out_dir, subject)
        rows, seen = [], set()
        for s in range(total_shards):
            path = os.path.join(d, f"{slug}_shard{s}of{total_shards}.jsonl")
            if not os.path.exists(path):
                problems.append((subject, f"{os.path.basename(path)} missing"))
                continue
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        if r["idx"] in seen:
                            problems.append((subject, f"idx {r['idx']} in more than one shard"))
                            continue
                        seen.add(r["idx"])
                        rows.append(r)
        if not rows:
            problems.append((subject, "no rows in any shard"))
            continue
        want = ids_by_subject[subject]
        if sorted(seen) != want:
            problems.append((subject, f"merged {len(seen)} ids, sample has {len(want)}; "
                                      f"missing {sorted(set(want) - seen)[:5]}, "
                                      f"extra {sorted(seen - set(want))[:5]}"))
        if len({json.dumps(r.get("config"), sort_keys=True) for r in rows}) > 1:
            problems.append((subject, "rows ran under more than one config"))
        failed = sum(1 for r in rows if r.get("failed"))
        if failed:
            problems.append((subject, f"{failed} failed row(s)"))
        thinking = sum(1 for r in rows if (r.get("config") or {}).get("forbid_thinking")
                       and thinking_signal(r))
        if thinking:
            problems.append((subject, f"{thinking} row(s) came back with reasoning on a model run "
                                      f"with thinking off — thinking came back on"))
        summaries.append(write_sample_subject(out_dir, subject, model_id, rows, len(want)))
    if summaries:
        write_sample_overall(out_dir, model_id, summaries)
    return summaries, problems
