"""Shared core for every DocVQA runner. Modelled on `Tasks_benchmarks/mmlu/mmlu_eval_core.py`.

**One scorer.** `score_response` and `anls_score` are the only scorers in this benchmark. They are
**copied verbatim** from the four runners that already produced results — `openai_eval.py`,
`gemini_eval.py`, `qwen_DocVQA/qwen_eval.py`, `gemma_DocVQA/gemma_eval_half*.py` — rather than
improved, because those four results stay valid only if a new model is judged by the same matcher
they were. Changing anything here means rescoring all six, not just scoring the new two.

**The prompt is byte-identical to those four runners too**, for the same reason: a different prompt
is a different condition, and DocVQA answers are verbatim spans, so wording moves the score.

**The task is defined on the document image.** The challenge's Task 1 page: "answer questions asked
on a document image", answers are "short text spans taken verbatim from the document". The OCR the
dataset ships is auxiliary — "participants are free to use any OCR" — and this project does not use
it (settled 2026-09-09; the reasoning is on the benchmark's page). So every runner here sends the
page image, and `load_image_bytes` is the only input path.

**ANLS takes the maximum over the answer list.** Each question carries several acceptable answers.
Scoring against one of them under-reports systematically.

**Two paths, and everything above `the 10% sample path` below is the first one, unchanged.** A run
without `--sample` goes through `run` / `retry` / `load_checkpoint` / `write_results` exactly as
before the sample path was added (2026-11-15) — same files, prompt and rows; the proof is
`test_docvqa_sample10.py`. A run with `--sample sample10` goes through `run_sample`: the 539
questions of `sample10.json`, written to `<model folder>/results_sample10/` (`smoke_sample10/`
under `--limit`), never `results/`, with the per-row usage fields of `evaluation-criteria.md`. Its
harness — `call_with_retries`, `openai_meta`, `negotiate_params`, the billing halts, `CutRetry` — is
ported from `LLMs-Planning_bench/plan_eval_core.py`, not imported: cores travel to Quest separately.
The prompt and the scorer are the same objects on both paths.
"""

import argparse
import base64
import collections
import csv
import hashlib
import json
import os
import random
import re
import signal
import socket
import statistics
import sys
import threading
import time

# Paths resolve from THIS FILE, never the cwd: a runner lives in <DocVQA>/DOCVQA_<Slot>/ while the
# data lives in <DocVQA>/docvqa_output/.
DOCVQA_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DOCVQA_ROOT, "docvqa_output", "docvqa_validation.json")


def _find_env():
    """This benchmark's own `.env` if it has one, else the nearest one above it.

    Quest keeps a `.env` per benchmark directory; the local tree keeps a single one at the repo
    root and DocVQA has none of its own. The old runners called `load_dotenv()` with no argument
    and got the same walk-up for free — this makes it explicit rather than dependent on the cwd
    the job happened to start in.
    """
    d = DOCVQA_ROOT
    while True:
        candidate = os.path.join(d, ".env")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.join(DOCVQA_ROOT, ".env")   # nothing found; report this path in the error
        d = parent


ENV_PATH = _find_env()

# Bumped whenever the matcher changes behaviour, and written onto every result row. A v1 number
# cannot be compared with a v2 one without rescoring — v2 added two branches on 2026-09-11, the
# history is on the benchmark's page.
SCORER_VERSION = "docvqa_lenient_v2"

# Byte-for-byte what the four finished runners sent. Do not tidy it.
PROMPT = (
    "You are reading a document image. Answer the question below using only "
    "information visible in the document.\n\n"
    "Question: {question}\n\n"
    "Give a short, direct answer — a word, number, or brief phrase. "
    'End your response with: "Final Answer: <your answer here>"'
)

AUTH_MARKERS = ("invalid_api_key", "incorrect api key", "unauthorized", "permission_denied",
                "api key not valid", "unauthenticated", "insufficient_quota")
EFFORT_FALLBACKS = ("low", "minimal", "medium")


def model_slug(model_id):
    return model_id.replace("/", "-").replace(".", "_")


def load_data():
    """The 5,349 validation questions. Each carries questionId, question, answers, image_path."""
    with open(DATA_PATH) as fh:
        return json.load(fh)


def build_prompt(example):
    return PROMPT.format(question=example["question"])


def load_image_bytes(image_path):
    """`image_path` in the data is repo-relative — './docvqa_output/images/49153.png'."""
    return open(os.path.join(DOCVQA_ROOT, image_path.lstrip("./")), "rb").read()


def load_image_b64(image_path):
    return base64.standard_b64encode(load_image_bytes(image_path)).decode("utf-8")


# ---------------------------------------------------------------- scoring


def extract_final_answer(model_output):
    if not isinstance(model_output, str):
        return ""
    m = re.search(r"Final Answer:\s*(.*)", model_output, re.IGNORECASE)
    return m.group(1).strip() if m else model_output.strip()


def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())


def strip_punct(text):
    return re.sub(r"[^\w\s]", "", text)


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def anls_score(prediction, gold_answers, threshold=0.5):
    """Average Normalized Levenshtein Similarity — DocVQA's official metric.

    The `max` over `gold_answers` is not a nicety: every question has several acceptable answers and
    the challenge's own metric takes the best of them. Scoring against one under-reports.
    The 0.5 threshold is the official one — below it the score is 0, not a small number.
    """
    pred = normalize(prediction)
    best = 0.0
    for g in gold_answers:
        g_norm = normalize(g)
        max_len = max(len(pred), len(g_norm))
        nl = 0.0 if max_len == 0 else levenshtein(pred, g_norm) / max_len
        sim = 0.0 if nl > threshold else 1.0 - nl
        best = max(best, sim)
    return best


def squash(text):
    """Every space and punctuation mark removed — the written form stripped to its characters.

    `"JAN 17 '68"` and `"jan17'68"` are the same answer written two ways, and so are `DR.W.J.DARBY`
    and `Dr. W. J. Darby`, `3. 28-2001` and `3-28-2001`. The existing branches normalise spacing OR
    punctuation; this normalises both at once, which is where those pairs fall through.
    """
    return re.sub(r"[^\w]", "", normalize(text))


def one_transposition(a, b, floor=8):
    """True when `a` and `b` differ by exactly one swap of two ADJACENT characters.

    An adjacent swap is the signature of a typing slip, and in this dataset it is usually the GOLD
    that has it: `Grocrey` for `Grocery`, `laways` for `always`, `Cigfil` for `Cigifl`. It is a
    different thing from a misreading — `Paul Saliman` for `Paul Saltman` is one substitution, not a
    swap, and is a wrong answer.

    **`floor` is not decoration.** Without it this branch credits `14` for `41`, which is a genuinely
    different number; measured over all 32,094 stored rows that was the single false gain it made,
    and 8 characters removes it while keeping all twelve real ones.
    """
    if len(a) != len(b) or a == b or len(a) < floor:
        return False
    d = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]


def score_response(model_output, gold_answers):
    """Lenient exact match. Branches are additive and tried in order, so one can only gain rows.

    Six branches. The first four are verbatim from the four standalone runners that produced this
    benchmark's earlier results; branches 5 and 6 were added on 2026-09-11 and each was measured
    over every stored row before being kept. **Every branch normalises how an answer is written.
    None of them tolerates an answer being slightly wrong** — see the page for the fuzzy branch that
    was measured and rejected for crediting `Paul Saliman` as `Paul Saltman`.
    """
    fa = extract_final_answer(model_output)
    if not fa:
        return 0
    if any(normalize(fa) == normalize(g) for g in gold_answers):
        return 1

    def tokenize(t):
        return normalize(t).replace(",", " ").split()

    if any(tokenize(fa) == tokenize(g) for g in gold_answers):
        return 1
    if any(normalize(strip_punct(fa)) == normalize(strip_punct(g)) for g in gold_answers):
        return 1
    fa_norm = normalize(fa)
    if any(fa_norm in normalize(g) or normalize(g) in fa_norm for g in gold_answers):
        return 1
    # 5 — spacing and punctuation removed together, +311 rows measured
    sq = squash(fa)
    if any(sq == squash(g) for g in gold_answers):
        return 1
    # 6 — one adjacent transposition, +12 rows measured, every one of them a typo in the gold
    if any(one_transposition(sq, squash(g)) for g in gold_answers):
        return 1
    return 0


# ---------------------------------------------------------------- output

FIELDS = ["questionId", "question", "gold_answers", "image_path", "model_response",
          "final_answer", "score", "anls", "config"]


def config_string(config):
    """The generation config, flattened onto every row — model-parameters.md rule 8. A config that
    lives only in a job script cannot be recovered from a result file later."""
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


def results_dir(model_dir):
    d = os.path.join(model_dir, "results")
    os.makedirs(d, exist_ok=True)
    return d


def shard_tag(shard, total_shards):
    """`_shard2of5`, or empty when total_shards is 1 — the empty case keeps an unsharded run writing
    the filename it always did. Without the tag every shard overwrites the last (`.claude/INDEX.md`).
    """
    return f"_shard{shard}of{total_shards}" if total_shards > 1 else ""


def shard_slice(items, shard, total_shards):
    """Contiguous block split, so a shard's rows stay in one range and a missing shard shows up as
    a gap rather than scattered holes."""
    if total_shards <= 1:
        return list(items)
    size = (len(items) + total_shards - 1) // total_shards
    return list(items)[shard * size: min((shard + 1) * size, len(items))]


def retry(fn, tries=3, base_sleep=2.0, label="", fatal=()):
    """Retries on exception AND on an empty string, returning "" when every attempt fails.

    **`fatal` is the reason this is not a generic retry.** DocVQA's one recorded incident was a
    daily-request-cap refusal retried three times an item: for every question answered, six were
    skipped because the retries had spent the quota, leaving 3,021 of 5,349 rows empty
    (`OPENAI_EVAL_NOTES.md`). A cap refusal is not transient — matching one of `fatal` stops the run
    at once rather than burning what is left of the day.
    """
    for attempt in range(tries):
        try:
            out = fn()
            if isinstance(out, str) and out.strip():
                return out.strip()
        except Exception as e:
            err = str(e)
            for marker, why in fatal:
                if marker in err.lower():
                    raise SystemExit(f"[{label}] {why}\n  {err[:400]}")
            print(f"[{label}] attempt {attempt + 1}/{tries} failed: {err[:200]}", flush=True)
        if attempt < tries - 1:
            time.sleep(base_sleep * (2 ** attempt))
    # An empty string at HTTP 200 raises nothing, so without this line every attempt is silent and
    # the run's only trace of the failure is a blank cell. gemini-3.5-flash-lite returned exactly
    # that on 5 of 5,349 images on 2026-09-10 — finish_reason MALFORMED_RESPONSE, thought tokens
    # spent, no output part — and the logs held not one word about it.
    print(f"[{label}] gave up after {tries} attempt(s), writing an empty row", flush=True)
    return ""


def load_checkpoint(out_path, config, verbose=True):
    """Rows already done, keyed by questionId — or a refusal if the config changed.

    **An empty row is not a done row.** The 3,021 empty rows the quota incident left behind had to
    be re-asked, and a plain resume would have skipped them forever.
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
            done[str(r["questionId"])] = r
        else:
            empty += 1
    if verbose and (done or empty):
        print(f"  resume: {len(done)} done, {empty} empty row(s) will be retried", flush=True)
    return done


def write_results(model_dir, model_id, records, config=None, tag=""):
    d = results_dir(model_dir)
    slug = model_slug(model_id) + tag
    ordered = sorted(records, key=lambda r: int(r["questionId"]))
    with open(os.path.join(d, f"{slug}.jsonl"), "w") as fh:
        for r in ordered:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(d, f"{slug}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    n = len(ordered)
    acc = sum(r["score"] for r in ordered) / n if n else ""
    anls = sum(r["anls"] for r in ordered) / n if n else ""
    blank = sum(1 for r in ordered if not str(r["model_response"]).strip())
    no_marker = sum(1 for r in ordered
                    if not re.search(r"Final Answer:", str(r["model_response"]), re.IGNORECASE))
    with open(os.path.join(d, f"{slug}_overall.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "n", "overall_accuracy", "anls", "no_marker", "empty_response",
                    "scorer", "config"])
        w.writerow([model_id, n, round(acc, 4) if n else "", round(anls, 4) if n else "",
                    no_marker, blank, SCORER_VERSION, config_string(config)])
    return {"model": model_id, "n": n, "overall_accuracy": round(acc, 4) if n else "",
            "anls": round(anls, 4) if n else "", "no_marker": no_marker, "empty_response": blank}


# ---------------------------------------------------------------- the loop


def run(model_dir, model_id, call, limit=0, config=None, save_every=20, resume=True,
        sleep_between=0.0, shard=0, total_shards=1, verbose=True):
    """`call(prompt, image_path)` returns the model's text, or "" when every attempt failed."""
    data = load_data()
    if limit:
        data = data[:limit]
        if verbose:
            print(f"--limit {limit}: smoke test, not a run", flush=True)
    config = dict(config or {})
    tag = shard_tag(shard, total_shards)
    out_path = os.path.join(results_dir(model_dir), f"{model_slug(model_id)}{tag}.jsonl")
    done = load_checkpoint(out_path, config, verbose) if (resume and not limit) else {}
    records = list(done.values())

    def flush(recs):
        with open(out_path, "w") as fh:
            for r in sorted(recs, key=lambda x: int(x["questionId"])):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    todo = [ex for ex in shard_slice(data, shard, total_shards)
            if str(ex["questionId"]) not in done]
    if verbose:
        print(f"shard {shard}/{total_shards}: {len(todo)} to ask, {len(done)} already done",
              flush=True)

    for k, ex in enumerate(todo, 1):
        gold = ex["answers"]
        if isinstance(gold, str):          # the JSON stores this as a stringified list
            gold = json.loads(gold.replace("'", '"'))
        try:
            resp = call(build_prompt(ex), ex["image_path"])
        except FileNotFoundError:
            print(f"[{ex['questionId']}] image missing: {ex['image_path']}", flush=True)
            resp = ""
        resp = resp if isinstance(resp, str) else ""
        records.append({
            "questionId": ex["questionId"], "question": ex["question"],
            "gold_answers": json.dumps(gold, ensure_ascii=False), "image_path": ex["image_path"],
            "model_response": resp, "final_answer": extract_final_answer(resp),
            "score": score_response(resp, gold), "anls": anls_score(extract_final_answer(resp), gold),
            "config": config_string(config)})
        if sleep_between:
            time.sleep(sleep_between)
        if save_every and k % save_every == 0:
            flush(records)
            if verbose:
                print(f"  {len(records)}/{len(todo) + len(done)}", flush=True)

    summary = write_results(model_dir, model_id, records, config=config, tag=tag)
    if verbose:
        print(f"done: n={summary['n']} acc={summary['overall_accuracy']} "
              f"anls={summary['anls']} no_marker={summary['no_marker']} "
              f"empty={summary['empty_response']}", flush=True)
    return summary


def negotiate(client, model, wanted, verbose=True):
    """Find which of `wanted` the model actually accepts, in one probe call. A refused VALUE is not
    a refused PARAMETER — conflating them drops a cap instead of correcting it. Ported verbatim
    from `mmlu_eval_core.negotiate`."""
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


# ======================================================================== the 10% sample path
# Added 2026-11-15 for the 10% sample rerun (`.claude/references/sample10-plan.md` § WP4). Nothing
# above this line changed: a run without --sample never calls anything below. Names that exist
# above (`negotiate`, `retry`, `load_checkpoint`, `results_dir`, `AUTH_MARKERS`, `EFFORT_FALLBACKS`)
# are NOT redefined here — the ported versions carry their own names, so the legacy path cannot
# pick one up by accident.
#
# What is ported from `LLMs-Planning_bench/plan_eval_core.py`, and kept in step with it by hand:
# the billing / daily-quota / failure halts, `call_with_retries`, `openai_meta`, `negotiate`
# (here `negotiate_params`) and `CutRetry`. Differences from there: `CutRetry.with_system` puts the
# ceiling FIRST, as the only system message (DocVQA has none to follow); `openai_meta` books every
# response onto a per-row ledger (BILLED_FIELDS, and `reasoning_chars` summed over every response of
# the row — Qwen's and Gemma's thinking must stay off, and a non-zero count is how a row shows it did
# not); and a resubmit keeps a stored failed row until its new row replaces it, folding its history
# in (RESUME_FIELDS).

SAMPLES = {"sample10": os.path.join(DOCVQA_ROOT, "sample10.json")}

# Every sample row carries these, filled or None — evaluation-criteria.md § Per-row fields.
# `reasoning_tokens` is None (empty in the CSV), never 0, where the provider reports no breakdown.
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "latency_s",
                "finish_reason", "n_attempts", "served_model")
# What the row COST, beside what the answer used (the fields above keep their meaning: the answered
# call's, or for a CutRetry row its stages summed). Summed over every HTTP call of the row — every
# attempt and every CutRetry stage — whose response carried a usage object, empty and interrupted
# replies included, since they are billed. An attempt retried after an HTTP 200 empty reply with
# 16,384 reasoning tokens would otherwise leave a row saying 50. `billed_cost` sums OpenRouter's
# `usage.cost`, `billed_estimated_cost` DeepInfra's `usage.estimated_cost`. Each `billed_*` is None
# when no call reported it. `attempts_without_usage` counts every HTTP call — CutRetry stages
# included — that returned no usage object or ended in an exception or timeout: its bill is
# unknown, not zero. **Cumulative across runs of the row**: a row that failed and was called again on
# a resubmit carries its earlier runs folded in (RESUME_FIELDS). Same names in the BBH and MMLU cores.
BILLED_FIELDS = ("billed_prompt_tokens", "billed_completion_tokens", "billed_reasoning_tokens",
                 "billed_cost", "billed_estimated_cost", "attempts_without_usage")
# A row called again after it was stored failed. `n_attempts` stays this run's; `prior_n_attempts` is
# the sum over the earlier runs and `prior_failed_runs` their count. 0, 0, False on every other row.
RESUME_FIELDS = ("prior_n_attempts", "prior_failed_runs", "resumed_after_failure")

# How the image travels — one layout for every model, recorded in each row's config. It is what the
# OpenAI slot already sent (`DOCVQA_GPT_5.6_Luna`): the stored PNG bytes unchanged, base64, at
# detail="high" (a document page is unreadable at low), image part before the text part.
MESSAGE_LAYOUT = "user[image_url(png bytes as stored, base64, detail=high), text(PROMPT)]"
PROMPT_SHA1 = hashlib.sha1(PROMPT.encode("utf-8")).hexdigest()

SAMPLE_FIELDS = ["questionId", "idx", "stratum", "question", "gold_answers", "image_path",
                 "model_response", "final_answer", "score", "anls", "parse_ok", "has_marker",
                 "empty", "error", *USAGE_FIELDS, *BILLED_FIELDS, *RESUME_FIELDS, "reasoning_chars",
                 "think_tag_calls", "halted",
                 "backend", "last_call_cost", "last_call_estimated_cost", "seed", "cut_retry",
                 "forced_answer", "sample", "scorer", "ts", "config"]


def image_file(image_path):
    """The file `load_image_bytes` opens — the same anchor (this file's directory), so a path that
    resolves here resolves for the call."""
    return os.path.join(DOCVQA_ROOT, image_path.lstrip("./"))


def build_messages(example):
    """The one message every sample runner sends: `build_prompt`'s text, unchanged, after the page."""
    b64 = load_image_b64(example["image_path"])
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
        {"type": "text", "text": build_prompt(example)}]}]


def load_sample(name):
    """(spec, rows): the sample's questions in data order, each {idx, stratum, example}.

    `idx` is the question's position in the whole validation set, so it indexes the benchmark and
    not the sample. Refuses — never trims — a sample file that disagrees with the data: an id the
    data lacks, a stratum that is not the question's `question_types[0]`, a count that is off."""
    with open(SAMPLES[name]) as fh:
        spec = json.load(fh)
    by_id = {str(ex["questionId"]): (i, ex) for i, ex in enumerate(load_data())}
    rows, problems = [], []
    for stratum, block in spec["strata"].items():
        if len(block["ids"]) != block["n"]:
            problems.append(f"{stratum}: n={block['n']} but {len(block['ids'])} ids")
        for qid in block["ids"]:
            if str(qid) not in by_id:
                problems.append(f"{stratum}: questionId {qid} is not in the data")
                continue
            idx, ex = by_id[str(qid)]
            if (ex.get("question_types") or [None])[0] != stratum:
                problems.append(f"{qid}: question_types[0]={ex.get('question_types')} != {stratum}")
            # anls_score gives an EMPTY prediction 1.0 against an empty gold answer, which would put
            # ANLS > 0 on a row with parse_ok False. None of the 5,349 has one (checked 2026-11-15);
            # refusing it here makes "parse_ok False => score 0 and ANLS 0" a guarantee, not luck.
            if any(normalize(g) == "" for g in ex.get("answers") or []):
                problems.append(f"{qid}: a gold answer is empty, and ANLS would credit an empty reply")
            rows.append({"idx": idx, "stratum": stratum, "example": ex})
    if len(rows) != spec.get("total_n") or len({r["idx"] for r in rows}) != len(rows):
        problems.append(f"{len(rows)} rows, {len({r['idx'] for r in rows})} distinct, "
                        f"total_n={spec.get('total_n')}")
    if problems:
        raise SystemExit(f"{SAMPLES[name]} does not match the data:\n  " + "\n  ".join(problems[:20]))
    rows.sort(key=lambda r: r["idx"])
    return spec, rows


def sample_dir(model_dir, sample, smoke=False):
    """results_<sample>/, or smoke_<sample>/ under --limit — never results/: a shared checkpoint would
    make the full run's resume skip sample rows, or a sample resume read full-run rows."""
    return os.path.join(model_dir, f"{'smoke' if smoke else 'results'}_{sample}")


def require_key(name):
    """The key's value, or a stop naming the .env read — never the value."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set (read {ENV_PATH}); add it there — do not copy .env files")
    return value


# ---------------------------------------------------------------- refusals that must stop a run
# plan_eval_core.py's, unchanged: only wordings that unambiguously mean money or a daily cap.

BILLING_SIGNATURES = (
    "insufficient_quota", "insufficient balance", "insufficient credits", "out of credits",
    "monthly budget", "used all available credits", "spending limit", "credit balance",
    "payment required", "arrears",
    # OpenRouter's out-of-credit 402 (provider-gotchas.md § Classifying a refusal)
    "requires more credits", "can only afford", "openrouter_credits",
    # DeepInfra's out-of-balance 402, "You need positive balance to do inference" (2026-11-15)
    "positive balance",
)
# A backend that cut the answer off: HTTP 200, the fragment as content. A failed attempt, never an
# answer. "length" is not here — that is the model reaching our cap, which the summary counts.
INTERRUPTED_FINISH = ("error", "insufficient_system_resource")
DAILY_QUOTA_SIGNATURES = ("per_model_per_day", "requests per day", "requests_per_day", "per day",
                          "daily limit", "daily quota")
DAILY_HALT_THRESHOLD_SECONDS = 600
FATAL_SIGNATURES = ("invalid_argument", "extra inputs are not permitted", "validationerror",
                    "unsupported parameter", "unsupported value", "unrecognized request argument",
                    "unexpected keyword argument")
SAMPLE_AUTH_MARKERS = ("authentication", "api key", "invalid_api_key", "incorrect api key")
AUTH_STATUS = ("401",)


def has_status(text, codes):
    """Whether `text` names one of the numeric HTTP statuses `codes` as a whole word (`\\b401\\b`).
    Matched as a substring, "401" also hits the hex request ids inside an unrelated 500 error and
    would stop the run as an auth failure (reviewer, 2026-11-15). Every numeric status tested in this
    section goes through here; wordings stay substrings."""
    return any(re.search(rf"\b{re.escape(code)}\b", str(text)) for code in codes)


def is_auth_failure(text):
    low = str(text).lower()
    return any(m in low for m in SAMPLE_AUTH_MARKERS) or has_status(low, AUTH_STATUS)

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
HALT_MARKERS = ("BILLING_HALT.txt", "QUOTA_HALT.txt", "FAILURE_HALT.txt", "FATAL_HALT.txt",
                "THINKING_HALT.txt")
# `sample10_` and, with more than one shard, `shard2_`: a sample run's marker can never be mistaken
# for (or cleared by) another run's. Set by run_sample, with the output directory a halt names.
MARKER_PREFIX = ""
SAMPLE_OUT_DIR = ""


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


def write_thinking_marker(row, model, model_dir):
    """THINKING_HALT.txt, for a row that shows thinking on a model that must run with it off; returns
    the one-line detail."""
    marker = os.path.join(model_dir, MARKER_PREFIX + "THINKING_HALT.txt")
    detail = (f"questionId {row['questionId']}: reasoning_chars {row.get('reasoning_chars')}, "
              f"think_tag_calls {row.get('think_tag_calls')}, reasoning_tokens "
              f"{row.get('reasoning_tokens')}, billed_reasoning_tokens {row.get('billed_reasoning_tokens')}")
    archive = SAMPLE_OUT_DIR or "the results_<sample>/ directory"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason=THINKING\n{detail}\n\nThis model must run with "
                     f"thinking off and a request turned it on. Fix the runner, move {archive} to a "
                     f"timestamped name, and rerun; a plain resubmit on this config halts again.\n")
    except OSError as error:
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! THINKING HALT [{model}] {stamp}\nSTOP: THINKING CAME BACK ON — {detail}\n",
          file=sys.stderr, flush=True)
    print(f"STOP: THINKING CAME BACK ON — {detail}; wrote {os.path.basename(marker)}", flush=True)
    return detail


def thinking_halt(row, model, model_dir):
    """A model that must run with thinking off (Qwen, Gemma) returned a row showing thinking: write
    THINKING_HALT.txt and stop with exit 3 at once — every further row would run on a config already
    known to be wrong. run_sample's `finally` saves the rows, this one included."""
    write_thinking_marker(row, model, model_dir)
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


# Consecutive failures catch a provider that stops serving; the window one that fails intermittently.
CONSECUTIVE_FAILURE_LIMIT = 10
FAILURE_WINDOW = 50
FAILURE_WINDOW_LIMIT = 0.5
_recent = collections.deque(maxlen=FAILURE_WINDOW)
_consecutive = [0]


def note_outcome(ok, model, model_dir):
    _recent.append(bool(ok))
    _consecutive[0] = 0 if ok else _consecutive[0] + 1
    if _consecutive[0] >= CONSECUTIVE_FAILURE_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt",
              f"{_consecutive[0]} consecutive rows failed every attempt", model, model_dir)
    if len(_recent) == FAILURE_WINDOW and _recent.count(False) / FAILURE_WINDOW >= FAILURE_WINDOW_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt",
              f"{_recent.count(False)} of the last {FAILURE_WINDOW} rows failed", model, model_dir)


# ---------------------------------------------------------------- one call, guarded


class CallTimeout(BaseException):
    """BaseException on purpose: an `except Exception` anywhere below would otherwise swallow the
    watchdog, and the rest of that call would run unprotected (provider-gotchas.md)."""


# Hard ceiling on one HTTP attempt, by SIGALRM — a runner sets it per provider (run_sample's
# call_timeout). It sits below the client's own timeout, so the watchdog, not the SDK, ends a hang.
HARD_CALL_TIMEOUT = 300


def _raise_timeout(signum, frame):
    raise CallTimeout(f"call exceeded {HARD_CALL_TIMEOUT}s hard limit")


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


# ---- billed usage: one ledger per row, booked by openai_meta as each response arrives

_LEDGER = threading.local()      # per thread, so a row's bill can never be booked to another row's
# An opening or closing think tag inside `content`, in any of the four spellings NEG_Gemma_DeepInfra
# strips (`<think>`, `</think>`, `<|think|>`, `<|/think|>`, any case, inner spaces allowed): a model
# that thinks inline, with no reasoning field and no reasoning tokens reported.
THINK_TAG = re.compile(r"<\|?\s*/?\s*think\s*\|?>", re.IGNORECASE)


def _start_call():
    """A new HTTP call of the open row begins: each attempt, and each CutRetry stage. It stays open
    until openai_meta books its response; a failure that finds it open adds one call without usage,
    and one that finds it closed adds none — the bill of a response already booked is known, even if
    `once` raised on reading it (HTTP 200, usage present, `choices` empty)."""
    ledger = getattr(_LEDGER, "row", None)
    if ledger is not None:
        ledger["call_open"] = True


def _close_failed_call(ledger):
    if ledger["call_open"]:
        ledger["without_usage"] += 1
        ledger["call_open"] = False


def take_interrupted():
    """The meta call_with_retries stashed when an exception it does not handle (a halt's SystemExit,
    SIGTERM's) ended a row mid-way, or None — read once, so it cannot be applied to another row."""
    stash, _LEDGER.interrupted = getattr(_LEDGER, "interrupted", None), None
    return stash
_LEDGER_KEYS = {"prompt": "billed_prompt_tokens", "completion": "billed_completion_tokens",
                "reasoning": "billed_reasoning_tokens", "cost": "billed_cost",
                "estimated_cost": "billed_estimated_cost"}


def _book(usage, reasoning_chars, think_tag=0):
    """Book one HTTP response onto the row being called. openai_meta calls this for every response
    it reads — the one function every runner's response passes through — so each response is
    counted exactly once, whatever happens to it next: an empty or interrupted reply that is
    retried, or a CutRetry stage kept for the next attempt (counting from the returned meta instead
    would miss a stage whose attempt later raised, and count a kept stage twice). A response with
    no usage object is a call whose bill is unknown. Outside call_with_retries (the negotiation
    probe) nothing is open and nothing is booked."""
    ledger = getattr(_LEDGER, "row", None)
    if ledger is None:
        return
    ledger["call_open"] = False                    # this call answered: its bill, if any, is below
    ledger["responses"] += 1
    ledger["reasoning_chars"] += reasoning_chars
    ledger["think_tag_calls"] += think_tag
    if usage is None:
        ledger["without_usage"] += 1
        return
    details = getattr(usage, "completion_tokens_details", None)
    for key, value in (("prompt", getattr(usage, "prompt_tokens", None)),
                       ("completion", getattr(usage, "completion_tokens", None)),
                       ("reasoning", getattr(details, "reasoning_tokens", None)),
                       ("cost", getattr(usage, "cost", None)),                       # OpenRouter
                       ("estimated_cost", getattr(usage, "estimated_cost", None))):  # DeepInfra
        kinds = (int, float) if key in ("cost", "estimated_cost") else (int,)   # tokens are counts
        if isinstance(value, kinds) and not isinstance(value, bool):
            ledger[key].append(value)


def _billed(ledger):
    def total(values):
        return round(sum(values), 10) if values else None
    out = {field: total(ledger[key]) for key, field in _LEDGER_KEYS.items()}
    out["attempts_without_usage"] = ledger["without_usage"]
    # Every response of the row — every attempt, every CutRetry stage — not just the one that
    # answered: thinking that came back on in a failed attempt or a cut stage is still thinking.
    out["reasoning_chars"] = ledger["reasoning_chars"]
    out["think_tag_calls"] = ledger["think_tag_calls"]
    return out


def call_with_retries(once, messages, model, model_dir, tries=5, fatal=()):
    """`once(messages) -> (text, meta)`. Retries an exception, a timeout AND an empty string — HTTP
    200 with an empty body raises nothing. Returns ("", meta) when every attempt failed; the row is
    then written empty, counted as failed, and retried on resume.

    `fatal` is the legacy `retry`'s, kept: (marker, why) pairs that stop the run at the first match
    rather than spending attempts against a wall — DocVQA's own incident was a daily-cap refusal
    retried until 3,021 of 5,349 rows were empty (`OPENAI_EVAL_NOTES.md`).

    The returned meta carries BILLED_FIELDS on every path — what every HTTP call of this run of the
    row cost, beside the usage of the call that answered — and `reasoning_chars` summed over every
    response. An attempt that raised ends in exactly one call that returned nothing (the exception
    propagates from it), so it adds one call without usage; an attempt that returned but booked no
    response at all (a `once` that skips openai_meta) adds one too, since at least one call ran."""
    meta, last_error = {}, None
    ledger = {key: [] for key in _LEDGER_KEYS}
    ledger.update(responses=0, without_usage=0, reasoning_chars=0, think_tag_calls=0, call_open=False)
    _LEDGER.row = ledger
    _LEDGER.interrupted = None
    attempt = -1
    try:
        for attempt in range(tries):
            started = time.time()
            wait = 5.0
            booked_before = ledger["responses"]
            has_alarm = hasattr(signal, "SIGALRM")
            previous = signal.signal(signal.SIGALRM, _raise_timeout) if has_alarm else None
            try:
                if has_alarm:
                    signal.alarm(HARD_CALL_TIMEOUT)
                _start_call()
                try:
                    text, meta = once(messages)
                finally:
                    if has_alarm:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, previous)
                ledger["call_open"] = False
                if ledger["responses"] == booked_before:
                    ledger["without_usage"] += 1
                # latency_s is this successful attempt alone; n_attempts every call spent on the row.
                # Both are honest only because every client is built with max_retries=0
                # (script-skeleton.md §4).
                meta = dict(meta or {}, latency_s=round(time.time() - started, 2), n_attempts=attempt + 1)
                text = text.strip() if isinstance(text, str) else ""
                if text and meta.get("finish_reason") in INTERRUPTED_FINISH:
                    STATS["interrupted"] += 1
                    last_error = f"interrupted: finish_reason={meta.get('finish_reason')}"
                    print(f"[{model}] INTERRUPTED ({attempt + 1}/{tries}) "
                          f"finish_reason={meta.get('finish_reason')}: backend cut the answer off", flush=True)
                    text = ""
                elif text:
                    STATS["ok"] += 1
                    note_outcome(True, model, model_dir)
                    return text, dict(meta, **_billed(ledger))
                else:
                    last_error = f"empty response: finish_reason={meta.get('finish_reason')}"
                STATS["empty"] += 1
                print(f"[{model}] empty response ({attempt + 1}/{tries}) "
                      f"finish_reason={meta.get('finish_reason')} "
                      f"completion_tokens={meta.get('completion_tokens')}", flush=True)
            except CallTimeout as error:
                _close_failed_call(ledger)
                STATS["timeouts"] += 1
                last_error = f"timeout: {error}"
                print(f"[{model}] TIMEOUT ({attempt + 1}/{tries}): {error}", flush=True)
            except Exception as error:
                _close_failed_call(ledger)
                STATS["errors"] += 1
                last_error = f"{type(error).__name__}: {str(error)[:400]}"
                print(f"[{model}] API error ({attempt + 1}/{tries}): {type(error).__name__}: {error}",
                      flush=True)
                low = str(error).lower()
                for marker, why in fatal:
                    if marker in low:
                        _halt("FATAL", "FATAL_HALT.txt", f"{why}\n{type(error).__name__}: {error}",
                              model, model_dir)
                halt_on_billing(error, model, model_dir)
                if is_fatal_request(error):
                    _halt("MALFORMED REQUEST", "FAILURE_HALT.txt",
                          f"{type(error).__name__}: {error}", model, model_dir)
                wait = retry_delay(error, attempt)
            if attempt + 1 < tries:
                time.sleep(wait)
        STATS["failed_rows"] += 1
        note_outcome(False, model, model_dir)
        print(f"[{model}] all {tries} attempts failed; row written empty, retried on resume", flush=True)
        return "", dict(meta or {}, latency_s=None, n_attempts=tries, error=last_error, **_billed(ledger))
    except BaseException as stop:
        # A halt's SystemExit (billing, quota, fatal, malformed, failure) or SIGTERM's ends the row
        # mid-way. The calls already made are billed: stash the row's meta for run_sample to write it
        # as a failed row (folded on the next resume), then let the exit go on. A call still open is
        # one whose bill is unknown — _close_failed_call counts it, and only it, so a call an inner
        # handler already counted is not counted twice.
        _close_failed_call(ledger)
        _LEDGER.interrupted = dict(meta or {}, latency_s=None, n_attempts=attempt + 1,
                                   error=f"interrupted: {type(stop).__name__}: {stop}"[:400],
                                   halted=f"{type(stop).__name__}: {stop}"[:400],
                                   **_billed(ledger))
        raise
    finally:
        _LEDGER.row = None


def openai_meta(response):
    """What an OpenAI-compatible response says about itself — written onto every row, because a
    truncation, a reasoning bill or a backend switch is invisible in the text. Also books the
    response onto the row's ledger (`_book`), first, so the bill is kept whatever happens next."""
    usage = getattr(response, "usage", None)
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
    content = getattr(message, "content", None)
    think_tag = int(isinstance(content, str) and bool(THINK_TAG.search(content)))
    _book(usage, reasoning_chars, think_tag)
    details = getattr(usage, "completion_tokens_details", None)
    finish = getattr(choice, "finish_reason", None)
    finish = getattr(finish, "value", finish)

    def as_int(value):
        return value if isinstance(value, int) else None

    meta = {
        "finish_reason": None if finish is None else str(finish),
        "prompt_tokens": as_int(getattr(usage, "prompt_tokens", None)),
        "completion_tokens": as_int(getattr(usage, "completion_tokens", None)),
        "reasoning_tokens": as_int(getattr(details, "reasoning_tokens", None)),
        "served_model": str(getattr(response, "model", "") or "") or None,
        "reasoning_chars": reasoning_chars,       # this call's; call_with_retries puts the row's sum
        "think_tag_calls": think_tag,             # likewise
    }
    provider = getattr(response, "provider", None)       # OpenRouter's answering backend
    if provider:
        meta["backend"] = str(provider)
    # This call's bill only — the row's, every call summed, is billed_cost / billed_estimated_cost.
    for field, key in (("last_call_cost", "cost"), ("last_call_estimated_cost", "estimated_cost")):
        value = getattr(usage, key, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            meta[field] = value
    return meta


# ---------------------------------------------------------------- parameter negotiation
# plan_eval_core.negotiate, unchanged in behaviour: `fixed` is sent on every call and never dropped;
# `required` stops the run if negotiation loses the parameter OR swaps its value (model-parameters.md
# rules 1 and 7). The legacy `negotiate` above stays as it was for the legacy path.

SAMPLE_EFFORT_FALLBACKS = ("minimal", "low", "medium", "high")
_TRANSIENT = ("rate limit", "timed out", "timeout", "connection", "overloaded", "unavailable")
TRANSIENT_STATUS = ("429", "500", "502", "503", "504")     # whole words only: has_status


def negotiate_params(client, model, wanted, model_dir, fixed=None, required=None):
    params, notes = dict(wanted), []
    fixed = dict(fixed or {})
    transient_left = 3
    for _ in range(len(wanted) + 3 + transient_left):
        has_alarm = hasattr(signal, "SIGALRM")
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
            if offender is None and transient_left > 0 and (
                    any(x in err.lower() for x in _TRANSIENT) or has_status(err, TRANSIENT_STATUS)):
                transient_left -= 1
                time.sleep(retry_delay(error, 3 - transient_left))
                continue
            if offender is None:
                raise SystemExit(f"probe refused and no negotiable parameter is named: {err}")
            if "unsupported value" in err.lower() or "does not support" in err.lower():
                m = re.search(r"[Ss]upported values are:?\s*([^.}]+)", err)
                options = re.findall(r"'([^']+)'", m.group(1)) if m else []
                pick = next((v for v in SAMPLE_EFFORT_FALLBACKS if v in options),
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

CUT_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")


def rearm_call_timeout():
    """Give the next HTTP call inside one `once` its own HARD_CALL_TIMEOUT (plan_eval_core's)."""
    if (hasattr(signal, "SIGALRM") and HARD_CALL_TIMEOUT
            and signal.getsignal(signal.SIGALRM) is _raise_timeout):
        signal.alarm(HARD_CALL_TIMEOUT)


def _usage_sum(metas):
    out = {}
    for key in CUT_USAGE_KEYS:
        parts = [m.get(key) for m in metas if isinstance(m.get(key), int)]
        out[key] = sum(parts) if parts else None
    return out


class CutRetry:
    """plan_eval_core.CutRetry, ported — the user's rule of 2026-11-13 for a model that deliberates
    in its visible answer (Qwen, reasoning off): a reply cut at the cap is called again, fresh, told
    to stay inside the limit; cut again, what it wrote goes back as an assistant turn and the model
    is forced to answer with a small cap. Read the original's docstring for what each row carries
    (`cut_retry`, `first_*`, `retry_*`, `forced_*`); the behaviour is the same.

    Two differences, both noted at the top of this section: the ceiling is the FIRST message (there
    is no shared system message to follow), and `reasoning_chars` stays on the first and second
    call's meta instead of being popped — though the row's value is call_with_retries' sum over
    every response, stages and failed attempts included.

    `call(sent, stage) -> (answer, handback, meta)`, stage "first", "retry" or "forced". Finished
    calls of the row being worked on are kept by the identity of the messages list run_sample builds
    once per row, so a retry by call_with_retries repeats only the stage that failed."""

    def __init__(self, ceiling, retry_ceiling, force_prompt, answer_max):
        self.ceiling, self.retry_ceiling = ceiling, retry_ceiling
        self.force_prompt, self.answer_max = force_prompt, answer_max
        self._kept = {}

    @staticmethod
    def with_system(messages, text):
        """The ceiling as a system message ahead of the shared user message, which stays untouched."""
        return [{"role": "system", "content": text}] + list(messages)

    def config(self, **forced_extra):
        """The keys this adds to the runner's extra_config — on every row's `config`."""
        return {"prompt_ceiling": self.ceiling,
                "cut_retry": {"prompt": self.retry_ceiling, "when": "the first call ends at length"},
                "forced_answer": dict({"prompt": self.force_prompt, "max_tokens": self.answer_max,
                                       "when": "the second call also ends at length, with text to "
                                               "hand back"}, **forced_extra)}

    def __call__(self, messages, call, model=""):
        kept = self._kept
        if kept.get("messages") is not messages:
            kept.clear()
            kept["messages"] = messages
        if "first" not in kept:
            started = time.time()
            _start_call()
            answer, handback, meta = call(self.with_system(messages, self.ceiling), "first")
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
            _start_call()
            answer, handback, meta = call(retry_sent, "retry")
            retry_latency = round(time.time() - started, 2)
            if meta.get("finish_reason") != "length" or not handback.strip():
                out = dict(meta, **_usage_sum([first_meta, meta]), **marks, retry_latency_s=retry_latency)
                print(f"[{model}] retry: {meta.get('completion_tokens')} tokens, "
                      f"finish={meta.get('finish_reason')}", flush=True)
                return answer, out
            kept["retry"] = (answer, handback, meta, retry_latency)
            print(f"[{model}] retry reached the cap too ({meta.get('completion_tokens')} tokens): "
                  f"forced answer", flush=True)
            rearm_call_timeout()
        retry_text, retry_handback, retry_meta, retry_latency = kept["retry"]
        started = time.time()
        _start_call()
        answer, _, meta = call(retry_sent + [{"role": "assistant", "content": retry_handback},
                                             {"role": "user", "content": self.force_prompt}], "forced")
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


# ---------------------------------------------------------------- checkpoint and files


def _atomic_write(path, write):
    # host AND pid: array tasks on different nodes share this GPFS directory
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        write(fh)
    os.replace(tmp, path)


def _canonical(obj):
    return json.loads(json.dumps(obj, sort_keys=True))


def load_sample_checkpoint(path, config):
    """(done, failed), each keyed by questionId — or a refusal if the stored config differs.

    An EMPTY or interrupted row is not a done row: it is called again, or a plain resume would skip
    it forever. But it is not thrown away either: run_sample keeps it in the file until the new row
    replaces it, and the new row folds its history in (`fold_prior`). Dropped at load, a resubmit
    killed mid-row would delete it from the file, and one that answered would claim `n_attempts=1`
    and one call's bill for a row that had already cost five (reviewer, 2026-11-15)."""
    if not os.path.exists(path):
        return {}, {}
    want = _canonical(config)
    done, failed = {}, {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("config") != want:
                stored = r.get("config") or {}
                diff = {k: (want.get(k), stored.get(k)) for k in set(want) | set(stored)
                        if want.get(k) != stored.get(k)}
                raise SystemExit(f"{path}\n  refusing to resume: stored rows ran under a different "
                                 f"config.\n  (wanted, stored): {diff}\n  Archive the directory to a "
                                 f"timestamped name and start clean.")
            if (str(r.get("model_response", "")).strip()
                    and r.get("finish_reason") not in INTERRUPTED_FINISH):
                done[str(r["questionId"])] = r
            else:
                failed[str(r["questionId"])] = r
    if done or failed:
        print(f"  resume {os.path.basename(path)}: {len(done)} done, {len(failed)} failed row(s) to call "
              f"again (kept in the file until replaced)", flush=True)
    return done, failed


def _add(a, b):
    """None means no call reported it; a sum of two reports keeps ints as ints."""
    if a is None:
        return b
    if b is None:
        return a
    total = a + b
    return total if isinstance(total, int) else round(total, 10)


def fold_prior(rec, prior):
    """`rec`, this run's row, replaces `prior`, the stored failed row: BILLED_FIELDS become cumulative
    over every run of the row (prior's are already cumulative over the runs before it), `n_attempts`
    stays this run's, and RESUME_FIELDS say how much came before."""
    for key in BILLED_FIELDS:
        rec[key] = _add(prior.get(key), rec.get(key))
    rec["prior_n_attempts"] = (prior.get("prior_n_attempts") or 0) + (prior.get("n_attempts") or 0)
    rec["prior_failed_runs"] = (prior.get("prior_failed_runs") or 0) + 1
    rec["resumed_after_failure"] = True
    return rec


def save_sample_rows(path, rows):
    def write(fh):
        for r in sorted(rows, key=lambda x: x["idx"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _atomic_write(path, write)


def sample_row(row, text, meta, config, sample, params, prior=None):
    """One result row. Scored by the same `score_response` / `anls_score` as every other run here.
    `prior` is the stored failed row this one replaces, if any (`fold_prior`)."""
    ex = row["example"]
    gold = ex["answers"]
    if isinstance(gold, str):          # as `run` does — the JSON stores lists, but keep the guard
        gold = json.loads(gold.replace("'", '"'))
    meta = dict(meta or {})
    fa = extract_final_answer(text)
    rec = {"questionId": str(ex["questionId"]), "idx": row["idx"], "stratum": row["stratum"],
           "question": ex["question"], "gold_answers": json.dumps(gold, ensure_ascii=False),
           "image_path": ex["image_path"], "model_response": text, "final_answer": fa,
           "score": score_response(text, gold), "anls": anls_score(fa, gold),
           # parse_ok: the parser produced a non-empty prediction. `extract_final_answer` falls back
           # to the whole reply when "Final Answer:" is missing — which is how every DocVQA number
           # was scored — so `has_marker` is written beside it for the strict reading.
           "parse_ok": bool(fa.strip()),
           "has_marker": bool(re.search(r"Final Answer:", text or "", re.IGNORECASE)),
           "empty": int(not text), "error": meta.pop("error", None)}
    # `halted`: None, or the stop that interrupted this row's call ("SystemExit: 143") on the failed
    # row written when a run stops mid-call — the same name as in the MMLU and BBH cores.
    for key in (*USAGE_FIELDS, *BILLED_FIELDS, "reasoning_chars", "think_tag_calls", "halted"):
        rec[key] = meta.pop(key, None)
    rec.update(meta)
    rec.update(prior_n_attempts=0, prior_failed_runs=0, resumed_after_failure=False)
    rec.update(seed=params.get("seed"), sample=sample, scorer=SCORER_VERSION,
               ts=time.strftime("%Y-%m-%dT%H:%M:%S"), config=config)
    return fold_prior(rec, prior) if prior else rec


def shows_thinking(r):
    """Reasoning text on any response of the row, a think tag inline in any reply's content, or
    reasoning tokens reported for it: thinking was on."""
    return ((r.get("reasoning_chars") or 0) > 0 or (r.get("think_tag_calls") or 0) > 0
            or (r.get("reasoning_tokens") or 0) > 0 or (r.get("billed_reasoning_tokens") or 0) > 0)


def _median(values):
    return round(statistics.median(values), 2) if values else ""


def write_sample_files(out_dir, model, records, tag=""):
    """`<slug><tag>.csv` beside the .jsonl, and `<slug><tag>_overall.csv`: one line per stratum plus
    ALL. Sample-level numbers, NOT scaled to the benchmark — that is sample10_report.py's job."""
    slug = model_slug(model) + tag
    ordered = sorted(records, key=lambda r: r["idx"])

    def write_rows(fh):
        w = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ordered:
            w.writerow({k: (json.dumps(r[k], sort_keys=True) if k == "config"
                            else "" if r.get(k) is None else r.get(k)) for k in SAMPLE_FIELDS})
    _atomic_write(os.path.join(out_dir, f"{slug}.csv"), write_rows)

    groups = collections.defaultdict(list)
    for r in ordered:
        groups[r["stratum"]].append(r)
    groups = dict(sorted(groups.items()))
    groups["ALL"] = ordered

    def total(rs, key):
        vals = [r.get(key) for r in rs
                if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
        return (sum(vals) if all(isinstance(v, int) for v in vals) else round(sum(vals), 10)) if vals else ""

    def write_overall(fh):
        w = csv.writer(fh)
        w.writerow(["model", "stratum", "n", "accuracy", "anls", "empty", "no_marker", "parse_fail",
                    "finish_length", "cut_retry", "forced_answer", "retried_rows", "prompt_tokens",
                    "completion_tokens", "reasoning_tokens", *BILLED_FIELDS, "reasoning_chars_rows",
                    "thinking_rows", "resumed_rows", "latency_median_s", "scorer"])
        for name, rs in groups.items():
            n = len(rs)
            w.writerow([model, name, n,
                        round(sum(r["score"] for r in rs) / n, 4) if n else "",
                        round(sum(r["anls"] for r in rs) / n, 4) if n else "",
                        sum(r["empty"] for r in rs),
                        sum(1 for r in rs if not r["has_marker"]),
                        sum(1 for r in rs if not r["parse_ok"]),
                        sum(1 for r in rs if r.get("finish_reason") == "length"),
                        sum(1 for r in rs if r.get("cut_retry")),
                        sum(1 for r in rs if r.get("forced_answer")),
                        sum(1 for r in rs if (r.get("n_attempts") or 0) > 1),
                        total(rs, "prompt_tokens"), total(rs, "completion_tokens"),
                        total(rs, "reasoning_tokens"),
                        *(total(rs, key) for key in BILLED_FIELDS),
                        sum(1 for r in rs if (r.get("reasoning_chars") or 0) > 0),
                        sum(1 for r in rs if shows_thinking(r)),
                        sum(1 for r in rs if r.get("resumed_after_failure")),
                        _median([r["latency_s"] for r in rs if isinstance(r.get("latency_s"), (int, float))]),
                        SCORER_VERSION])
    _atomic_write(os.path.join(out_dir, f"{slug}_overall.csv"), write_overall)
    n = len(ordered)
    return {"n": n, "empty": sum(r["empty"] for r in ordered),
            "reasoning_chars_rows": sum(1 for r in ordered if (r.get("reasoning_chars") or 0) > 0),
            "thinking_rows": sum(1 for r in ordered if shows_thinking(r)),
            "accuracy": round(sum(r["score"] for r in ordered) / n, 4) if n else "",
            "anls": round(sum(r["anls"] for r in ordered) / n, 4) if n else ""}


# ---------------------------------------------------------------- the sample run


def positive_int(text):
    """argparse type: an integer >= 1. `--save-every 0` would divide by zero mid-run, after paying for
    the first row."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def check_sample_args(ap, args):
    """For a runner whose parser also serves the legacy path (the GPT runner), where `--save-every 0`
    stays legal without --sample: the same refusal, still at parse time, through `ap.error`."""
    if args.save_every < 1:
        ap.error(f"argument --save-every: must be 1 or more with --sample, got {args.save_every}")
    if args.limit < 0:
        ap.error(f"argument --limit: must be 0 or more, got {args.limit}")


def add_sample_args(ap, required=False):
    ap.add_argument("--sample", choices=sorted(SAMPLES), required=required, default=None,
                    help="run only the questions of <DocVQA>/<sample>.json (sample10: 539, stratified "
                         "by question_types[0]), written to results_<sample>/ (smoke_<sample>/ under "
                         "--limit) — never results/")


def sample_parser(description, default_model, default_sleep=2.0):
    """The CLI of a runner that exists only for the sample: --sample is required."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default=default_model,
                    help="model id; it is BOTH what is called and what the result files are named after")
    add_sample_args(ap, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N sample questions, written to smoke_<sample>/ — a smoke test, not a run")
    ap.add_argument("--sleep", type=float, default=default_sleep,
                    help="seconds after every successful call")
    ap.add_argument("--save-every", dest="save_every", type=positive_int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="at 1 the filenames carry no shard tag. Shards share one key's quota "
                         "(docvqa.md trap 1); 539 questions do not need them")
    return ap


def run_sample(args, *, model_dir, client, once, wanted, route, fixed=None, required=None,
               call_timeout=300, extra_config=None, fatal=(), forbid_reasoning=False):
    """`once(messages, model, params) -> (text, meta)`. Ends with SystemExit, every applicable message
    printed first and the code the most serious of them: 3 when `forbid_reasoning` and a row carries
    reasoning text or reasoning tokens (`shows_thinking`: thinking came back on — a config failure,
    which no resubmit fixes; the loop stops at the first such row, writing THINKING_HALT.txt), 1 when
    any row is empty (resubmit: resume calls exactly those, folding
    each one's earlier runs into its new row), 2 when a forced answer filled its own cap or reasoned
    again (read it before scoring), else 0."""
    global HARD_CALL_TIMEOUT, MARKER_PREFIX, SAMPLE_OUT_DIR
    sample, model, smoke = args.sample, args.model, bool(args.limit)
    if args.total_shards < 1 or not 0 <= args.shard < args.total_shards:
        raise SystemExit(f"--shard {args.shard} is outside --total-shards {args.total_shards}")
    if args.save_every < 1:                # parsers refuse it first; this guards a direct caller
        raise SystemExit(f"--save-every must be 1 or more, got {args.save_every}")
    _, rows = load_sample(sample)
    missing = [r["example"]["image_path"] for r in rows
               if not os.path.exists(image_file(r["example"]["image_path"]))]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} sample images do not resolve from "
                         f"{DOCVQA_ROOT} (first: {missing[0]}) — no call made")
    rows = rows[args.shard::args.total_shards]          # interleaved: every shard gets every stratum
    if smoke:
        rows = rows[:args.limit]

    HARD_CALL_TIMEOUT = int(call_timeout)
    # smoke_sample10_ for a --limit run, so a smoke can never clear, or be mistaken for, the real
    # run's marker (reviewer, 2026-11-15).
    MARKER_PREFIX = (("smoke_" if smoke else "") + f"{sample}_"
                     + (f"shard{args.shard}_" if args.total_shards > 1 else ""))
    SAMPLE_OUT_DIR = os.path.basename(sample_dir(model_dir, sample, smoke)) + "/"
    # SLURM ends a job at its wall-clock limit (and on scancel) with SIGTERM, whose default kills the
    # process WITHOUT running `finally`; as SystemExit, the final save below still happens.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    clear_stale_halt_markers(model_dir)

    print(f"negotiating parameter surface for {model} ...", flush=True)
    params, notes = negotiate_params(client, model, wanted, model_dir, fixed=fixed, required=required)
    print(f"  accepted: {params}" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    if "seed" in wanted and "seed" not in params:
        print("  WARNING: seed was dropped — this run is not reproducible", flush=True)
    # The run config, on every row; a resume across any change of it is refused.
    config = _canonical(dict(extra_config or {}, model=model, route=route, params=params,
                             fixed=fixed or {}, prompt_sha1=PROMPT_SHA1, message=MESSAGE_LAYOUT,
                             sample=sample))

    out_dir = sample_dir(model_dir, sample, smoke)
    os.makedirs(out_dir, exist_ok=True)
    tag = shard_tag(args.shard, args.total_shards)
    path = os.path.join(out_dir, f"{model_slug(model)}{tag}.jsonl")
    _atomic_write(os.path.join(out_dir, f"{model_slug(model)}_negotiated_params.json"),
                  lambda fh: json.dump({"asked": wanted, "accepted": params, "notes": notes,
                                        "config": config}, fh, indent=2))
    done, failed = ({}, {}) if smoke else load_sample_checkpoint(path, config)
    mine = {str(r["example"]["questionId"]) for r in rows}
    foreign = (set(done) | set(failed)) - mine
    if foreign:
        raise SystemExit(f"{path}\n  holds {len(foreign)} row(s) not in this shard (first: "
                         f"{sorted(foreign)[0]}) — written under other shard boundaries. Saving would "
                         f"drop them; archive the file and look at it first.")
    records = [done[str(r["example"]["questionId"])] for r in rows
               if str(r["example"]["questionId"]) in done]
    todo = [r for r in rows if str(r["example"]["questionId"]) not in done]
    # A stored failed row stays in the file until its new row replaces it: a resubmit killed (or
    # halted) before reaching it, or in the middle of it, must not delete it.
    held = {q: failed[q] for q in (str(r["example"]["questionId"]) for r in todo) if q in failed}

    def to_save():
        # A held row whose new row is already in `records` is not written twice.
        replaced = {r["questionId"] for r in records}
        return records + [r for q, r in held.items() if q not in replaced]
    stored_thinking = [r for r in to_save() if shows_thinking(r)] if forbid_reasoning else []
    if stored_thinking:
        # A resubmit after a THINKING halt, same config: the stored row is the evidence already. Stop
        # before calling anything, rather than run the rest of the shard on it. (A fixed runner has a
        # different config, which load_sample_checkpoint refuses to resume.)
        thinking_halt(stored_thinking[0], model, model_dir)
    print(f"{len(rows)} sample rows in this {'smoke test' if smoke else 'shard'}, {len(todo)} to call "
          f"-> {path}", flush=True)
    started = time.time()
    inflight = {}                                      # the row being called, until it is in records
    try:
        for n, row in enumerate(todo, 1):
            qid = str(row["example"]["questionId"])
            inflight = {"row": row, "qid": qid}
            take_interrupted()                             # nothing stale can attach to this row
            messages = build_messages(row["example"])          # built once per row: CutRetry keys on it
            text, meta = call_with_retries(lambda m: once(m, model, params), messages, model,
                                           model_dir, fatal=fatal)
            inflight.update(text=text, meta=meta)
            # get, append, THEN pop: an exit between any two of these must not lose the stored row
            rec = sample_row(row, text, meta, config, sample, params, prior=held.get(qid))
            records.append(rec)
            held.pop(qid, None)
            inflight = {}
            if forbid_reasoning and shows_thinking(rec):
                thinking_halt(rec, model, model_dir)           # exit 3 now; `finally` saves this row
            if n % args.save_every == 0:
                save_sample_rows(path, to_save())
                print(f"  {len(records)}/{len(rows)}  [{row['stratum']}]", flush=True)
            if text and args.sleep:
                time.sleep(args.sleep)
    finally:
        # A halt, or SIGTERM at the wall clock, must not lose the rows paid for since the last save.
        # A second SIGTERM (a wrapper forwarding one, scancel twice) would interrupt this save itself
        # and leave only a .tmp file — seen on a local smoke, 2026-11-15 — so ignore it from here.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # The row in flight when a halt or SIGTERM struck: its calls are billed, so it is written —
        # as answered if its call had returned, else failed from call_with_retries' stash — folding
        # the stored failed row it replaces, if any. A resume folds it again in turn.
        if inflight and inflight["qid"] not in {r["questionId"] for r in records}:
            try:
                got = ((inflight["text"], inflight["meta"]) if "meta" in inflight
                       else ("", take_interrupted()))
                if got[1] is not None:
                    records.append(sample_row(inflight["row"], got[0], got[1], config, sample, params,
                                              prior=held.get(inflight["qid"])))
                    held.pop(inflight["qid"], None)
            except BaseException as error:             # the save below must happen regardless
                print(f"[{model}] could not write the row in flight ({inflight['qid']}): "
                      f"{type(error).__name__}: {error}; its stored row, if any, is kept", flush=True)
        save_sample_rows(path, to_save())
        summary = write_sample_files(out_dir, model, to_save(), tag)
        print(f"\ncalls: {STATS}  elapsed {time.time() - started:.0f}s\n"
              f"rows: n={summary['n']} empty={summary['empty']} acc={summary['accuracy']} "
              f"anls={summary['anls']} reasoning_chars>0: {summary['reasoning_chars_rows']} "
              f"thinking (reasoning text, think tags or tokens): {summary['thinking_rows']}", flush=True)
    final = to_save()
    codes = []
    # A backstop: the loop halts at the first such row it calls, and a stored one halts the run before
    # its first call; nothing should reach here.
    thinking = [r for r in final if shows_thinking(r)]
    if forbid_reasoning and thinking:
        write_thinking_marker(thinking[0], model, model_dir)
        print(f"{len(thinking)} of {len(final)} row(s) carry reasoning text or reasoning tokens; fix the "
              f"runner, archive this directory, rerun.", flush=True)
        codes.append(3)
    bad = [r for r in final if r["empty"]]
    if bad:
        print(f"\n{len(bad)} row(s) of this shard are empty — resubmit: resume calls exactly those. "
              f"First: {bad[0]['questionId']} ({bad[0].get('error')})", flush=True)
        codes.append(1)
    odd = [r for r in final if r.get("forced_answer")
           and (r.get("finish_reason") == "length" or r.get("forced_reasoning_chars"))]
    if odd:
        print(f"\nSTOP: {len(odd)} forced answer(s) filled their own cap or reasoned again — read "
              f"them before scoring. First: {odd[0]['questionId']}", flush=True)
        codes.append(2)
    raise SystemExit(codes[0] if codes else 0)


def sample_main(*, description, default_model, model_dir, client, once, wanted, route, fixed=None,
                required=None, call_timeout=300, extra_config=None, fatal=(), default_sleep=2.0,
                forbid_reasoning=False):
    """Entry point of a runner that exists only for the sample."""
    ap = sample_parser(description, default_model, default_sleep)
    args = ap.parse_args()
    check_sample_args(ap, args)
    run_sample(args, model_dir=model_dir, client=client, once=once, wanted=wanted, route=route,
               fixed=fixed, required=required, call_timeout=call_timeout,
               extra_config=extra_config, fatal=fatal, forbid_reasoning=forbid_reasoning)
