"""Shared generation core for every PlanBench runner — `plan-bench/prompts`, 53 prompt files, 15,568 rows.

A runner holds its client, its model id and the parameters `model-parameters.md` requires of that
model. Everything else lives here, so it cannot differ between models: which rows are sent, the
messages, the retry/timeout/billing harness, the checkpoint, sharding and the generation-health
summary.

**This file makes calls and never scores.** Scoring needs the VAL validator and `tarski`, neither of
which is on Quest's shared Python env, so it is `plan_scorer.py`, run offline on stored rows. Every
row keeps `raw_response` verbatim — a scorer change is a rescore, never a rerun.

Read before changing anything: `.claude/references/benchmarks/transition/planbench.md`. The harness
half (billing/quota halts, `call_with_retries`, `negotiate`, checkpoint, shards, atomic writes) is
ported from `Awareness_in_LLM/aware_eval_core.py`, itself from `NegotiationToM/neg_eval_core.py` and
`mmlu/mmlu_eval_core.py`; each piece is cited where it is reused.
"""

import argparse
import collections
import csv
import filecmp
import glob
import hashlib
import json
import os
import random
import re
import shutil
import signal
import socket
import statistics
import sys
import time

PB_ROOT = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(PB_ROOT, "plan-bench")
PROMPTS_DIR = os.path.join(UPSTREAM, "prompts")
ENV_PATH = os.path.join(PB_ROOT, ".env")

# Upstream's own message for chat models (plan-bench/utils/llm_utils.py, the `_chat` branch of
# send_query), sent unchanged. The user message is the shipped `query`, byte for byte. Upstream sent
# no `stop` to chat models, and neither do we: `[PLAN END]` is cut by the scorer, not the API.
SYSTEM_PROMPT = "You are the planner assistant who comes up with correct plans."

# One slice = one shipped prompt file, `<domain>/<task>`. (full, pilot10) rows, measured 2026-11-09
# from plan-bench/prompts. pilot10 = every 10th instance by sorted instance_id, i.e. ceil(n/10).
# A run producing other counts has a bug in sampling, sharding or resume.
EXPECTED = {
    "blocksworld/task_1_plan_generation": (500, 50),
    "blocksworld/task_2_plan_optimality": (500, 50),
    "blocksworld/task_3_plan_verification": (500, 50),
    "blocksworld/task_4_plan_reuse": (500, 50),
    "blocksworld/task_5_plan_generalization": (500, 50),
    "blocksworld/task_6_replanning": (500, 50),
    "blocksworld/task_7_plan_execution": (500, 50),
    "blocksworld/task_8_1_goal_shuffling": (500, 50),
    "blocksworld/task_8_2_full_to_partial": (500, 50),
    "blocksworld/task_8_3_partial_to_full": (500, 50),
    "blocksworld_3/task_1_plan_generation": (100, 10),
    "blocksworld_3/task_2_plan_optimality": (100, 10),
    "blocksworld_3/task_3_plan_verification": (100, 10),
    "blocksworld_3/task_3_plan_verification_with_llm_plans": (99, 10),
    "blocksworld_3/task_4_plan_reuse": (100, 10),
    "blocksworld_3/task_5_plan_generalization": (100, 10),
    "blocksworld_3/task_6_replanning": (100, 10),
    "blocksworld_3/task_7_plan_execution": (100, 10),
    "blocksworld_3/task_8_1_goal_shuffling": (100, 10),
    "blocksworld_3/task_8_2_full_to_partial": (100, 10),
    "blocksworld_3/task_8_3_partial_to_full": (100, 10),
    "depots/task_1_plan_generation": (500, 50),
    "logistics/task_1_plan_generation": (285, 29),
    "logistics/task_2_plan_optimality": (285, 29),
    "logistics/task_3_plan_verification": (285, 29),
    "logistics/task_4_plan_reuse": (285, 29),
    "logistics/task_5_plan_generalization": (12, 2),
    "logistics/task_6_replanning": (285, 29),
    "logistics/task_7_plan_execution": (285, 29),
    "logistics/task_8_1_goal_shuffling": (285, 29),
    "logistics/task_8_2_full_to_partial": (285, 29),
    "logistics/task_8_3_partial_to_full": (285, 29),
    "mystery_blocksworld/task_1_plan_generation": (500, 50),
    "mystery_blocksworld/task_2_plan_optimality": (500, 50),
    "mystery_blocksworld/task_3_plan_verification": (500, 50),
    "mystery_blocksworld/task_4_plan_reuse": (500, 50),
    "mystery_blocksworld/task_5_plan_generalization": (500, 50),
    "mystery_blocksworld/task_6_replanning": (500, 50),
    "mystery_blocksworld/task_7_plan_execution": (500, 50),
    "mystery_blocksworld/task_8_1_goal_shuffling": (500, 50),
    "mystery_blocksworld/task_8_2_full_to_partial": (500, 50),
    "mystery_blocksworld/task_8_3_partial_to_full": (500, 50),
    "mystery_blocksworld_3/task_1_plan_generation": (100, 10),
    "mystery_blocksworld_3/task_2_plan_optimality": (100, 10),
    "mystery_blocksworld_3/task_3_plan_verification": (100, 10),
    "mystery_blocksworld_3/task_4_plan_reuse": (100, 10),
    "mystery_blocksworld_3/task_6_replanning": (100, 10),
    "mystery_blocksworld_3/task_7_plan_execution": (100, 10),
    "mystery_blocksworld_3/task_8_1_goal_shuffling": (100, 10),
    "mystery_blocksworld_3/task_8_2_full_to_partial": (100, 10),
    "mystery_blocksworld_3/task_8_3_partial_to_full": (100, 10),
    "obfuscated_deceptive_logistics/task_1_plan_generation": (285, 29),
    "obfuscated_deceptive_logistics/task_2_plan_optimality": (207, 21),
}
SLICES = list(EXPECTED)
# Every k-th instance by sorted instance_id. The sample decides which rows are called, not how, so
# it is not part of the run config; it decides the output directory instead (results_dir).
SAMPLES = {"full": 1, "pilot10": 10}
SAMPLE_COLUMN = {"full": 0, "pilot10": 1}
# Plan-producing tasks: a usable answer ends its plan with [PLAN END], as every in-context example
# does. Reported per slice as generation health; the scorer does not require it.
PLAN_TASKS = ("task_1", "task_2", "task_4", "task_5", "task_6", "task_8")


def model_slug(model_id):
    return model_id.replace("/", "-").replace(".", "_")


def is_plan_task(task):
    return task.startswith(PLAN_TASKS)


# ---------------------------------------------------------------- the dataset


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_rows(sample, slices=None):
    """Rows of the chosen sample, in (slice, instance_id) order, `idx` assigned over the WHOLE
    sample before any shard or slice filter — so shards and filtered runs never reuse an idx.
    The query itself is not stored on the row (15,568 prompts of 2-6 KB each would bloat every
    result file); `query_sha1` pins it, and the scorer reloads it from the prompt file."""
    step = SAMPLES[sample]
    on_disk = sorted(os.path.relpath(p, PROMPTS_DIR)[:-5]
                     for p in glob.glob(os.path.join(PROMPTS_DIR, "*", "*.json")))
    if on_disk != sorted(SLICES):
        raise SystemExit(f"prompt files differ from EXPECTED: extra {sorted(set(on_disk) - set(SLICES))}, "
                         f"missing {sorted(set(SLICES) - set(on_disk))}")
    rows, idx = [], 0
    for sl in SLICES:
        domain, task = sl.split("/")
        with open(os.path.join(PROMPTS_DIR, domain, f"{task}.json")) as fh:
            instances = sorted(json.load(fh)["instances"], key=lambda r: r["instance_id"])
        ids = [r["instance_id"] for r in instances]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"{sl}: duplicate instance_id in the shipped prompts")
        chosen = instances[::step]
        if len(chosen) != EXPECTED[sl][SAMPLE_COLUMN[sample]]:
            raise SystemExit(f"{sl}: {len(chosen)} rows in sample {sample}, expected "
                             f"{EXPECTED[sl][SAMPLE_COLUMN[sample]]}")
        for inst in chosen:
            rows.append({"uid": f"{domain}|{task}|{inst['instance_id']}", "idx": idx, "slice": sl,
                         "domain": domain, "task": task, "instance_id": inst["instance_id"],
                         "query": inst["query"], "query_sha1": _sha1(inst["query"])})
            idx += 1
    if slices:
        rows = [r for r in rows if any(r["slice"] == s or r["domain"] == s for s in slices)]
    return rows


def build_messages(row):
    """The shared messages, identical for every model. A model-specific addition (DeepSeek's prompt
    ceiling) is made in that model's runner, never here."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["query"]}]


# ---------------------------------------------------------------- refusals that must stop a run
# Ported from aware_eval_core.py (and before it NegotiationToM/neg_eval_core.py, where each signature
# is explained by the incident that added it). Only wordings that unambiguously mean money.

BILLING_SIGNATURES = (
    "insufficient_quota", "insufficient balance", "insufficient credits", "out of credits",
    "monthly budget", "used all available credits", "spending limit", "credit balance",
    "payment required", "arrears",
    # OpenRouter's out-of-credit 402 wordings (neg_eval_core.py, judge_core.py; provider-gotchas.md
    # § Classifying a refusal). Without them Gemini running dry retries ~2 minutes a row and stops
    # as FAILURE_HALT, not BILLING_HALT.
    "requires more credits", "can only afford", "openrouter_credits",
)
# A backend that cut the answer off — not the model, not the cap. OpenRouter normalises an upstream
# failure to finish_reason "error" on HTTP 200 with the fragment as content (Gemini, MultiChallenge
# full run, 2026-11-01); DeepSeek documents "insufficient_system_resource". Judge_core.INTERRUPTED_FINISH.
# Such a reply is a failed attempt, never an answer; "length" is not here — that is the model hitting
# our cap, a measurement the health files report (`finish_length`).
INTERRUPTED_FINISH = ("error", "insufficient_system_resource")
DAILY_QUOTA_SIGNATURES = ("per_model_per_day", "requests per day", "requests_per_day", "per day",
                          "daily limit", "daily quota")
DAILY_HALT_THRESHOLD_SECONDS = 600
FATAL_SIGNATURES = ("invalid_argument", "extra inputs are not permitted", "validationerror",
                    "unsupported parameter", "unsupported value", "unrecognized request argument",
                    "unexpected keyword argument")
AUTH_MARKERS = ("authentication", "api key", "invalid_api_key", "incorrect api key", "401")

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
    return any(sig in text for sig in FATAL_SIGNATURES) or any(m in text for m in AUTH_MARKERS)


STATS = {"ok": 0, "empty": 0, "interrupted": 0, "errors": 0, "timeouts": 0, "failed_rows": 0}
HALT_MARKERS = ("BILLING_HALT.txt", "QUOTA_HALT.txt", "FAILURE_HALT.txt")
# Tagged by shard (`shard2_FAILURE_HALT.txt`) when there is more than one: untagged, a shard starting
# late — or one resubmitted alone — would delete another shard's marker at startup. Every marker
# still matches *_HALT.txt. Set by run_cli.
MARKER_PREFIX = ""


def _halt(kind, marker_name, detail, model, model_dir):
    """Write a marker an agent can find without reading logs, say why on both streams, exit."""
    marker = os.path.join(model_dir, MARKER_PREFIX + marker_name)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    spoiled = STATS["empty"] + STATS["errors"] + STATS["timeouts"]
    advice = ("Rows written empty are retried on resume (load_checkpoint skips only non-empty "
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


# Consecutive failures catch a provider that stops serving; the rolling window catches one that
# fails intermittently — the shape that fills a full-length result set with empties.
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


# Hard ceiling on one HTTP attempt, by SIGALRM — a backstop for a hung connection; work itself is
# bounded by the token cap. Runners set it per provider through run_cli(call_timeout=...).
HARD_CALL_TIMEOUT = 300


def _raise_timeout(signum, frame):
    raise CallTimeout(f"call exceeded {HARD_CALL_TIMEOUT}s hard limit")


def retry_delay(error, attempt, default=5.0):
    """Provider's hint as a floor, exponential growth, jitter so shards do not retry in lockstep,
    capped at a quarter of the watchdog (aware_eval_core.retry_delay)."""
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


def call_with_retries(once, messages, model, model_dir, tries=5):
    """`once(messages) -> (text, meta)`. Retries an exception, a timeout AND an empty string — HTTP
    200 with an empty body raises nothing and is this project's most common failure. Returns
    ("", meta) when every attempt failed; the row is then written empty and retried on resume."""
    meta = {}
    for attempt in range(tries):
        started = time.time()
        wait = 5.0
        has_alarm = hasattr(signal, "SIGALRM")
        previous = signal.signal(signal.SIGALRM, _raise_timeout) if has_alarm else None
        try:
            if has_alarm:
                signal.alarm(HARD_CALL_TIMEOUT)
            try:
                text, meta = once(messages)
            finally:
                if has_alarm:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, previous)
            # latency_s is this successful attempt alone; n_attempts counts every call spent on the
            # row. Both are honest only because every client is built with max_retries=0
            # (script-skeleton.md §4).
            meta = dict(meta or {}, latency_s=round(time.time() - started, 2), n_attempts=attempt + 1)
            text = text.strip() if isinstance(text, str) else ""
            if text and meta.get("finish_reason") in INTERRUPTED_FINISH:
                # A fragment on HTTP 200: non-empty, so every check below would accept it. Retried in
                # a branch of its own, never through the halt classifiers; if every attempt ends this
                # way the row is written EMPTY (retried on resume), never as a scored fragment.
                STATS["interrupted"] += 1
                print(f"[{model}] INTERRUPTED ({attempt + 1}/{tries}) "
                      f"finish_reason={meta.get('finish_reason')}: backend cut the answer off", flush=True)
                text = ""
            elif text:
                STATS["ok"] += 1
                note_outcome(True, model, model_dir)
                return text, meta
            STATS["empty"] += 1
            print(f"[{model}] empty response ({attempt + 1}/{tries}) "
                  f"finish_reason={meta.get('finish_reason')} "
                  f"completion_tokens={meta.get('completion_tokens')}", flush=True)
        except CallTimeout as error:
            STATS["timeouts"] += 1
            print(f"[{model}] TIMEOUT ({attempt + 1}/{tries}): {error}", flush=True)
        except Exception as error:
            STATS["errors"] += 1
            print(f"[{model}] API error ({attempt + 1}/{tries}): {type(error).__name__}: {error}",
                  flush=True)
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
    return "", dict(meta or {}, latency_s=None, n_attempts=tries)


def openai_meta(response):
    """What an OpenAI-compatible response says about itself — written onto every row, because a
    truncation, a reasoning bill or a backend switch is invisible in the text."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    choice = response.choices[0] if getattr(response, "choices", None) else None
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
    }
    provider = getattr(response, "provider", None)       # OpenRouter's answering backend
    if provider:
        meta["backend"] = str(provider)
    cost = getattr(usage, "cost", None)                  # OpenRouter's per-call bill
    if isinstance(cost, (int, float)):
        meta["cost"] = cost
    return meta


# ---------------------------------------------------------------- parameter negotiation
# aware_eval_core.negotiate, unchanged in behaviour: `fixed` is sent on every call and never dropped;
# `required` ({parameter: value}) stops the run if negotiation loses the parameter OR swaps its value
# (model-parameters.md rules 1 and 7).

EFFORT_FALLBACKS = ("minimal", "low", "medium", "high")
_TRANSIENT = ("rate limit", "429", "500", "502", "503", "504", "timed out", "timeout",
              "connection", "overloaded", "unavailable")


def negotiate(client, model, wanted, model_dir, fixed=None, required=None, probe_kwargs=None):
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
                    **(probe_kwargs or {}), **fixed, **params)
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
            if any(m in err.lower() for m in AUTH_MARKERS):
                raise SystemExit(f"authentication failure, not negotiated around: {err}")
            offender = next((k for k in params if f"'{k}'" in err or f'"{k}"' in err
                             or f"`{k}`" in err), None)
            if offender is None and transient_left > 0 and any(x in err.lower() for x in _TRANSIENT):
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


# ---------------------------------------------------------------- checkpoint, shards, files


def shard_tag(shard, total_shards):
    """`_shard2of5`, EMPTY at one shard — so an unsharded run keeps its plain filename."""
    return f"_shard{shard}of{total_shards}" if total_shards > 1 else ""


def shard_slice(rows, shard, total_shards, interleave=False):
    """Which of the idx-ordered rows one shard calls; idx was assigned over the whole sample first.

    The pilot's shards are contiguous blocks, and stay so: its stored shard files (and `--force-cut`
    on them) depend on that. The full set is INTERLEAVED — every `total_shards`-th row — because
    contiguous blocks put whole domains in one shard: in the pilot the mystery_blocksworld shards
    took 4-6x the others, so at 10x the rows the longest Qwen shard would have been ~131 h against
    ~88 h interleaved (Gemma 41 h against 22 h), the other shards' streams idle meanwhile — both
    before CutRetry's second and third calls, which in Qwen's pilot would have added up to ~60%."""
    if total_shards <= 1:
        return list(rows)
    if interleave:
        return list(rows)[shard::total_shards]
    size = (len(rows) + total_shards - 1) // total_shards
    return list(rows)[shard * size: min((shard + 1) * size, len(rows))]


def results_dir(model_dir, sample, smoke=False):
    """smoke/ for a --limit run (never results: a smoke row would be resumed past by the real run);
    results_pilot10/ for the 10% pilot; results/ for the full set. The pilot is kept apart so its
    audit is unambiguous and the full run starts from a clean checkpoint."""
    if smoke:
        return os.path.join(model_dir, "smoke")
    return os.path.join(model_dir, "results" if sample == "full" else f"results_{sample}")


def rows_path(out_dir, model, tag=""):
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{model_slug(model)}{tag}.jsonl")


def _atomic_write(path, write):
    # host AND pid: array tasks on different nodes share this GPFS directory
    tmp = f"{path}.tmp.{socket.gethostname()}.{os.getpid()}"
    with open(tmp, "w", newline="") as fh:
        write(fh)
    os.replace(tmp, path)


def save_rows(path, rows):
    def write(fh):
        for r in sorted(rows, key=lambda x: x["idx"]):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    _atomic_write(path, write)


def forced_reasoned(row):
    """A forced answer whose call reasoned again — the one thing a forced answer may not do."""
    return bool(row.get("forced_answer") and (row.get("forced_reasoning_chars")
                                              or row.get("forced_reasoning_tokens")))


def is_cut(row):
    """A stored row the output cap cut off — answered or empty — that no forced answer replaced."""
    return row.get("finish_reason") == "length" and not row.get("forced_answer")


# ---------------------------------------------------------------- a reply cut at the output cap

# The config keys the cut-at-the-cap handling adds. `--force-cut` carries a stored row written
# without them under the config that has them (load_checkpoint).
CUT_KEYS = ("forced_answer", "cut_retry")
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens")


def rearm_call_timeout():
    """Give the next HTTP call inside one `once` its own HARD_CALL_TIMEOUT. call_with_retries arms
    the alarm once per attempt and clears it when `once` returns, so a re-arm cannot outlive it.
    Only while call_with_retries' handler is installed: a probe calling `once` directly would get an
    alarm under SIGALRM's default handler, which kills the process."""
    if (hasattr(signal, "SIGALRM") and HARD_CALL_TIMEOUT
            and signal.getsignal(signal.SIGALRM) is _raise_timeout):
        signal.alarm(HARD_CALL_TIMEOUT)


def _usage_sum(metas):
    out = {}
    for key in USAGE_KEYS:
        parts = [m.get(key) for m in metas if isinstance(m.get(key), int)]
        out[key] = sum(parts) if parts else None
    return out


class CutRetry:
    """What a row whose reply hits the output cap gets — the user, 2026-11-13: "对于那种就是跑一次直接
    撞到上限的问题…让他第二次重新跑的时候，不要超出这个思考的上限。…如果说他就是不受约束，那就直接回答".

    1. **The call as configured**, with the runner's prompt ceiling as a second system message. A
       reply that ends any way but `length` is the row's answer, as on every other model.
    2. **Cut at the cap: called again, fresh**, with `retry_ceiling` in place of the ceiling — the
       model is told its first attempt overran the limit and must stay inside it. A reply that ends
       any way but `length` is the answer. The row carries `cut_retry: true` and `first_*`, among
       them `first_response`, the cut call's answer text in full (its `content`, not the reasoning
       handed back): scored on its own it gives the single-call result GPT, Gemini and Gemma are
       measured by, and nothing else can (reviewer, 2026-11-13).
    3. **Cut again: the forced answer.** What the second attempt wrote goes back as an assistant
       turn, then `force_prompt`; the runner makes that call with its small answer cap and thinking
       off where a switch exists. The row carries `forced_answer: true`, `retry_*` (with
       `retry_response`, in full) and `forced_*`; its `finish_reason` is the forced call's, so
       `length` there means the answer filled its cap.

    **The truncation rate is the `cut_retry` rows** — every row whose first call was cut; the
    forced ones and the ones still `length` are among them. `latency_s` is the last attempt's, as on
    every row; `first_latency_s`, `retry_latency_s` and `forced_latency_s` time each call, so a stage
    kept from an earlier attempt is still timed. `served_model` and `backend` are the last call's.

    A second attempt that is cut with nothing to hand back stays a cut row (`finish_reason=length`,
    no `forced_answer`): the shard exits 1 and `--force-cut` calls it again from step 1.

    `call(sent, stage) -> (answer, handback, meta)` is the runner's client call, stage "first",
    "retry" or "forced": `answer` is the text that would be the row's response, `handback` what goes
    back if the reply was cut — the visible text, or `reasoning_content` for a model that hides its
    reasoning — and `meta` is openai_meta's plus `reasoning_chars`, the length of any
    `reasoning_content` (which a forced answer must not have: `forced_reasoned`).

    Usage on the row is every call of the attempt summed, so `completion_tokens` can pass the cap; a
    row's calls are one to three. The finished calls of the row being worked on are kept, so a retry
    by call_with_retries (a timeout, an exception, an empty answer) repeats only the stage that
    failed. Kept by the identity of the messages list run_rows builds once per row — held here, so
    that identity cannot be reused — never by text: PlanBench ships rows with byte-identical queries
    (101 groups in the full set), and a text key would hand one row's reasoning to the next."""

    def __init__(self, ceiling, retry_ceiling, force_prompt, answer_max):
        self.ceiling, self.retry_ceiling = ceiling, retry_ceiling
        self.force_prompt, self.answer_max = force_prompt, answer_max
        self._kept = {}

    @staticmethod
    def with_system(messages, text):
        """After the shared system message, before the shared user message: those stay untouched."""
        return messages[:1] + [{"role": "system", "content": text}] + messages[1:]

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


def load_checkpoint(path, config, force_cut=False):
    """Rows already done, keyed by uid — or a refusal if the stored config differs. An EMPTY row is
    not a done row: it is returned for retry, or a plain resume would skip it forever.

    `force_cut` (the `--force-cut` repair, the user's rule of 2026-11-13: call only the rows that
    hit the cap, not the run again): the runner's config now holds a `forced_answer`, and the rows
    on disk were written before it had one. Rows under exactly this config WITHOUT that key are
    accepted and carried under the new config; a row that ended at `length` with no forced answer
    is not done and is called again, through the runner's CutRetry from its first call. Returns (done, cut,
    copy_first): `cut` are those stored rows, which stay in the file until their own call replaces
    them; `copy_first` says the file still holds rows from before the forced answer."""
    if not os.path.exists(path):
        return ({}, {}, False) if force_cut else {}
    want = json.loads(json.dumps(config, sort_keys=True))
    base = {k: v for k, v in want.items() if k not in CUT_KEYS} if force_cut else None
    done, cut, empty, upgraded = {}, {}, 0, 0
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if force_cut and r.get("config") == base and base != want:
                r["config"] = want
                upgraded += 1
            elif r.get("config") != want:
                stored = r.get("config") or {}
                diff = {k: (want.get(k), stored.get(k)) for k in set(want) | set(stored)
                        if want.get(k) != stored.get(k)}
                raise SystemExit(f"{path}\n  refusing to resume: stored rows ran under a different "
                                 f"config.\n  (wanted, stored): {diff}\n  Archive the results "
                                 f"directory to a timestamped name and start clean.")
            if force_cut and is_cut(r):
                cut[r["uid"]] = r
            elif (str(r.get("raw_response", "")).strip()
                    and r.get("finish_reason") not in INTERRUPTED_FINISH):
                done[r["uid"]] = r
            else:
                empty += 1
    if done or empty or cut:
        print(f"  resume {os.path.basename(path)}: {len(done)} done, {empty} empty row(s) to retry"
              + (f", {len(cut)} cut at the cap to force, {upgraded} row(s) from before the forced "
                 f"answer" if force_cut else ""), flush=True)
    return (done, cut, bool(upgraded)) if force_cut else done


def archive_before_forced(model_dir, path):
    """Copy a shard file aside before `--force-cut` rewrites it, once: its rows are the run without
    the forced answer, and the cut rows in it are about to be replaced."""
    name = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path))
    held = glob.glob(os.path.join(model_dir, "results_archive_before_forced_answer_*", name))
    if held:
        if all(filecmp.cmp(h, path, shallow=False) for h in held):
            return                 # a run that stopped before its first save: the copy is this file
        raise SystemExit(f"{held[0]} already exists, differs from {path}, and that file still holds "
                         f"rows from before the forced answer — look at both before going on")
    target = os.path.join(model_dir, "results_archive_before_forced_answer_"
                          + time.strftime("%Y%m%d"), name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(path, target)
    print(f"  stored rows copied to {target}", flush=True)


# ---------------------------------------------------------------- the run


def run_rows(model_dir, model, once, config, sample, slices=None, shard=0, total_shards=1,
             limit=0, save_every=20, sleep=2.0, force_cut=False):
    rows = load_rows(sample, slices)
    smoke = bool(limit)
    if smoke:
        # First `limit` rows of each slice — a smoke test touches every slice it is given.
        per, kept = collections.Counter(), []
        for r in rows:
            if per[r["slice"]] < limit:
                kept.append(r)
                per[r["slice"]] += 1
        rows = kept
    rows = shard_slice(rows, shard, total_shards, interleave=(sample == "full"))
    path = rows_path(results_dir(model_dir, sample, smoke), model, shard_tag(shard, total_shards))
    if force_cut:
        # the stored rows ARE the input, smoke or not: nothing here starts from an empty file
        if not os.path.exists(path):
            raise SystemExit(f"--force-cut: {path} does not exist — there are no stored rows to force")
        done, cut, copy_first = load_checkpoint(path, config, force_cut=True)
        if copy_first and not smoke:
            archive_before_forced(model_dir, path)
    else:
        done, cut = ({} if smoke else load_checkpoint(path, config)), {}
    # A stored row this shard would not call is a file written under other shard boundaries (e.g. a
    # contiguous full-set shard before interleaving). Saving would drop it without a word — the
    # reviewer dropped 2,491 of 3,114 paid rows that way, 2026-11-13 — so refuse instead.
    foreign = (set(done) | set(cut)) - {r["uid"] for r in rows}
    if foreign:
        raise SystemExit(f"{path}\n  holds {len(foreign)} row(s) that are not in this shard "
                         f"(first: {sorted(foreign)[0]}) — written under other shard boundaries. "
                         f"Saving would drop them; archive the file and look at it first.")
    records = [done[r["uid"]] for r in rows if r["uid"] in done]
    todo = [r for r in rows if r["uid"] not in done]
    # --force-cut: a stored row the cap cut stays in the file until its OWN call returns an answer.
    # A call that fails, or a job stopped mid-loop, must not turn it into an empty row or drop it.
    held = {r["uid"]: cut[r["uid"]] for r in todo if r["uid"] in cut}

    def to_save():
        return records + list(held.values())
    print(f"{len(rows)} rows in this {'smoke test' if smoke else 'shard'} "
          f"({len({r['slice'] for r in rows})} slices), {len(todo)} to call -> {path}", flush=True)
    try:
        for n, row in enumerate(todo, 1):
            text, meta = call_with_retries(once, build_messages(row), model, model_dir)
            if not text and row["uid"] in held:
                print(f"  {row['uid']}: no answer from the forced call — the stored row is kept as "
                      f"it was", flush=True)
                continue
            held.pop(row["uid"], None)
            rec = {k: v for k, v in row.items() if k != "query"}
            rec.update(raw_response=text, empty=int(not text),
                       has_plan_end=int("[PLAN END]" in text.upper()) if is_plan_task(row["task"]) else None,
                       config=config, sample=sample, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), **meta)
            records.append(rec)
            if n % save_every == 0:
                save_rows(path, to_save())
                print(f"  {len(records)}/{len(rows)}  [{row['slice']}]", flush=True)
            if text and sleep:
                time.sleep(sleep)
    finally:
        # A halt's SystemExit, or a SIGTERM at the wall-clock limit (run_cli turns it into
        # SystemExit), must not lose the rows already paid for since the last periodic save.
        save_rows(path, to_save())
    return path, to_save()


# ---------------------------------------------------------------- generation health


def collect(out_dir, model):
    """Every row for one model in one results directory, shards merged. Returns (rows, problems):
    a missing shard is reported and the rest merged — never silently (script-skeleton.md 7b)."""
    slug = model_slug(model)
    plain = os.path.join(out_dir, f"{slug}.jsonl")
    sharded = sorted(glob.glob(os.path.join(out_dir, f"{slug}_shard*of*.jsonl")))
    files = ([plain] if os.path.exists(plain) else []) + sharded
    problems = []
    if os.path.exists(plain) and sharded:
        problems.append("both an unsharded file and shard files exist — one is stale")
    totals = {int(m.group(2)) for f in sharded for m in [re.search(r"_shard(\d+)of(\d+)\.jsonl$", f)] if m}
    if len(totals) > 1:
        problems.append(f"shard files from different shard counts: {sorted(totals)}")
    for total in totals:
        have = {int(re.search(r"_shard(\d+)of", f).group(1)) for f in sharded if f.endswith(f"of{total}.jsonl")}
        missing = sorted(set(range(total)) - have)
        if missing:
            problems.append(f"shard(s) {missing} of {total} missing")
    rows = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    if r["uid"] in rows:
                        problems.append(f"uid {r['uid']} in more than one file")
                    rows[r["uid"]] = r
    return sorted(rows.values(), key=lambda r: r["idx"]), problems


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else ""


def summarise_health(model_dir, model, sample, smoke=False):
    """Generation health per slice — no scoring. Whole = the sample's expected count in every slice,
    no empty row, one config. Written to <slug>_health.csv and <slug>_health_overall.csv."""
    out_dir = results_dir(model_dir, sample, smoke)
    rows, problems = collect(out_dir, model)
    configs = {json.dumps(r.get("config"), sort_keys=True) for r in rows}
    if len(configs) > 1:
        problems.append(f"rows ran under {len(configs)} different configs")
    by_slice = collections.defaultdict(list)
    for r in rows:
        by_slice[r["slice"]].append(r)
    col = SAMPLE_COLUMN[sample]
    records = []
    for sl in SLICES:
        rs = by_slice.get(sl, [])
        plan_rs = [r for r in rs if r.get("has_plan_end") is not None]
        finishes = collections.Counter(str(r.get("finish_reason")) for r in rs)
        records.append({
            "slice": sl, "rows": len(rs), "expected": EXPECTED[sl][col],
            "empty": sum(r.get("empty", 0) for r in rs),
            "finish_length": finishes.get("length", 0),
            "finish_interrupted": sum(finishes.get(f, 0) for f in INTERRUPTED_FINISH),
            "plan_end_rate": round(sum(r["has_plan_end"] for r in plan_rs) / len(plan_rs), 4) if plan_rs else "",
            "completion_tokens_median": _median([r.get("completion_tokens") for r in rs]),
            "completion_tokens_max": max([r["completion_tokens"] for r in rs
                                          if isinstance(r.get("completion_tokens"), int)], default=""),
            "reasoning_tokens_median": _median([r.get("reasoning_tokens") for r in rs]),
            "latency_s_median": _median([r.get("latency_s") for r in rs]),
            "attempts_over_1": sum(1 for r in rs if (r.get("n_attempts") or 1) > 1),
        })
    n = len(rows)
    expected_total = sum(v[col] for v in EXPECTED.values())
    overall = {
        "model": model, "sample": sample, "rows": n, "expected": expected_total,
        "empty": sum(r.get("empty", 0) for r in rows),
        "finish_length": sum(1 for r in rows if str(r.get("finish_reason")) == "length"),
        "finish_interrupted": sum(1 for r in rows if str(r.get("finish_reason")) in INTERRUPTED_FINISH),
        "fallback": sum(1 for r in rows if r.get("fallback")),
        "cut_retry": sum(1 for r in rows if r.get("cut_retry")),
        "forced_answer": sum(1 for r in rows if r.get("forced_answer")),
        "forced_reasoned": sum(1 for r in rows if forced_reasoned(r)),
        "configs": len(configs), "problems": " | ".join(problems),
    }
    # `complete` = every count met, no empty row, no fragment a backend cut off, one config. It does
    # NOT cover `finish_length`: a row that hit our output cap is a measurement (does the cap bind?),
    # not a missing row, so read that column before trusting a number.
    overall["complete"] = int(not smoke and not problems and n == expected_total
                              and overall["empty"] == 0 and overall["finish_interrupted"] == 0
                              and all(rec["rows"] == rec["expected"] for rec in records))
    slug = model_slug(model)
    os.makedirs(out_dir, exist_ok=True)
    for name, fields, recs in (("health", list(records[0]), records),
                               ("health_overall", list(overall), [overall])):
        def write(fh, fields=fields, recs=recs):
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(recs)
        _atomic_write(os.path.join(out_dir, f"{slug}_{name}.csv"), write)
    print(f"\nhealth [{model}] {sample}{' smoke' if smoke else ''}: rows={n}/{expected_total} "
          f"empty={overall['empty']} length={overall['finish_length']} "
          f"cut_retry={overall['cut_retry']} forced={overall['forced_answer']} forced_reasoned={overall['forced_reasoned']} "
          f"interrupted={overall['finish_interrupted']} configs={len(configs)} "
          f"COMPLETE={overall['complete']}", flush=True)
    for p in problems:
        print(f"  PROBLEM: {p}", flush=True)
    return overall, problems


# ---------------------------------------------------------------- CLI shared by every runner


def run_cli(*, description, default_model, model_dir, client, once, wanted, fixed=None,
            required=None, call_timeout=300, extra_config=None, probe_kwargs=None):
    """`once(messages, model, params) -> (text, meta)`. Everything model-specific is an argument;
    the rows, the messages and the files are identical for every model."""
    global HARD_CALL_TIMEOUT, MARKER_PREFIX
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", default=default_model,
                    help="model id; it is BOTH what is called and what the result files are named after")
    ap.add_argument("--sample", default="pilot10", choices=sorted(SAMPLES),
                    help="pilot10 = every 10th instance of every slice (1,563 rows), written to "
                         "results_pilot10/; full = all 15,568, written to results/")
    ap.add_argument("--slices", default="",
                    help="comma-separated domains or <domain>/<task> slices; default every slice")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--total-shards", dest="total_shards", type=int, default=1,
                    help="at 1 the filenames carry no shard tag")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N rows of each slice, written to smoke/ — a smoke test, not a run")
    ap.add_argument("--save-every", dest="save_every", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds after every successful call")
    ap.add_argument("--health-only", dest="health_only", action="store_true",
                    help="rewrite the health files from stored rows; makes no calls")
    ap.add_argument("--force-cut", dest="force_cut", action="store_true",
                    help="call only the stored rows that ended at the output cap (and any empty "
                         "one), through the runner's CutRetry from its first call; every other row "
                         "is kept")
    args = ap.parse_args()
    smoke = bool(args.limit)
    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    known = set(SLICES) | {s.split("/")[0] for s in SLICES}
    unknown = [s for s in slices if s not in known]
    if unknown:
        raise SystemExit(f"unknown slice(s) {unknown}")
    if slices and not smoke:
        # A filter moves the shard boundaries and the checkpoint keeps only the rows inside it, so
        # save_rows would rewrite the shard file without every paid row outside the selection
        # (the reviewer simulated 313 rows -> 10). --slices is for --limit smoke tests only.
        raise SystemExit("--slices is only allowed with --limit (a smoke test): on a real run it "
                         "would overwrite the shard's checkpoint. Re-run the whole shard instead — "
                         "resume skips the rows already done.")
    if args.health_only:
        _, problems = summarise_health(model_dir, args.model, args.sample, smoke)
        if problems:
            raise SystemExit(1)
        return

    if args.force_cut and "forced_answer" not in (extra_config or {}):
        raise SystemExit("--force-cut: this runner defines no forced answer (no `forced_answer` in "
                         "its config)")
    HARD_CALL_TIMEOUT = int(call_timeout)
    MARKER_PREFIX = f"shard{args.shard}_" if args.total_shards > 1 else ""
    # SLURM ends a job at its wall-clock limit (and on scancel) with SIGTERM, whose Python default
    # kills the process WITHOUT running `finally` — so run_rows' final save would never happen.
    # Turning it into SystemExit makes that save run; resume then continues from it.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    clear_stale_halt_markers(model_dir)
    print(f"negotiating parameter surface for {args.model} ...", flush=True)
    params, notes = negotiate(client, args.model, wanted, model_dir, fixed=fixed,
                              required=required, probe_kwargs=probe_kwargs)
    print(f"  accepted: {params}" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    if "seed" not in params and "seed" in wanted:
        print("  WARNING: seed was dropped — this run is not reproducible", flush=True)
    # The run config, written onto every row; a resume across any change of it is refused.
    config = json.loads(json.dumps(dict(extra_config or {}, model=args.model, params=params,
                                        fixed=fixed or {}, system_prompt=SYSTEM_PROMPT,
                                        prompts="plan-bench/prompts"), sort_keys=True))
    out_dir = results_dir(model_dir, args.sample, smoke)
    os.makedirs(out_dir, exist_ok=True)
    _atomic_write(os.path.join(out_dir, f"{model_slug(args.model)}_negotiated_params.json"),
                  lambda fh: json.dump({"asked": wanted, "accepted": params, "notes": notes,
                                        "config": config}, fh, indent=2))
    started = time.time()
    records = []
    try:
        _, records = run_rows(model_dir, args.model, lambda m: once(m, args.model, params), config,
                              args.sample, slices=slices, shard=args.shard,
                              total_shards=args.total_shards, limit=args.limit,
                              save_every=args.save_every, sleep=args.sleep,
                              force_cut=args.force_cut)
    finally:
        print(f"\ncalls: {STATS}  elapsed {time.time() - started:.0f}s", flush=True)
    summarise_health(model_dir, args.model, args.sample, smoke)
    # Exit 1 on THIS shard's own bad rows, so SLURM does not report COMPLETED 0:0 over a shard that
    # wrote empties below the 10-in-a-row halt. Not on the health `problems`: those include sibling
    # shards still running, which is no fault of this one.
    # A runner with a forced answer: no row may be left cut at the cap without one. Checked BEFORE
    # the empty rows, because a plain resubmit retries an empty row but skips a cut row that has
    # text — the advice below would then leave it cut for good (reviewer, 2026-11-13).
    still = [r for r in records if is_cut(r)] if "forced_answer" in config else []
    if still:
        print(f"\n{len(still)} row(s) of this shard are cut at the cap with no forced answer — run "
              f"the shard again WITH --force-cut (a plain resubmit skips a cut row that has text). "
              f"First: {still[0]['uid']}", flush=True)
        raise SystemExit(1)
    bad = [r for r in records if r.get("empty") or r.get("finish_reason") in INTERRUPTED_FINISH]
    if bad:
        print(f"\n{len(bad)} row(s) of this shard are empty or interrupted — resubmit the shard: "
              f"resume retries exactly those. First: {bad[0]['uid']}", flush=True)
        raise SystemExit(1)
    # A forced answer that reasoned again, or filled its own cap, is not a known complete answer.
    odd = [r for r in records if r.get("forced_answer")
           and (r.get("finish_reason") == "length" or forced_reasoned(r))]
    if odd:
        print(f"\nSTOP: {len(odd)} forced answer(s) of this shard filled their own cap or reasoned "
              f"again — read them before scoring. First: {odd[0]['uid']}", flush=True)
        raise SystemExit(2)
