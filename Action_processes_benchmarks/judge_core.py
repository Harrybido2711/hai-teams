"""Shared core for the LLM-as-judge pilot on the two judged benchmarks — MultiChallenge and Wonderbread.

The pilot (asked for 2026-10-19): generate `gpt-5.6-luna` responses on a stratified 10% sample of
each benchmark, have each of the four candidate judges grade them with the benchmark's own judge
prompt, and compare the judges. The advisor's rule for choosing: if the judges agree, take the cheap
one; if they differ a lot and the expensive one is better, take the expensive one.
Why these four: `LLM_as_judge/JUDGE_SELECTION.md` §5; how they were configured: §6.1.

What lives here, because both benchmarks need it identically:
- `JUDGES` — the four configurations, probed on OpenRouter 2026-10-19 (below);
- the call infrastructure, ported from `Awareness_in_LLM/aware_eval_core.py` (itself ported from
  `neg_eval_core.py`): billing/quota halts, the SIGALRM watchdog, retries on an empty body, the
  per-row usage fields, parameter negotiation for the generating model;
- the checkpoint: one JSON row per call, appended and flushed as it is written, keyed by a stable
  uid, one writer per file (`lock_output`).

What does NOT live here: any prompt or scorer. Those are each benchmark's own, imported from its
vendored upstream code by `mc_eval_core.py` / `wb_eval_core.py`, so the judge model is the only thing
that changes (JUDGE_SELECTION.md F5).

Nothing here imports `openai` at module level, so the pure functions can be tested on any Python.
"""

import collections
import hashlib
import json
import os
import random
import re
import signal
import sys
import time

GROUP_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- the four candidate judges
# Probed 2026-10-19 with each benchmark's real judge prompts (JUDGE_SELECTION.md §6.1 has the table):
# - REASONING is pinned at the lowest level each allows: off, or effort "low" for Sonnet 5.5, whose
#   endpoint refuses to disable it ("Reasoning is mandatory"). Defaults differ — Haiku 5.5 and Kimi
#   K2.6 think by default — so leaving it unset would compare thinking budgets, not models.
# - TEMPERATURE 0 only where the pinned backend lists it. Sent where it is not, OpenRouter silently
#   drops it; `require_parameters` turns that into a 404 instead.
# - BACKEND pinned per judge, no fallback. Kimi K2.6 is served by 17 providers at int4, fp4, fp8 and
#   bf16 — unpinned, one "judge" is a mix of weights (the smoke rows showed two backends billing
#   differently). Moonshot AI's own endpoint returns whitespace for MultiChallenge's JSON-mode call
#   (3 of 3, schema and object mode) and refuses temperature, so Kimi runs on Crusoe — bf16, the
#   unquantised weights, temperature accepted, three identical answers in three. The Claude models
#   run on Anthropic's own endpoint.
# - Sonnet 5.5 REFUSES SOP Generation's line-matching prompt, on every backend tried (Anthropic,
#   Google, Azure, Claude Platform on AWS): content filter, "reverse engineering or duplicating model
#   outputs". Recorded as status `refused`, never scored.
# Prices: OpenRouter, the pinned endpoint's list price, read 2026-10-19, USD per million tokens. The
# comparison sums the per-call `cost` OpenRouter bills; these are for estimates only.
JUDGES = {
    "haiku-4.5": {"model": "anthropic/claude-haiku-4.5", "provider": "Anthropic",
                  "price_in": 1.00, "price_out": 5.00,
                  "params": {"temperature": 0, "reasoning": {"enabled": False}}},
    "haiku-5.5": {"model": "anthropic/claude-haiku-5.5", "provider": "Anthropic",
                  "price_in": 0.10, "price_out": 0.50,
                  "params": {"reasoning": {"enabled": False}}},
    "sonnet-5.5": {"model": "anthropic/claude-sonnet-5.5", "provider": "Anthropic",
                   "price_in": 2.00, "price_out": 10.00,
                   "params": {"reasoning": {"effort": "low"}}},
    "kimi-k2.6": {"model": "moonshotai/kimi-k2.6", "provider": "Crusoe",
                  "price_in": 0.70, "price_out": 3.50,
                  "params": {"temperature": 0, "reasoning": {"enabled": False}}},
}
PRICES_READ = "2026-10-19"
# Every upstream judge call in both benchmarks passes max_tokens=4096 (LLM_as_judge/JUDGE_SELECTION.md §7.2-7.4, C1).
JUDGE_MAX_TOKENS = 4096
# Replicates per judged unit (LLM_as_judge/JUDGE_SELECTION.md §6.4). A unit's label needs a strict
# majority of its replicates; run rep-major, so a stopped run still holds complete replicates.
N_REPS = 3
# A judge whose answers cannot be read is not a judge: the run exits non-zero above this share of
# parse failures among the rows it called, so `COMPLETED 0:0` can never hide it.
PARSE_FAIL_EXIT_RATE = 0.05
# A judge refusing most of a set cannot judge it; that stage reports failure even when too few calls
# were left for REFUSAL_HALT to trigger (a resumed run). Its rows stay, and the report shows them.
REFUSAL_EXIT_RATE = 0.5

# The generating model — the project's GPT, settled config (model-parameters.md § Settled).
# max_completion_tokens counts reasoning too; 8192 leaves upstream's 4096-token visible cap
# (Wonderbread `_fetch_openai_completion`) plus headroom for `low` reasoning. A declared cap is a
# hypothesis: a generation run with any finish_reason=length exits 2, so judging cannot start on
# truncated answers without someone deciding it should.
GEN_MODEL = "gpt-5.6-luna"
GEN_WANTED = {"max_completion_tokens": 8192, "reasoning_effort": "low", "seed": 42}
GEN_REQUIRED = {"reasoning_effort": "low", "max_completion_tokens": 8192}

SAMPLE_SEED = 20261019
SAMPLE_FRACTION = 0.10     # the user's bound is 5-10%; the top, because MultiChallenge has 273 items


def stratified_sample(units, stratum_of, fraction=SAMPLE_FRACTION, seed=SAMPLE_SEED):
    """Simple random sample within each stratum, round(fraction * size) and at least 1. `units` are
    uids; the draw is sorted first so it depends only on the uid set, never on file order."""
    strata = collections.defaultdict(list)
    for uid in units:
        strata[stratum_of(uid)].append(uid)
    rng = random.Random(seed)
    picked = {}
    for name in sorted(strata):
        pool = sorted(strata[name])
        k = max(1, int(fraction * len(pool) + 0.5))
        picked[name] = sorted(rng.sample(pool, k))
    return picked


def write_sample_file(path, picked, sizes, note):
    doc = {"seed": SAMPLE_SEED, "fraction": SAMPLE_FRACTION, "note": note,
           "strata": {k: {"population": sizes[k], "sampled": len(v), "uids": v}
                      for k, v in picked.items()}}
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def read_sample_file(path):
    with open(path) as fh:
        doc = json.load(fh)
    return {k: v["uids"] for k, v in doc["strata"].items()}


def text_sha(*parts):
    """Fingerprint of the text a judge row judged — stored on the row, checked on resume and by the
    comparison, so rows can never be matched to a regenerated answer they did not see."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- refusals that must stop a run
# From aware_eval_core.py (committed). Only wordings that unambiguously mean money. The last two are
# OpenRouter's out-of-credit 402, which matched nothing when AwareBench's Gemini ran dry
# (provider-gotchas.md § Classifying a refusal); its third wording ("Retry after in-flight requests
# settle") invites a retry and is left out, as that section leaves it undecided.
BILLING_SIGNATURES = (
    "insufficient_quota", "insufficient balance", "insufficient credits", "out of credits",
    "monthly budget", "used all available credits", "spending limit", "credit balance",
    "payment required", "arrears", "requires more credits", "can only afford",
)
DAILY_QUOTA_SIGNATURES = ("per_model_per_day", "requests per day", "requests_per_day", "per day",
                          "daily limit", "daily quota")
DAILY_HALT_THRESHOLD_SECONDS = 600
# A malformed request fails identically on every item; retrying it spends the run to learn one
# fact. "no endpoints found" is OpenRouter under require_parameters / a pinned provider: no backend
# honours the config.
FATAL_SIGNATURES = ("invalid_argument", "extra inputs are not permitted", "validationerror",
                    "unsupported parameter", "unsupported value", "unrecognized request argument",
                    "unexpected keyword argument", "no endpoints found",
                    "reasoning is mandatory")
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


STATS = {"ok": 0, "empty": 0, "errors": 0, "timeouts": 0, "failed_rows": 0, "refused": 0,
         "parse_fail": 0}
HALT_MARKERS = ("BILLING_HALT.txt", "QUOTA_HALT.txt", "FAILURE_HALT.txt", "REFUSAL_HALT.txt",
                "PARSE_HALT.txt")
# Set by each runner to name the run a marker belongs to (`gpt-5.6-luna_`, `anchor-human-sample_`,
# `qa_`, `smoke_` …): stages that share a folder then never clear each other's markers on startup.
# Every marker still matches *_HALT.txt.
MARKER_PREFIX = ""


def _halt(kind, marker_name, detail, model, out_dir):
    """Write a marker an agent can find without reading logs, say why on both streams, exit."""
    os.makedirs(out_dir, exist_ok=True)
    marker = os.path.join(out_dir, MARKER_PREFIX + marker_name)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    advice = ("Rows that failed every attempt are retried on resume (load_done skips only finished "
              "rows), so a plain resubmit is safe once the cause is fixed.")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason={kind}\n{detail}\n\n{advice}\n")
    except OSError as error:
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! {kind} HALT [{model}] {stamp}\n{detail}\n", file=sys.stderr, flush=True)
    raise SystemExit(f"aborting: {model} stopped on {kind.lower()}. {advice}")


def halt_on_billing(error, model, out_dir):
    """Called first in every except block. Billing and an exhausted daily cap stop the run."""
    detail = f"{type(error).__name__}: {error}"
    if is_daily_quota_failure(error):
        _halt("DAILY QUOTA", "QUOTA_HALT.txt", detail, model, out_dir)
    if is_billing_failure(error):
        _halt("BILLING", "BILLING_HALT.txt", detail, model, out_dir)


def clear_stale_halt_markers(out_dir):
    for name in HALT_MARKERS:
        path = os.path.join(out_dir, MARKER_PREFIX + name)
        if os.path.exists(path):
            os.remove(path)
            print(f"[startup] cleared stale {MARKER_PREFIX + name}", flush=True)


# Consecutive failures catch a provider that stops serving; the rolling window catches one that
# fails intermittently. Refusals and unreadable answers have their own limits: both are
# deterministic for a prompt shape, so twenty in a row is the evidence and the rest would only
# restate it at full price.
CONSECUTIVE_FAILURE_LIMIT = 10
FAILURE_WINDOW = 50
FAILURE_WINDOW_LIMIT = 0.5
CONSECUTIVE_REFUSAL_LIMIT = 20
CONSECUTIVE_PARSE_FAIL_LIMIT = 20
_recent = collections.deque(maxlen=FAILURE_WINDOW)
_streak = {"failed": 0, "refused": 0, "parse_fail": 0}


def note_outcome(status, model, out_dir):
    """Every called row's status, once, after the row is appended. Each streak resets on any other
    outcome."""
    for kind in _streak:
        _streak[kind] = _streak[kind] + 1 if status == kind else 0
    if _streak["refused"] >= CONSECUTIVE_REFUSAL_LIMIT:
        _halt("REFUSAL", "REFUSAL_HALT.txt",
              f"{_streak['refused']} consecutive calls refused by the model's content filter",
              model, out_dir)
    if _streak["parse_fail"] >= CONSECUTIVE_PARSE_FAIL_LIMIT:
        _halt("PARSE FAILURE", "PARSE_HALT.txt",
              f"{_streak['parse_fail']} consecutive rows answered but unreadable after every re-ask",
              model, out_dir)
    if status in ("refused", "parse_fail"):
        return
    _recent.append(status == "ok")
    if _streak["failed"] >= CONSECUTIVE_FAILURE_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt",
              f"{_streak['failed']} consecutive rows failed every attempt", model, out_dir)
    if len(_recent) == FAILURE_WINDOW and _recent.count(False) / FAILURE_WINDOW >= FAILURE_WINDOW_LIMIT:
        _halt("FAILURE", "FAILURE_HALT.txt",
              f"{_recent.count(False)} of the last {FAILURE_WINDOW} rows failed", model, out_dir)


# ---------------------------------------------------------------- one call, guarded


class CallTimeout(BaseException):
    """BaseException on purpose: an `except Exception` anywhere below would otherwise swallow the
    watchdog, and the rest of that call would run unprotected (provider-gotchas.md)."""


class Refused(Exception):
    """The model answered with a refusal instead of content (OpenRouter `message.refusal`, or
    finish_reason `content_filter` with no content). Not retried, not scored: a status of its own."""

    def __init__(self, text, meta):
        super().__init__(text)
        self.text, self.meta = text, meta


# Hard ceiling on one HTTP attempt, by SIGALRM: `timeout=` is not a guard (provider-gotchas.md §
# Timeouts). Generation with many screenshots is slower than judging; runners override it.
HARD_CALL_TIMEOUT = 200

# Summed over EVERY attempt a row spent — empties, refusals and re-asks are billed too, and a judge
# that needs re-asking must not look cheaper than one that answers first time.
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost")


def _add_usage(total, meta):
    for k in USAGE_FIELDS:
        v = (meta or {}).get(k)
        if isinstance(v, (int, float)):
            total[k] = (total.get(k) or 0) + v
    return total


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


def _guarded(fn):
    has_alarm = hasattr(signal, "SIGALRM")
    previous = signal.signal(signal.SIGALRM, _raise_timeout) if has_alarm else None
    try:
        if has_alarm:
            signal.alarm(HARD_CALL_TIMEOUT)
        return fn()
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def call_with_retries(once, model, out_dir, tries=5):
    """`once() -> (text, meta)`. Retries an exception, a timeout AND an empty string — HTTP 200 with
    an empty body raises nothing. Returns (text, meta, status): status is "ok", "refused" (not
    retried) or "failed" (every attempt spent; the row is retried on resume). `meta` is the last
    response's, with USAGE_FIELDS summed over every attempt. Does not record the row's outcome:
    the runner calls `note_outcome` once, AFTER appending the row, so the row that trips a halt is
    written — with its bill — before the halt exits."""
    meta, used = {}, {}
    for attempt in range(tries):
        started = time.time()
        wait = 5.0
        try:
            text, meta = _guarded(once)
            _add_usage(used, meta)
            # latency_s is this successful attempt alone; n_attempts counts every call spent. Both
            # honest only because every client is built with max_retries=0 (script-skeleton.md §4).
            meta = dict(meta or {}, **used, latency_s=round(time.time() - started, 2),
                        n_attempts=attempt + 1)
            text = text.strip() if isinstance(text, str) else ""
            if text:
                STATS["ok"] += 1
                return text, meta, "ok"
            STATS["empty"] += 1
            print(f"[{model}] empty response ({attempt + 1}/{tries}) "
                  f"finish_reason={meta.get('finish_reason')} "
                  f"completion_tokens={meta.get('completion_tokens')}", flush=True)
        except CallTimeout as error:
            STATS["timeouts"] += 1
            print(f"[{model}] TIMEOUT ({attempt + 1}/{tries}): {error}", flush=True)
        except Refused as refusal:
            STATS["refused"] += 1
            _add_usage(used, refusal.meta)
            meta = dict(refusal.meta or {}, **used, latency_s=round(time.time() - started, 2),
                        n_attempts=attempt + 1, refusal=refusal.text)
            print(f"[{model}] REFUSED: {refusal.text[:160]}", flush=True)
            return "", meta, "refused"
        except Exception as error:
            STATS["errors"] += 1
            print(f"[{model}] API error ({attempt + 1}/{tries}): {type(error).__name__}: {error}",
                  flush=True)
            halt_on_billing(error, model, out_dir)
            if is_fatal_request(error):
                _halt("MALFORMED REQUEST", "FAILURE_HALT.txt",
                      f"{type(error).__name__}: {error}", model, out_dir)
            wait = retry_delay(error, attempt)
        if attempt + 1 < tries:
            time.sleep(wait)
    STATS["failed_rows"] += 1
    print(f"[{model}] all {tries} attempts failed; row written as failed, retried on resume",
          flush=True)
    return "", dict(meta or {}, **used, latency_s=None, n_attempts=tries), "failed"


def call_and_parse(once, parse, model, out_dir, parse_tries=3):
    """Second retry layer (script-skeleton.md §5): a NON-EMPTY answer the parser cannot read is
    re-asked, up to `parse_tries` calls. Returns (text, parsed, meta, status) with status "ok",
    "parse_fail" (answered, unreadable every time — raw text kept), "refused" or "failed". Usage and
    attempts are summed over every call. The runner records the outcome (`note_outcome`)."""
    spent, used = 0, {}
    for k in range(parse_tries):
        text, meta, status = call_with_retries(once, model, out_dir)
        spent += meta.get("n_attempts") or 0
        for f in USAGE_FIELDS:                     # call_with_retries already summed its attempts
            if isinstance(meta.get(f), (int, float)):
                used[f] = (used.get(f) or 0) + meta[f]
        meta = dict(meta, **used, n_attempts=spent, n_parse_attempts=k + 1)
        if status != "ok":
            return text, None, meta, status
        parsed = parse(text)
        if parsed is not None:
            return text, parsed, meta, "ok"
        print(f"[{model}] unparseable answer ({k + 1}/{parse_tries}): {text[:120]!r}", flush=True)
    STATS["parse_fail"] += 1
    return text, None, meta, "parse_fail"


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


def _answer(response):
    """(content, meta) from one response; a refusal raises `Refused`."""
    meta = openai_meta(response)
    message = response.choices[0].message if getattr(response, "choices", None) else None
    content = getattr(message, "content", None) or ""
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise Refused(str(refusal), meta)
    if meta.get("finish_reason") == "content_filter" and not content.strip():
        raise Refused("finish_reason=content_filter with no content", meta)
    return content, meta


# ---------------------------------------------------------------- clients


def _key(name):
    """Never let the SDK pick a key: openai falls back to OPENAI_API_KEY when api_key is None, which
    would send the OpenAI key to openrouter.ai."""
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set — check this benchmark's .env")
    return value


def _openai():
    """The real openai client class. wb_eval_core stands an absent library in for upstream's
    module-level imports; a client must never be built from that stand-in."""
    import openai
    if getattr(openai, "__absent__", False):
        raise SystemExit("the openai package is not installed in this interpreter")
    return openai.OpenAI


def judge_client():
    OpenAI = _openai()
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie.
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_key("OPENROUTER_API_KEY"),
                  timeout=180, max_retries=0)


def judge_routing(judge):
    return {"only": [JUDGES[judge]["provider"]], "allow_fallbacks": False, "require_parameters": True}


def judge_config(judge):
    """The settings a judge row was produced under — stored on every row, compared on resume."""
    cfg = JUDGES[judge]
    return {"judge": judge, "judge_model": cfg["model"], "params": cfg["params"],
            "provider": judge_routing(judge), "max_tokens": JUDGE_MAX_TOKENS, "n_reps": N_REPS}


def judge_once(client, judge, messages, response_format=None):
    """One judge call, as a zero-argument closure for `call_with_retries`. Everything sent is in
    `judge_config` except the messages and the response format, which the runner stores."""
    cfg = JUDGES[judge]
    params = dict(cfg["params"])
    extra = {"provider": judge_routing(judge)}
    if "reasoning" in params:
        extra["reasoning"] = params.pop("reasoning")
    kwargs = dict(model=cfg["model"], messages=messages, max_tokens=JUDGE_MAX_TOKENS,
                  extra_body=extra, **params)
    if response_format is not None:
        kwargs["response_format"] = response_format
    return lambda: _answer(client.chat.completions.create(**kwargs))


def gen_client():
    OpenAI = _openai()
    return OpenAI(api_key=_key("OPENAI_API_KEY"), timeout=300, max_retries=0)


EFFORT_FALLBACKS = ("minimal", "low", "medium", "high")
_TRANSIENT = ("rate limit", "429", "500", "502", "503", "504", "timed out", "timeout",
              "connection", "overloaded", "unavailable")


def negotiate(client, model, wanted, out_dir, required=None):
    """From aware_eval_core.negotiate: find the accepted parameter set with one probe, never letting
    a required VALUE change (a refused value is not a refused parameter — model-parameters.md 7)."""
    params, notes = dict(wanted), []
    transient_left = 3
    for _ in range(len(wanted) + 3 + transient_left):
        try:
            _guarded(lambda: client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with: ok"}], **params))
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
            halt_on_billing(error, model, out_dir)
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


def gen_once(client, model, messages, params):
    return lambda: _answer(client.chat.completions.create(model=model, messages=messages, **params))


# ---------------------------------------------------------------- checkpoint
# One file per (producer, task, set). A row is appended and flushed the moment its call returns, so
# a crash loses at most the call in flight — upstream Wonderbread wrote its CSV only after the loop
# (LLM_as_judge/JUDGE_SELECTION.md §7.3, C1). On resume the LAST row for a uid wins.

FINISHED = ("ok", "parse_fail", "refused", "skipped")
_locks = []


def out_path(base_dir, smoke, *parts):
    """`smoke/` for a --limit run, never `results/`: a smoke row left in results/ would be resumed
    past by the real run and reported as part of it."""
    path = os.path.join(base_dir, "smoke" if smoke else "results", *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def lock_output(path):
    """One writer per output file, held until the process exits. POSIX record locks (lockf), which
    GPFS honours across nodes: a resubmit running beside a live job refuses to start rather than
    interleaving rows and paying twice."""
    import fcntl
    fh = open(path + ".lock", "a")
    try:
        fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"{path} is being written by another process (lock held on {path}.lock)")
    _locks.append(fh)
    return fh


def load_done(path, config):
    """Rows already finished, keyed by uid — or a refusal if a stored row ran under another config.
    Resuming across a config change puts two conditions in one result set; archive instead. A
    `failed` row is not finished: it is returned for another try."""
    if not os.path.exists(path):
        return {}
    want = json.loads(json.dumps(config, sort_keys=True))
    rows = {}
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
                                 f"config.\n  (wanted, stored): {diff}\n  Archive the results "
                                 f"directory to a timestamped name and start clean.")
            rows[r["uid"]] = r
    done = {u: r for u, r in rows.items() if r.get("status") in FINISHED}
    retry = len(rows) - len(done)
    if rows:
        print(f"  resume {os.path.basename(path)}: {len(done)} finished, {retry} to retry", flush=True)
    return done


def check_judged(done, expected_sha, path):
    """Refuse to resume if a finished judge row judged a different text than the one now on disk —
    the answers were regenerated under it. Archive the judge rows instead."""
    stale = [u for u, r in done.items() if u in expected_sha and r.get("judged_sha") != expected_sha[u]]
    if stale:
        raise SystemExit(f"{path}\n  refusing to resume: {len(stale)} stored row(s) judged a different "
                         f"text than the responses now on disk (e.g. {stale[:2]}). Archive the judge "
                         f"rows to a timestamped name and start clean.")


def append_row(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_rows(path):
    """Last row per uid, in file order of first appearance."""
    rows = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows[r["uid"]] = r
    return list(rows.values())


def status_counts(rows):
    return dict(collections.Counter(r.get("status") for r in rows))


def judge_exit_code(rows, expected):
    """Non-zero unless every expected row exists, none failed, parse failures stay under
    PARSE_FAIL_EXIT_RATE and refusals under REFUSAL_EXIT_RATE of the rows that were called. An
    occasional refusal is a finding, not a crash."""
    called = [r for r in rows if r.get("status") != "skipped"]
    parse_fail = sum(r.get("status") == "parse_fail" for r in called)
    refused = sum(r.get("status") == "refused" for r in called)
    problems = []
    if len(rows) < expected:
        problems.append(f"{expected - len(rows)} row(s) missing")
    if any(r.get("status") == "failed" for r in rows):
        problems.append("failed row(s) present")
    if called and parse_fail / len(called) > PARSE_FAIL_EXIT_RATE:
        problems.append(f"parse_fail {parse_fail}/{len(called)} > {PARSE_FAIL_EXIT_RATE:.0%}")
    if called and refused / len(called) > REFUSAL_EXIT_RATE:
        problems.append(f"refused {refused}/{len(called)} > {REFUSAL_EXIT_RATE:.0%}")
    if problems:
        print("EXIT 1: " + "; ".join(problems), flush=True)
        return 1
    return 0


def strip_fences(text):
    """```json … ``` is packaging: Claude wraps JSON-mode answers in it on OpenRouter (probed)."""
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", t, re.DOTALL)
    return m.group(1).strip() if m else t


def first_json_object(text):
    """The first top-level {...} that parses, or None. Lenient about packaging, strict about JSON."""
    t = strip_fences(text)
    try:
        value = json.loads(t)
        return value if isinstance(value, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch == "{":
            try:
                value, _ = decoder.raw_decode(t[i:])
                if isinstance(value, dict):
                    return value
            except (json.JSONDecodeError, ValueError):
                continue
    return None
