"""Shared objective evaluator for the six NegotiationToM model runners."""

import argparse
import collections
import json
import os
import random
import re
import signal
import sys
import time

import pandas as pd
from sklearn.metrics import f1_score


# Per-1M-token USD prices, filled in by whoever runs the pilot. Left empty on purpose so the
# pilot never reports an invented cost — with no entry it reports token counts only.
PRICE_PER_1M = {
    # "gpt-4o-mini": {"in": 0.0, "out": 0.0},
}

# Runtime counters, reported by the pilot. Runners update these through record_* below.
#
# The three list-valued entries exist because a *sum* cannot size a budget. `completion_tokens`
# alone gives a mean, and setting max_tokens from a mean truncates roughly half the calls — the
# failure mode already recorded for Gemini, where "a 256 cap once truncated the JSON mid-object".
# Sizing needs the tail, so keep every observation and report percentiles.
STATS = {
    "calls": 0, "empty": 0, "errors": 0, "parse_fail": 0,
    "prompt_tokens": 0, "completion_tokens": 0, "usage_seen": 0,
    "completion_lengths": [],   # billed output tokens, one entry per response that reported usage
    "call_seconds": [],         # wall-clock of each call that returned, hangs excluded
    "truncated": 0,             # responses cut off by max_tokens (finish_reason == "length")
    "timeouts": 0,              # calls killed by the SIGALRM watchdog, i.e. that never returned
    # Counts responses whose finish_reason the runner actually passed in. Without this,
    # "truncated: 0" is ambiguous: five of the six runners still call record_call(*usage_from(r))
    # and never supply a finish_reason, so their zero means "never looked", not "looked and found
    # none". Reporting never-looked as a measured zero is how a truncating run gets a clean bill —
    # and Gemini is the model whose history is "a 256 cap truncated the JSON mid-object".
    "finish_reason_seen": 0,
}


def record_call(prompt_tokens=0, completion_tokens=0, usage_seen=False, finish_reason=None):
    """Called by each runner after a successful API call.

    `finish_reason` is optional and defaults to None so the six existing runners, which call this
    as `record_call(*usage_from(response))`, keep working untouched. Pass it to make truncation
    visible: a response cut off at max_tokens is otherwise indistinguishable from a wrong answer,
    which is how a too-tight budget silently lowers a score.
    """
    STATS["calls"] += 1
    STATS["prompt_tokens"] += prompt_tokens or 0
    STATS["completion_tokens"] += completion_tokens or 0
    if usage_seen:
        STATS["usage_seen"] += 1
        STATS["completion_lengths"].append(completion_tokens or 0)
    if finish_reason is not None:
        STATS["finish_reason_seen"] += 1
        if finish_reason == "length":
            STATS["truncated"] += 1


def record_usage(prompt_tokens=0, completion_tokens=0, usage_seen=False, finish_reason=None):
    """Record billed tokens from an API response that did not yield a usable answer.

    Empty responses are especially expensive for reasoning models: Qwen can consume its entire
    output budget and return empty content. They must count toward cost without being reported as
    successful calls.
    """
    STATS["prompt_tokens"] += prompt_tokens or 0
    STATS["completion_tokens"] += completion_tokens or 0
    if usage_seen:
        STATS["usage_seen"] += 1
        STATS["completion_lengths"].append(completion_tokens or 0)
    if finish_reason is not None:
        STATS["finish_reason_seen"] += 1
        if finish_reason == "length":
            STATS["truncated"] += 1


def finish_reason_from(response):
    """Best-effort finish_reason, normalised to lowercase. SDKs disagree on the shape."""
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError):
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        name = getattr(reason, "name", None) or str(reason or "")
        # Gemini spells the same state MAX_TOKENS; normalise to the OpenAI wording.
        return "length" if name.upper() in ("MAX_TOKENS", "LENGTH") else name.lower() or None
    reason = getattr(choice, "finish_reason", None) or getattr(choice, "stop_reason", None)
    if reason is None:
        return None
    name = getattr(reason, "name", None) or str(reason)
    name = name.lower()
    # Anthropic-style wording for the same state.
    return "length" if name in ("max_tokens", "length") else name


def percentiles(values):
    """median / p90 / p99 / max of a list, or None if it is empty. No numpy dependency needed."""
    if not values:
        return None
    ordered = sorted(values)
    def at(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"median": at(0.50), "p90": at(0.90), "p99": at(0.99), "max": ordered[-1],
            "n": len(ordered)}


def record_empty():
    STATS["empty"] += 1


def record_error():
    STATS["errors"] += 1


# A run in which every call fails should stop, not spend hours producing empty rows that look like
# a completed job. Grok's credits ran out mid-run and the job carried on for five more hours, wrote
# 9,378 empty rows, and exited COMPLETED with 0:0 — Belief_EM 0.0 was an unpaid invoice, not a
# result. Nothing in the pipeline noticed.
#
# Two separate mechanisms, because the two failures are not alike:
#
#   * A billing refusal is deterministic and clears only when a human tops the account up. Retrying
#     it is pure waste, so `halt_on_billing` stops at the FIRST one — zero tolerance.
#   * An unknown failure (outage, network, a model emitting garbage) might pass on the next item, so
#     it gets a consecutive-failure budget. 40 was too generous: at ~16s per failed item that is
#     ~11 minutes and 120 API calls. 10 costs under 3 minutes, and aborting is nearly free because
#     the checkpoint means a resubmit resumes exactly where it stopped.
CONSECUTIVE_FAILURE_LIMIT = 10
_consecutive_failures = 0

# The consecutive counter only catches a provider that fails *continuously*. A rolling window is
# what catches one that fails intermittently — the shape that produces a full-length result set
# quietly stuffed with zeros. 50% of the last 50 items is far above any healthy run measured here
# (Qwen's fixed pilot: 0 of 1,121) and far below the rates that ruined the xAI and Gemini runs.
FAILURE_WINDOW = 50
FAILURE_WINDOW_LIMIT = 0.5
_recent_outcomes = collections.deque(maxlen=FAILURE_WINDOW)

# Only wordings that unambiguously mean *money*.
#
# The bare word "billing" is deliberately NOT here, and that is the whole subtlety. Gemini's
# ordinary rate limit reads:
#
#   429 RESOURCE_EXHAUSTED. You exceeded your current quota, please check your plan and billing
#   details.
#
# It mentions billing while being a throughput limit that clears on its own — matching it would
# hard-stop a perfectly healthy run. OpenAI's genuine billing refusal is *also* a 429, so the
# status code cannot separate them either. Only the provider-specific tokens below can.
BILLING_SIGNATURES = (
    "insufficient_quota",          # OpenAI — billing, despite arriving as a 429
    "insufficient balance",        # DeepSeek
    "insufficient credits",        # Together
    "out of credits",              # Together
    "monthly budget",              # Together
    "used all available credits",  # xAI
    "spending limit",              # xAI
    "credit balance",
    "payment required",
    "arrears",
)

# Throughput limits, kept for documentation and for `retry_delay`. They are deliberately NOT used to
# veto a billing match any more, and removing that veto was a correctness fix, not a simplification.
#
# The veto existed only to stop the bare word "billing" matching Gemini's rate-limit boilerplate
# ("check your plan and billing details"). Once that word was dropped from BILLING_SIGNATURES, none
# of the messages below match a billing signature on their own, so the veto protected nothing — but
# it did break the provider it was written for. xai_sdk raises grpc errors unwrapped, and grpc's
# str() always embeds `status = StatusCode.<CODE>`; a credit refusal arriving as RESOURCE_EXHAUSTED
# or UNAVAILABLE would have been vetoed before "used all available credits" was ever tested,
# reproducing the eight-hour, 9,378-empty-row incident exactly.
TRANSIENT_SIGNATURES = (
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "requests per day",
    "requests per minute",
    "please retry",
    "try again",
    "503",
    "unavailable",
    "overloaded",
)


# A daily cap is neither of the above: no human has to pay anything, but it will not clear for
# hours, so retrying is as useless as retrying a billing refusal. Gemini's cap is 10,000
# requests/day for gemini-2.5-flash while one full run needs 14,138 — the run cannot finish in a
# day at that tier, and once the cap hit, the remaining items failed three times each and were
# written as 1,963 empty rows instead of being left for tomorrow.
DAILY_QUOTA_SIGNATURES = (
    "per_model_per_day",
    "requests per day",
    "requests_per_day",
    "per day",
    "daily limit",
    "daily quota",
)

# Wording alone cannot separate a daily cap from an ordinary rolling-window limit — both say
# "requests per day". OpenAI's healthy RPD throttle reads
#     ... on requests per day (RPD): Limit 10000, Used 10000. Please try again in 8.64s.
# and Gemini's exhausted daily cap reads
#     ... generate_requests_per_model_per_day, limit: 10000. Please retry in 18h7m17s.
# The discriminator is the wait the provider itself asks for: seconds means wait, hours means stop.
# Matching on the phrase alone would hard-stop every healthy GPT run that ever touches its RPD.
DAILY_HALT_THRESHOLD_SECONDS = 600

_RETRY_PHRASE = re.compile(r"(?:try again|retry|available again)\s+in\s+([0-9hms.\s]+)", re.I)
_DURATION_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.I)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def retry_after_seconds(error):
    """How long the provider asked us to wait, in seconds, or None if it did not say.

    Handles both shapes seen in practice: a single token ("8.64s", "20ms") and a compound one
    ("18h7m17.167193977s").
    """
    match = _RETRY_PHRASE.search(str(error))
    if not match:
        return None
    total, found = 0.0, False
    for value, unit in _DURATION_TOKEN.findall(match.group(1)):
        total += float(value) * _UNIT_SECONDS[unit.lower()]
        found = True
    return total if found else None


def is_daily_quota_failure(error):
    """True when the provider says the *daily* allowance is gone and will not return soon."""
    if not any(sig in str(error).lower() for sig in DAILY_QUOTA_SIGNATURES):
        return False
    wait = retry_after_seconds(error)
    if wait is None:
        return True  # named a daily cap but gave no ETA; assume it is the cap, not a throttle
    return wait >= DAILY_HALT_THRESHOLD_SECONDS


def is_billing_failure(error):
    """True when the provider is refusing over money rather than over throughput.

    Every signature here names money explicitly, so no transient veto is applied — see the note on
    TRANSIENT_SIGNATURES for why adding one silently disarmed the guard for xAI.
    """
    text = str(error).lower()
    return any(sig in text for sig in BILLING_SIGNATURES)


# run_cli fills this in so the halt paths can name the model and put their marker in the model's own
# folder. Without it `note_outcome` — which has no arguments — could only exit silently, leaving the
# two abort paths with different evidence.
_RUN_CONTEXT = {"model": "unknown", "script_dir": "."}


def set_run_context(model, script_dir):
    _RUN_CONTEXT["model"] = model
    _RUN_CONTEXT["script_dir"] = script_dir


def clear_stale_halt_markers(script_dir):
    """Remove markers from a previous run so a fresh start never inherits an old verdict.

    A marker that is never cleared claims a halt that no longer applies, and the next operator (or
    watcher) reads it as current.
    """
    for name in ("BILLING_HALT.txt", "QUOTA_HALT.txt", "FAILURE_HALT.txt"):
        path = os.path.join(script_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"[startup] cleared stale {name}", flush=True)
            except OSError as error:
                print(f"[startup] could not clear {name}: {error}", flush=True)


def _prune_advice():
    """Tell the operator to prune only when rows really were written empty before this halt.

    Saying "nothing needs pruning" unconditionally was wrong: any earlier item that exhausted its
    retries on a *different* error already persisted a row with an empty raw_response and a zero
    score, and `load_checkpoint` would skip it forever on resume.
    """
    spoiled = STATS["empty"] + STATS["errors"] + STATS["parse_fail"]
    if spoiled:
        return (f"{spoiled} earlier call(s) already failed, so rows may have been written empty — "
                "run prune_failed_rows.py before resubmitting, or resume will skip them forever.")
    return "No failed rows were written before this point, so a plain resume is safe."


def _halt(kind, marker_name, detail, advice):
    """Write the marker, say why on both streams, then exit. Shared by every abort path."""
    model = _RUN_CONTEXT["model"]
    marker = os.path.join(_RUN_CONTEXT["script_dir"], marker_name)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(marker, "w") as fh:
            fh.write(f"{stamp}  model={model}  reason={kind}\n{detail}\n\n{advice}\n")
    except OSError as error:
        # Logged, not swallowed: a silent bookkeeping failure is how the last incident stayed
        # invisible. The halt itself still happens.
        print(f"[{model}] could not write {marker}: {error}", flush=True)
    print(f"\n!!! {kind} HALT [{model}] {stamp}\n{detail}\n", file=sys.stderr, flush=True)
    print(f"[{model}] {kind} HALT — stopping immediately, marker at {marker}", flush=True)
    raise SystemExit(f"aborting: {model} stopped on {kind.lower()}. {advice}")


def halt_on_billing(error, model, script_dir):
    """Stop at the first unrecoverable refusal, leaving a marker an agent can find without logs.

    Two kinds qualify, and both are pointless to retry:

      * **Billing.** xAI's wording — "has either used all available credits or reached its monthly
        spending limit" — matched neither `insufficient_quota` nor `requests per day`, the only two
        strings every runner checked, so it fell through to the generic retry path and the run
        spent eight hours writing empty rows.
      * **Daily quota.** Gemini's cap is 10,000 requests/day against a 14,138-request run. Once hit
        it returns "Please retry in 18h7m", and every remaining item failed three times in ~15s and
        was written as an empty row. Stopping leaves those items untouched for a resubmit after the
        quota resets, and resume picks them up.
    """
    set_run_context(model, script_dir)
    detail = f"{type(error).__name__}: {error}"
    if is_daily_quota_failure(error):
        _halt("DAILY QUOTA", "QUOTA_HALT.txt", detail,
              "The daily allowance is gone. Wait for the provider's reset window, then resubmit. "
              + _prune_advice())
    if is_billing_failure(error):
        _halt("BILLING", "BILLING_HALT.txt", detail,
              "Top the account up, then resubmit. " + _prune_advice())


def note_outcome(ok):
    """Abort when the provider stops serving — by a run of failures OR by a bad enough rate.

    The consecutive counter alone was not enough, and that gap is the whole reason this function
    was revisited. `record_empty(); continue` in every runner means an HTTP-200-with-empty-body
    never reaches `halt_on_billing`, so this is the only defence on that path — and a counter that
    resets on *any* success never fires against an intermittent failure. At a 50% empty rate the run
    would finish with a perfect 14,138 rows, half of them scored zero, and exit COMPLETED: exactly
    the original incident with a smaller constant.

    The rolling window closes it. It also catches late-onset failure — a quota wall two thirds of
    the way through — which a cumulative rate would dilute below any sane threshold.
    """
    global _consecutive_failures
    _recent_outcomes.append(bool(ok))
    if ok:
        _consecutive_failures = 0
    else:
        _consecutive_failures += 1
        if _consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
            _halt("CONSECUTIVE FAILURE", "FAILURE_HALT.txt",
                  f"{_consecutive_failures} consecutive items returned nothing. The provider is "
                  "almost certainly refusing every call (credits, quota or an outage).",
                  _prune_advice())

    if len(_recent_outcomes) >= FAILURE_WINDOW:
        failed = sum(1 for outcome in _recent_outcomes if not outcome)
        rate = failed / len(_recent_outcomes)
        if rate >= FAILURE_WINDOW_LIMIT:
            _halt("FAILURE RATE", "FAILURE_HALT.txt",
                  f"{failed} of the last {len(_recent_outcomes)} items returned nothing "
                  f"({rate:.0%}). The provider is failing intermittently, which produces a "
                  "full-length result set full of zeros rather than an obvious crash.",
                  _prune_advice())


def usage_from(response):
    """Best-effort token extraction; SDKs disagree on the shape, so tolerate all of them.

    **Reasoning tokens must be counted as output.** They are billed at the output rate but reported
    in a separate field, so reading only the visible-answer count understates the bill badly: one
    measured gemini-2.5-flash call was 16 visible tokens against 143 thinking tokens — a 10x
    undercount, which is why an early cost projection for Gemini came in an order of magnitude
    below the actual invoice.
    """
    usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0, False
    prompt = (getattr(usage, "prompt_tokens", None)
              or getattr(usage, "prompt_token_count", None)
              or getattr(usage, "input_tokens", None) or 0)
    completion_total = (getattr(usage, "completion_tokens", None)
                        or getattr(usage, "output_tokens", None))
    if completion_total is not None:
        # OpenAI-compatible APIs include reasoning tokens inside completion_tokens. Their
        # completion_tokens_details.reasoning_tokens value is a subset, not an additional charge.
        completion = completion_total
    else:
        # Gemini reports visible candidates and hidden thoughts separately.
        completion = getattr(usage, "candidates_token_count", None) or 0
        completion += getattr(usage, "thoughts_token_count", None) or 0
        if not completion:
            completion = (getattr(usage, "reasoning_tokens", None)
                          or getattr(usage, "reasoning_token_count", None) or 0)
    return prompt, completion, True


# Surface-form aliases, in the spirit of bbh/xai_eval.py::score_response: be generous about how an
# answer is written, strict about what it says. Keys are matched after the cleanup in norm_item().
ITEM_NORM = {
    "food": "Food", "foods": "Food",
    "water": "Water", "waters": "Water",
    "firewood": "Firewood", "fire wood": "Firewood", "fire-wood": "Firewood",
    "wood": "Firewood", "woods": "Firewood",
    # "not revealed yet" — the model has several ways of saying it
    "not given": "Not Given", "notgiven": "Not Given", "not_given": "Not Given",
    "not-given": "Not Given", "not specified": "Not Given", "unspecified": "Not Given",
    "unknown": "Not Given", "n/a": "Not Given", "na": "Not Given",
    # The sentinel. In gold it marks an unannotated sample and those rows are excluded from the
    # metrics, so in any scored row gold is one of Food/Water/Firewood/Not Given and a model
    # answering "None" is simply wrong — which is correct, since the prompt no longer offers it.
    "none": "None", "null": "None",
}
INTENT_LABELS = [
    "Build-Rapport", "Callout-Fairness", "Describe-Need", "Discover-Preference",
    "No-Intention", "No-Need", "Promote-Coordination", "Show-Empathy",
    "Undermine-Requirements",
]
INTENT_NORM = {label.lower(): label for label in INTENT_LABELS}


def parse_json(text):
    """Parse bare or fenced JSON; return None instead of raising."""
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = match.group(1) if match else text
    try:
        return json.loads(candidate.strip())
    except (TypeError, ValueError):
        return None


def retry_delay(error, default=5.0):
    match = re.search(r"try again in ([\d.]+)(ms|s)", str(error), re.IGNORECASE)
    if not match:
        return default
    delay = float(match.group(1))
    if match.group(2).lower() == "ms":
        delay /= 1000
    return max(delay + 1, 1)


class CallTimeout(BaseException):
    """Derives from BaseException on purpose, like KeyboardInterrupt and SystemExit.

    Every runner's call_api ends in `except Exception`, which would otherwise swallow the watchdog
    and treat it as an ordinary API error. The alarm has already fired and been cleared by then, so
    the remaining attempts inside that same call_api run with no watchdog at all — the protection
    silently evaporates after its first use. Deriving from BaseException lets it propagate past
    those handlers to call_and_parse, which retries deliberately.
    """


# Hard ceiling on one call_api invocation, enforced with SIGALRM so it does not depend on the SDK
# honouring its own timeout= argument. Together's client did not: Gemma sat inside a single request
# for over two hours with timeout=18000, and again produced nothing for ~70 minutes with
# timeout=300, in both cases with SLURM still reporting RUNNING and an empty log. SIGALRM
# interrupts the blocking read regardless of what the library does.
# This is a backstop for a hung connection, not the mechanism for bounding work. Bound work with
# max_tokens: the server enforces it, it returns promptly, and finish_reason says what happened.
# A wall-clock kill tells you nothing about why.
#
# Sizing: the ceiling must sit above the slowest *legitimate* call, or it silently truncates good
# work. Measured worst case is a full max_tokens=8192 generation at ~90 tok/s, about 110s, so 200
# leaves real margin. An earlier 150 was too tight against a 32768 budget — every runaway was cut
# by the clock before it could report Length, which destroyed the diagnostic signal.
HARD_CALL_TIMEOUT = 200


def set_call_timeout(seconds):
    """Per-model override of the watchdog ceiling, called by a runner before run_cli.

    The right ceiling is a property of the provider, not of the benchmark, so one shared constant
    cannot fit all six. Sizing rule: it must sit above the slowest *legitimate* call, which is
    max_tokens divided by the model's observed tokens/second, with margin. Below that it truncates
    good work; far above it, every hung connection costs the full ceiling before the retry can run.

    Measured on Gemma/Together 2026-08-04: successful calls returned in ~10s at ~57 tok/s while
    roughly 9% of attempts never returned at all. At the shared 200s ceiling those hangs consumed
    67% of the job's wall clock, which is why a run projected at 19.6h was tracking towards 97h.
    """
    global HARD_CALL_TIMEOUT
    HARD_CALL_TIMEOUT = int(seconds)


def _raise_timeout(signum, frame):
    raise CallTimeout(f"call exceeded {HARD_CALL_TIMEOUT}s hard limit")


def guarded_call(call_api, messages, model):
    """Run call_api under a SIGALRM watchdog. Raises CallTimeout if it overruns.

    Also times every call. The latency distribution is what separates the two failure modes that
    look identical from the outside: a provider that is merely slow shows a long tail of calls that
    still return, while a provider whose connections hang shows fast successes plus a set of calls
    that never return at all. The fixes are opposite — cap max_tokens for the first, lower the
    ceiling for the second — so guessing between them wastes a run.
    """
    if not hasattr(signal, "SIGALRM"):
        return call_api(messages, model)          # not POSIX; fall back
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(HARD_CALL_TIMEOUT)
    started = time.time()
    try:
        result = call_api(messages, model)
        STATS["call_seconds"].append(time.time() - started)
        return result
    except CallTimeout:
        STATS["timeouts"] += 1
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        # Pulse from here, not from save_checkpoint, so the diagnostic is driven by the clock
        # rather than by progress — a run that completes no rows at all must still report.
        _pulse()


def call_and_parse(call_api, messages, model, max_parse_retries=3):
    last_raw = None
    for attempt in range(max_parse_retries):
        try:
            raw = guarded_call(call_api, messages, model)
        except CallTimeout as error:
            # A hang is usually transient, so it is worth another attempt rather than losing the
            # item — unlike call_api returning None, which means its own retries are exhausted.
            record_error()
            print(f"[{model}] {error} ({attempt + 1}/{max_parse_retries}), retrying", flush=True)
            continue
        if raw is None:
            note_outcome(False)
            return None, None
        last_raw = raw
        parsed = parse_json(raw)
        if isinstance(parsed, dict):
            note_outcome(True)
            return raw, parsed
        print(f"[{model}] JSON parse failed ({attempt + 1}/{max_parse_retries}); calling again",
              flush=True)
    STATS["parse_fail"] += 1
    note_outcome(False)
    return last_raw, None


def format_dialogue(turns):
    return "\n".join(turns)


def desire_messages(dialogue, agent):
    # Gold labels are cutoff-aware: a priority the dialogue has not revealed yet is "Not Given",
    # and "None" means the agent expressed no preference at that level. The prompt has to offer
    # both, otherwise the model can never match those labels.
    system = (
        "Infer an agent's desires in a negotiation over Food, Water, and Firewood, using only "
        "what the dialogue so far reveals. Use 'Not Given' when a priority has not been revealed. "
        "Return only valid JSON with no markdown: {\"high\":\"<item or Not Given>\","
        '"medium":"<item or Not Given>","low":"<item or Not Given>"}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"What are {agent}'s High, Medium, and Low item preferences?"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def belief_messages(dialogue, agent, opponent):
    system = (
        "Infer one negotiator's belief about the other negotiator's preferences over Food, "
        "Water, and Firewood. Use 'Not Given' when a priority is not revealed. Return only "
        'valid JSON with no markdown: {"high":"<item or Not Given>",'
        '"medium":"<item or Not Given>","low":"<item or Not Given>"}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"What does {agent} believe about {opponent}'s High, Medium, and Low preferences?"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def intention_messages(dialogue, target):
    labels = "\n".join(f"- {label}" for label in INTENT_LABELS)
    system = (
        "Classify all intents expressed by the target negotiation utterance. Return only valid "
        'JSON with no markdown: {"intents":["label1","label2"]}'
    )
    user = (
        f"Negotiation dialogue:\n{format_dialogue(dialogue)}\n\n"
        f"Target utterance: {target}\n\nChoose one or more labels:\n{labels}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def clean_surface(value):
    """Strip the packaging a model puts around an answer, following the bbh scripts.

    Quotes and backticks (`.strip("\\"'`")` in bbh/xai_eval.py), runs of whitespace, and trailing
    punctuation are all noise. Applied to model output only — gold labels are already canonical and
    must never be rewritten.
    """
    if not isinstance(value, str):
        return ""
    text = value.strip().strip("\"'`").strip()
    text = re.sub(r"\s+", " ", text)          # "Fire   wood" -> "Fire wood"
    return text.strip(" .,;:!")


def norm_item(value):
    text = clean_surface(value)
    if not text:
        return ""
    return ITEM_NORM.get(text.lower(), text.title())


def norm_intent(value):
    """Normalise one intent label: surface cleanup, then space/underscore -> hyphen."""
    text = clean_surface(value)
    if not text:
        return ""
    key = re.sub(r"[\s_]+", "-", text.lower())
    return INTENT_NORM.get(key, text)


def pred_item(prediction, key):
    if not isinstance(prediction, dict):
        return ""
    return norm_item(prediction.get(key) or prediction.get(key.title()) or "")


def is_unannotated(values):
    """True when every priority slot is the sentinel string "None".

    The dataset writes "None" in all three slots at once — 84 samples for agent_1 and 72 for
    agent_2, identically in the desire and belief fields, and never mixed with a real label — while
    the agent{N}_desire dict still holds real values. So "None" marks an unannotated sample rather
    than "the agent wants nothing", exactly like utterance2_intent == "None" on the intention side.
    Such rows are unanswerable and are excluded from the metrics (the rows are still written out).
    """
    return all(str(v).strip() == "None" for v in values)


def desire_em(prediction, gold):
    return int(bool(prediction) and all(
        pred_item(prediction, key.lower()) == gold.get(key, "")
        for key in ("High", "Medium", "Low")
    ))


def belief_em(prediction, high, medium, low):
    return int(bool(prediction) and (
        pred_item(prediction, "high") == high
        and pred_item(prediction, "medium") == medium
        and pred_item(prediction, "low") == low
    ))


def intent_bitmask(labels):
    if isinstance(labels, str):
        labels = labels.split(",")
    normalized = {
        norm_intent(label)
        for label in (labels or []) if isinstance(label, str) and label.strip()
    }
    return [int(label in normalized) for label in INTENT_LABELS]


def model_slug(model):
    return model.split("/")[-1].replace(".", "_").replace("/", "-")


def shard_slice(rows, shard, total_shards):
    size = (len(rows) + total_shards - 1) // total_shards
    return rows[shard * size:min((shard + 1) * size, len(rows))]


def load_checkpoint(path):
    done, rows = set(), []
    if not os.path.exists(path):
        return done, rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add(row["uid"])
            rows.append(row)
    return done, rows


_last_pulse = [0.0]
PULSE_EVERY = 600          # seconds between one-line progress pulses


def save_checkpoint(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pulse():
    """One line every PULSE_EVERY seconds: rate, hang rate, truncation, and the token tail so far.

    Called from guarded_call, i.e. once per API call, NOT from save_checkpoint. Driving it from
    checkpoints tied the diagnostic to progress: at --save-every 20 and a slow model the first line
    would not arrive for ~22 minutes, and the slower the run — exactly when the line is most needed
    — the later it came. Driving it from calls makes it a clock, so a run that is failing every
    call still reports on time.

    It must print with zero successful calls. An earlier version returned early when nothing had
    succeeded yet, which made a 100%-hang run look identical to a run that had not started. That is
    this project's dominant bug class, and reintroducing it inside the diagnostic built to catch it
    would have been the worst place for it.
    """
    now = time.time()
    if now - _last_pulse[0] < PULSE_EVERY:
        return
    _last_pulse[0] = now
    secs, toks = STATS["call_seconds"], STATS["completion_lengths"]
    done, hangs = len(secs), STATS["timeouts"]
    attempts = done + hangs
    if not attempts:
        return                      # genuinely nothing has been attempted yet
    bad = (f"{hangs}/{attempts} hung ({100 * hangs / attempts:.0f}%), "
           f"trunc {STATS['truncated']}, empty {STATS['empty']}, err {STATS['errors']}")
    if not done:
        # Every attempt so far has hung. Say so loudly rather than saying nothing.
        print(f"[pulse] *** {attempts} attempts, ZERO returned — {bad}; "
              f"ceiling {HARD_CALL_TIMEOUT}s", flush=True)
        return
    mean_s = sum(secs) / done
    # Effective seconds per completed row, hang cost included — the number that actually predicts
    # the finish time, unlike the mean latency of the calls that happened to succeed.
    per_row = (sum(secs) + hangs * HARD_CALL_TIMEOUT) / done
    tail = f", tokens p99~{sorted(toks)[min(len(toks) - 1, int(0.99 * len(toks)))]}" if toks else ""
    print(f"[pulse] {done} calls, {bad}, mean {mean_s:.1f}s, "
          f"effective {per_row:.1f}s/row -> {60 / per_row:.2f} rows/min{tail}", flush=True)


def output_paths(script_dir, task, model, shard, total_shards, pilot=False):
    # Pilot output is kept under results/pilot/ so it can never overwrite a full run.
    parts = [script_dir, "results"] + (["pilot"] if pilot else []) + [task]
    out_dir = os.path.join(*parts)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_shard{shard}of{total_shards}" if total_shards > 1 else ""
    stem = model_slug(model) + tag
    return out_dir, stem, os.path.join(out_dir, stem + ".jsonl")


def scorable(task, rows):
    """Drop rows the dataset never annotated — they are unanswerable, not wrong answers.

    See is_unannotated(). Returns (kept_rows, dropped_count).
    """
    if task == "desire":
        keep = [r for r in rows if not is_unannotated((r["gold_desire"] or {}).values())]
    elif task == "belief":
        keep = [r for r in rows
                if not is_unannotated((r["gold_high"], r["gold_med"], r["gold_low"]))]
    else:
        keep = rows
    return keep, len(rows) - len(keep)


def write_task_outputs(task, rows, out_dir, stem):
    """Write columns in exactly the same order as Output_template."""
    df = pd.DataFrame(rows)
    scored, dropped = scorable(task, rows)
    if dropped:
        print(f"[{task}] excluded {dropped}/{len(rows)} unannotated rows from the metrics",
              flush=True)
    sdf = pd.DataFrame(scored)

    if task == "desire":
        columns = ["uid", "dialogue_id", "agent", "gold_desire", "pred", "raw_response", "desire_em"]
        # Desire_EM is the headline number and excludes unannotated rows. Desire_EM_all keeps them
        # (where the model scores 0 by construction) purely so the figure can be lined up against a
        # published table that may have used the other convention. Both come from the same rows —
        # no extra API calls, and the raw output is unchanged either way.
        metrics = [{"metric": "Desire_EM", "score": sdf["desire_em"].mean()},
                   {"metric": "Desire_EM_all", "score": df["desire_em"].mean()}]
    elif task == "belief":
        columns = ["uid", "dialogue_id", "agent", "opponent", "gold_high", "gold_med",
                   "gold_low", "pred", "belief_em", "raw_response"]
        metrics = [{"metric": "Belief_EM", "score": sdf["belief_em"].mean()},
                   {"metric": "Belief_EM_all", "score": df["belief_em"].mean()}]
    else:
        columns = ["uid", "dialogue_id", "utt_idx", "target_utterance", "gold_intent",
                   "gold_bitmask", "pred_intents", "pred_bitmask", "raw_response"]
        gold = list(sdf["gold_bitmask"])
        pred = list(sdf["pred_bitmask"])
        metrics = [
            {"metric": "Intent_Micro_F1", "score": f1_score(gold, pred, average="micro", zero_division=0)},
            {"metric": "Intent_Macro_F1", "score": f1_score(gold, pred, average="macro", zero_division=0)},
        ]
    metrics.append({"metric": f"{task}_scored_rows", "score": len(scored)})
    # every row is still written out; only the metrics exclude the unannotated ones
    df.reindex(columns=columns).to_csv(os.path.join(out_dir, stem + ".csv"), index=False)
    pd.DataFrame(metrics, columns=["metric", "score"]).to_csv(
        os.path.join(out_dir, stem + "_overall.csv"), index=False
    )


def run_desire(data, call_api, model, script_dir, shard, total_shards, save_every, pilot=False):
    rows = []
    for sample in data:
        for agent, prefix in (("agent_1", "agent1_desire"), ("agent_2", "agent2_desire")):
            # Use the cutoff-aware flat fields, not the sample["agent{N}_desire"] dict: the dict is
            # always a complete Food/Water/Firewood permutation, so it can never express the
            # "Not Given" / "None" labels the benchmark's Desire label set requires.
            gold_desire = {
                "High": sample[prefix + "_high"],
                "Medium": sample[prefix + "_medium"],
                "Low": sample[prefix + "_low"],
            }
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": sample["dialogue"],
                         "agent": agent, "gold_desire": gold_desire})
    rows = shard_slice(rows, shard, total_shards)
    out_dir, stem, checkpoint = output_paths(script_dir, "desire", model, shard, total_shards, pilot)
    done, results = load_checkpoint(checkpoint)
    try:
        for item in rows:
            uid = f"{item['dialogue_id']}_{item['agent']}_desire"
            if uid in done:
                continue
            raw, pred = call_and_parse(call_api, desire_messages(item["dialogue"], item["agent"]), model)
            results.append({"uid": uid, "dialogue_id": item["dialogue_id"], "agent": item["agent"],
                            "gold_desire": item["gold_desire"], "pred": pred,
                            "raw_response": raw or "", "desire_em": desire_em(pred, item["gold_desire"])})
            done.add(uid)
            if len(results) % save_every == 0:
                save_checkpoint(checkpoint, results)
    finally:
        # Runs even when a halt raises SystemExit mid-loop, so an abort never
        # discards the up-to-save_every-1 items completed since the last write.
        save_checkpoint(checkpoint, results)
    write_task_outputs("desire", results, out_dir, stem)


def run_belief(data, call_api, model, script_dir, shard, total_shards, save_every, pilot=False):
    rows = []
    for sample in data:
        for agent, opponent, prefix in (
            ("agent_1", "agent_2", "agent1_belief"), ("agent_2", "agent_1", "agent2_belief")
        ):
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": sample["dialogue"],
                         "agent": agent, "opponent": opponent,
                         "gold_high": sample[prefix + "_high"],
                         "gold_med": sample[prefix + "_medium"],
                         "gold_low": sample[prefix + "_low"]})
    rows = shard_slice(rows, shard, total_shards)
    out_dir, stem, checkpoint = output_paths(script_dir, "belief", model, shard, total_shards, pilot)
    done, results = load_checkpoint(checkpoint)
    try:
        for item in rows:
            uid = f"{item['dialogue_id']}_{item['agent']}_belief"
            if uid in done:
                continue
            messages = belief_messages(item["dialogue"], item["agent"], item["opponent"])
            raw, pred = call_and_parse(call_api, messages, model)
            results.append({"uid": uid, "dialogue_id": item["dialogue_id"], "agent": item["agent"],
                            "opponent": item["opponent"], "gold_high": item["gold_high"],
                            "gold_med": item["gold_med"], "gold_low": item["gold_low"], "pred": pred,
                            "belief_em": belief_em(pred, item["gold_high"], item["gold_med"], item["gold_low"]),
                            "raw_response": raw or ""})
            done.add(uid)
            if len(results) % save_every == 0:
                save_checkpoint(checkpoint, results)
    finally:
        # Runs even when a halt raises SystemExit mid-loop, so an abort never
        # discards the up-to-save_every-1 items completed since the last write.
        save_checkpoint(checkpoint, results)
    write_task_outputs("belief", results, out_dir, stem)


def run_intention(data, call_api, model, script_dir, shard, total_shards, save_every, pilot=False):
    rows = []
    for sample in data:
        turns = sample["dialogue"]
        # Only even-length dialogues end on a complete exchange and carry two annotated targets.
        # Odd-length ones (the final, uncut sample of a dialogue) annotate the last utterance only
        # and set utterance2_intent to the string "None" — which is not one of the 9 labels, so
        # scoring it would silently produce an all-zero gold bitmask.
        if sample["utterance2_intent"] == "None":
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": turns, "utt_idx": 1,
                         "target": turns[-1], "gold_intent": sample["utterance1_intent"]})
        else:
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": turns, "utt_idx": 1,
                         "target": turns[-2], "gold_intent": sample["utterance1_intent"]})
            rows.append({"dialogue_id": sample["dialogue_id"], "dialogue": turns, "utt_idx": 2,
                         "target": turns[-1], "gold_intent": sample["utterance2_intent"]})
    rows = shard_slice(rows, shard, total_shards)
    out_dir, stem, checkpoint = output_paths(script_dir, "intention", model, shard, total_shards, pilot)
    done, results = load_checkpoint(checkpoint)
    try:
        for item in rows:
            uid = f"{item['dialogue_id']}_utt{item['utt_idx']}_intention"
            if uid in done:
                continue
            raw, pred = call_and_parse(call_api, intention_messages(item["dialogue"], item["target"]), model)
            pred_intents = (pred or {}).get("intents", [])
            if isinstance(pred_intents, str):
                pred_intents = [part.strip() for part in pred_intents.split(",") if part.strip()]
            results.append({"uid": uid, "dialogue_id": item["dialogue_id"], "utt_idx": item["utt_idx"],
                            "target_utterance": item["target"], "gold_intent": item["gold_intent"],
                            "gold_bitmask": intent_bitmask(item["gold_intent"]),
                            "pred_intents": pred_intents, "pred_bitmask": intent_bitmask(pred_intents),
                            "raw_response": raw or ""})
            done.add(uid)
            if len(results) % save_every == 0:
                save_checkpoint(checkpoint, results)
    finally:
        # Runs even when a halt raises SystemExit mid-loop, so an abort never
        # discards the up-to-save_every-1 items completed since the last write.
        save_checkpoint(checkpoint, results)
    write_task_outputs("intention", results, out_dir, stem)


def pilot_report(model, tasks, script_dir, stem, n_samples, n_full, elapsed):
    """Print the go/no-go summary for a pilot run: health counters, scores, and projections."""
    line = "=" * 68
    print(f"\n{line}\n PILOT REPORT — {model}\n{line}")

    print(f"\n[sample]  {n_samples} / {n_full} dialogues "
          f"({n_samples / n_full * 100:.1f}%), seed-fixed so every model sees the same subset")

    calls = STATS["calls"]
    print(f"\n[health]  successful calls : {calls}")
    print(f"          empty responses  : {STATS['empty']}")
    print(f"          API errors       : {STATS['errors']}")
    print(f"          JSON parse fails : {STATS['parse_fail']}")
    if calls == 0:
        print("\n  *** NO SUCCESSFUL CALLS — check the key name, SDK and model id before scaling up.")
    else:
        bad = STATS["empty"] + STATS["errors"] + STATS["parse_fail"]
        rate = bad / (calls + bad) * 100
        verdict = "looks healthy" if rate < 5 else "HIGH FAILURE RATE — investigate before scaling"
        print(f"          failure rate     : {rate:.1f}%  ({verdict})")

    print("\n[scores]")
    for task in tasks:
        path = os.path.join(script_dir, "results", "pilot", task, stem + "_overall.csv")
        if os.path.exists(path):
            for row in pd.read_csv(path).to_dict("records"):
                print(f"          {row['metric']:<18} {row['score']:.4f}")

    scale = n_full / n_samples if n_samples else 0
    print(f"\n[cost]    elapsed          : {elapsed / 60:.1f} min for {calls} calls")
    print(f"          projected full   : {elapsed * scale / 3600:.1f} h single-threaded, "
          f"{elapsed * scale / 3600 / 5:.1f} h across 5 shards")
    if STATS["usage_seen"]:
        pin, pout = STATS["prompt_tokens"], STATS["completion_tokens"]
        print(f"          tokens           : {pin:,} in / {pout:,} out "
              f"({STATS['usage_seen']} API responses reported usage)")
        print(f"          output/response  : {pout / STATS['usage_seen']:.1f} tokens average")
        print(f"          projected full   : {pin * scale:,.0f} in / {pout * scale:,.0f} out")
        price = PRICE_PER_1M.get(model)
        if price:
            cost = (pin * price["in"] + pout * price["out"]) / 1e6
            print(f"          projected cost   : ${cost * scale:.2f} USD")
        else:
            print(f"          projected cost   : add '{model}' to PRICE_PER_1M to get a figure")
    else:
        print("          tokens           : this SDK did not report usage")

    budget_report()
    print(f"{line}\n")


def budget_report():
    """Print what the run measured about its own token and latency budget.

    Called from the pilot report and again at the end of a full run. A full run is the largest
    sample this benchmark will ever take of a provider's behaviour, and printing it only for
    pilots threw that away — the numbers that finally explained Gemma came from a full run's logs,
    reconstructed by hand from row counts and timeout lines because nothing recorded them.
    """
    # The mean cannot size max_tokens: half of all calls exceed it by construction. Size from
    # the tail, and size the watchdog from the observed latency of calls that actually returned.
    tokens = percentiles(STATS["completion_lengths"])
    seconds = percentiles(STATS["call_seconds"])
    # Print the header unconditionally. Gating it on "did anything succeed" meant a run in which
    # every single call hung produced no budget block at all — silence, which reads as "fine".
    print("\n[budget]  measured, for the next run's max_tokens and call timeout")
    if not seconds and STATS["timeouts"]:
        print(f"          *** {STATS['timeouts']} attempts, ZERO returned. No latency or token "
              f"distribution exists to size from. Every call hit the {HARD_CALL_TIMEOUT}s ceiling; "
              f"the provider, the model id or the ceiling itself is wrong.")
    if not seconds and not STATS["timeouts"]:
        print("          no calls were made")
    if tokens:
        print(f"          output tokens    : median {tokens['median']}  p90 {tokens['p90']}  "
              f"p99 {tokens['p99']}  max {tokens['max']}  (n={tokens['n']})")
        suggested = int(tokens["p99"] * 1.5)
        print(f"          suggested max_tokens : {suggested}  (p99 x 1.5)")
        if STATS["truncated"]:
            # The distribution is censored: every truncated call was cut off AT the current cap, so
            # the true p99 is unknown and larger than what was observed. Treating the suggestion as
            # a target rather than a lower bound would lock in the truncation it is meant to fix.
            print(f"          truncated at the cap : {STATS['truncated']}  *** these answers were "
                  f"cut off, so the tail above is CENSORED and the suggestion is a LOWER BOUND — "
                  f"raise the cap and re-measure")
        elif STATS["finish_reason_seen"]:
            print(f"          truncated at the cap : 0 of {STATS['finish_reason_seen']} checked "
                  f"(measured — the tail above is real)")
        else:
            # The distinction that stops a never-measured zero from reading as a clean bill.
            print("          truncated at the cap : UNKNOWN — this runner never passes "
                  "finish_reason, so truncation was never checked. The tail above may be censored. "
                  "Pass finish_reason_from(response) to record_call to find out.")
    if seconds:
        print(f"          call latency (s) : median {seconds['median']:.1f}  p90 {seconds['p90']:.1f}"
              f"  p99 {seconds['p99']:.1f}  max {seconds['max']:.1f}  (n={seconds['n']})")
    if seconds and STATS["timeouts"]:
        attempts = seconds["n"] + STATS["timeouts"]
        hang_rate = STATS["timeouts"] / attempts
        wasted = STATS["timeouts"] * HARD_CALL_TIMEOUT
        print(f"          watchdog kills   : {STATS['timeouts']} of {attempts} attempts "
              f"({hang_rate * 100:.1f}%), {wasted / 60:.0f} min at the {HARD_CALL_TIMEOUT}s ceiling")
        # The diagnosis that is easy to get backwards. If the slowest call that RETURNED is far
        # under the ceiling, the killed calls were not slow generations approaching the limit —
        # they were connections that produced nothing. Capping max_tokens does not touch those;
        # only a lower ceiling does. Getting this wrong costs a whole run.
        # Latency alone cannot separate the two, and reading it that way is what produced a wrong
        # call here once already. A run whose answers are being truncated shows FAST successes —
        # the cap cuts them short — which looks exactly like "hung connections" if you only look
        # at how long the survivors took. Truncation and the token tail have to be consulted too.
        tok_rate = (sum(STATS["completion_lengths"]) / sum(STATS["call_seconds"])
                    if STATS["call_seconds"] and STATS["completion_lengths"] else 0)
        if tok_rate:
            print(f"          throughput       : {tok_rate:.1f} output tok/s over calls that "
                  f"returned — a call using the full budget takes budget/{tok_rate:.1f} s, which "
                  f"must sit well inside the {HARD_CALL_TIMEOUT}s ceiling")
        if STATS["truncated"]:
            print(f"          diagnosis        : INCONCLUSIVE — {STATS['truncated']} responses "
                  f"were truncated, so the observed latency is of shortened calls and cannot be "
                  f"read as evidence about the killed ones. Raise max_tokens, re-measure, then "
                  f"diagnose.")
        elif seconds["max"] < HARD_CALL_TIMEOUT * 0.5:
            floor = max(20, int(seconds["max"] * 2))
            print(f"          diagnosis        : HUNG CONNECTIONS, not slow generation — the "
                  f"slowest call that returned took {seconds['max']:.0f}s, well under the "
                  f"{HARD_CALL_TIMEOUT}s ceiling, and nothing was truncated")
            print(f"          suggested timeout: set_call_timeout({floor})  (2x the slowest real "
                  f"call); at this hang rate that recovers "
                  f"{STATS['timeouts'] * (HARD_CALL_TIMEOUT - floor) / 60:.0f} min")
        else:
            print("          diagnosis        : calls are genuinely running to the ceiling — cap "
                  "max_tokens first, and only then lower the timeout")


def run_cli(call_api, default_model, script_file):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    benchmark_root = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--task", default="all", choices=["desire", "belief", "intention", "all"])
    parser.add_argument("--data", default=os.path.join(benchmark_root, "NegotiationToM.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--pilot", action="store_true",
                        help="run a fraction of the data, write to results/pilot/, print a report")
    parser.add_argument("--pilot-frac", type=float, default=0.10)
    parser.add_argument("--pilot-seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.shard < args.total_shards:
        parser.error("--shard must be in [0, --total-shards)")
    if args.pilot and not 0 < args.pilot_frac <= 1:
        parser.error("--pilot-frac must be in (0, 1]")

    with open(args.data, encoding="utf-8") as handle:
        data = json.load(handle)
    n_full = len(data)

    if args.pilot:
        k = max(1, round(n_full * args.pilot_frac))
        # Shuffle once with a fixed seed, then take a prefix — so a smaller --pilot-frac yields a
        # strict subset of a larger one. (random.sample would return an unrelated set instead,
        # making a cheaper pilot for a slow model incomparable with the others.)
        order = list(range(n_full))
        random.Random(args.pilot_seed).shuffle(order)
        data = [data[i] for i in order[:k]]
        print(f"[pilot] {k}/{n_full} dialogues ({k / n_full * 100:.1f}%), seed={args.pilot_seed}")

    # Name the run before any call can fail, so every halt path can write its marker into this
    # model's own folder, and drop any marker left by a previous run rather than inheriting it.
    set_run_context(args.model, script_dir)
    clear_stale_halt_markers(script_dir)

    tasks = ["desire", "belief", "intention"] if args.task == "all" else [args.task]
    started = time.time()
    for task in tasks:
        globals()[f"run_{task}"](
            data, call_api, args.model, script_dir, args.shard, args.total_shards,
            args.save_every, args.pilot,
        )
    elapsed = time.time() - started

    tag = f"_shard{args.shard}of{args.total_shards}" if args.total_shards > 1 else ""
    stem = model_slug(args.model) + tag
    if args.task == "all":
        results_root = os.path.join(script_dir, "results", "pilot" if args.pilot else "")
        summary = []
        for task in ("desire", "belief", "intention"):
            path = os.path.join(results_root, task, stem + "_overall.csv")
            if os.path.exists(path):
                summary.extend(pd.read_csv(path).to_dict("records"))
        summary_path = os.path.join(results_root, stem + "_negotiation_overall.csv")
        pd.DataFrame(summary, columns=["metric", "score"]).to_csv(summary_path, index=False)

    if args.pilot:
        pilot_report(args.model, tasks, script_dir, stem, len(data), n_full, elapsed)
    else:
        # A full run is the largest sample of provider behaviour this benchmark ever takes. Print
        # the same budget block so the next run's max_tokens and timeout can be set from it rather
        # than reconstructed from row counts and log lines after the fact.
        print(f"\n[{args.model}] shard {args.shard} finished in {elapsed / 60:.1f} min")
        budget_report()
