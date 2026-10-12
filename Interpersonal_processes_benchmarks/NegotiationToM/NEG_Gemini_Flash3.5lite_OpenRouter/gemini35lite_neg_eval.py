"""Gemini 3.5 Flash-Lite via OpenRouter — NegotiationToM runner. **The project's Gemini slot.**

**There is no scorer in this file.** Every metric is computed by `neg_eval_core`, the one lenient
matcher all NegotiationToM models are judged by. This file supplies `call_api` and nothing else.

The folder carries the route because there are two and only one is settled: OpenRouter, set by the
user 2026-08-23 (`.claude/references/model-calls.md`). The native Google AI Studio route measured
the same speed but takes a different config shape, and Quest's google-genai is a version behind —
its `ThinkingConfig` has no `thinking_level` field, which raises a pydantic ValidationError on
every call rather than failing once.

This slot has never been run on NegotiationToM. `NEG_Gemini/` holds `gemini-2.5-flash` on the
native SDK, which is superseded **and whose rows are not a usable result**: its daily quota ran out
mid-run (`NEG_Gemini/QUOTA_HALT.txt`, 2026-07-29) and 1,963 of its rows came back empty, which
score as wrong answers. That is the column this runner is meant to replace.

**`minimal` is the thinking floor; there is no off.** `thinking_budget=0` is rejected 400
INVALID_ARGUMENT on this model. `reasoning.effort` goes through `extra_body`, so OpenRouter passes
it to the backend rather than validating it here — it is not part of the negotiation.
**Omit `temperature`, `top_p`, `top_k`** — 3.x guidance is to remove, not tune.

**OpenRouter switches backend mid-run.** It serves this model from Google AI Studio *and* Google
Vertex with failover; over one finished 400-item run the split was 110/90. A seed alone therefore
does not reproduce a row. The response carries `provider`, so this runner tallies it and writes the
count beside the results — a score that moved may be a backend change rather than the model.
"""

import atexit
import collections
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    backoff_sleep, enable_row_fields, finish_reason_from, halt_on_billing, negotiate, parse_cli,
    reasoning_from, record_call, record_empty, record_error, record_usage, retry_delay, run_cli,
    run_tag_of, usage_from,
)

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"

# `reasoning.effort` is not a named parameter, so it bypasses negotiate() — OpenRouter forwards it.
EXTRA_BODY = {"reasoning": {"effort": "minimal"}}

# CHOSEN, NOT MEASURED on NegotiationToM prompts — run run_pilot.sh and read `truncated` in the
# budget block first. 8192 matches what the other runners here give the visible JSON, and stays
# under the core's 200s SIGALRM watchdog at ~90 tok/s. Thinking is billed at the output rate but
# measured 0 tokens over 400 items at `minimal`, so this cap is effectively the visible budget.
WANTED = {"max_tokens": 8192, "seed": 42}

PARAMS = {}
BACKENDS = collections.Counter()
BACKENDS_WRITE_EVERY = 50
RUN_TAG = ["full"]       # pilot | shardNofM | full, set in __main__ before any call


def _write_backends(quiet=False):
    """Which backend answered is not recoverable from the rows afterwards, so the tally is written
    every BACKENDS_WRITE_EVERY calls and again at exit. atexit alone was not enough: it does not run
    on the SIGTERM a SLURM wall-clock kill sends, so such a run left no file at all. One file per
    shard: an untagged file was last-writer-wins and showed one shard's split as the run's."""
    if not BACKENDS:
        return
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    path = os.path.join(SCRIPT_DIR, "results", f"backend_providers_{RUN_TAG[0]}.json")
    with open(path, "w") as handle:
        json.dump(dict(BACKENDS), handle, indent=2)
    if not quiet:
        print("backends:", dict(BACKENDS), flush=True)


load_dotenv(os.path.join(ROOT, ".env"))
# max_retries=0: the SDK otherwise retries 429/5xx/timeouts twice inside one create(), silently, so
# a row's n_attempts would undercount and its latency_s would include the back-off
# (`.claude/references/script-skeleton.md` §4). call_api's loop is the only retry layer, with 5
# attempts to make up for the two the SDK no longer spends.
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"), timeout=300, max_retries=0)


def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            started = time.monotonic()
            response = client.chat.completions.create(
                model=model, messages=messages, extra_body=EXTRA_BODY, **PARAMS)
            latency = round(time.monotonic() - started, 3)
            reasoning = reasoning_from(response)
            # Counted before the content check: a backend that answers only with empty bodies is
            # exactly what this tally is for, and skipping it there would hide the culprit.
            BACKENDS[getattr(response, "provider", None) or "unreported"] += 1
            if sum(BACKENDS.values()) % BACKENDS_WRITE_EVERY == 0:
                _write_backends(quiet=True)
            content = (response.choices[0].message.content or "").strip()
            reason = finish_reason_from(response)
            if not content:
                record_usage(*usage_from(response), finish_reason=reason, reasoning_tokens=reasoning)
                record_empty()
                print(f"[{model}] empty response ({attempt + 1}/{max_retries}), "
                      f"finish_reason={reason}, retrying", flush=True)
                # backoff_sleep, not time.sleep: a deliberate wait must not count against the
                # core's 200s watchdog, which exists to catch a dead connection.
                backoff_sleep(5)
                continue
            record_call(*usage_from(response), finish_reason=reason, reasoning_tokens=reasoning,
                        latency_s=latency)
            time.sleep(2)
            return content
        except Exception as error:
            record_error()
            print(f"[{model}] API error ({attempt + 1}/{max_retries}): "
                  f"{type(error).__name__}: {error}", flush=True)
            # A real daily cap halts inside halt_on_billing. No "requests per day" early return
            # after it: what reaches this line is the seconds-scale RPD throttle ("try again in
            # 8.64s"), and returning there wrote an empty row after one request.
            halt_on_billing(error, model, SCRIPT_DIR)
            if attempt + 1 < max_retries:
                backoff_sleep(retry_delay(error, attempt=attempt))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    # The whole command line, parsed and validated before the probe: a rejected command spends no
    # call and writes no file. The probe negotiates against the model that will be called.
    args = parse_cli(DEFAULT_MODEL, __file__)
    model = args.model
    # Every shard negotiates and tallies for itself, so each records its own.
    RUN_TAG[0] = run_tag_of(args)

    enable_row_fields()
    print(f"negotiating parameter surface for {model} ...", flush=True)
    try:
        # The probe writes no row, so it keeps the SDK's own retries: a transient 5xx here would
        # otherwise end the job before the first item.
        PARAMS, notes = negotiate(client.with_options(max_retries=2), model, WANTED)
    except Exception as error:
        halt_on_billing(error, model, SCRIPT_DIR)    # an OpenRouter 402 leaves BILLING_HALT.txt
        raise
    print(f"  accepted: {PARAMS}" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    if "seed" not in PARAMS:
        print("  WARNING: seed was dropped — this run is NOT reproducible "
              "(model-parameters.md rule 8)", flush=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    params_path = os.path.join(SCRIPT_DIR, "results", f"negotiated_params_{RUN_TAG[0]}.json")
    if args.tag and os.path.exists(params_path):
        # A tag resumes only on the parameters its first start accepted: a provider that has
        # since stopped honouring seed would otherwise mix seeded and unseeded rows in one tag.
        with open(params_path, encoding="utf-8") as handle:
            first = json.load(handle).get("accepted")
        if first != PARAMS:
            raise SystemExit(f"results/{args.tag} started with {first}, the provider now "
                             f"accepts {PARAMS}; archive the tag before rerunning.")
    with open(params_path, "w") as handle:
        json.dump({"model": model, "asked": WANTED, "accepted": PARAMS,
                   "extra_body": EXTRA_BODY, "notes": notes}, handle, indent=2)

    atexit.register(_write_backends)
    run_cli(call_api, DEFAULT_MODEL, __file__)
