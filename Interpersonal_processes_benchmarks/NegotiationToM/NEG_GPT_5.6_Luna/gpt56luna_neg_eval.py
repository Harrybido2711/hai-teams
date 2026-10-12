"""GPT-5.6-Luna — NegotiationToM runner. **The project's OpenAI slot.**

**There is no scorer in this file.** Every metric is computed by `neg_eval_core`, the one lenient
matcher all NegotiationToM models are judged by. This file supplies `call_api` and nothing else.

This slot has never been run on NegotiationToM. `NEG_GPT/` holds `gpt-4o-mini`, which is
superseded — re-pointing the OpenAI column means running this, not editing a header.

**The parameter surface is negotiated at startup, not hardcoded.** This model refuses `max_tokens`
by name (it must be `max_completion_tokens`) and refuses the *value* `reasoning_effort="minimal"`
while supporting the parameter perfectly. A rejection is permanent, so discovering it per item
costs the run; and dropping `reasoning_effort` because one value was refused would run all 14,138
rows at the model default `medium`, uncapped, breaching the standing thinking-cap rule with nothing
in the log saying so. `core.negotiate` handles both cases and the run aborts if the parameter is
dropped outright. Never ask for `reasoning_effort="none"` — that is removal, not a cap, and costs
9-11 points (`.claude/references/model-parameters.md`).
"""

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

DEFAULT_MODEL = "gpt-5.6-luna"

# What we ask for; what is actually used is whatever negotiate() settles on, written to
# results/negotiated_params_<pilot|shardNofM>.json beside the numbers.
#
# `max_completion_tokens` counts reasoning tokens as well as visible ones. A NegotiationToM answer
# is a small JSON object and `low` effort measured a median of 203 reasoning tokens on an
# EmoBench-shaped item, so 8192 is generous. It is NOT 16384 (what the bbh runner asks for) on
# purpose: the core's SIGALRM watchdog fires at 200s and a full 16384-token generation at ~90 tok/s
# would outlive it, turning a truncation — which finish_reason reports — into a recorded hang,
# which reports nothing. CHOSEN, NOT MEASURED: run run_pilot.sh first and read `truncated` in the
# budget block before committing the full run to it.
WANTED = {"max_completion_tokens": 8192, "reasoning_effort": "low", "seed": 42}

PARAMS = {}

load_dotenv(os.path.join(ROOT, ".env"))
# timeout=300 deliberately, not the 18000 the older runners inherited: five hours per request means
# one hung call stalls the job with SLURM still reporting RUNNING
# (`.claude/references/provider-gotchas.md`).
#
# max_retries=0: the SDK otherwise retries 429/5xx/timeouts twice inside one create(), silently, so
# a row's n_attempts would undercount and its latency_s would include the back-off
# (`.claude/references/script-skeleton.md` §4). call_api's loop is the only retry layer, with 5
# attempts to make up for the two the SDK no longer spends.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=300, max_retries=0)


def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            started = time.monotonic()
            response = client.chat.completions.create(model=model, messages=messages, **PARAMS)
            latency = round(time.monotonic() - started, 3)
            reasoning = reasoning_from(response)
            content = (response.choices[0].message.content or "").strip()
            # finish_reason is passed on BOTH paths on purpose. Five of the six older runners call
            # record_call(*usage_from(response)) and never supply it, so their "truncated: 0" means
            # "never looked" rather than "looked and found none" — and an empty response from a
            # reasoning model is most often a budget that was spent entirely on thinking, which is
            # exactly the case this field distinguishes from a wrong answer.
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
    # Every shard negotiates for itself, so each records its own: one untagged file was
    # last-writer-wins and could not show a shard that settled on something different.
    run_tag = run_tag_of(args)

    enable_row_fields()
    print(f"negotiating parameter surface for {model} ...", flush=True)
    try:
        # The probe writes no row, so it keeps the SDK's own retries: a transient 5xx here would
        # otherwise end the job before the first item.
        PARAMS, notes = negotiate(client.with_options(max_retries=2), model, WANTED)
    except Exception as error:
        halt_on_billing(error, model, SCRIPT_DIR)    # leaves BILLING_HALT.txt / QUOTA_HALT.txt
        raise
    print(f"  accepted: {PARAMS}" + (f"  [{'; '.join(notes)}]" if notes else ""), flush=True)
    if "reasoning_effort" not in PARAMS:
        raise SystemExit("reasoning_effort was dropped entirely — that runs uncapped at the model "
                         "default and breaches the thinking-cap rule. Fix the request rather than "
                         "proceeding.")
    if "seed" not in PARAMS:
        print("  WARNING: seed was dropped — this run is NOT reproducible "
              "(model-parameters.md rule 8)", flush=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    params_path = os.path.join(SCRIPT_DIR, "results", f"negotiated_params_{run_tag}.json")
    if args.tag and os.path.exists(params_path):
        # A tag resumes only on the parameters its first start accepted: a provider that has
        # since stopped honouring seed would otherwise mix seeded and unseeded rows in one tag.
        with open(params_path, encoding="utf-8") as handle:
            first = json.load(handle).get("accepted")
        if first != PARAMS:
            raise SystemExit(f"results/{args.tag} started with {first}, the provider now "
                             f"accepts {PARAMS}; archive the tag before rerunning.")
    with open(params_path, "w") as handle:
        json.dump({"model": model, "asked": WANTED, "accepted": PARAMS, "notes": notes},
                  handle, indent=2)

    run_cli(call_api, DEFAULT_MODEL, __file__)
