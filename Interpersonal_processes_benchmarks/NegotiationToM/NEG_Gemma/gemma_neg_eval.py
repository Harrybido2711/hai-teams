import os
import sys
import time

from dotenv import load_dotenv
from together import Together

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    finish_reason_from, halt_on_billing, record_call, record_empty, record_error, record_usage,
    backoff_sleep, retry_delay, run_cli, set_call_timeout, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
# Short per-request timeout: with the inherited 18000s the first pilot hung inside a single call
# for over 2 hours, still reported as RUNNING by SLURM and with nothing in the log. A stuck request
# must fail fast so the retry loop can do its job. Together ignores this argument — the real guard
# is the SIGALRM watchdog below — but it costs nothing to state the intent.
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=300)


# gemma-4-31B-it is a Together hybrid reasoning model, and leaving reasoning on is what made job
# 8526978 unusable: 132 rows per shard in 4h20m, 80% of the wall clock inside calls that never
# returned, and five shards running only 1.05x faster than one.
#
# Measured directly (measure_gemma_reasoning.py, 2026-08-04, 12 calls per arm on production
# prompts, max_tokens=8192 held constant in both):
#
#                        succeeded   median latency   median output tokens
#   reasoning ON             6/12            9.3 s                    517
#   reasoning OFF           11/12            0.3 s                     16
#
# 29.6x faster, 32x fewer output tokens, hang rate 50% -> 8%.
#
# Two earlier attempts at this were wrong and are recorded so they are not retried:
#
#   * "cap max_tokens" — there is nothing to cap. Every call that returned stopped on its own at
#     254-795 tokens, at most 9.7% of the 8192 budget, with finish_reason=stopsequence. Lowering
#     the cap changes no call's behaviour.
#   * "lower the watchdog" — the ceiling only makes each hang cheaper. The hangs produce ZERO
#     tokens in 240s, so they are not slow generations, and the rate stays the same.
#
# The lever is which serving path the request takes, and on Together that is a boolean: there is
# no thinking-budget parameter on chat.completions.create, only this passthrough toggle. Gemini's
# "cap it rather than disable it" has no equivalent here.
#
# This matches NEG_Qwen exactly, which reached 14,138 rows at 1h57m per shard with the same
# provider, the same client and the same 8192 budget — the one difference being this argument.
# It is also closer to the paper's baseline: NegotiationToM (arXiv 2404.13627) reports zero-shot,
# CoT and few-shot as three separate settings, our prompts carry no CoT wording, and hidden
# provider-side reasoning matches none of the three.
MAX_TOKENS = 8192

# Watchdog ceiling, 200s -> 120s. A hung call returns ZERO tokens, so most of a 200s wait buys no
# information; on the cancelled job 8526978 that waiting was 80% of the entire wall clock.
#
# 120 and not 60, which was the first choice and was too aggressive. 60 came from a single
# 11-success micro-probe whose slowest legitimate call took 50.1s — a 20% margin on a sample that
# small. The production reasoning-off log (log_pilot_ab.txt, 171 calls that returned) shows a mean
# of 18.3s but never printed a max, because the job was cancelled before budget_report ran, so the
# tail is genuinely unmeasured. Under a 60s ceiling every legitimate call in that unmeasured tail
# would be recorded as a hang, and after three of them the row is written with raw_response="" and
# scored zero. That corrupts both the data and the hang rate this run exists to interpret.
#
# 120s still removes 40% of the old wait while sitting far above anything observed. The pulse now
# prints latency percentiles, so the NEXT revision can be set from this run's own p99 instead of
# from an 11-point probe.
set_call_timeout(120)


# Together returns an empty string for this model intermittently at HTTP 200, so the retry budget
# is larger here than for the other providers.
def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=MAX_TOKENS,
                reasoning={"enabled": False},
            )
            finish = finish_reason_from(response)
            content = (response.choices[0].message.content or "").strip()
            if not content:
                record_usage(*usage_from(response), finish_reason=finish)
                record_empty()
                print(f"[{model}] empty response ({attempt + 1}/{max_retries}), "
                      f"finish_reason={finish}, retrying", flush=True)
                backoff_sleep(5)
                continue
            record_call(*usage_from(response), finish_reason=finish)
            time.sleep(2)
            return content
        except Exception as error:
            record_error()
            text = str(error).lower()
            print(f"[{model}] API error ({attempt + 1}/{max_retries}): "
                  f"{type(error).__name__}: {error}", flush=True)
            halt_on_billing(error, model, SCRIPT_DIR)
            if "requests per day" in text:
                print(f"[{model}] daily request quota exhausted", flush=True)
                return None
            if attempt + 1 < max_retries:
                # Pass the attempt so the wait grows. Together's limits are dynamic and shaped by
                # our traffic, so a fixed 5s retry against a throttled endpoint is what keeps it
                # throttled — and with many shards, an unjittered fixed wait makes them retry in
                # lockstep.
                backoff_sleep(retry_delay(error, attempt=attempt))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "google/gemma-4-31B-it", __file__)
