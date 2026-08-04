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
    retry_delay, run_cli, set_call_timeout, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
# Short per-request timeout: with the inherited 18000s the first pilot hung inside a single call
# for over 2 hours, still reported as RUNNING by SLURM and with nothing in the log. A stuck request
# must fail fast so the retry loop can do its job. Together ignores this argument — the real guard
# is the SIGALRM watchdog below — but it costs nothing to state the intent.
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=300)


# Measured on job 8526978 (2026-08-04, 4h20m of full run, five shards) before it was cancelled:
#
#   successful call    ~10s, at ~57 tok/s for ~567 billed output tokens
#   hung calls         ~9% of attempts returned nothing at all and were killed by the watchdog
#   cost of that       9% x 200s = 67% of the job's entire wall clock
#
# The run was tracking towards 97h against a 19.6h projection, and the cause was not slow
# generation: at 57 tok/s even a runaway to the full 8192 budget returns in 144s, inside the 200s
# ceiling. These were hung connections, the failure Together has shown here before (ISSUES.md:
# one request stuck for over two hours at timeout=18000).
#
# So the lever is the ceiling, not the token budget. Both are set below, because they interact:
# the ceiling can only be lowered safely once the slowest *legitimate* call is bounded, and that
# bound is max_tokens / observed tokens-per-second.
#
#   MAX_TOKENS 2048  -> 3.6x the 567-token mean; worst legitimate call ~36s at 57 tok/s, and ~100s
#                       even if the server halves its speed. The visible answer is ~15 tokens, so
#                       this is headroom for hidden reasoning, not for the JSON.
#   timeout    90s   -> 2.5x that worst legitimate call, so it cannot cut good work, while a hang
#                       now costs 90s instead of 200s.
#
# Expected effect: mean seconds per completed row falls from ~30s to ~19s, roughly 1.6x.
# These are the first values sized this way rather than inherited; budget_report() now prints the
# real latency and token percentiles at every checkpoint and at the end, so the next revision is a
# measurement rather than another estimate. If truncation shows up in that report, raise MAX_TOKENS
# first — a censored tail makes every other number in the block a lower bound.
MAX_TOKENS = 2048
set_call_timeout(90)


# Together returns an empty string for this model intermittently at HTTP 200, so the retry budget
# is larger here than for the other providers.
def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=MAX_TOKENS
            )
            finish = finish_reason_from(response)
            content = (response.choices[0].message.content or "").strip()
            if not content:
                record_usage(*usage_from(response), finish_reason=finish)
                record_empty()
                print(f"[{model}] empty response ({attempt + 1}/{max_retries}), "
                      f"finish_reason={finish}, retrying", flush=True)
                time.sleep(5)
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
                time.sleep(retry_delay(error))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "google/gemma-4-31B-it", __file__)
