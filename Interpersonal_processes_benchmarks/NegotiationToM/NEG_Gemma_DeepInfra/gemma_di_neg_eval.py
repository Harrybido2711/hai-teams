import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from Interpersonal_processes_benchmarks.NegotiationToM.neg_eval_core import (  # noqa: E402
    backoff_sleep, finish_reason_from, halt_on_billing, record_call, record_empty, record_error,
    record_usage, retry_delay, run_cli, set_call_timeout, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
# DeepInfra is OpenAI-compatible, so this is the same client shape as NEG_Deepseek and NEG_GPT —
# only the base_url and the key differ. 300s rather than NEG_Deepseek's 7200: DeepSeek queues
# rather than rejecting and can legitimately take half an hour, while DeepInfra's own docs describe
# a 200-concurrent limit that returns 429 instead of holding the connection open.
#
# max_retries=5 is NOT the SDK default and is not decoration. The two clients ship different hidden
# retry budgets — together 1.5.30 uses MAX_RETRIES=5, openai 2.6.1 uses DEFAULT_MAX_RETRIES=2 — so
# leaving both unset would make one harness "call" up to 6 HTTP attempts on Together and 3 here.
# That is not merely unequal, it is unequal in a way that is invisible afterwards and that biases
# toward this experiment's hypothesis: a 429 the Together SDK absorbs never reaches the harness, so
# it stays out of STATS["errors"] and its retry seconds are recorded as call latency, while the same
# event here raises, counts as an error, and takes a backoff_sleep that is subtracted from
# call_seconds. Identical provider behaviour, different columns of the comparison table. Matching
# Together's 5 rather than zeroing both keeps the Together side continuous with the per-shard
# production runs this is meant to be compared against.
client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai",
    timeout=300,
    max_retries=5,
)

# Measured 2026-08-06 on three production belief prompts: completion_tokens 15-17 against a
# 17-token answer, i.e. no reasoning overhead at all, and no <think> tags in the output. DeepInfra
# appears to serve this model with thinking already off, so unlike every other provider of a
# reasoning-capable model in this repo there is no knob to set:
#
#   Together   (NEG_Gemma, this same checkpoint)  on by default  -> reasoning={"enabled": False}
#   DeepSeek   (NEG_Deepseek)                     on, DELIBERATELY LEFT ON
#   Gemini 2.5 (NEG_Gemini)                       on by default  -> ThinkingConfig(thinking_budget=0)
#   DeepInfra  (here)                             already off    -> nothing
#
# Do not "helpfully" add a reasoning parameter here. DeepInfra accepts reasoning_effort, and passing
# it would turn on the very thing the other runners spend paragraphs disabling.
#
# The evidence for "already off" is weaker than the Together row's: that one rests on
# measure_gemma_reasoning.py with 12 calls per arm and a committed log, this one on three probe
# calls whose script was never committed. It does not need to be stronger, because the claim cannot
# fail silently — finish_reason reaches record_call on every path and budget_report prints the
# output-token median/p90/p99/max, so hidden reasoning shows up as a token tail far above 15-17.
# Check that tail in the pilot report before trusting this comment.
MAX_TOKENS = 8192

# Watchdog. 120 < the client's 300s socket timeout, so SIGALRM is the PRIMARY guard here and the
# socket timeout is unreachable by construction. That is deliberate: NEG_Gemma pairs the identical
# 120/300, so a hang is classified the same way on both sides (STATS["timeouts"], not
# STATS["errors"]) and the comparison stays symmetric. retry_delay_cap() is max(5, min(60, 120/4))
# = 30s on both sides for the same reason.
set_call_timeout(120)

# The model page warns that with thinking disabled the model "will still generate the tags but with
# an empty thought block". That did not happen in the probe, but an empty <think></think> prefix
# would make parse_json fail on an otherwise correct answer, and stripping it costs nothing.
#
# Matches everything up to and including the LAST closing think tag, anchored at the start. Written
# this way rather than as <think>...</think> because that shape is only one of the four this family
# actually emits, and the other three each cost three deterministic parse retries (temperature=0, so
# the retries return byte-identical text and cannot succeed) plus a row scored zero:
#
#   <think>reasoning</think>{json}     the documented shape
#   reasoning</think>{json}            no opening tag — what a chat template that pre-fills
#                                      "<think>" produces, and the most likely shape here
#   <think>a<think>b</think>{json}     duplicated opener; a non-greedy match stops at the first
#                                      close and leaves a dangling tag that breaks parse_json
#   <|think|>reasoning<|/think|>{json} pipe-delimited spelling
#
# Greedy .* takes the last close, which handles the nested case. `^` keeps it a prefix strip, so an
# answer that merely mentions a tag after the JSON is never touched. The residual risk — a literal
# closing tag inside a legitimate string value — cannot arise here: all three prompts request
# fixed-shape JSON whose only values are item names and intent labels.
THINK_TAG = re.compile(r"^.*<\|?\s*/\s*think\s*\|?>", re.DOTALL | re.IGNORECASE)

# NEG_Deepseek's results are labelled "deepseek-reasoner", which is an alias rather than a model,
# so the served model behind those 14k rows is not recoverable from the data. Log it once here and
# again whenever it changes, so this run always carries evidence of what actually answered.
_seen_served = set()


# max_retries, the post-success sleep, backoff_sleep and the exponential back-off below are all
# copied from NEG_Gemma/gemma_neg_eval.py rather than chosen independently. This runner exists to be
# timed against that one, so every constant that moves elapsed time, latency or the empty-response
# count has to be identical on both sides; the provider client is meant to be the only difference.
# An earlier draft used 0.2s pacing and max_retries=3, which alone would have shifted the measured
# speed ratio by 10-30% and made DeepInfra look both faster and less accurate for reasons that were
# purely harness configuration.
def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=MAX_TOKENS,
            )

            served = getattr(response, "model", None)
            if served and served not in _seen_served:
                _seen_served.add(served)
                print(f"[{model}] served by: {served}", flush=True)

            # Passed through so the health report can tell a short answer apart from one truncated
            # at MAX_TOKENS. Without it the measured output-token tail is silently censored and the
            # next run's max_tokens gets set from a distribution that excluded its own failures.
            finish = finish_reason_from(response)
            raw = (response.choices[0].message.content or "").strip()
            content = THINK_TAG.sub("", raw).strip()

            # Only the strip may have emptied it. Keep the original so the row records what the
            # model actually said: an answer wrapped entirely in think tags is a mis-shaped answer,
            # not an empty response, and counting it as empty would corrupt the one health metric
            # that distinguishes these two providers (Together's known defect is intermittent empty
            # HTTP-200 bodies). Returning the raw text lets call_and_parse fail at parse_json with
            # the body preserved, which is what the other five runners do.
            if raw and not content:
                print(f"[{model}] response was entirely inside think tags; "
                      f"returning it unstripped for parse_json", flush=True)
                record_call(*usage_from(response), finish_reason=finish)
                time.sleep(2)
                return raw

            # An HTTP 200 with an empty body raises nothing and is the most common failure mode in
            # this project, so it is retried rather than written as a scored zero.
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
            # DeepInfra documents no daily cap; a 429 here means the 200-concurrent ceiling. Through
            # backoff_sleep so the wait is not charged to the watchdog or to the recorded latency,
            # and with attempt= so it grows — the expected error on this provider is a concurrency
            # 429, and a flat retry against a throttled endpoint is what keeps it throttled.
            if attempt + 1 < max_retries:
                backoff_sleep(retry_delay(error, attempt=attempt))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "google/gemma-4-31B-it", __file__)
