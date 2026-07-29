import os
import sys
import time

from dotenv import load_dotenv
from together import Together

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    record_call, record_empty, record_error, retry_delay, run_cli, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
# Qwen3.5 is a thinking model and its reasoning length is unstable: on the same prompt at
# temperature=0 it ranged from ~700 to over 32,768 output tokens, and when the budget runs out
# mid-thought Together returns finish_reason=Length with empty `content` (the reasoning is in a
# separate field). Two measures work together, since neither is enough alone:
#   * BREVITY_HINT below asks for terse reasoning, which lowers the typical cost
#   * a large max_tokens absorbs whatever tail remains
# The short client timeout stays as a floor, backed by the SIGALRM watchdog in neg_eval_core —
# Together's client has not reliably honoured `timeout` on its own.
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=180)

# Qwen-specific, deliberately not in neg_eval_core's shared prompt builders: the other five models
# answer the identical prompt, and only Qwen needs its reasoning reined in. It asks for brevity
# rather than banning reasoning outright, so the model keeps whatever short chain it needs.
BREVITY_HINT = (
    "\n\nBe concise. Keep any reasoning to at most one short sentence, "
    "then output the JSON object."
)

# Measured on real pilot items with the brevity hint in place, at ~90 output tok/s:
#
#   budget   succeeded   a success costs   a failure costs
#   8192     2 of 5      48-51s, ~3.5-4.7k tokens   ~91s, Length
#   16384    0 of 3      —                          ~150-184s, Length
#
# **A larger budget makes things worse, not better** — the model tends to spend whatever it is
# given. Item #80 is the proof: it finished cleanly at 8192 using 4,665 tokens, and burned all
# 16,384 on the identical prompt at temperature=0. So the budget is not a safety margin for a
# fixed amount of work; it shapes how much work the model does.
#
# 8192 therefore accommodates every successful call ever observed (max 5,615 tokens) while capping
# a runaway at ~91s. Failures return promptly with finish_reason=Length, and the retry gets a fresh
# draw — with a ~40% per-attempt success rate, call_api's 5 attempts give ~92% per item.
#
# (Without the brevity hint, 8192 truncated almost everything. The hint is what makes a small
# budget viable; the two changes only work together.)
MAX_TOKENS = 8192


def with_brevity_hint(messages):
    """Append the hint to the last user turn, leaving the shared prompt untouched."""
    out = [dict(m) for m in messages]
    for message in reversed(out):
        if message.get("role") == "user":
            message["content"] = message["content"] + BREVITY_HINT
            break
    return out


def call_api(messages, model, max_retries=5):
    messages = with_brevity_hint(messages)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=MAX_TOKENS
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                record_empty()
                finish = getattr(response.choices[0], "finish_reason", "?")
                used = getattr(getattr(response, "usage", None), "completion_tokens", "?")
                print(f"[{model}] empty response ({attempt + 1}/{max_retries}), "
                      f"finish={finish} tokens={used}, retrying", flush=True)
                time.sleep(5)
                continue
            record_call(*usage_from(response))
            time.sleep(2)
            return content
        except Exception as error:
            record_error()
            text = str(error).lower()
            print(f"[{model}] API error ({attempt + 1}/{max_retries}): "
                  f"{type(error).__name__}: {error}", flush=True)
            if "insufficient_quota" in text:
                raise SystemExit("Together quota exhausted") from error
            if "requests per day" in text:
                print(f"[{model}] daily request quota exhausted", flush=True)
                return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "Qwen/Qwen3.5-9B", __file__)
