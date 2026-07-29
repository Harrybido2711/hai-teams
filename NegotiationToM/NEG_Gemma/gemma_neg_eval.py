import os
import sys
import time

from dotenv import load_dotenv
from together import Together

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    record_call, record_empty, record_error, record_usage, retry_delay, run_cli, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
# Short per-request timeout: with the inherited 18000s the first pilot hung inside a single call
# for over 2 hours, still reported as RUNNING by SLURM and with nothing in the log. A stuck request
# must fail fast so the retry loop can do its job.
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=300)


# Together returns an empty string for this model intermittently at HTTP 200, so the retry budget
# is larger here than for the other providers.
def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=8192
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                record_usage(*usage_from(response))
                record_empty()
                print(f"[{model}] empty response ({attempt + 1}/{max_retries}), retrying", flush=True)
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
    run_cli(call_api, "google/gemma-4-31B-it", __file__)
