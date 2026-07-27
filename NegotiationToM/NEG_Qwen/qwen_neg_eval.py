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
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=18000)


def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            # Qwen3.5 is a thinking model and its reasoning length is highly variable: measured on
            # NegotiationToM prompts it ranged over 25k-53k characters for the *same* prompt at
            # temperature=0. Whenever the budget runs out mid-thought, Together returns
            # finish_reason=Length with an empty `content` (the thinking sits in `reasoning`).
            # 8192 — which is enough for the short BBH prompts — truncates most calls here, so the
            # budget is 32768. A successful call typically spends only ~7k tokens; the headroom is
            # there for the long tail, not the average.
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=32768
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
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
    run_cli(call_api, "Qwen/Qwen3.5-9B", __file__)
