import os
import sys
import time

from dotenv import load_dotenv
from together import Together

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from Interpersonal_processes_benchmarks.NegotiationToM.neg_eval_core import (  # noqa: E402
    halt_on_billing, record_call, record_empty, record_error, record_usage, retry_delay, run_cli, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
# Qwen3.5-9B is a hybrid reasoning model. Its default thinking mode previously consumed 8,192
# tokens repeatedly and returned empty content. Together supports a real per-request reasoning
# toggle, so disable reasoning instead of trying to shape hidden CoT with prompt hints and retries.
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=180)
MAX_TOKENS = 8192


def call_api(messages, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=MAX_TOKENS,
                reasoning={"enabled": False},
            )
            content = (response.choices[0].message.content or "").strip()
            if not content:
                record_usage(*usage_from(response))
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
            halt_on_billing(error, model, SCRIPT_DIR)
            if "requests per day" in text:
                print(f"[{model}] daily request quota exhausted", flush=True)
                return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "Qwen/Qwen3.5-9B", __file__)
