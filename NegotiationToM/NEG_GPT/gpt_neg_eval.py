import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    record_call, record_empty, record_error, retry_delay, run_cli, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"), timeout=18000)


def call_api(messages, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=0, max_tokens=8192
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
            message = str(error).lower()
            print(f"[{model}] API error ({attempt + 1}/{max_retries}): "
                  f"{type(error).__name__}: {error}", flush=True)
            if "insufficient_quota" in message:
                raise SystemExit("OpenAI quota exhausted") from error
            if "requests per day" in message:
                print(f"[{model}] daily request quota exhausted", flush=True)
                return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "gpt-4o-mini", __file__)
