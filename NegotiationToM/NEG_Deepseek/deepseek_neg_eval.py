import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    halt_on_billing, record_call, record_empty, record_error, record_usage, retry_delay, run_cli, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=7200
)


def call_api(messages, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            # deepseek-reasoner is a reasoning model; that is the point of using it here, and it is
            # what bbh/ and EmoBench-master/ run too. Reasoning is not disabled: this benchmark
            # measures theory-of-mind inference, and switching it off would change what is being
            # measured rather than making it cheaper to measure.
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=8192,
            )
            message = response.choices[0].message
            content = (message.content or "").strip()
            # The answer sometimes arrives in reasoning_content with content left empty.
            if not content:
                content = (getattr(message, "reasoning_content", None) or "").strip()
                if content:
                    print(f"[{model}] content empty, fell back to reasoning_content", flush=True)
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
            halt_on_billing(error, model, SCRIPT_DIR)
            if "requests per day" in text:
                print(f"[{model}] daily request quota exhausted", flush=True)
                return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "deepseek-reasoner", __file__)
