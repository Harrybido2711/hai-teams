import os
import sys
import time

from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import system as xai_system
from xai_sdk.chat import user as xai_user

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    record_call, record_empty, record_error, record_usage, retry_delay, run_cli, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
# The .env files in this repo name this key GROK_API_KEY (same as bbh/ and EmoBench-master/).
client = Client(api_key=os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"), timeout=3600)


def call_api(messages, model, max_retries=3):
    # xai_sdk does not take message dicts: system/user are appended as objects before sampling.
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    for attempt in range(max_retries):
        try:
            chat = client.chat.create(model=model, max_tokens=8192, temperature=0)
            chat.append(xai_system(system))
            chat.append(xai_user(user))
            response = chat.sample()
            content = (response.content or "").strip()
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
                raise SystemExit("xAI quota exhausted") from error
            if "requests per day" in text:
                print(f"[{model}] daily request quota exhausted", flush=True)
                return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "grok-3-mini", __file__)
