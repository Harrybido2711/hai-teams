import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import (  # noqa: E402
    record_call, record_empty, record_error, retry_delay, run_cli, usage_from,
)

load_dotenv(os.path.join(ROOT, ".env"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def call_api(messages, model, max_retries=3):
    # Gemini takes no "system" role in the message list; the system prompt goes in the config.
    system = next(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0,
                    response_mime_type="application/json",
                    # max_output_tokens omitted on purpose: a 256 cap once truncated the JSON
                    # mid-object, so let Gemini use its provider maximum.
                ),
            )
            content = (response.text or "").strip()
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
                raise SystemExit("Gemini quota exhausted") from error
            if "per day" in text:
                print(f"[{model}] daily request quota exhausted", flush=True)
                return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    print(f"[{model}] all {max_retries} attempts failed, giving up on this item", flush=True)
    return None


if __name__ == "__main__":
    run_cli(call_api, "gemini-2.5-flash", __file__)
