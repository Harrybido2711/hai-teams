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


# Thinking is billed at the output rate but reported separately, so leaving it unbounded is an
# open cheque. Every answer this benchmark asks for is a fixed-shape JSON object of roughly 20
# tokens, so the budget is set as a multiple of that rather than an arbitrary number: thinking may
# cost several times the answer, not hundreds of times it.
#
# Measured on 15 belief items: thinking off scored 7/15 at 15 output tokens per call, and a 512
# budget scored the same 7/15 at 396 — identical accuracy for 26x the tokens. Unbounded, a single
# call was observed spending 1,165 thinking tokens on a 14-token answer, i.e. 98.8% of the billed
# output discarded. 256 sits well above the point where accuracy stopped improving while capping
# the tail that produced the surprise invoice.
ANSWER_TOKENS = 32          # a 3-field JSON object, with slack
THINKING_MULTIPLE = 8
THINKING_BUDGET = ANSWER_TOKENS * THINKING_MULTIPLE      # 256


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
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
                    # max_output_tokens stays unset: a 256 cap once truncated the JSON mid-object.
                    # The thinking budget is the right lever — it bounds the expensive part without
                    # risking the answer itself.
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
