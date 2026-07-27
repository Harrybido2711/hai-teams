import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)
from neg_eval_core import retry_delay, run_cli  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def call_api(messages, model, max_retries=3):
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
                ),
            )
            # Intentionally omit max_output_tokens so Gemini can use its provider maximum.
            content = (response.text or "").strip()
            if not content:
                time.sleep(5)
                continue
            time.sleep(2)
            return content
        except Exception as error:
            text = str(error).lower()
            if "insufficient_quota" in text:
                raise SystemExit("Gemini quota exhausted") from error
            if "requests per day" in text or "resource_exhausted" in text:
                if "per day" in text:
                    return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    return None


if __name__ == "__main__":
    run_cli(call_api, "gemini-2.5-flash", __file__)
