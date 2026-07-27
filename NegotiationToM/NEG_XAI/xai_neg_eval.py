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
from neg_eval_core import retry_delay, run_cli  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))
client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=3600)


def call_api(messages, model, max_retries=3):
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
                time.sleep(5)
                continue
            time.sleep(2)
            return content
        except Exception as error:
            text = str(error).lower()
            if "insufficient_quota" in text:
                raise SystemExit("xAI quota exhausted") from error
            if "requests per day" in text:
                return None
            if attempt + 1 < max_retries:
                time.sleep(retry_delay(error))
    return None


if __name__ == "__main__":
    run_cli(call_api, "grok-3-mini", __file__)
