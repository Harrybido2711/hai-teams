"""Gemini 3.5 Flash-Lite via OpenRouter — AwareBench runner. **The project's Gemini slot.**

No scorer, no prompt and no run loop here: all of that is `aware_eval_core.py`, identical for every
model. This file is the client and the parameters this model is settled on.

**Settled config** (`model-parameters.md` § Settled — the project's Gemini): OpenRouter, thinking at
`effort: "minimal"`, `max_tokens=2048`, `seed=42`, `temperature` unset. Measured on EmoBench, whose
items are the same shape as AwareEval's. `minimal` is the floor — there is no off on this model —
and it spent zero thinking tokens over 400 items. Omit `temperature`, `top_p`, `top_k`.

The thinking cap travels in `extra_body`, so it is `fixed`: sent on every call, never negotiated
away.

**OpenRouter switches backend mid-run** (Google AI Studio and Vertex, with failover), and the answer
follows the backend, so a seed alone does not reproduce. The answering backend is written on every
row (`backend`), not assumed per run.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import aware_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
# max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
# makes up to 5 attempts itself (script-skeleton.md §4).
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                timeout=300, max_retries=0)

WANTED = {"max_tokens": 2048, "seed": 42}
FIXED = {"extra_body": {"reasoning": {"effort": "minimal"}}}


def once(prompt, model, params):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], **FIXED, **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="AwareBench — Gemini 3.5 Flash-Lite (OpenRouter)",
                 default_model="google/gemini-3.5-flash-lite", model_dir=MODEL_DIR, client=client,
                 once=once, wanted=WANTED, fixed=FIXED, required={"max_tokens": 2048}, call_timeout=200)
