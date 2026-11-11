"""Gemini 3.5 Flash-Lite via OpenRouter — PlanBench runner. **The project's Gemini slot.**

No prompt, no scorer and no run loop here: all of that is `plan_eval_core.py` (generation) and
`plan_scorer.py` (offline scoring), identical for every model. This file is the client and the
parameters this model is settled on.

**Settled config** (`model-parameters.md` § Settled — the project's Gemini): OpenRouter, thinking at
`effort: "minimal"` (the floor; there is no off on this model), `seed=42`, `temperature` unset. The
thinking cap travels in `extra_body`, so it is `fixed`: sent on every call, never negotiated away.

**Output cap `max_tokens=8192` — chosen, not measured**: bbh's declared cap for this model, raised
from AwareBench's 2048 because a PlanBench answer is a multi-step plan. The 10% pilot's
`finish_length` column confirms or refutes it.

**OpenRouter switches backend mid-run**, and the answer follows the backend, so the answering
backend is written on every row (`backend`), not assumed per run.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import plan_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                timeout=400, max_retries=0)

WANTED = {"max_tokens": 8192, "seed": 42}
FIXED = {"extra_body": {"reasoning": {"effort": "minimal"}}}


def once(messages, model, params):
    r = client.chat.completions.create(model=model, messages=messages, **FIXED, **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="PlanBench — Gemini 3.5 Flash-Lite (OpenRouter)",
                 default_model="google/gemini-3.5-flash-lite", model_dir=MODEL_DIR, client=client,
                 once=once, wanted=WANTED, fixed=FIXED, required={"max_tokens": 8192},
                 call_timeout=300)
