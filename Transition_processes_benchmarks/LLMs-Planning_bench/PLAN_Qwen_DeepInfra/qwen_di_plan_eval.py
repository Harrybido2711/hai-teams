"""Qwen3.5-9B via DeepInfra — PlanBench runner. **The project's Qwen slot.**

No prompt, no scorer and no run loop here: all of that is `plan_eval_core.py` (generation) and
`plan_scorer.py` (offline scoring), identical for every model. This file is the client and the
parameters this model is called with.

**DeepInfra, not Together** — new Qwen runs use DeepInfra from 2026-11-04 (`model-calls.md`).

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}`. Thinking is ON by
default on this route too; without the switch a one-line question ran 51 s into `finish=length` with
an empty `content`. On the OpenAI SDK it must travel in `extra_body` — top-level it raises TypeError
(`model-calls.md`, 2026-11-09). It is `fixed`: sent on every call, never negotiated away.

Output cap `max_tokens=4096` (rule 2) — chosen, not measured, as twice AwareBench's 2048 because a
PlanBench answer is a multi-step plan; the 10% pilot's `finish_length` column shows whether it binds.
`temperature=0` and `seed=42` pinned (rules 4 and 8), the seed negotiated.

This route ran 17-111 s a call on 2026-11-09 (`model-calls.md`), so the SIGALRM ceiling is 600 s,
as in `Wonderbread_bench/WB_Qwen_DeepInfra/`.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import plan_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai", timeout=900, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 4096, "seed": 42}
FIXED = {"extra_body": {"reasoning": {"enabled": False}}}


def once(messages, model, params):
    r = client.chat.completions.create(model=model, messages=messages, **FIXED, **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="PlanBench — Qwen3.5-9B (DeepInfra, reasoning off)",
                 default_model="Qwen/Qwen3.5-9B", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, fixed=FIXED, required={"max_tokens": 4096}, call_timeout=600,
                 extra_config={"route": "deepinfra"})
