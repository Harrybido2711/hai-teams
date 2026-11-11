"""Gemma 4 31B via DeepInfra — PlanBench runner. **The project's Gemma slot.**

No prompt, no scorer and no run loop here: all of that is `plan_eval_core.py` (generation) and
`plan_scorer.py` (offline scoring), identical for every model. This file is the client and the
parameters this model is called with.

**DeepInfra serves this checkpoint with thinking already off. Pass no reasoning parameter here**:
DeepInfra accepts `reasoning_effort`, and passing it turns thinking back ON (`model-parameters.md`).

A model that does not reason gets an explicit output cap (rule 2): `max_tokens=4096` — chosen, not
measured, as twice AwareBench's 2048 because a PlanBench answer is a multi-step plan. A `length`
finish means something is wrong; the 10% pilot's `finish_length` column shows whether it binds.
`temperature=0` and `seed=42` are set even where they equal a default (rule 4); the seed is
negotiated, because whether DeepInfra honours it is unestablished (rule 8).
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
                base_url="https://api.deepinfra.com/v1/openai", timeout=300, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 4096, "seed": 42}


def once(messages, model, params):
    r = client.chat.completions.create(model=model, messages=messages, **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="PlanBench — Gemma 4 31B (DeepInfra)",
                 default_model="google/gemma-4-31B-it", model_dir=MODEL_DIR, client=client,
                 once=once, wanted=WANTED, required={"max_tokens": 4096}, call_timeout=240)
