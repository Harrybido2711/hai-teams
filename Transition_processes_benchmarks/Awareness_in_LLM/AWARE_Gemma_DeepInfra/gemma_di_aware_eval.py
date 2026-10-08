"""Gemma 4 31B via DeepInfra — AwareBench runner. **The project's Gemma slot.**

No scorer, no prompt and no run loop here: all of that is `aware_eval_core.py`, identical for every
model. This file is the client and the parameters this model is called with.

**DeepInfra, not Together** (`PLAN.md`, `model-calls.md`). DeepInfra serves this checkpoint with
thinking already off — measured 15-17-token answers, no `<think>` tags. **Pass no reasoning
parameter here**: DeepInfra accepts `reasoning_effort`, and passing it turns thinking back ON. The
same checkpoint on Together is the opposite, which is why NEG_Gemma and this file differ.

A model that does not reason gets an explicit output cap (`model-parameters.md` rule 2):
`max_tokens=2048`, far above an answer-only reply, so a `length` finish means something is wrong.
`temperature=0` and `seed=42` are set even where they equal a default (rule 4); the seed is
negotiated, because whether DeepInfra accepts it is unestablished (rule 8).

SIGALRM at 120 s is the primary guard, below the 300 s socket timeout (`model-calls.md`).
`max_retries=0`, unlike NEG_Gemma_DeepInfra's 5: the core makes the attempts itself, so every one
is counted in `n_attempts` (`script-skeleton.md` §4).
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import aware_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai", timeout=300, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 2048, "seed": 42}


def once(prompt, model, params):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="AwareBench — Gemma 4 31B (DeepInfra)",
                 default_model="google/gemma-4-31B-it", model_dir=MODEL_DIR, client=client,
                 once=once, wanted=WANTED, required={"max_tokens": 2048}, call_timeout=120)
