"""GPT-5.6-Luna — AwareBench runner. **The project's OpenAI slot.**

No scorer, no prompt and no run loop here: all of that is `aware_eval_core.py`, identical for every
model. This file is the client and the parameters this model is settled on.

**Settled config** (`model-parameters.md` § Settled — the project's GPT): `reasoning_effort="low"`,
`max_completion_tokens=2048`, `seed=42`; `temperature` is not settable. Measured on EmoBench, whose
items are the same shape as AwareEval's — a short scenario and a one-token answer. `low`, never
`none`: `none` removes thinking rather than capping it, and cost 9-11 points there.

**The surface is negotiated at startup.** This model refuses `max_tokens` by name and refuses the
VALUE `reasoning_effort="minimal"`. Dropping the parameter because one value was refused would run
all 4,015 rows at the default `medium`, uncapped — so `required` pins the VALUE: anything but `low`
after negotiation stops the run.
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=300, max_retries=0)

# `max_completion_tokens` counts reasoning as well as the visible answer.
WANTED = {"max_completion_tokens": 2048, "reasoning_effort": "low", "seed": 42}


def once(prompt, model, params):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="AwareBench — GPT-5.6-Luna (OpenAI platform)",
                 default_model="gpt-5.6-luna", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, required={"reasoning_effort": "low", "max_completion_tokens": 2048}, call_timeout=200)
