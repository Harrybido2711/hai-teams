"""GPT-5.6-Luna — PlanBench runner. **The project's OpenAI slot.**

No prompt, no scorer and no run loop here: all of that is `plan_eval_core.py` (generation) and
`plan_scorer.py` (offline scoring), identical for every model. This file is the client and the
parameters this model is settled on.

**Settled config** (`model-parameters.md` § Settled — the project's GPT): `reasoning_effort="low"`,
`seed=42`; `temperature` is not settable. `low`, never `none`: `none` removes thinking rather than
capping it. The surface is negotiated at startup and `required` pins the VALUE `low`, so a refused
value can never fall back silently to the default `medium`.

**Output cap `max_completion_tokens=16384` — chosen, not measured.** It counts reasoning as well as
the visible answer. PlanBench answers are multi-step plans after a 1-2k-token one-shot prompt, not
the one-token answers AwareBench's 2048 was measured on; 16384 is bbh's declared cap for this model.
The 10% pilot's `finish_length` column is what confirms or refutes it.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import plan_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
# max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
# makes up to 5 attempts itself (script-skeleton.md §4).
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=900, max_retries=0)

WANTED = {"max_completion_tokens": 16384, "reasoning_effort": "low", "seed": 42}


def once(messages, model, params):
    r = client.chat.completions.create(model=model, messages=messages, **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="PlanBench — GPT-5.6-Luna (OpenAI platform)",
                 default_model="gpt-5.6-luna", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, required={"reasoning_effort": "low", "max_completion_tokens": 16384},
                 call_timeout=600)
