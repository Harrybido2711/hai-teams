"""Qwen3.5-9B via Together — AwareBench runner. **The project's Qwen slot.**

No scorer, no prompt and no run loop here: all of that is `aware_eval_core.py`, identical for every
model. This file is the client and the parameters this model is called with.

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}`
(`model-parameters.md`). Qwen3.5-9B is a hybrid model whose thinking length varied from ~700 to past
32,768 tokens for the same prompt at `temperature=0`; one pilot produced 60 rows in 7 hours. Measured
off: 11/12 completions against 6/12, 16 output tokens against 517. The switch is `fixed` — sent on
every call, never negotiated away — and the prompt stays the shared one.

Output cap `max_tokens=2048` (rule 2), far above an answer-only reply. `temperature=0` and `seed=42`
are pinned (rules 4 and 8); the seed is negotiated, since Together's acceptance of it on this model
is unestablished.

Together's client has ignored `timeout=` outright, so the core's SIGALRM watchdog is the real guard.
"""

import os
import sys

from dotenv import load_dotenv
from together import Together

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import aware_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
# max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
# makes up to 5 attempts itself (script-skeleton.md §4).
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=180, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 2048, "seed": 42}
FIXED = {"reasoning": {"enabled": False}}


def once(prompt, model, params):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], **FIXED, **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="AwareBench — Qwen3.5-9B (Together, reasoning off)",
                 default_model="Qwen/Qwen3.5-9B", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, fixed=FIXED, required={"max_tokens": 2048}, call_timeout=200)
