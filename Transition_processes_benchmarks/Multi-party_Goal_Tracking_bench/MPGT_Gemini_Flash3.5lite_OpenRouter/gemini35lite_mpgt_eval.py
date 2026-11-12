"""Gemini 3.5 Flash-Lite via OpenRouter — mpgt-eval, zero-shot DST and goal tracking. **The project's
Gemini slot.**

No prompt, loop or scorer here: all of that is `../mpgt_eval_core.py::run_cli`, identical for every
model. Default: both tasks x all six upstream prompt styles x 29 dialogues = 348 calls.

**Settled config** (`model-parameters.md` § Settled — the project's Gemini): OpenRouter, thinking at
`effort: "minimal"` — the floor, there is no off — in `extra_body`, so it is `fixed`: sent on every
call, never negotiated away. `seed=42`; `temperature` unset (omit it on 3.x). Output cap
`max_tokens=8192` (`judge_core.GEN_MAX_TOKENS`, the same for every model here).
**OpenRouter switches backend mid-run** and the answer follows the backend; the answering backend is
on every row (`backend`). **Visible reasoning:** the README says nothing; nothing is added.

    python gemini35lite_mpgt_eval.py --limit 1     # smoke, writes smoke/zero-shot/
    python gemini35lite_mpgt_eval.py               # all 348, writes results/zero-shot/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mpgt_eval_core as mp  # noqa: E402
jc = mp.jc


def client():
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
    # makes up to 5 attempts itself (script-skeleton.md §4).
    return jc.openai_class()(base_url="https://openrouter.ai/api/v1",
                             api_key=jc.api_key("OPENROUTER_API_KEY"), timeout=300, max_retries=0)


GEN = jc.Generator(
    "google/gemini-3.5-flash-lite", client,
    wanted={"max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    fixed={"extra_body": {"reasoning": {"effort": "minimal"}}},
    call_timeout=300)

if __name__ == "__main__":
    mp.run_cli(GEN, MODEL_DIR, "mpgt-eval zero-shot — Gemini 3.5 Flash-Lite (OpenRouter)")
