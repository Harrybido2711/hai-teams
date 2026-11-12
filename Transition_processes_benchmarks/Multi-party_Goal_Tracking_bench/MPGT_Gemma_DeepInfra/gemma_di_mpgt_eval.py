"""Gemma 4 31B via DeepInfra — mpgt-eval, zero-shot DST and goal tracking. **The project's Gemma
slot.**

No prompt, loop or scorer here: all of that is `../mpgt_eval_core.py::run_cli`, identical for every
model. Default: both tasks x all six upstream prompt styles x 29 dialogues = 348 calls.

**DeepInfra serves this checkpoint with thinking already off. Pass no reasoning parameter:**
DeepInfra accepts `reasoning_effort`, and passing it turns thinking back ON (`model-calls.md`).
A model that does not reason gets an explicit output cap (`model-parameters.md` rule 2):
`max_tokens=8192` (`judge_core.GEN_MAX_TOKENS`). `temperature=0` and `seed=42` set even where they
equal a default (rule 4); the seed is negotiated. **Visible reasoning:** the README says nothing;
nothing is added.

    python gemma_di_mpgt_eval.py --limit 1     # smoke, writes smoke/zero-shot/
    python gemma_di_mpgt_eval.py               # all 348, writes results/zero-shot/
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
    return jc.openai_class()(base_url="https://api.deepinfra.com/v1/openai",
                             api_key=jc.api_key("DEEPINFRA_API_KEY"), timeout=300, max_retries=0)


GEN = jc.Generator(
    "google/gemma-4-31B-it", client,
    wanted={"temperature": 0, "max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    call_timeout=300)

if __name__ == "__main__":
    mp.run_cli(GEN, MODEL_DIR, "mpgt-eval zero-shot — Gemma 4 31B (DeepInfra)")
