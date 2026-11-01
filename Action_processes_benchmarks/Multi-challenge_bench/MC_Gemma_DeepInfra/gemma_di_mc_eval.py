"""Gemma 4 31B via DeepInfra — MultiChallenge response generation. **The project's Gemma slot.**

No prompt, loop or scorer here: all of that is `../mc_eval_core.py::run_cli`, identical for every
model. This file is the client and the parameters this model is called with. Judging is
`../MC_Judges/mc_judge.py`.

**DeepInfra, not Together** (`model-calls.md`). DeepInfra serves this checkpoint with thinking
already off. **Pass no reasoning parameter here**: DeepInfra accepts `reasoning_effort`, and passing
it turns thinking back ON.

A model that does not reason gets an explicit output cap (`model-parameters.md` rule 2):
`max_tokens=8192` (`judge_core.GEN_MAX_TOKENS`, the same for every model here — answers are
free-form, sometimes long rewrites). `temperature=0` and `seed=42` are set even where they equal a
default (rule 4); the seed is negotiated, because whether DeepInfra accepts it is unestablished.
**SIGALRM at 300 s, not the recipe's 120:** the recipe was measured on one-token answers, and 8,192
tokens of rewrite at this model's speed can need most of 300 s. **Visible reasoning:** the README
gives no instruction; nothing is added.

    python gemma_di_mc_eval.py --limit 3     # smoke, writes smoke/full/
    python gemma_di_mc_eval.py               # all 273 items, writes results/full/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mc_eval_core as mc  # noqa: E402
jc = mc.jc


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
    mc.run_cli(GEN, MODEL_DIR, "MultiChallenge — Gemma 4 31B responses (DeepInfra)")
