"""Gemini 3.5 Flash-Lite via OpenRouter — MultiChallenge response generation. **The project's Gemini
slot.**

No prompt, loop or scorer here: all of that is `../mc_eval_core.py::run_cli`, identical for every
model. This file is the client and the parameters this model is settled on. Judging is
`../MC_Judges/mc_judge.py`.

**Settled config** (`model-parameters.md` § Settled — the project's Gemini): OpenRouter, thinking at
`effort: "minimal"` — the floor, there is no off — `seed=42`, `temperature` unset (omit
`temperature`, `top_p`, `top_k` on 3.x). The thinking cap travels in `extra_body`, so it is `fixed`:
sent on every call, never negotiated away. Output cap `max_tokens=8192`, the same for every model
here (`judge_core.GEN_MAX_TOKENS`): answers are free-form, sometimes long rewrites.

**OpenRouter switches backend mid-run** (Google AI Studio and Vertex, with failover) and the answer
follows the backend, so a seed alone does not reproduce. The answering backend is on every row
(`backend`). **Visible reasoning:** the README gives no instruction; nothing is added.

    python gemini35lite_mc_eval.py --limit 3     # smoke, writes smoke/full/
    python gemini35lite_mc_eval.py               # all 273 items, writes results/full/
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
    return jc.openai_class()(base_url="https://openrouter.ai/api/v1",
                             api_key=jc.api_key("OPENROUTER_API_KEY"), timeout=300, max_retries=0)


GEN = jc.Generator(
    "google/gemini-3.5-flash-lite", client,
    wanted={"max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    fixed={"extra_body": {"reasoning": {"effort": "minimal"}}},
    call_timeout=300)

if __name__ == "__main__":
    mc.run_cli(GEN, MODEL_DIR, "MultiChallenge — Gemini 3.5 Flash-Lite responses (OpenRouter)")
