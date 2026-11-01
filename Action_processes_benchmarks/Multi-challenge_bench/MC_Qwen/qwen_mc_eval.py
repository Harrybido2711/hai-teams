"""Qwen3.5-9B via Together — MultiChallenge response generation. **The project's Qwen slot.**

No prompt, loop or scorer here: all of that is `../mc_eval_core.py::run_cli`, identical for every
model. This file is the client and the parameters this model is called with. Judging is
`../MC_Judges/mc_judge.py`.

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}`
(`model-parameters.md`). Qwen3.5-9B is a hybrid model whose thinking length varied from ~700 to past
32,768 tokens for the same prompt at `temperature=0`; one pilot produced 60 rows in 7 hours. The
switch is `fixed` — sent on every call, never negotiated away.

Output cap `max_tokens=8192` (rule 2; `judge_core.GEN_MAX_TOKENS`, the same for every model here —
answers are free-form, sometimes long rewrites). `temperature=0` and `seed=42` are pinned (rules 4
and 8); the seed is negotiated, since Together's acceptance of it on this model is unestablished.
Together's client has ignored `timeout=` outright, so the core's SIGALRM watchdog is the real guard.
**Visible reasoning:** the README gives no instruction; nothing is added.

    python qwen_mc_eval.py --limit 3     # smoke, writes smoke/full/
    python qwen_mc_eval.py               # all 273 items, writes results/full/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mc_eval_core as mc  # noqa: E402
jc = mc.jc


def client():
    from together import Together
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
    # makes up to 5 attempts itself (script-skeleton.md §4).
    return Together(api_key=jc.api_key("TOGETHER_API_KEY"), timeout=180, max_retries=0)


GEN = jc.Generator(
    "Qwen/Qwen3.5-9B", client,
    wanted={"temperature": 0, "max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    fixed={"reasoning": {"enabled": False}},
    call_timeout=300)

if __name__ == "__main__":
    mc.run_cli(GEN, MODEL_DIR, "MultiChallenge — Qwen3.5-9B responses (Together, reasoning off)")
