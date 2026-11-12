"""Qwen3.5-9B via DeepInfra — mpgt-eval, zero-shot DST and goal tracking. **The project's Qwen slot.**

DeepInfra, not Together: new Qwen runs use DeepInfra from 2026-11-04 (`model-calls.md`). No prompt,
loop or scorer here: all of that is `../mpgt_eval_core.py::run_cli`, identical for every model.
Default: both tasks x all six upstream prompt styles x 29 dialogues = 348 calls.

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}` in `extra_body` —
`fixed`, sent on every call, never negotiated away. **On DeepInfra thinking is ON by default for this
model**, so the switch is not optional. In `extra_body`, not top-level: the OpenAI SDK refuses an
unknown `reasoning=` kwarg with a TypeError. Output cap `max_tokens=8192`
(`judge_core.GEN_MAX_TOKENS`); `temperature=0` and `seed=42` pinned (rules 4 and 8).
**This route is slow** — 17-111 s a call on Wonderbread, 2026-11-09 — so the SIGALRM ceiling is
600 s, as on `WB_Qwen_DeepInfra`. `route` is in the config: the model id is the same on Together.
**Visible reasoning:** the README says nothing; nothing is added.

    python qwen_di_mpgt_eval.py --limit 1     # smoke, writes smoke/zero-shot/
    python qwen_di_mpgt_eval.py               # all 348, writes results/zero-shot/
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
                             api_key=jc.api_key("DEEPINFRA_API_KEY"), timeout=600, max_retries=0)


GEN = jc.Generator(
    "Qwen/Qwen3.5-9B", client,
    wanted={"temperature": 0, "max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    fixed={"extra_body": {"reasoning": {"enabled": False}}},
    call_timeout=600, extra_config={"route": "deepinfra"})

if __name__ == "__main__":
    mp.run_cli(GEN, MODEL_DIR, "mpgt-eval zero-shot — Qwen3.5-9B (DeepInfra, reasoning off)")
