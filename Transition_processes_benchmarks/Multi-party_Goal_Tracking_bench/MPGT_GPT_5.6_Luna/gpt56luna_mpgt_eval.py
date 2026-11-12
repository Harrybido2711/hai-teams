"""GPT-5.6-Luna — mpgt-eval, zero-shot DST and goal tracking. **The project's OpenAI slot.**

No prompt, loop or scorer here: all of that is `../mpgt_eval_core.py::run_cli`, identical for every
model. Default: both tasks x all six upstream prompt styles x 29 dialogues = 348 calls, into
`results/zero-shot/`.

**Settled config** (`model-parameters.md` § Settled — the project's GPT), taken whole from
`judge_core` as MultiChallenge does: `reasoning_effort="low"`, `max_completion_tokens=8192`,
`seed=42`; `temperature` is not settable on this model (upstream ran `gpt-3.5-turbo` at 0.4 — a
declared deviation). `required` pins the VALUES, so `low` cannot quietly become `medium`.
**Visible reasoning:** the README says nothing; upstream's own `reasoning` style asks the model to
step through, and that is the prompt, not an addition. Nothing is added.

    python gpt56luna_mpgt_eval.py --dry-run                      # build every prompt, call nothing
    python gpt56luna_mpgt_eval.py --limit 1                      # smoke: 1 dialogue x 6 styles x 2 tasks
    python gpt56luna_mpgt_eval.py                                # all 348, writes results/zero-shot/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mpgt_eval_core as mp  # noqa: E402
jc = mp.jc

GEN = jc.Generator(jc.GEN_MODEL, jc.gen_client, jc.GEN_WANTED, jc.GEN_REQUIRED, call_timeout=300)

if __name__ == "__main__":
    mp.run_cli(GEN, MODEL_DIR, "mpgt-eval zero-shot — gpt-5.6-luna (OpenAI platform)")
