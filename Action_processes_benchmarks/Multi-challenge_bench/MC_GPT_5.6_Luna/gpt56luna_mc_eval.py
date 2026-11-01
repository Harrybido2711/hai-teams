"""GPT-5.6-Luna — MultiChallenge response generation. **The project's OpenAI slot.**

No prompt, loop or scorer here: all of that is `../mc_eval_core.py::run_cli`, identical for every
model. `--set full` (default) answers all 273 items into `results/full/`. `--set sample` is the
judge pilot's 27, whose rows from job 4480406 are `results/gpt-5.6-luna.jsonl`; this code refuses to
resume them (its config carries keys the pilot's did not), so a sample run needs that file archived.
Judging is `../MC_Judges/mc_judge.py`.

**Settled config** (`model-parameters.md` § Settled — the project's GPT): `reasoning_effort="low"`,
`seed=42`, `temperature` not settable (upstream ran its models at temp=0 — a declared deviation).
`max_completion_tokens=8192` (`judge_core.GEN_WANTED`): MultiChallenge answers are free-form,
sometimes long rewrites; upstream's OpenAI path set no cap at all. The surface is negotiated at
startup, and `required` pins the VALUES: `low` quietly becoming `medium` would be uncapped.
**Visible reasoning:** the README gives no instruction and the conversation is sent as stored, so
nothing is added to ask for or suppress it.

    python gpt56luna_mc_eval.py --limit 3     # smoke, writes smoke/full/
    python gpt56luna_mc_eval.py               # all 273 items, writes results/full/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mc_eval_core as mc  # noqa: E402
jc = mc.jc

GEN = jc.Generator(jc.GEN_MODEL, jc.gen_client, jc.GEN_WANTED, jc.GEN_REQUIRED, call_timeout=300)

if __name__ == "__main__":
    mc.run_cli(GEN, MODEL_DIR, "MultiChallenge — gpt-5.6-luna responses (OpenAI platform)")
