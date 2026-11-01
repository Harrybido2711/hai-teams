"""GPT-5.6-Luna — Wonderbread QA answers and generated SOPs. **The project's OpenAI slot.**

No prompt, loop or scorer here: all of that is `../wb_eval_core.py::run_cli`, identical for every
model — what the model sees is built by upstream's own `generate_response` / `run`. `--set full`
(default): 120 QA answers, 162 rank-1 demos x 3 ablations = 486 SOPs, into `results/full/`.
`--set sample` is the judge pilot's (12 + 48), whose rows from job 4480407 are `results/{qa,sop}/`;
this code refuses to resume them (its config adds `fixed` and the image policy, which the pilot's
full-size screenshots did not have), so a sample run needs them archived. Judging is
`../WB_Judges/wb_judge.py`.

**Settled config** (`model-parameters.md` § Settled — the project's GPT): `reasoning_effort="low"`,
`seed=42`; `temperature` is unsettable (upstream's GPT-4 ran at 0 — declared deviation).
`max_completion_tokens=8192` (`judge_core.GEN_WANTED`): upstream capped the visible answer at 4096,
and this cap also has to hold `low` reasoning. **Visible reasoning:** upstream's prompts ask for a
2-3 sentence answer (QA) or an SOP; nothing is added.

    python gpt56luna_wb_eval.py --task qa --dry-run     # build every request, call nothing
    python gpt56luna_wb_eval.py --task sop --limit 1    # smoke: 1 demo x 3 ablations, smoke/full/
    python gpt56luna_wb_eval.py --task sop              # all 162 demos, writes results/full/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import wb_eval_core as wb  # noqa: E402
jc = wb.jc

GEN = jc.Generator(jc.GEN_MODEL, jc.gen_client, jc.GEN_WANTED, jc.GEN_REQUIRED, call_timeout=300)

if __name__ == "__main__":
    wb.run_cli(GEN, MODEL_DIR, "Wonderbread — gpt-5.6-luna generation (OpenAI platform)")
