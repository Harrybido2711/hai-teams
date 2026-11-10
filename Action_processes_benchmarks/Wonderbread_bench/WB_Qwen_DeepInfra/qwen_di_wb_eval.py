"""Qwen3.5-9B via DeepInfra — Wonderbread QA answers and generated SOPs. **The project's Qwen slot.**

Replaces `../WB_Qwen/` (Together) for Wonderbread: the user's decision of 2026-11-09, to lift the image
cap from Together's 10 to DeepInfra's 30 and rerun every unit on one route. `../WB_Qwen/` and its
results stay as the record of the Together run; they are not mixed with these.

No prompt, loop or scorer here: all of that is `../wb_eval_core.py::run_cli`, identical for every
model — what the model sees is built by upstream's own `generate_response` / `run`, screenshots as
base64 `image_url` parts (this model takes images, `model-calls.md`). `--set full` (default): 120 QA
answers, 162 rank-1 demos x 3 ablations = 486 SOPs, into `results/full/`. Judging is
`../WB_Judges/wb_judge.py --producer WB_Qwen_DeepInfra`.

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}` in `extra_body`
(`model-parameters.md`) — `fixed`, sent on every call, never negotiated away. **On DeepInfra thinking
is ON by default for this model** (probed 2026-11-04: without the switch a one-line question ran 51 s
and hit the cap with empty `content`), so this switch is not optional — unlike Gemma's route. Output
cap `max_tokens=8192` (rule 2; `judge_core.GEN_MAX_TOKENS`, the same for every model here).
`temperature=0` and `seed=42` pinned (rules 4 and 8); DeepInfra accepts the seed on this model.
**Visible reasoning:** upstream's prompts ask for a 2-3 sentence answer (QA) or an SOP; nothing is
added.

**At most 30 images per request — DeepInfra's cap for this model** ("Too many images in request:
32 > 30", HTTP 400, probed 2026-11-04 with small PNGs; real downscaled screenshots probed
2026-11-09, `provider-gotchas.md`). `max_images=30`: a request over it is never sent; the unit is
written `over_image_limit` and is not judged — the user's rule of 2026-10-31. On the full set: 7 of
120 QA items and 2 of 162 demos in each image ablation. `route` is written into the config, since the
model id is the same on Together and a row must say which provider produced it.

    python qwen_di_wb_eval.py --task qa --dry-run     # build every request, call nothing
    python qwen_di_wb_eval.py --task qa --limit 6     # smoke, writes smoke/full/
    python qwen_di_wb_eval.py --task sop              # all 162 demos, writes results/full/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import wb_eval_core as wb  # noqa: E402
jc = wb.jc


def client():
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
    # makes up to 5 attempts itself (script-skeleton.md §4).
    return jc.openai_class()(base_url="https://api.deepinfra.com/v1/openai",
                             api_key=jc.api_key("DEEPINFRA_API_KEY"), timeout=600, max_retries=0)


GEN = jc.Generator(
    "Qwen/Qwen3.5-9B", client,
    wanted={"temperature": 0, "max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    # In `extra_body`, not a top-level kwarg: the OpenAI SDK refuses an unknown `reasoning=` with a
    # TypeError (Together's SDK accepted it, which is how ../WB_Qwen/ passes it).
    fixed={"extra_body": {"reasoning": {"enabled": False}}},
    # 600 s, not the siblings' 300: on 2026-11-09 this route took 102 s for a 335-token SOP, and
    # Together's Qwen SOPs reach 1,240 tokens at p99 — at that rate near or past 300 s, which would
    # spend 5 x 300 s and write `failed`. A real hang still ends here; it just costs more wall clock.
    call_timeout=600, max_images=30, extra_config={"route": "deepinfra"})

if __name__ == "__main__":
    wb.run_cli(GEN, MODEL_DIR, "Wonderbread — Qwen3.5-9B generation (DeepInfra, reasoning off)")
