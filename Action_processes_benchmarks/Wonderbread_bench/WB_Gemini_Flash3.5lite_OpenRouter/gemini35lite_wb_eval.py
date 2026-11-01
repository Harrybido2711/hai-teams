"""Gemini 3.5 Flash-Lite via OpenRouter — Wonderbread QA answers and generated SOPs. **The project's
Gemini slot.**

No prompt, loop or scorer here: all of that is `../wb_eval_core.py::run_cli`, identical for every
model — what the model sees is built by upstream's own `generate_response` / `run`, screenshots as
base64 `image_url` parts (this model takes images). `--set full` (default): 120 QA answers, 162
rank-1 demos x 3 ablations = 486 SOPs, into `results/full/`. Judging is `../WB_Judges/wb_judge.py`.

**Settled config** (`model-parameters.md` § Settled — the project's Gemini): OpenRouter, thinking at
`effort: "minimal"` — the floor, there is no off — `seed=42`, `temperature` unset. The thinking cap
travels in `extra_body`, so it is `fixed`: sent on every call, never negotiated away. Output cap
`max_tokens=8192` (`judge_core.GEN_MAX_TOKENS`, the same for every model here; upstream capped the
visible answer at 4096). **OpenRouter switches backend mid-run**; the answering backend is on every
row (`backend`). **Visible reasoning:** upstream's prompts ask for a 2-3 sentence answer (QA) or an
SOP; nothing is added.

    python gemini35lite_wb_eval.py --task qa --dry-run     # build every request, call nothing
    python gemini35lite_wb_eval.py --task qa --limit 6     # smoke, writes smoke/full/
    python gemini35lite_wb_eval.py --task sop              # all 162 demos, writes results/full/
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
    return jc.openai_class()(base_url="https://openrouter.ai/api/v1",
                             api_key=jc.api_key("OPENROUTER_API_KEY"), timeout=300, max_retries=0)


GEN = jc.Generator(
    "google/gemini-3.5-flash-lite", client,
    wanted={"max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    fixed={"extra_body": {"reasoning": {"effort": "minimal"}}},
    call_timeout=300)

if __name__ == "__main__":
    wb.run_cli(GEN, MODEL_DIR, "Wonderbread — Gemini 3.5 Flash-Lite generation (OpenRouter)")
