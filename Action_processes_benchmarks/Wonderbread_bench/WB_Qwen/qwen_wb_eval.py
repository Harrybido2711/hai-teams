"""Qwen3.5-9B via Together — Wonderbread QA answers and generated SOPs. **The project's Qwen slot.**

No prompt, loop or scorer here: all of that is `../wb_eval_core.py::run_cli`, identical for every
model — what the model sees is built by upstream's own `generate_response` / `run`, screenshots as
base64 `image_url` parts (this model takes images, `model-calls.md`). `--set full` (default): 120 QA
answers, 162 rank-1 demos x 3 ablations = 486 SOPs, into `results/full/`. Judging is
`../WB_Judges/wb_judge.py`.

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}`
(`model-parameters.md`) — `fixed`, sent on every call, never negotiated away. Thinking on this hybrid
model ran from ~700 to past 32,768 tokens for one prompt at `temperature=0`. Output cap
`max_tokens=8192` (rule 2; `judge_core.GEN_MAX_TOKENS`, the same for every model here; upstream
capped the visible answer at 4096). `temperature=0` and `seed=42` pinned (rules 4 and 8); the seed
is negotiated. Together's client has ignored `timeout=`, so the core's SIGALRM watchdog is the real
guard. **Visible reasoning:** upstream's prompts ask for a 2-3 sentence answer (QA) or an SOP;
nothing is added.

**At most 10 images per request — Together's cap for this model** ("model spec qwen3_vl supports at
most 10 image inputs", HTTP 400, probed 2026-10-31). `max_images=10`: a request over it is never
sent; the unit is written `over_image_limit` and is not judged, and Qwen is reported on the units
within its cap (the user's decision, 2026-10-31). On the full set that is ~37 of 120 QA items and
~53 of 162 demos in each image ablation (td_kf, td_kf_act). Together's other limit, 255 MiB of
decoded image payload per request, is met by the shared downscale (`wb_eval_core.IMAGE_POLICY`):
at full size, six screenshots already came to 556 MiB.

    python qwen_wb_eval.py --task qa --dry-run     # build every request, call nothing
    python qwen_wb_eval.py --task qa --limit 6     # smoke, writes smoke/full/
    python qwen_wb_eval.py --task sop              # all 162 demos, writes results/full/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import wb_eval_core as wb  # noqa: E402
jc = wb.jc


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
    call_timeout=300, max_images=10)

if __name__ == "__main__":
    wb.run_cli(GEN, MODEL_DIR, "Wonderbread — Qwen3.5-9B generation (Together, reasoning off)")
