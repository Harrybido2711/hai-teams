"""Gemma 4 31B via DeepInfra — Wonderbread QA answers and generated SOPs. **The project's Gemma slot.**

No prompt, loop or scorer here: all of that is `../wb_eval_core.py::run_cli`, identical for every
model — what the model sees is built by upstream's own `generate_response` / `run`, screenshots as
base64 `image_url` parts (this model takes images, `model-calls.md`). `--set full` (default): 120 QA
answers, 162 rank-1 demos x 3 ablations = 486 SOPs, into `results/full/`. Judging is
`../WB_Judges/wb_judge.py`.

**DeepInfra, not Together.** DeepInfra serves this checkpoint with thinking already off. **Pass no
reasoning parameter here**: DeepInfra accepts `reasoning_effort`, and passing it turns thinking back
ON. Output cap `max_tokens=8192` (rule 2; `judge_core.GEN_MAX_TOKENS`, the same for every model
here; upstream capped the visible answer at 4096). `temperature=0` and `seed=42` set even where they
equal a default (rule 4); the seed is negotiated. **SIGALRM at 300 s, not the recipe's 120:** the
recipe was measured on one-token answers, and these requests carry up to dozens of screenshots.
**Visible reasoning:** upstream's prompts ask for a 2-3 sentence answer (QA) or an SOP; nothing is
added.

**At most 32 images per request — DeepInfra's cap** ("Too many images in request: 55 > 32", HTTP
400, probed 2026-10-31 with the full set's largest requests). `max_images=32`: a request over it is
never sent; the unit is written `over_image_limit` and is not judged — the rule the user set for
Qwen's cap on 2026-10-31, applied here the same way. On the full set: 5 of 120 QA items and 2 of 162
demos in each image ablation. At full size, 30 screenshots hung past the watchdog instead of being
refused; the shared downscale (`wb_eval_core.IMAGE_POLICY`) cleared that.

    python gemma_di_wb_eval.py --task qa --dry-run     # build every request, call nothing
    python gemma_di_wb_eval.py --task qa --limit 6     # smoke, writes smoke/full/
    python gemma_di_wb_eval.py --task sop              # all 162 demos, writes results/full/
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
                             api_key=jc.api_key("DEEPINFRA_API_KEY"), timeout=300, max_retries=0)


GEN = jc.Generator(
    "google/gemma-4-31B-it", client,
    wanted={"temperature": 0, "max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    call_timeout=300, max_images=32)

if __name__ == "__main__":
    wb.run_cli(GEN, MODEL_DIR, "Wonderbread — Gemma 4 31B generation (DeepInfra)")
