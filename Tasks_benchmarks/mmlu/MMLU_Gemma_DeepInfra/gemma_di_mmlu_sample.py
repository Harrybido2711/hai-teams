"""Gemma 4 31B via DeepInfra — MMLU runner for the 10% sample rerun. **The project's Gemma slot.**

Exists only for the sample (`.claude/references/sample10-plan.md`, WP3): the 401 items of
`../sample10.json`, written to `results_sample10/` (a `--limit` smoke to `smoke_sample10/`), every row
carrying the per-row usage fields. No prompt, no scorer and no run loop here — all of that is
`mmlu_eval_core.py`, identical for every model: `PROMPT_V2` as the user turn, `score_response` as the
one scorer. **`MMLU_Gemma/` keeps the legacy script and the reported rows, untouched**: they ran on
Together with thinking on; this is today's settled config instead (the user's D1 = B and D2 =
DeepInfra, 2026-11-15), so its accuracy may not match the workbook.

**DeepInfra serves this checkpoint with thinking already off. Pass no reasoning parameter here**:
DeepInfra accepts `reasoning_effort`, and passing it turns thinking back ON (`model-calls.md`,
`model-parameters.md`). A model that does not reason gets an explicit output cap (rule 2):
**`max_tokens=4096` — chosen, not measured on MMLU** (PlanBench's settled cap for this model);
`finish_reason` on the rows says whether it binds. `temperature=0` and `seed=42` pinned; all three
`required`. No prompt ceiling and no `CutRetry`: neither is part of this model's settled config.

**Thinking off is checked, not assumed.** Every row carries `reasoning_chars`, the reasoning text of
every call of the row summed by the core, `reasoning_tokens`, and `think_tag_calls` (inline `<think>`
tags in the content). With thinking off all must be 0: any above 0 means thinking came back on — a request that grew a reasoning parameter, or DeepInfra
changing the default — so the shard stops at that row: THINKING CAME BACK ON, the row saved,
`sample10_[shard<N>_]THINKING_HALT.txt`, exit 3; a resubmit that finds the row stops again
before calling any (`run_sample(forbid_thinking=True)`). The merge
reports such rows too.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mmlu_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
# max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
# makes up to 5 attempts itself (script-skeleton.md §4).
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai", timeout=300, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 4096, "seed": 42}


def once(messages, model, params):
    r = core.create(client, model=model, messages=messages, **params)
    meta = core.openai_meta(r)           # first: the usage is billed even if the body is malformed
    return r.choices[0].message.content, meta


if __name__ == "__main__":
    args = core.sample_parser("MMLU 10% sample — Gemma 4 31B (DeepInfra, no reasoning parameter)",
                              "google/gemma-4-31B-it").parse_args()
    raise SystemExit(core.run_sample(
        args, model_dir=MODEL_DIR, client=client, once=once, wanted=WANTED,
        required=dict(WANTED), call_timeout=240, extra_config={"route": "deepinfra"},
        forbid_thinking=True))
