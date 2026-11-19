"""Gemma 4 31B via DeepInfra — BIG-Bench Hard, **the 10% sample rerun only** (2026-11-15).

**There is no scorer, prompt or loop in this file.** Scoring is `bbh_eval_core.score_response`, the
one lenient matcher; the prompt is the core's `v1`, the one every bbh model answered; the loop,
retries, halts and files are `bbh_eval_core.run_sample`. This file is the client and the settled
config — copied from `LLMs-Planning_bench/PLAN_Gemma_DeepInfra/gemma_di_plan_eval.py`, the user's
choice for this rerun (D1 = B, today's settled config; D2 = DeepInfra; `sample10-plan.md`).

**Sample-only on purpose.** `--sample` is required and rows go to `results_sample10/` (or
`smoke_sample10/` under `--limit`). `BBH_Gemma/` keeps the reported run and its script untouched:
its 4,833 rows are 15 tasks on Together and 5 on DeepInfra at `max_tokens=12000`, so this config is
not the one behind the reported number — the sample's drift check says by how much.

**Pass no reasoning parameter here.** DeepInfra serves this checkpoint with thinking already off,
and it *accepts* `reasoning_effort` — passing one turns thinking back ON (`model-calls.md`). So every
row carries `reasoning_chars` (the `reasoning_content` returned, summed over every call of the row)
and `think_tag_calls` (calls with a `<think>` tag in the content), and the run is declared
`thinking_off`: the first row with either above 0, or reasoning tokens > 0, halts the run there,
with `sample10_THINKING_HALT.txt` and exit 3 — the config is wrong, and the other rows would not
be the measurement.

**Output cap `max_tokens=4096` — chosen, not measured on bbh.** It is PlanBench's settled cap for
this model; bbh's reported rows ran at 12000. A model that does not reason gets an explicit output
cap (rule 2), and a `length` finish means something is wrong. `temperature=0`, `seed=42`
(negotiated: whether DeepInfra honours it is unestablished, rule 8).
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import bbh_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
# max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
# makes up to 5 attempts itself (script-skeleton.md §4).
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai", timeout=300, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 4096, "seed": 42}


def once(messages, model, params):
    r = client.chat.completions.create(model=model, messages=messages, **params)
    # First: openai_meta books the bill before choices[0] can raise, and records what shows thinking
    # stayed off — `reasoning_chars` (reasoning_content) and a <think> tag in the content
    # (`think_tag_calls`).
    meta = core.openai_meta(r)
    return r.choices[0].message.content, meta


if __name__ == "__main__":
    core.sample_main(description="BIG-Bench Hard sample rerun — Gemma 4 31B (DeepInfra)",
                     default_model="google/gemma-4-31B-it", model_dir=MODEL_DIR, client=client,
                     once=once, wanted=WANTED, required={"max_tokens": 4096}, call_timeout=240,
                     extra_config={"route": "deepinfra"}, default_sleep=1.0, thinking_off=True)
