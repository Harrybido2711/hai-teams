"""Qwen3.5-9B via DeepInfra — BIG-Bench Hard, **the 10% sample rerun only** (2026-11-15).

**There is no scorer, prompt or loop in this file.** Scoring is `bbh_eval_core.score_response`, the
one lenient matcher; the prompt is the core's `v1`, the one every bbh model answered; the loop,
retries, halts and files are `bbh_eval_core.run_sample`. This file is the client and the settled
config — copied from `LLMs-Planning_bench/PLAN_Qwen_DeepInfra/qwen_di_plan_eval.py`, the user's
choice for this rerun (D1 = B, today's settled config; D2 = DeepInfra; `sample10-plan.md`).

**Sample-only on purpose.** `--sample` is required and rows go to `results_sample10/` (or
`smoke_sample10/` under `--limit`). `BBH_Qwen/` keeps the reported run and its script untouched: its
4,833 rows are 13 tasks on DeepInfra at `max_tokens=12500` and 7 on Together with thinking on, so
this config is not the one behind the reported number — the sample's drift check says by how much.

**Reasoning off, by the provider's own switch:** `extra_body={"reasoning": {"enabled": False}}`,
`fixed` so it is sent on every call and never negotiated away. Thinking is ON by default on this
route; without the switch a one-line question ran 51 s into `finish=length` with an empty `content`
(`model-calls.md`). With it off the model still deliberates at length in its visible answer. The
run is declared `thinking_off`, as Gemma's: a call that returns `reasoning_content`
(`reasoning_chars > 0`), a `<think>` tag in its content (`think_tag_calls > 0`) or reasoning tokens
halts the run at that row with `sample10_THINKING_HALT.txt` and exit 3 — the switch did not hold.

**Output cap `max_tokens=8192` — chosen, not measured on bbh.** It is PlanBench's settled cap for
this model (doubled from 4096 there after it bound on 27% of rows); bbh's reported rows ran at
12500. `temperature=0`, `seed=42` (negotiated).

**The prompt ceiling and CutRetry** (the user's rule of 2026-11-13, `prompt-ceiling.md`): the first
call carries `PROMPT_CEILING` as a system message ahead of the shared user prompt; a reply cut at
the cap is called again with `RETRY_CEILING`, and if that is cut too its visible text goes back with
`FORCE_ANSWER` at `ANSWER_MAX`. All four texts are PlanBench's, word for word — none of them names
a plan. What each row carries is `bbh_eval_core.CutRetry`'s docstring.

The route's speed varies by day (14-52 tokens/s measured on PlanBench); at 12 tokens/s a reply that
fills 8192 takes ~680 s, so the SIGALRM ceiling is 900 s and the client's own timeout sits above it.
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
                base_url="https://api.deepinfra.com/v1/openai", timeout=1500, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 8192, "seed": 42}
FIXED = {"extra_body": {"reasoning": {"enabled": False}}}

PROMPT_CEILING = ("Your whole reply, every bit of reasoning plus the final answer, must fit within 8192 "
                  "tokens; anything past that is cut off and lost. Keep your reasoning short and do not "
                  "re-check the same step again and again. As soon as you have a solution, write the "
                  "final answer in the exact format the task asks for. If you are running out of room, "
                  "stop reasoning and give your best answer immediately.")
RETRY_CEILING = ("Your previous attempt at this task reasoned past the 8192-token limit and was cut off "
                 "before it gave an answer. This time you must stay within the limit: keep your "
                 "reasoning under 4000 tokens, do not re-check steps you have already settled, and if "
                 "you notice you are repeating yourself, stop at once. Then write the final answer in "
                 "the exact format the task asks for.")
ANSWER_MAX = 4096         # the forced answer's max_tokens; a bbh answer is one line
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")
CUT = core.CutRetry(PROMPT_CEILING, RETRY_CEILING, FORCE_ANSWER, ANSWER_MAX)


def _call(sent, stage, model, params):
    """One HTTP call. The visible text is both the answer and what goes back if it was cut."""
    r = client.chat.completions.create(
        model=model, messages=sent, **FIXED,
        **(dict(params, max_tokens=ANSWER_MAX) if stage == "forced" else params))
    # First: openai_meta books the bill before choices[0] can raise, and records reasoning_content
    # (`reasoning_chars`) and a <think> tag in the content (`think_tag_calls`) — either halts this
    # thinking-off run.
    meta = core.openai_meta(r)
    text = (r.choices[0].message.content or "").strip()
    return text, text, meta


def once(messages, model, params):
    return CUT(messages, lambda sent, stage: _call(sent, stage, model, params), model)


if __name__ == "__main__":
    core.sample_main(description="BIG-Bench Hard sample rerun — Qwen3.5-9B (DeepInfra, reasoning off)",
                     default_model="Qwen/Qwen3.5-9B", model_dir=MODEL_DIR, client=client, once=once,
                     wanted=WANTED, fixed=FIXED, required={"max_tokens": 8192}, call_timeout=900,
                     extra_config=dict(CUT.config(), route="deepinfra"), default_sleep=1.0,
                     thinking_off=True)
