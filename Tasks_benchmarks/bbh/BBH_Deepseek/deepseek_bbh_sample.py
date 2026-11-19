"""DeepSeek flash — BIG-Bench Hard, **the 10% sample rerun only** (2026-11-15).

**There is no scorer, prompt or loop in this file.** Scoring is `bbh_eval_core.score_response`, the
one lenient matcher; the prompt is the core's `v1`, the one every bbh model answered; the loop,
retries, halts and files are `bbh_eval_core.run_sample`. This file is the client and the settled
config — copied from `LLMs-Planning_bench/PLAN_Deepseek/deepseek_plan_eval.py`, the user's choice for
this rerun (D1 = B, today's settled config; `sample10-plan.md`).

**Sample-only, beside the legacy runner.** `deepseek_bbh_eval.py` in this folder produced the
reported 4,833 rows (`deepseek-reasoner`, `temperature=0`, no cap, no seed, no ceiling) and is left
exactly as it was; this script needs `--sample` and writes `results_sample10/` (or `smoke_sample10/`
under `--limit`), never `results/`. Result files are named after the model, so this one's are
`deepseek-flash.*` and cannot be mistaken for the legacy `deepseek-reasoner.*`.

**Called by its real id, `deepseek-flash`** (user, 2026-10-12); `served_model` on every row says
what answered. `temperature=0`, `seed=42` (negotiated).

**Output cap `max_tokens=16384` — chosen, not measured on bbh.** It bounds hidden reasoning and the
answer together; it is PlanBench's settled cap for this model.

**No thinking knob exists, so the cap goes in the prompt, and a cut reply is retried then forced**
(the user's rule of 2026-11-13: "让他第二次重新跑的时候，不要超出这个思考的上限…如果说他就是不受约束，那就直接回答";
`prompt-ceiling.md`). `PROMPT_CEILING` goes as a system message ahead of the shared user prompt. A
reply cut at the cap is called again with `RETRY_CEILING`; if that is cut too, its
`reasoning_content` goes back with `FORCE_ANSWER`, thinking **disabled**, `max_tokens=ANSWER_MAX`.
What each row carries is `bbh_eval_core.CutRetry`'s docstring; `forced_reasoning_chars` must be 0.

**One word differs from PlanBench's texts.** Its `RETRY_CEILING` ends "stop reasoning as soon as you
have a plan"; bbh asks for an answer, not a plan, so here it reads "as soon as you have an answer".
Every other sentence of the three texts is PlanBench's, word for word, and all three are on every
row's config.

**The answer sometimes arrives in `reasoning_content` with `content` empty**: fallen back to, logged
and marked `fallback` — never when `finish_reason` is `length` (a chain of thought cut off, not an
answer) and never on a forced answer. DeepSeek queues rather than refusing, hence the long client
timeout; the SIGALRM ceiling is 900 s for each of a row's up to three calls.
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
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com",
                timeout=7200, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 16384, "seed": 42}
PROMPT_CEILING = ("Think briefly. Use at most five sentences of reasoning before you answer. Your whole "
                  "reply, reasoning included, must fit within 16384 tokens; anything past that is cut off "
                  "and lost. If you are running out of room, stop reasoning and give your best answer "
                  "immediately.")
RETRY_CEILING = ("Your previous attempt at this task reasoned past the 16384-token limit and was cut off "
                 "before it gave an answer. This time you must stay within the limit: keep your "
                 "reasoning under 8000 tokens, do not re-check steps you have already settled, and stop "
                 "reasoning as soon as you have an answer. Your whole reply, reasoning included, must fit "
                 "within 16384 tokens.")
ANSWER_MAX = 4096         # the forced answer's max_tokens; a bbh answer is one line
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")
NO_THINKING = {"thinking": {"type": "disabled"}}
CUT = core.CutRetry(PROMPT_CEILING, RETRY_CEILING, FORCE_ANSWER, ANSWER_MAX)


def _call(sent, stage, model, params, tag):
    """One HTTP call. The answer is `content`; what goes back if it was cut is `reasoning_content`.
    `reasoning_chars` on the meta is openai_meta's: the length of reasoning_content."""
    if stage == "forced":
        r = client.chat.completions.create(model=model, messages=sent,
                                           **dict(params, max_tokens=ANSWER_MAX), extra_body=NO_THINKING)
    else:
        r = client.chat.completions.create(model=model, messages=sent, **params)
    meta = core.openai_meta(r)            # first: it books the bill before choices[0] can raise
    message = r.choices[0].message
    content = (message.content or "").strip()
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    if stage != "forced" and not content and reasoning and meta.get("finish_reason") != "length":
        print(f"[{tag}] content empty, fell back to reasoning_content", flush=True)
        content = reasoning
        meta["fallback"] = "reasoning_content"
    return content, reasoning, meta


def once(messages, model, params):
    return CUT(messages, lambda sent, stage: _call(sent, stage, model, params, model), model)


if __name__ == "__main__":
    core.sample_main(description="BIG-Bench Hard sample rerun — DeepSeek flash (prompt ceiling on thinking)",
                     default_model="deepseek-flash", model_dir=MODEL_DIR, client=client, once=once,
                     wanted=WANTED, required={"max_tokens": 16384}, call_timeout=900,
                     extra_config=dict(CUT.config(extra_body=NO_THINKING), route="deepseek"),
                     default_sleep=0.0)
