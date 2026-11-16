"""Qwen3.5-9B via DeepInfra — MMLU runner for the 10% sample rerun. **The project's Qwen slot.**

Exists only for the sample (`.claude/references/sample10-plan.md`, WP3): the 401 items of
`../sample10.json`, written to `results_sample10/` (a `--limit` smoke to `smoke_sample10/`), every row
carrying the per-row usage fields. No prompt, no scorer and no run loop here — all of that is
`mmlu_eval_core.py`, identical for every model: `PROMPT_V2` as the user turn, `score_response` as the
one scorer. This file is the client, the parameters and the wording of this model's three
instructions. **`MMLU_Qwen/` keeps the legacy script and the reported rows, untouched**: they ran on
Together with thinking on and no cap; this is today's settled config instead (the user's D1 = B and
D2 = DeepInfra, 2026-11-15), so its accuracy may not match the workbook — the drift check says by
how much.

**DeepInfra, not Together** — new Qwen runs use DeepInfra from 2026-11-04 (`model-calls.md`).
**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}`, in `extra_body` (a
top-level kwarg raises TypeError on the OpenAI SDK). It is `fixed`: sent on every call, never
negotiated away. With it off this model still deliberates in its visible answer, which PROMPT_V2
asks for anyway ("Please show your reasoning").

**Output cap `max_tokens=8192` — chosen, not measured on MMLU**: PlanBench's settled cap for this
model (4096 bound there on 27% of rows). `temperature=0` and `seed=42` pinned; all three `required`.

**The prompt ceiling and `CutRetry`** (the user's rule of 2026-11-13; `prompt-ceiling.md`): the first
call carries `PROMPT_CEILING` as a system message before the question; a reply cut at the cap is
called again with `RETRY_CEILING`; cut again, its visible text goes back with `FORCE_ANSWER`,
`max_tokens=ANSWER_MAX`, reasoning still off. All three wordings and `ANSWER_MAX` are PlanBench's
(`PLAN_Qwen_DeepInfra/qwen_di_plan_eval.py`), word for word — none mentions planning. What each row
carries is `CutRetry`'s docstring; the ceilings are on every row's `config`.

The SIGALRM ceiling is 900 s for each of a row's up to three calls (this route has run at 12-88
tokens/s; 8192 tokens at 12/s is ~680 s), and the client's own timeout sits above it.

**Thinking off is checked, not assumed** (`run_sample(forbid_thinking=True)`, as for Gemma). DeepInfra
reports no `reasoning_tokens` for this model, so the evidence is `reasoning_chars`: the
`reasoning_content` of every call of the row, every stage included, summed by the core — and inline
`<think>` tags in the content (`think_tag_calls`). Either above 0 and the switch did not hold: the shard stops at that row — THINKING CAME BACK ON, the row saved,
`sample10_[shard<N>_]THINKING_HALT.txt`, exit 3; a resubmit that finds the row stops again
before calling any — rather than run the rest on a config known wrong.
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
ANSWER_MAX = 4096         # the forced answer's max_tokens — PlanBench's; chosen, not measured
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")
CUT = core.CutRetry(PROMPT_CEILING, RETRY_CEILING, FORCE_ANSWER, ANSWER_MAX)


def _call(sent, stage, model, params):
    """One HTTP call. The visible text is both the answer and what goes back if it was cut."""
    r = core.create(client, model=model, messages=sent, **FIXED,
                    **(dict(params, max_tokens=ANSWER_MAX) if stage == "forced" else params))
    meta = core.openai_meta(r)           # first: the usage is billed even if the body is malformed
    text = (r.choices[0].message.content or "").strip()
    return text, text, meta


def once(messages, model, params):
    return CUT(messages, lambda sent, stage: _call(sent, stage, model, params), model)


if __name__ == "__main__":
    args = core.sample_parser("MMLU 10% sample — Qwen3.5-9B (DeepInfra, reasoning off)",
                              "Qwen/Qwen3.5-9B").parse_args()
    raise SystemExit(core.run_sample(
        args, model_dir=MODEL_DIR, client=client, once=once, wanted=WANTED, fixed=FIXED,
        required=dict(WANTED), call_timeout=900,
        extra_config=dict(CUT.config(), route="deepinfra"), forbid_thinking=True))
