"""DeepSeek flash — MMLU runner for the 10% sample rerun. **The project's Deepseek slot.**

Exists only for the sample (`.claude/references/sample10-plan.md`, WP3): the 401 items of
`../sample10.json`, written to `results_sample10/` (a `--limit` smoke to `smoke_sample10/`), every row
carrying the per-row usage fields. No prompt, no scorer and no run loop here — all of that is
`mmlu_eval_core.py`, identical for every model. **`deepseek_eval.py` and the flat `deepseek_*.csv`
beside it are the legacy run and stay untouched**: that ran `deepseek-reasoner` on prompt v1 with no
cap. **This runs `PROMPT_V2`** — one prompt for all five models, the user's decision of 2026-11-15 —
under today's settled config (D1 = B), so its accuracy may not match the workbook.

**No thinking knob exists, so the cap goes in the prompt** (`model-parameters.md` rules 1 and 5,
`prompt-ceiling.md`): `PROMPT_CEILING`, a system message before the question. It is a REQUEST, not a
limit; `reasoning_tokens` on the rows says whether it held. **Output cap `max_tokens=16384` — chosen,
not measured on MMLU** (PlanBench's settled cap for this model; it bounds reasoning and answer
together). `temperature=0`, `seed=42`; all three `required`.

**A reply cut at the cap: called again, then forced** (`core.CutRetry`, the user's rule of
2026-11-13, in `PLAN_Deepseek/deepseek_plan_eval.py`'s docstring). Called again fresh with
`RETRY_CEILING`; cut again, its `reasoning_content` goes back as an assistant turn with
`FORCE_ANSWER`, thinking **disabled**, `max_tokens=ANSWER_MAX`. `forced_reasoning_chars` must be 0;
anything else is `forced_reasoned` and the shard exits 2. The ceiling, `FORCE_ANSWER` and
`ANSWER_MAX` are PlanBench's word for word; **`RETRY_CEILING` changes one phrase** — PlanBench's ends
"stop reasoning as soon as you have a plan", here "… as soon as you have an answer".

**Called by its real id, `deepseek-flash`** (user, 2026-10-12); `served_model` on every row says what
answered. **The answer sometimes arrives in `reasoning_content` with `content` empty**: fallen back
to, logged and marked (`fallback`) — never when `finish_reason` is `length`, and never on a forced
answer. DeepSeek queues rather than refusing, hence the long client timeout; the SIGALRM ceiling is
900 s for each of a row's up to three calls.
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
                 "reasoning as soon as you have an answer. Your whole reply, reasoning included, must "
                 "fit within 16384 tokens.")
ANSWER_MAX = 4096         # the forced answer's max_tokens — PlanBench's; chosen, not measured
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")
NO_THINKING = {"thinking": {"type": "disabled"}}
CUT = core.CutRetry(PROMPT_CEILING, RETRY_CEILING, FORCE_ANSWER, ANSWER_MAX)


def _call(sent, stage, model, params, tag):
    """One HTTP call. The answer is `content`; what goes back if it was cut is `reasoning_content`."""
    if stage == "forced":
        r = core.create(client, model=model, messages=sent, **dict(params, max_tokens=ANSWER_MAX),
                        extra_body=NO_THINKING)
    else:
        r = core.create(client, model=model, messages=sent, **params)
    meta = core.openai_meta(r)           # also this call's reasoning_chars, for forced_reasoned
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
    args = core.sample_parser("MMLU 10% sample — DeepSeek flash (prompt ceiling on thinking)",
                              "deepseek-flash").parse_args()
    raise SystemExit(core.run_sample(
        args, model_dir=MODEL_DIR, client=client, once=once, wanted=WANTED,
        required=dict(WANTED), call_timeout=900,
        extra_config=dict(CUT.config(extra_body=NO_THINKING), route="deepseek")))
