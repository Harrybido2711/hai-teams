"""DeepSeek flash — PlanBench runner. **The project's Deepseek slot.**

No scorer and no run loop here: `plan_eval_core.py` (generation) and `plan_scorer.py` (offline
scoring) are identical for every model. The shared messages go out unchanged; the one addition is a
second system message, below.

**No thinking knob exists, so the cap goes in the prompt** (`model-parameters.md` rules 1 and 5,
`prompt-ceiling.md`) — AwareBench's wording plus the token budget (below), written on every row
through `extra_config`. It is a REQUEST, not a limit: verify it from `reasoning_tokens` on the rows.
The pilot's first run used AwareBench's sentence alone; 11 task_2 rows spent all 16,384 tokens on
reasoning and came back empty, so the user had the budget added and the pilot rerun (2026-11-12).

**Output cap `max_tokens=16384` — chosen, not measured.** It bounds reasoning and answer together.
AwareBench and MultiChallenge ran this model at 8192 on short answers; a PlanBench answer is a
multi-step plan reasoned over first, so the cap is doubled. The 10% pilot's `finish_length` column
confirms or refutes it.

**Called by its real id, `deepseek-flash`** (user, 2026-10-12); `served_model` on every row says
what answered. **The answer sometimes arrives in `reasoning_content` with `content` empty**: fallen
back to, logged and marked (`fallback`) — never when `finish_reason` is `length`, which is a chain of
thought cut off, not an answer, and never on a forced answer, whose reasoning must be empty.

**A reply cut at the cap: called again, then forced** (`core.CutRetry`, the user's rule of
2026-11-13: "让他第二次重新跑的时候，不要超出这个思考的上限…如果说他就是不受约束，那就直接回答"). The
prompt ceiling does not reach the hidden reasoning: the pilot with the budget in the prompt still had
20 rows end at `length`, 17 of them with no answer at all. Such a row is called again, fresh, with
`RETRY_CEILING` in place of the ceiling; a first call cut with no answer and no reasoning gets that
second call too. If the second is cut as well, its `reasoning_content` goes back as an assistant
turn with FORCE_ANSWER, thinking **disabled**, `max_tokens=ANSWER_MAX` — the model answers from the
reasoning it had. A second call cut with no reasoning has nothing to hand back and stays a cut row.
What each row carries is `CutRetry`'s docstring; `forced_reasoning_chars` must be 0 (with thinking
disabled the API reports no reasoning tokens), anything else is `forced_reasoned` and the shard exits 2.

**`--force-cut` repairs stored rows** (`plan_eval_core`): rows that ended at `length` are called
again from the first call — the reasoning was never stored — and every other row is kept. The first
call is a new draw, so a row can come back finished on its own (`cut_retry` absent).

DeepSeek queues rather than refusing, hence the long client timeout; the SIGALRM ceiling is 900 s —
16,384 tokens at ~30 tok/s with margin — for each of a row's up to three calls.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import plan_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com",
                timeout=7200, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 16384, "seed": 42}
# The first sentence is AwareBench's wording. The rest states the budget `max_tokens` enforces: the
# pilot's first run, with the first sentence alone, had 11 rows whose reasoning used all 16,384
# tokens and left no answer. A REQUEST, not a limit — `finish_reason` and `reasoning_tokens` say
# whether it worked.
PROMPT_CEILING = ("Think briefly. Use at most five sentences of reasoning before you answer. Your whole "
                  "reply, reasoning included, must fit within 16384 tokens; anything past that is cut off "
                  "and lost. If you are running out of room, stop reasoning and give your best answer "
                  "immediately.")
# The second call of a row whose first reply was cut at the cap: told not to exceed the limit this
# time, with half the budget so the answer has room after the reasoning.
RETRY_CEILING = ("Your previous attempt at this task reasoned past the 16384-token limit and was cut off "
                 "before it gave an answer. This time you must stay within the limit: keep your "
                 "reasoning under 8000 tokens, do not re-check steps you have already settled, and stop "
                 "reasoning as soon as you have a plan. Your whole reply, reasoning included, must fit "
                 "within 16384 tokens.")
ANSWER_MAX = 4096         # the forced answer's max_tokens; a plan is a few hundred tokens
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")   # MPGT_Deepseek's, unchanged
NO_THINKING = {"thinking": {"type": "disabled"}}
CUT = core.CutRetry(PROMPT_CEILING, RETRY_CEILING, FORCE_ANSWER, ANSWER_MAX)


def _call(sent, stage, model, params, tag):
    """One HTTP call. The answer is `content`; what goes back if it was cut is `reasoning_content`."""
    if stage == "forced":
        r = client.chat.completions.create(model=model, messages=sent,
                                           **dict(params, max_tokens=ANSWER_MAX), extra_body=NO_THINKING)
    else:
        r = client.chat.completions.create(model=model, messages=sent, **params)
    meta = core.openai_meta(r)
    message = r.choices[0].message
    content = (message.content or "").strip()
    reasoning = (getattr(message, "reasoning_content", None) or "").strip()
    meta["reasoning_chars"] = len(reasoning)
    if stage != "forced" and not content and reasoning and meta.get("finish_reason") != "length":
        print(f"[{tag}] content empty, fell back to reasoning_content", flush=True)
        content = reasoning
        meta["fallback"] = "reasoning_content"
    return content, reasoning, meta


def once(messages, model, params):
    return CUT(messages, lambda sent, stage: _call(sent, stage, model, params, model), model)


if __name__ == "__main__":
    core.run_cli(description="PlanBench — DeepSeek flash (prompt ceiling on thinking)",
                 default_model="deepseek-flash", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, required={"max_tokens": 16384}, call_timeout=900,
                 extra_config=CUT.config(extra_body=NO_THINKING))
