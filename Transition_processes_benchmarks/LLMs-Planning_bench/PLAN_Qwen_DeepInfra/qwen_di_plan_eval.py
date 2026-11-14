"""Qwen3.5-9B via DeepInfra — PlanBench runner. **The project's Qwen slot.**

No prompt, no scorer and no run loop here: all of that is `plan_eval_core.py` (generation) and
`plan_scorer.py` (offline scoring), identical for every model. This file is the client, the
parameters this model is called with, and the wording of its three instructions.

**DeepInfra, not Together** — new Qwen runs use DeepInfra from 2026-11-04 (`model-calls.md`).

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}`. Thinking is ON by
default on this route too; without the switch a one-line question ran 51 s into `finish=length` with
an empty `content`. On the OpenAI SDK it must travel in `extra_body` — top-level it raises TypeError
(`model-calls.md`, 2026-11-09). It is `fixed`: sent on every call, never negotiated away. With it
off, this model still deliberates at length in its visible answer — that analysis is where its plans
come from (below).

Output cap `max_tokens=8192` (rule 2). It was 4096 first, and the pilot measured that it bound (27%
of rows cut); doubled on the user's decision, 2026-11-11. `temperature=0` and `seed=42` pinned
(rules 4 and 8), the seed negotiated.

**First call: the budget sentence** (`PROMPT_CEILING`, the user's instruction of 2026-11-12) — the
configuration of the pilot that scored 687 of 1,513, archived as `results_archive_budget_prompt_
20261113`. An answer-only instruction was tried instead on 2026-11-13 and cut its plans: 448 of 1,513,
plan-writing tasks 405 -> 114 of 907 (`planbench.md`); that run is archived as
`results_archive_answer_only_20261113`.

**A reply cut at the cap: called again, then forced** (`core.CutRetry`, the user's rule of
2026-11-13). In that pilot 367 of 1,563 rows ended at `length`. Such a row is called again, fresh,
with `RETRY_CEILING` in place of the budget sentence; if that is cut too, its visible text goes back
as an assistant turn with FORCE_ANSWER, `max_tokens=ANSWER_MAX`, reasoning still off. What each row
carries is `CutRetry`'s docstring. On MPGT the forced answer did not rescue a repetition loop: the
model looped again to its cap, and the row keeps `finish_reason=length` (the shard then exits 2).

This route's speed is not constant: median 14 tokens/s in the 4096 run, 31 in the 8192 run (hourly
medians 16-88), 52 for a lone stream afterwards (`planbench.md`). At 12 tokens/s a reply that fills
8192 tokens takes ~680 s, so the SIGALRM ceiling is 900 s — for each of a row's up to three calls,
which `CutRetry` re-arms — and the client's own timeout sits above that so the watchdog, not the
SDK, ends a hung call.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import plan_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai", timeout=1500, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 8192, "seed": 42}
FIXED = {"extra_body": {"reasoning": {"enabled": False}}}

# The prompt ceiling (`prompt-ceiling.md`): tells the model the budget `max_tokens` enforces, because
# this model deliberates in its visible answer. A REQUEST, not a limit — `finish_reason` and
# `completion_tokens` say whether it worked. Unchanged from the 2026-11-12 pilot, word for word.
PROMPT_CEILING = ("Your whole reply, every bit of reasoning plus the final answer, must fit within 8192 "
                  "tokens; anything past that is cut off and lost. Keep your reasoning short and do not "
                  "re-check the same step again and again. As soon as you have a solution, write the "
                  "final answer in the exact format the task asks for. If you are running out of room, "
                  "stop reasoning and give your best answer immediately.")
# The second call of a row whose first reply was cut at the cap: the user asked that it be told not
# to exceed the limit this time. It keeps the reasoning (an answer-only instruction removed it) and
# asks for half the budget, so the answer has room after it.
RETRY_CEILING = ("Your previous attempt at this task reasoned past the 8192-token limit and was cut off "
                 "before it gave an answer. This time you must stay within the limit: keep your "
                 "reasoning under 4000 tokens, do not re-check steps you have already settled, and if "
                 "you notice you are repeating yourself, stop at once. Then write the final answer in "
                 "the exact format the task asks for.")
ANSWER_MAX = 4096         # the forced answer's max_tokens; MPGT's forced answers were <= 612 tokens
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")   # MPGT's, unchanged
CUT = core.CutRetry(PROMPT_CEILING, RETRY_CEILING, FORCE_ANSWER, ANSWER_MAX)


def _call(sent, stage, model, params):
    """One HTTP call. The visible text is both the answer and what goes back if it was cut."""
    r = client.chat.completions.create(
        model=model, messages=sent, **FIXED,
        **(dict(params, max_tokens=ANSWER_MAX) if stage == "forced" else params))
    message = r.choices[0].message
    text = (message.content or "").strip()
    meta = core.openai_meta(r)
    meta["reasoning_chars"] = len(getattr(message, "reasoning_content", None) or "")
    return text, text, meta


def once(messages, model, params):
    return CUT(messages, lambda sent, stage: _call(sent, stage, model, params), model)


if __name__ == "__main__":
    core.run_cli(description="PlanBench — Qwen3.5-9B (DeepInfra, reasoning off)",
                 default_model="Qwen/Qwen3.5-9B", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, fixed=FIXED, required={"max_tokens": 8192}, call_timeout=900,
                 extra_config=dict(CUT.config(), route="deepinfra"))
