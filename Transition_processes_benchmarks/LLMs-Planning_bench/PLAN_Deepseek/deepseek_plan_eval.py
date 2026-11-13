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
thought cut off, not an answer.

**A reply that reaches the cap gets one forced-answer turn — the user, 2026-11-13** ("如果到了那个
上界依然没有想出来就直接回答"). The prompt ceiling does not reach the hidden reasoning: the pilot
with the budget in the prompt still had 20 rows end at `length`, 17 of them with no answer at all.
So, as on `MPGT_Deepseek`: a first call that ends at `length` has its `reasoning_content` handed back
as an assistant turn, followed by FORCE_ANSWER, with thinking **disabled** and
`max_tokens=ANSWER_MAX`. The model answers from the reasoning it had. A partial answer the cap cut
off is not the answer; it is kept as `first_content`. A cut call that returns no reasoning has
nothing to hand back and stays as it was (an empty `length` row).

**What a row says.** A forced row carries `forced_answer: true`, `first_finish_reason`,
`first_reasoning_tokens`, `first_content`, and the second call's `forced_finish_reason`,
`forced_completion_tokens`, `forced_reasoning_tokens` (None when the API reports none — with
thinking disabled it reports none) and `forced_reasoning_chars`, which must be 0: anything else is a
forced answer that reasoned again. Its `finish_reason` is the second call's, so `length` on a forced
row means the answer filled ANSWER_MAX. **Neither is an answer to trust unread: the core counts both
(`forced_reasoned`, `finish_length`) and the shard exits 2.** Usage is both calls summed — of the
last attempt, as on every PlanBench row (the core keeps the last attempt's usage, not the sum over
retries) — so `completion_tokens` on a forced row can pass 16,384, and `n_attempts` counts attempts,
not calls. **The truncation rate is therefore `forced_answer` rows plus `finish_reason=length` rows**,
not `length` alone. The cut first call is kept while the core retries the row, so a retry repeats
only the forced answer.

**`--force-cut` repairs stored rows** (`plan_eval_core`): rows that ended at `length` are called
again — both calls, since the reasoning was never stored — and every other row is kept. The first
call is a new draw, so a row can also come back finished on its own (`forced_answer` absent).

DeepSeek queues rather than refusing, hence the long client timeout; the SIGALRM ceiling is 900 s —
16,384 tokens at ~30 tok/s with margin.
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


ANSWER_MAX = 4096         # the forced answer's max_tokens; a plan is a few hundred tokens
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")   # MPGT_Deepseek's, unchanged
NO_THINKING = {"thinking": {"type": "disabled"}}
USAGE = ("prompt_tokens", "completion_tokens", "reasoning_tokens")

# The cut first call of the row being worked on. The core retries a row by calling `once` again with
# the SAME messages list (`run_rows` builds it once per row), so keeping the one cut call means a
# retry repeats only the forced answer. Keyed on that list object, held here so its identity cannot
# be reused — never on the text: PlanBench ships rows with byte-identical queries (the task 5 rows
# of blocksworld and blocksworld_3; 101 groups in the full set), and a text key would hand one
# row's reasoning to the next.
_kept = {}


def _forced(sent, model, params, first, first_meta):
    """The forced-answer call for a first call that ended at `length`: its reasoning handed back,
    thinking disabled. Returns (answer, meta) with both calls' usage summed."""
    second = client.chat.completions.create(
        model=model,
        messages=sent + [{"role": "assistant", "content": first["reasoning"]},
                         {"role": "user", "content": FORCE_ANSWER}],
        **dict(params, max_tokens=ANSWER_MAX), extra_body=NO_THINKING)
    second_meta = core.openai_meta(second)
    message = second.choices[0].message
    meta = dict(second_meta)
    for key in USAGE:
        parts = [m.get(key) for m in (first_meta, second_meta) if isinstance(m.get(key), int)]
        meta[key] = sum(parts) if parts else None
    meta.update(forced_answer=True,
                first_finish_reason=first_meta.get("finish_reason"),
                first_reasoning_tokens=first_meta.get("reasoning_tokens"),
                first_content=first["content"],
                forced_finish_reason=second_meta.get("finish_reason"),
                forced_completion_tokens=second_meta.get("completion_tokens"),
                forced_reasoning_tokens=second_meta.get("reasoning_tokens"),
                forced_reasoning_chars=len(getattr(message, "reasoning_content", None) or ""))
    print(f"[{model}] reply reached the cap ({first_meta.get('reasoning_tokens')} reasoning "
          f"tokens): forced answer, {meta['forced_completion_tokens']} tokens, "
          f"finish={meta['forced_finish_reason']}, second-call reasoning "
          f"{meta['forced_reasoning_tokens']} tokens / {meta['forced_reasoning_chars']} chars",
          flush=True)
    return (message.content or "").strip(), meta


def once(messages, model, params):
    # The ceiling goes after the shared system message and before the shared user message, so the
    # shared messages themselves are untouched.
    sent = messages[:1] + [{"role": "system", "content": PROMPT_CEILING}] + messages[1:]
    if _kept.get("messages") is not messages:
        _kept.clear()
        r = client.chat.completions.create(model=model, messages=sent, **params)
        meta = core.openai_meta(r)
        message = r.choices[0].message
        content = (message.content or "").strip()
        reasoning = (getattr(message, "reasoning_content", None) or "").strip()
        if meta.get("finish_reason") != "length" or not reasoning:
            if not content and meta.get("finish_reason") != "length":
                content = reasoning
                if content:
                    print(f"[{model}] content empty, fell back to reasoning_content", flush=True)
                    meta["fallback"] = "reasoning_content"
            return content, meta
        _kept.update(messages=messages, first={"reasoning": reasoning, "content": content},
                     meta=meta)
    return _forced(sent, model, params, _kept["first"], _kept["meta"])


if __name__ == "__main__":
    core.run_cli(description="PlanBench — DeepSeek flash (prompt ceiling on thinking)",
                 default_model="deepseek-flash", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, required={"max_tokens": 16384}, call_timeout=900,
                 extra_config={"prompt_ceiling": PROMPT_CEILING,
                               "forced_answer": {"prompt": FORCE_ANSWER, "max_tokens": ANSWER_MAX,
                                                 "extra_body": NO_THINKING,
                                                 "when": "the first call ends at length with "
                                                         "reasoning_content"}})
