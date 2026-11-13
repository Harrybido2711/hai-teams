"""DeepSeek flash — mpgt-eval, zero-shot DST and goal tracking. **The project's Deepseek slot.**

No loop or scorer here: all of that is `../mpgt_eval_core.py::run_cli`, identical for every model.
The prompt is upstream's, byte for byte; the additions below are a system message and, on a row
whose reasoning reaches the budget, one forced-answer turn. Default: both tasks x all six upstream
prompt styles x 29 dialogues = 348 calls (rows).

**Reasoning may not exceed REASONING_BUDGET = 8192 tokens — a hard limit, the user's, 2026-11-12**
("不能让它有太多的推理"). DeepSeek exposes no thinking budget (`model-parameters.md`), and the
prompt ceiling below is a request it ignores: the first full run on Quest (job 8428390, 23 rows,
cancelled) reasoned for a median ~9k tokens and 4 rows spent all of `max_tokens=16384` on reasoning
with no answer. So the limit is built from two calls (`BudgetedGenerator`):

1. The call as before, with `max_tokens=REASONING_BUDGET`. A clean finish is the row's answer.
2. If it ends at `length` — reasoning cut at the budget, or finished with the answer cut — the
   reasoning it wrote (`reasoning_content`, returned on a cut: probed 2026-11-12) goes back as an
   assistant turn, followed by FORCE_ANSWER, with thinking **disabled** and `max_tokens=ANSWER_MAX`.
   The model answers from the reasoning it had; it may not reason further.

Probed on `dst/basic/00` with the first call cut at 2048: the second call returned no
`reasoning_content`, ended at `stop` with the whole window annotated (428 tokens), and scored 15/23 —
what the uncut 9k–14k-token rows of that dialogue scored. **With thinking disabled DeepSeek reports
no reasoning-token count at all** (`completion_tokens_details` absent: `forced_reasoning_tokens` is
None), so the evidence that a forced answer did not reason is `forced_reasoning_chars == 0`.

**What a row says.** A forced row carries `forced_answer: true`, the first call's
`first_reasoning_tokens` / `first_finish_reason`, and the second call's own `forced_finish_reason`,
`forced_completion_tokens`, `forced_reasoning_tokens` (None when the API did not report it — never
coerced to 0) and `forced_reasoning_chars`. Its `finish_reason` is the second call's, so `length` on
a row now means the forced answer hit ANSWER_MAX (or a cut call returned no reasoning to hand back),
not that the budget bound — `forced_answer` counts that. Usage fields are summed over both calls and
over the core's attempts; a cut first call is kept across retries, so a retry repeats only the
forced answer and the first call is billed once (`first_call_billed`). **A second call that reasons
at all breaks the budget**: its row's `finish_reason` becomes `budget_exceeded`, which is not a
known complete answer, so the core prints STOP and the run exits 1. On ~74% of the cancelled run's
rows the reasoning was over 8192 — most rows here are expected to be forced. All of this is in the
config (`reasoning_budget`, `forced_answer`), so rows written under the old 16384 setting are
refused on resume rather than mixed.

**The prompt ceiling stays** (`prompt-ceiling.md`), worded as on AwareBench and MultiChallenge: the
same request every Deepseek runner makes. It is on every row.

**Called by its real id, `deepseek-flash`** (the user, 2026-10-12): `deepseek-reasoner` is an alias
that answers as it; `served_model` on every row says what answered. **The answer sometimes arrives in
`reasoning_content` with `content` empty** — fallen back to only on a clean `stop`, logged, and
marked on the row (`fallback`). The long client timeout is deliberate: DeepSeek queues rather than
refusing. SIGALRM ceiling 600 s, over both calls of a row.

    python deepseek_mpgt_eval.py --limit 1     # smoke, writes smoke/zero-shot/
    python deepseek_mpgt_eval.py               # all 348, writes results/zero-shot/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mpgt_eval_core as mp  # noqa: E402
jc = mp.jc

PROMPT_CEILING = "Think briefly. Use at most five sentences of reasoning before you answer."
REASONING_BUDGET = 8192   # the first call's max_tokens: reasoning can never pass it (docstring)
ANSWER_MAX = 4096         # the forced answer's max_tokens; answers seen so far are <= 600 tokens
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")
NO_THINKING = {"thinking": {"type": "disabled"}}


def client():
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
    # makes up to 5 attempts itself (script-skeleton.md §4).
    return jc.openai_class()(base_url="https://api.deepseek.com",
                             api_key=jc.api_key("DEEPSEEK_API_KEY"), timeout=7200, max_retries=0)


def with_ceiling(messages):
    return [{"role": "system", "content": PROMPT_CEILING}] + list(messages)


def answer(response):
    text, meta = jc.read_answer(response)
    # only a clean stop: a cut-off chain (length, or a finish nothing here has met) is never the answer
    if not text.strip() and meta.get("finish_reason") == "stop":
        message = response.choices[0].message
        text = (getattr(message, "reasoning_content", None) or "").strip()
        if text:
            print(f"[{meta.get('served_model')}] content empty, fell back to reasoning_content",
                  flush=True)
            meta["fallback"] = "reasoning_content"
    return text, meta


BUDGET_EXCEEDED = "budget_exceeded"   # not in jc.EXPECTED_FINISH: the run STOPs and exits 1


def _forced_meta(first, first_meta, second, second_meta, bill_first, has_answer):
    """The row's meta for a forced answer. Usage is the second call's plus — on the one attempt
    that made it — the first call's; a value the API did not report stays None, never 0. A second
    call that reasoned at all (tokens reported above 0, or any `reasoning_content`) and still gave
    an answer breaks the budget: its finish becomes BUDGET_EXCEEDED, which the core treats as not a
    complete answer. The markers say it either way (`forced_reasoning_chars`)."""
    meta = dict(second_meta or {})
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
        parts = [m.get(key) for m in ((first_meta,) if bill_first else ()) + (second_meta or {},)]
        parts = [v for v in parts if isinstance(v, int)]
        meta[key] = sum(parts) if parts else None
    message = second.choices[0].message if second is not None and second.choices else None
    forced_reasoning = (second_meta or {}).get("reasoning_tokens")
    forced_chars = len(getattr(message, "reasoning_content", None) or "")
    meta.update(forced_answer=True,
                first_finish_reason=first_meta.get("finish_reason"),
                first_reasoning_tokens=first_meta.get("reasoning_tokens"),
                first_content_chars=len(getattr(first.choices[0].message, "content", None) or ""),
                first_call_billed=bill_first,
                forced_finish_reason=(second_meta or {}).get("finish_reason"),
                forced_completion_tokens=(second_meta or {}).get("completion_tokens"),
                forced_reasoning_tokens=forced_reasoning,
                forced_reasoning_chars=forced_chars)
    reasoned = (isinstance(forced_reasoning, int) and forced_reasoning > 0) or forced_chars
    if reasoned and has_answer:
        # only a row that will be scored: an empty one keeps its own finish, so an empty `length`
        # still stops the core's retries at once (judge_core.call_with_retries) and is `failed`
        meta["finish_reason"] = BUDGET_EXCEEDED
    return meta


class BudgetedGenerator(jc.Generator):
    """`once` makes the first call and, when it ends at `length`, the forced-answer call. The
    closure is built once per row and reused by every retry of it (`run_cli`), so a cut first call
    is kept: a retry repeats only the forced answer, and the first call is billed on the row once."""

    def once(self, client, model, messages, params):
        if self.fixed:
            raise SystemExit("BudgetedGenerator sends no `fixed` fields; it was given some")
        read = self.answer or answer
        kept = {}                         # the cut first call, once there is one

        def call():
            if "first" not in kept:
                first = client.chat.completions.create(model=model, messages=messages, **params)
                first_meta = jc.openai_meta(first)
                if first_meta.get("finish_reason") != "length":
                    return read(first)
                reasoning = (getattr(first.choices[0].message, "reasoning_content", None)
                             or "").strip()
                if not reasoning:
                    return read(first)    # nothing to hand back: an empty `length` row, as before
                kept.update(first=first, meta=first_meta, reasoning=reasoning, billed=False)
            first, first_meta = kept["first"], kept["meta"]
            bill_first = not kept["billed"]
            forced = list(messages) + [{"role": "assistant", "content": kept["reasoning"]},
                                       {"role": "user", "content": FORCE_ANSWER}]
            second = client.chat.completions.create(
                model=model, messages=forced, **dict(params, max_tokens=ANSWER_MAX),
                extra_body=NO_THINKING)
            try:
                text, second_meta = read(second)
            except (jc.Refused, jc.Interrupted) as stopped:
                # the core adds this meta's usage and keeps it on the row: carry the first call's
                # bill and the forced markers with it
                stopped.meta = _forced_meta(first, first_meta, second, stopped.meta, bill_first,
                                            has_answer=False)
                kept["billed"] = True
                raise
            meta = _forced_meta(first, first_meta, second, second_meta, bill_first,
                                has_answer=bool((text or "").strip()))
            kept["billed"] = True
            print(f"[{model}] reasoning reached {REASONING_BUDGET} "
                  f"({first_meta.get('reasoning_tokens')} reasoning tokens): forced answer, "
                  f"{meta['forced_completion_tokens']} tokens, "
                  f"finish={meta['forced_finish_reason']}, second-call reasoning "
                  f"{meta['forced_reasoning_tokens']} tokens / {meta['forced_reasoning_chars']} chars"
                  + (" — BUDGET EXCEEDED" if meta["finish_reason"] == BUDGET_EXCEEDED else ""),
                  flush=True)
            return text, meta
        return call


GEN = BudgetedGenerator(
    "deepseek-flash", client,
    wanted={"temperature": 0, "max_tokens": REASONING_BUDGET, "seed": 42},
    required={"max_tokens": REASONING_BUDGET},
    prepare=with_ceiling, answer=answer, call_timeout=600,
    extra_config={"prompt_ceiling": PROMPT_CEILING,
                  "reasoning_budget": REASONING_BUDGET,
                  "forced_answer": {"prompt": FORCE_ANSWER, "max_tokens": ANSWER_MAX,
                                    "extra_body": NO_THINKING,
                                    "when": "the first call ends at length with reasoning_content"}})

if __name__ == "__main__":
    mp.run_cli(GEN, MODEL_DIR, "mpgt-eval zero-shot — DeepSeek flash (reasoning budget "
               f"{REASONING_BUDGET}, forced answer past it)")
