"""Qwen3.5-9B via DeepInfra — mpgt-eval, zero-shot DST and goal tracking. **The project's Qwen slot.**

DeepInfra, not Together: new Qwen runs use DeepInfra from 2026-11-04 (`model-calls.md`). No prompt,
loop or scorer here: all of that is `../mpgt_eval_core.py::run_cli`, identical for every model.
Default: both tasks x all six upstream prompt styles x 29 dialogues = 348 calls.

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}` in `extra_body` —
`fixed`, sent on every call, never negotiated away. **On DeepInfra thinking is ON by default for this
model**, so the switch is not optional. In `extra_body`, not top-level: the OpenAI SDK refuses an
unknown `reasoning=` kwarg with a TypeError. Output cap `max_tokens=8192`
(`judge_core.GEN_MAX_TOKENS`); `temperature=0` and `seed=42` pinned (rules 4 and 8).
**This route is slow** — 17-111 s a call on Wonderbread, 2026-11-09 — so the SIGALRM ceiling is
600 s, as on `WB_Qwen_DeepInfra`. `route` is in the config: the model id is the same on Together.
**Visible reasoning:** the README says nothing; nothing is added.

**A reply that reaches the output cap gets one forced-answer turn — the user, 2026-11-13** ("如果到了
那个上界依然没有想出来就直接回答"). With thinking off this model still deliberates in its visible
answer: 45 of the full run's 348 rows were cut at 8192 with no final answer. So, as on
`MPGT_Deepseek` (`ForcedAnswerGenerator`):

1. The call as before. A reply that ends any way but `length` is the row's answer.
2. If it ends at `length`, the text it wrote goes back as an assistant turn, followed by
   FORCE_ANSWER (Deepseek's wording), with `max_tokens=ANSWER_MAX`. The second reply is the row's
   answer; the cut text is kept on the row as `first_response`.

There is no thinking switch left to turn — `reasoning.enabled=False` is already on both calls — so
only the request stops it deliberating again, and `forced_finish_reason` says whether it did: a
forced answer that ends at `length` filled ANSWER_MAX too. Probed 2026-11-13, second call only, on
16 stored cut rows (4 here, 10 on PlanBench, plus 2 loops): the 14 cut mid-deliberation all ended
`stop` in 17-612 tokens; **the 2 repetition loops looped again to the cap** — a forced answer does
not rescue a loop (17 of the 45), and such a row keeps `finish_reason=length`.

**What a row says.** A forced row carries `forced_answer: true`, `first_response`,
`first_finish_reason`, `first_completion_tokens`, `forced_finish_reason`,
`forced_completion_tokens`, and `first_call_reused`. Its `finish_reason` is the second call's.
Usage fields are summed over both calls; a cut first call is kept across retries and billed once.
`n_attempts` counts attempts, and a forced attempt is two calls. **The truncation rate is therefore
`forced_answer` rows plus `finish_reason=length` rows**, not `length` alone.

**`--force-cut` forces the rows already stored** (`mpgt_eval_core.force_cut`): only rows that ended
at `length` are called, and only their second call — the first is the stored row
(`first_call_reused: true`, its own `latency_s` / `n_attempts` kept as `first_latency_s` /
`first_n_attempts`; the row's `latency_s` and `n_attempts` are then the forced call's alone). The
route is not reproducible at `temperature=0, seed=42`, so remaking the first call would replace a
stored row by another draw. The forced answer is in the config, so the file is first copied to
`results_archive_before_forced_answer_<date>/` and every row then carries the one new config.

    python qwen_di_mpgt_eval.py --limit 1     # smoke, writes smoke/zero-shot/
    python qwen_di_mpgt_eval.py               # all 348, writes results/zero-shot/
    python qwen_di_mpgt_eval.py --force-cut   # stored rows cut at the cap: forced answer only
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mpgt_eval_core as mp  # noqa: E402
jc = mp.jc


def client():
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
    # makes up to 5 attempts itself (script-skeleton.md §4).
    return jc.openai_class()(base_url="https://api.deepinfra.com/v1/openai",
                             api_key=jc.api_key("DEEPINFRA_API_KEY"), timeout=600, max_retries=0)


ANSWER_MAX = 4096         # the forced answer's max_tokens; the probe's answers were <= 612 tokens
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")   # MPGT_Deepseek's, unchanged


def _forced_meta(first_meta, cut_text, second_meta, bill_first, reused):
    """The row's meta for a forced answer: the second call's, with usage summed with the first
    call's on the one attempt that bills it. A value neither call reported stays None, never 0."""
    second_meta = second_meta or {}
    meta = dict(second_meta)
    for key in jc.USAGE_FIELDS:
        parts = [m.get(key) for m in ((first_meta,) if bill_first else ()) + (second_meta,)]
        parts = [v for v in parts if isinstance(v, (int, float))]
        if parts:
            meta[key] = sum(parts)
    meta.update(forced_answer=True, first_response=cut_text,
                first_finish_reason=first_meta.get("finish_reason"),
                first_completion_tokens=first_meta.get("completion_tokens"),
                first_call_reused=reused,
                forced_finish_reason=second_meta.get("finish_reason"),
                forced_completion_tokens=second_meta.get("completion_tokens"))
    return meta


class ForcedAnswerGenerator(jc.Generator):
    """`once` makes the call and, when it ends at `length`, the forced-answer call. The closure is
    built once per row and reused by every retry of it (`run_cli`), so a cut first call is kept: a
    retry repeats only the forced answer, and the first call is billed on the row once.
    `force_stored` is the same second call for a row already on disk (`--force-cut`)."""

    def _force(self, client, model, messages, params, cut_text, first_meta, kept, reused):
        bill_first = not kept["billed"]
        forced = list(messages) + [{"role": "assistant", "content": cut_text},
                                   {"role": "user", "content": FORCE_ANSWER}]
        second = client.chat.completions.create(
            model=model, messages=forced, **self.fixed, **dict(params, max_tokens=ANSWER_MAX))
        try:
            text, second_meta = (self.answer or jc.read_answer)(second)
        except (jc.Refused, jc.Interrupted) as stopped:
            # the core adds this meta's usage and keeps it on the row: carry the first call's bill
            # and the forced markers with it
            stopped.meta = _forced_meta(first_meta, cut_text, stopped.meta, bill_first, reused)
            kept["billed"] = True
            raise
        meta = _forced_meta(first_meta, cut_text, second_meta, bill_first, reused)
        kept["billed"] = True
        print(f"[{model}] reply reached the cap ({first_meta.get('completion_tokens')} tokens"
              f"{', stored' if reused else ''}): forced answer, "
              f"{meta['forced_completion_tokens']} tokens, finish={meta['forced_finish_reason']}",
              flush=True)
        return text, meta

    def once(self, client, model, messages, params):
        read = self.answer or jc.read_answer
        kept = {"billed": False}          # plus the cut first call, once there is one

        def call():
            if "text" not in kept:
                first = client.chat.completions.create(model=model, messages=messages,
                                                       **self.fixed, **params)
                text, meta = read(first)
                if meta.get("finish_reason") != "length" or not (text or "").strip():
                    return text, meta     # an empty `length` has nothing to hand back
                # stripped, as the core stores a response: `force_stored` hands back the same text
                kept.update(text=text.strip(), meta=meta)
            return self._force(client, model, messages, params, kept["text"], kept["meta"], kept,
                               reused=False)
        return call

    def force_stored(self, client, model, messages, params, stored):
        """The forced answer for a stored row cut at the cap: its `response` is the first call."""
        kept = {"billed": False}
        first_meta = {k: stored.get(k) for k in ("finish_reason",) + tuple(jc.USAGE_FIELDS)}
        return lambda: self._force(client, model, messages, params, stored["response"], first_meta,
                                   kept, reused=True)


GEN = ForcedAnswerGenerator(
    "Qwen/Qwen3.5-9B", client,
    wanted={"temperature": 0, "max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    fixed={"extra_body": {"reasoning": {"enabled": False}}},
    call_timeout=600,
    extra_config={"route": "deepinfra",
                  "forced_answer": {"prompt": FORCE_ANSWER, "max_tokens": ANSWER_MAX,
                                    "hands_back": "content",
                                    "when": "the first call ends at length with content"}})

if __name__ == "__main__":
    mp.run_cli(GEN, MODEL_DIR, "mpgt-eval zero-shot — Qwen3.5-9B (DeepInfra, reasoning off)")
