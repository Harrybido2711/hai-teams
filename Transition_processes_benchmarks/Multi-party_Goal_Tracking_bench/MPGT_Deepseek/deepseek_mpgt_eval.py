"""DeepSeek flash — mpgt-eval, zero-shot DST and goal tracking. **The project's Deepseek slot.**

No loop or scorer here: all of that is `../mpgt_eval_core.py::run_cli`, identical for every model.
The prompt is upstream's, byte for byte; the one addition below is a system message. Default: both
tasks x all six upstream prompt styles x 29 dialogues = 348 calls.

**No thinking knob exists, so the cap goes in the prompt** (`model-parameters.md` rules 1 and 5,
`prompt-ceiling.md`): a system message ahead of upstream's user turn, worded as on AwareBench and
MultiChallenge. It is a REQUEST, not a limit: verify it from `reasoning_tokens` on the rows. Its
wording is part of the run config, so it is on every row.

**`max_tokens=16384` — twice the project default (`jc.GEN_MAX_TOKENS`), set by the user
2026-11-11.** It bounds reasoning and answer together. At 8192 the smoke (dialogue 00, 12 calls) had
7 rows end at `length` with no answer, and the 5 that finished spent 5.6k–7.4k reasoning tokens: one
call fills a whole window — 23 masks in that dialogue, up to 27 — and the ceiling does not keep the
reasoning short. The user asked for a modest raise, not 32768; whether 16384 is enough is read from
`reasoning_tokens` on the smoke rows.

**Called by its real id, `deepseek-flash`** (the user, 2026-10-12): `deepseek-reasoner` is an alias
that answers as it; `served_model` on every row says what answered. **The answer sometimes arrives in
`reasoning_content` with `content` empty** — fallen back to only on a clean `stop`, logged, and
marked on the row (`fallback`). The long client timeout is deliberate: DeepSeek queues rather than
refusing. SIGALRM ceiling 600 s.

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
MAX_TOKENS = 16384   # see the docstring: 8192 cut 7 of 12 smoke rows off


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


GEN = jc.Generator(
    "deepseek-flash", client,
    wanted={"temperature": 0, "max_tokens": MAX_TOKENS, "seed": 42},
    required={"max_tokens": MAX_TOKENS},
    prepare=with_ceiling, answer=answer, call_timeout=600,
    extra_config={"prompt_ceiling": PROMPT_CEILING})

if __name__ == "__main__":
    mp.run_cli(GEN, MODEL_DIR, "mpgt-eval zero-shot — DeepSeek flash (prompt ceiling on thinking)")
