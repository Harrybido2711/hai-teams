"""DeepSeek flash — MultiChallenge response generation. **The project's Deepseek slot.**

No loop or scorer here: all of that is `../mc_eval_core.py::run_cli`, identical for every model.
The conversation is upstream's, byte for byte; the one addition below is a system message. Judging
is `../MC_Judges/mc_judge.py`. Deepseek is text-only, so it runs MultiChallenge and not Wonderbread
(the user's decision, 2026-10-31).

**No thinking knob exists, so the cap goes in the prompt** (`model-parameters.md` rules 1 and 5,
`prompt-ceiling.md`) — the user's decision for this benchmark, 2026-10-31, knowing it departs from
upstream, which sends the stored conversation with no system prompt. It is a system message ahead of
the conversation, worded as AwareBench's, so it asks about *reasoning* and not about the visible
answer. It is a REQUEST, not a limit: verify it from `reasoning_tokens` on the rows, never from how
long the answers look. Its wording is part of the run config, so it is on every row.
`max_tokens=8192` bounds reasoning and answer together.

**Called by its real id, `deepseek-flash`** — the user's decision, 2026-10-12, made for AwareBench.
`deepseek-reasoner` is an alias that answers as `deepseek-flash`; `served_model` on every row says
what answered.

**The answer sometimes arrives in `reasoning_content` with `content` empty.** Fallen back to
explicitly, logged, and marked on the row (`fallback`). Never when `finish_reason` is `length`: that
is chain of thought cut off mid-way, not an answer, so the row is treated as empty and retried.
The long client timeout is deliberate: DeepSeek queues rather than refusing. The SIGALRM ceiling is
600 s — 8,192 tokens at ~30 tok/s with margin.

    python deepseek_mc_eval.py --limit 3     # smoke, writes smoke/full/
    python deepseek_mc_eval.py               # all 273 items, writes results/full/
"""

import os
import sys

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import mc_eval_core as mc  # noqa: E402
jc = mc.jc

PROMPT_CEILING = "Think briefly. Use at most five sentences of reasoning before you answer."


def client():
    # max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
    # makes up to 5 attempts itself (script-skeleton.md §4).
    return jc.openai_class()(base_url="https://api.deepseek.com",
                             api_key=jc.api_key("DEEPSEEK_API_KEY"), timeout=7200, max_retries=0)


def with_ceiling(conversation):
    return [{"role": "system", "content": PROMPT_CEILING}] + list(conversation)


def answer(response):
    text, meta = jc.read_answer(response)
    if not text.strip() and meta.get("finish_reason") != "length":
        message = response.choices[0].message
        text = (getattr(message, "reasoning_content", None) or "").strip()
        if text:
            print(f"[{meta.get('served_model')}] content empty, fell back to reasoning_content",
                  flush=True)
            meta["fallback"] = "reasoning_content"
    return text, meta


GEN = jc.Generator(
    "deepseek-flash", client,
    wanted={"temperature": 0, "max_tokens": jc.GEN_MAX_TOKENS, "seed": 42},
    required={"max_tokens": jc.GEN_MAX_TOKENS},
    prepare=with_ceiling, answer=answer, call_timeout=600,
    extra_config={"prompt_ceiling": PROMPT_CEILING})

if __name__ == "__main__":
    mc.run_cli(GEN, MODEL_DIR, "MultiChallenge — DeepSeek flash responses (prompt ceiling on thinking)")
