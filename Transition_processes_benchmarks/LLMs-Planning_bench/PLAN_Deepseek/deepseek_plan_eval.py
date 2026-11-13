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


def once(messages, model, params):
    # The ceiling goes after the shared system message and before the shared user message, so the
    # shared messages themselves are untouched.
    r = client.chat.completions.create(
        model=model, messages=messages[:1] + [{"role": "system", "content": PROMPT_CEILING}]
        + messages[1:], **params)
    meta = core.openai_meta(r)
    message = r.choices[0].message
    content = (message.content or "").strip()
    if not content and meta.get("finish_reason") != "length":
        content = (getattr(message, "reasoning_content", None) or "").strip()
        if content:
            print(f"[{model}] content empty, fell back to reasoning_content", flush=True)
            meta["fallback"] = "reasoning_content"
    return content, meta


if __name__ == "__main__":
    core.run_cli(description="PlanBench — DeepSeek flash (prompt ceiling on thinking)",
                 default_model="deepseek-flash", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, required={"max_tokens": 16384}, call_timeout=900,
                 extra_config={"prompt_ceiling": PROMPT_CEILING})
