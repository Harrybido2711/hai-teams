"""DeepSeek reasoner — AwareBench runner. **The project's Deepseek slot.**

No scorer and no run loop here: all of that is `aware_eval_core.py`, identical for every model. The
user prompt is the shared one, byte for byte; the one addition below is a system message.

**No thinking knob exists, so the cap goes in the prompt** (`model-parameters.md` rules 1 and 5,
`prompt-ceiling.md`). Hidden reasoning is always capped, and an unestablished knob counts as no
knob. The ceiling is a system message, so the shared user prompt — and the answer format the
dataset asks for — stays identical across models. It is a REQUEST, not a limit: verify it from
`reasoning_tokens` on the rows, never from how long the answers look. Its wording is part of the run
config and so is written on every row. `max_tokens=8192` bounds reasoning and answer together.

**`deepseek-reasoner` is an alias, not a model id** — what it resolves to can change under you. The
resolved model comes back on every row as `served_model`.

**The answer sometimes arrives in `reasoning_content` with `content` empty.** Fallen back to
explicitly, logged, and marked on the row (`fallback`), counted per task in the summaries. Never
when `finish_reason` is `length`: that is chain of thought cut off mid-way, not an answer, so the row
is treated as empty and retried.

The long client timeout is deliberate: DeepSeek queues rather than refusing. The SIGALRM ceiling is
600 s — 8,192 tokens at ~30 tok/s with margin.
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import aware_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
# max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
# makes up to 5 attempts itself (script-skeleton.md §4).
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com",
                timeout=7200, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 8192, "seed": 42}
PROMPT_CEILING = "Think briefly. Use at most five sentences of reasoning before you answer."


def once(prompt, model, params):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "system", "content": PROMPT_CEILING},
                               {"role": "user", "content": prompt}], **params)
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
    core.run_cli(description="AwareBench — DeepSeek reasoner (prompt ceiling on thinking)",
                 default_model="deepseek-reasoner", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, required={"max_tokens": 8192}, call_timeout=600, extra_config={"prompt_ceiling": PROMPT_CEILING})
