"""Qwen3.5-9B via DeepInfra — PlanBench runner. **The project's Qwen slot.**

No prompt, no scorer and no run loop here: all of that is `plan_eval_core.py` (generation) and
`plan_scorer.py` (offline scoring), identical for every model. This file is the client and the
parameters this model is called with.

**DeepInfra, not Together** — new Qwen runs use DeepInfra from 2026-11-04 (`model-calls.md`).

**Reasoning off, by the provider's own switch:** `reasoning={"enabled": False}`. Thinking is ON by
default on this route too; without the switch a one-line question ran 51 s into `finish=length` with
an empty `content`. On the OpenAI SDK it must travel in `extra_body` — top-level it raises TypeError
(`model-calls.md`, 2026-11-09). It is `fixed`: sent on every call, never negotiated away.

Output cap `max_tokens=8192` (rule 2). **It was 4096 first, and the pilot measured that it bound:**
at 7 h 42 min that run had 480 rows, 131 of them (27%) `finish=length` — 49 of 60 in the
mystery_blocksworld shard — and only 10 of the 131 reached `[PLAN END]`. They were long visible
chain-of-thought cut mid-sentence (no repetition loops), so 4096 measured the cap, not the model.
Doubled on the user's decision, 2026-11-11; those rows are archived, not mixed in.
`temperature=0` and `seed=42` pinned (rules 4 and 8), the seed negotiated.

This route's speed is not constant: median 14 tokens/s in the 4096 run, 31 in the 8192 run (hourly
medians 16-88), 52 for a lone stream afterwards (`planbench.md`). At 12 tokens/s a reply that fills
8192 tokens takes ~680 s, so the SIGALRM ceiling is 900 s (it was 600 s at 4096), and the client's
own timeout sits above that so the watchdog, not the SDK, ends a hung call.
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
# this model deliberates in its visible answer and the pilot's first 8,192 run was cut off on 23% of
# its rows. A REQUEST, not a limit — `finish_reason` and `completion_tokens` say whether it worked.
PROMPT_CEILING = ("Your whole reply, every bit of reasoning plus the final answer, must fit within 8192 "
                  "tokens; anything past that is cut off and lost. Keep your reasoning short and do not "
                  "re-check the same step again and again. As soon as you have a solution, write the "
                  "final answer in the exact format the task asks for. If you are running out of room, "
                  "stop reasoning and give your best answer immediately.")


def once(messages, model, params):
    # After the shared system message, before the shared user message: the shared messages are untouched.
    r = client.chat.completions.create(
        model=model, messages=messages[:1] + [{"role": "system", "content": PROMPT_CEILING}]
        + messages[1:], **FIXED, **params)
    return r.choices[0].message.content, core.openai_meta(r)


if __name__ == "__main__":
    core.run_cli(description="PlanBench — Qwen3.5-9B (DeepInfra, reasoning off)",
                 default_model="Qwen/Qwen3.5-9B", model_dir=MODEL_DIR, client=client, once=once,
                 wanted=WANTED, fixed=FIXED, required={"max_tokens": 8192}, call_timeout=900,
                 extra_config={"route": "deepinfra", "prompt_ceiling": PROMPT_CEILING})
