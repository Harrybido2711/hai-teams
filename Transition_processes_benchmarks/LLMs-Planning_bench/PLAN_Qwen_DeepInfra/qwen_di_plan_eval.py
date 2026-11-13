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

**Answer-only, by instruction** (user's decision, 2026-11-13, after two 8,192 runs — see
`PROMPT_CEILING`): the model's long analysis is its own style with thinking off, not a switch that
failed, and a budget sentence barely moved it (349 -> 320 truncated rows). On 60 rows, asking for the
final answer alone cut the median reply from 4,645 to 64 tokens and the truncated rows from 24 to 2
of the 24 that had been cut, with 23 correct against 19. Qwen is now measured like Gemma and Gemini:
a model that answers directly.

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

# The answer-only instruction, carried as the project's "prompt ceiling" (`prompt-ceiling.md`).
# With thinking off this model still writes a long analysis in its visible answer, and that is what
# `max_tokens` cut off: 23% of the pilot's rows at 8,192 (21% with a budget sentence added). Every
# way of switching thinking off gave the same long answers (checked 2026-11-13); asking for the answer
# alone is what shortened them. A REQUEST, not a limit — `finish_reason` says whether it worked.
PROMPT_CEILING = ("Do not explain, analyse or think step by step. Reply with the final answer only, in "
                  "exactly the format the task asks for. Your whole reply must fit within 8192 tokens.")


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
