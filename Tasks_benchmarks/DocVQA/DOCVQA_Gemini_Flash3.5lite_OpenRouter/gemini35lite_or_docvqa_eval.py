"""DocVQA 10% sample runner — `google/gemini-3.5-flash-lite` via **OpenRouter**. The project's Gemini.

**Exists for the 10% sample rerun only** (`sample10-plan.md` WP4): `--sample sample10` is required,
and rows go to `results_sample10/` (`smoke_sample10/` under `--limit`), never `results/`. The
reported DocVQA Gemini number came from the **native Google route**
(`../DOCVQA_Gemini_Flash3.5lite_Google/`, `thinking_budget=128`); the user chose today's settled
config for the sample (D1 = B, 2026-11-15), which is this route. So this measures a different
condition from that number, and the drift check is what says by how much.

No prompt, no scorer, no loop here: `docvqa_eval_core` builds the message (`build_messages`: the
page image, then `build_prompt`'s text, byte-identical to every DocVQA run) and scores with
`score_response` / `anls_score`, the one matcher of this benchmark.

**Settled config** (`model-parameters.md`, the PlanBench runner `PLAN_Gemini_Flash3.5lite_OpenRouter`
is the template): thinking at `reasoning: {effort: "minimal"}` — the floor, there is no off — sent in
`extra_body` as `fixed`, so it is never negotiated away; `seed=42`; `temperature` unset.
**`max_tokens=8192` is chosen, not measured**: the cap bbh and PlanBench declare for this model.
DocVQA answers are a short span, so a `length` finish here would mean something is wrong.

**OpenRouter switches backend mid-run** (Google AI Studio / Vertex), so the answering backend is
written on every row, with OpenRouter's `usage.cost`: `last_call_cost` for the answering call and
`billed_cost` summed over every call of the row. **One sample question is a known hard failure on the
native route**: 59607 came back `MALFORMED_RESPONSE` with no output part, deterministically, in the
full run (`docvqa.md`). If it reproduces here it is written as a failed row, not dropped.

    python gemini35lite_or_docvqa_eval.py --sample sample10 --limit 3   # smoke -> smoke_sample10/
    sbatch run_docvqa_sample10.sh                                        # on Quest -> results_sample10/
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(MODEL_DIR))
import docvqa_eval_core as core  # noqa: E402

load_dotenv(core.ENV_PATH)
# max_retries=0: the SDK's own hidden retries would make n_attempts and latency_s lie; the core
# makes up to 5 attempts itself (script-skeleton.md §4). SIGALRM at 300 s sits below timeout=400.
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=core.require_key("OPENROUTER_API_KEY"),
                timeout=400, max_retries=0)

WANTED = {"max_tokens": 8192, "seed": 42}
FIXED = {"extra_body": {"reasoning": {"effort": "minimal"}}}


def once(messages, model, params):
    r = client.chat.completions.create(model=model, messages=messages, **FIXED, **params)
    meta = core.openai_meta(r)              # first: it books the bill even if `choices` is empty
    return r.choices[0].message.content, meta


if __name__ == "__main__":
    core.sample_main(description="DocVQA 10% sample — Gemini 3.5 Flash-Lite (OpenRouter)",
                     default_model="google/gemini-3.5-flash-lite", model_dir=MODEL_DIR,
                     client=client, once=once, wanted=WANTED, fixed=FIXED,
                     required={"max_tokens": 8192}, route="openrouter", call_timeout=300)
