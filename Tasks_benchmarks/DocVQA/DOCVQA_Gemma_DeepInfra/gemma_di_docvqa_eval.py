"""DocVQA 10% sample runner — `google/gemma-4-31B-it` via **DeepInfra**. The project's Gemma.

**Exists for the 10% sample rerun only** (`sample10-plan.md` WP4): `--sample sample10` is required,
and rows go to `results_sample10/` (`smoke_sample10/` under `--limit`), never `results/`. The
reported DocVQA Gemma number came from `../gemma_DocVQA/` on **Together**, `max_tokens=12000`, no
reasoning switch (thinking on there). The user chose today's settled config for the sample (D1 = B,
D2 = DeepInfra, 2026-11-15), so this measures a different condition from that number; the drift
check is what says by how much.

No prompt, no scorer, no loop here: `docvqa_eval_core` builds the message (`build_messages`: the
page image, then `build_prompt`'s text, byte-identical to every DocVQA run) and scores with
`score_response` / `anls_score`, the one matcher of this benchmark.

**Settled config** (template: `PLAN_Gemma_DeepInfra`). **No reasoning parameter at all**: DeepInfra
serves this checkpoint with thinking already off, and passing one (`reasoning_effort`) turns it back
ON (`model-calls.md`). `temperature=0` and `seed=42` set even where they equal a default (rule 4);
the seed is negotiated, because whether DeepInfra honours it is unestablished (rule 8).
**`max_tokens=4096` is chosen, not measured** — PlanBench's cap for this model; a DocVQA answer is a
short span, so a `length` finish would mean thinking came back on or the model looped.
**Thinking must stay off on every row** (`forbid_reasoning=True`): a row with reasoning text
(`reasoning_chars`, summed over every call of the row), a think tag inline in a reply
(`think_tag_calls` — the four spellings NEG_Gemma_DeepInfra strips), or reasoning tokens
(`reasoning_tokens`, `billed_reasoning_tokens` > 0 — DeepInfra reports 0 for this model) means it
came back on: the run
stops at that row, writes `sample10_THINKING_HALT.txt` and exits 3.

One image a request — far below DeepInfra's 32-image cap for this model (`provider-gotchas.md`);
the largest sample page is 3.1 MB, sent as stored.

    python gemma_di_docvqa_eval.py --sample sample10 --limit 3   # smoke -> smoke_sample10/
    sbatch run_docvqa_sample10.sh                                 # on Quest -> results_sample10/
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
# makes up to 5 attempts itself (script-skeleton.md §4). SIGALRM at 240 s sits below timeout=300.
client = OpenAI(api_key=core.require_key("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai", timeout=300, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 4096, "seed": 42}


def once(messages, model, params):
    r = client.chat.completions.create(model=model, messages=messages, **params)
    meta = core.openai_meta(r)              # first: it books the bill even if `choices` is empty
    return r.choices[0].message.content, meta


if __name__ == "__main__":
    core.sample_main(description="DocVQA 10% sample — Gemma 4 31B (DeepInfra)",
                     default_model="google/gemma-4-31B-it", model_dir=MODEL_DIR, client=client,
                     once=once, wanted=WANTED, required={"max_tokens": 4096}, route="deepinfra",
                     call_timeout=240, forbid_reasoning=True)
