"""DocVQA 10% sample runner — `Qwen/Qwen3.5-9B` via **DeepInfra**, reasoning off. The project's Qwen.

**Exists for the 10% sample rerun only** (`sample10-plan.md` WP4): `--sample sample10` is required,
and rows go to `results_sample10/` (`smoke_sample10/` under `--limit`), never `results/`. The
reported DocVQA Qwen number came from `../qwen_DocVQA/` on **Together**, `max_tokens=12000`, no
reasoning switch (thinking on). The user chose today's settled config for the sample (D1 = B,
D2 = DeepInfra, 2026-11-15), so this measures a different condition from that number; the drift
check is what says by how much.

No scorer and no loop here: `docvqa_eval_core` builds the message (`build_messages`: the page
image, then `build_prompt`'s text, byte-identical to every DocVQA run) and scores with
`score_response` / `anls_score`, the one matcher of this benchmark. The prompt ceiling below is
this model's addition, made here and never in the core.

**Settled config** (template: `PLAN_Qwen_DeepInfra`). Reasoning off by the provider's switch,
`reasoning={"enabled": False}`, in `extra_body` (top-level it raises TypeError on the OpenAI SDK) and
`fixed`, so never negotiated away — thinking is ON by default on this route too. `temperature=0`,
`seed=42` (negotiated). **Checked, not assumed** (`forbid_reasoning=True`): DeepInfra reports no
`reasoning_tokens` for this model, so the signals are `reasoning_chars` — the reasoning text of every
call of the row, CutRetry stages and failed attempts included — and `think_tag_calls`, a think tag
inline in a reply's content. A row with either means thinking came
back on: the run stops at that row, writes `sample10_THINKING_HALT.txt` and exits 3, as Gemma's does.
**`max_tokens=8192` is chosen, not measured for DocVQA**: PlanBench measured
that 4096 bound for multi-step plans and the user doubled it (2026-11-11); a DocVQA answer is a
short span, so it is headroom here, not a measurement.

**The prompt ceiling** (`PROMPT_CEILING`, PlanBench's sentence word for word, `prompt-ceiling.md`) is
sent as a system message ahead of the shared user message: with thinking off this model deliberates
in its visible answer, and the sentence tells it the budget `max_tokens` enforces. A request, not a
limit. **A reply cut at the cap is called again, then forced** (`core.CutRetry`, the user's rule of
2026-11-13): fresh with `RETRY_CEILING`; cut again, its visible text goes back as an assistant turn
with `FORCE_ANSWER` and `max_tokens=ANSWER_MAX`, reasoning still off. All three texts are
PlanBench's, unchanged, and are in every row's config.

SIGALRM at 900 s per call (re-armed by CutRetry for each of a row's up to three calls), below the
client's 1500 s: this route's speed varies by day — 17-46 s for a 12-token reply on 2026-11-09
(`model-calls.md`) — and a reply that fills 8192 tokens at 12 tokens/s takes ~680 s.

    python qwen_di_docvqa_eval.py --sample sample10 --limit 3   # smoke -> smoke_sample10/
    sbatch run_docvqa_sample10.sh                                # on Quest -> results_sample10/
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
# makes up to 5 attempts itself (script-skeleton.md §4).
client = OpenAI(api_key=core.require_key("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai", timeout=1500, max_retries=0)

WANTED = {"temperature": 0, "max_tokens": 8192, "seed": 42}
FIXED = {"extra_body": {"reasoning": {"enabled": False}}}

# PlanBench's three texts (PLAN_Qwen_DeepInfra/qwen_di_plan_eval.py), word for word.
PROMPT_CEILING = ("Your whole reply, every bit of reasoning plus the final answer, must fit within 8192 "
                  "tokens; anything past that is cut off and lost. Keep your reasoning short and do not "
                  "re-check the same step again and again. As soon as you have a solution, write the "
                  "final answer in the exact format the task asks for. If you are running out of room, "
                  "stop reasoning and give your best answer immediately.")
RETRY_CEILING = ("Your previous attempt at this task reasoned past the 8192-token limit and was cut off "
                 "before it gave an answer. This time you must stay within the limit: keep your "
                 "reasoning under 4000 tokens, do not re-check steps you have already settled, and if "
                 "you notice you are repeating yourself, stop at once. Then write the final answer in "
                 "the exact format the task asks for.")
ANSWER_MAX = 4096         # the forced answer's max_tokens
FORCE_ANSWER = ("Your thinking has reached its limit. Do not reason any further: give your final "
                "answer now, in exactly the format the task asks for.")
CUT = core.CutRetry(PROMPT_CEILING, RETRY_CEILING, FORCE_ANSWER, ANSWER_MAX)


def _call(sent, stage, model, params):
    """One HTTP call. The visible text is both the answer and what goes back if it was cut."""
    r = client.chat.completions.create(
        model=model, messages=sent, **FIXED,
        **(dict(params, max_tokens=ANSWER_MAX) if stage == "forced" else params))
    meta = core.openai_meta(r)              # first: it books the bill even if `choices` is empty
    text = (r.choices[0].message.content or "").strip()
    return text, text, meta


def once(messages, model, params):
    return CUT(messages, lambda sent, stage: _call(sent, stage, model, params), model)


if __name__ == "__main__":
    core.sample_main(description="DocVQA 10% sample — Qwen3.5-9B (DeepInfra, reasoning off)",
                     default_model="Qwen/Qwen3.5-9B", model_dir=MODEL_DIR, client=client, once=once,
                     wanted=WANTED, fixed=FIXED, required={"max_tokens": 8192}, route="deepinfra",
                     call_timeout=900, extra_config=CUT.config(), forbid_reasoning=True)
