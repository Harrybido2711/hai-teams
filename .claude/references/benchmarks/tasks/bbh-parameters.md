# BBH — model call parameters, as they are today

<!-- size-budget: 6000 -->
<!-- One job: the as-is parameter record for eight runners. The prose that would be cut to fit
     is the two-values-were-wrong note, which is the reason the page is trustworthy at all. -->

This is the **as-is record** for bbh's eight runners. What every runner *must* be changed to is
[model-parameters.md](../../model-parameters.md), which is keyed by model and covers every
benchmark. Do not read this page as a target.

Read from the code on **2026-08-29**, after the runners were rewritten onto
`Tasks_benchmarks/bbh/bbh_eval_core.py`. **The values did not change in that rewrite** — each
runner kept exactly what the script its job actually submitted was passing, including two that the
previous version of this page got wrong (see the note under the table).

- **`—` means the parameter is never passed**, so the provider or SDK default applies. It does not
  mean `0`, `False`, or unsupported.
- **There is one runner per model now.** The `_eval.py` / `_finish.py` split is gone, and with it
  the trap where a job submitted the twin rather than the file its name implied.

| Model (exact id in code) | API / SDK | Runner | `temperature` | `max_tokens` | `seed` | `reasoning` | client `timeout` | `stream` | `--sleep` |
|---|---|---|---:|---:|---|---|---:|---|---:|
| `gpt-4o-mini-2024-07-18` *(superseded)* | OpenAI Chat Completions | [`openai_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_GPT_4o_mini/openai_bbh_eval.py#L28) | `0` | — | — | — | — | — | `0` |
| `gemini-2.5-flash` *(superseded)* | Google Gen AI `generate_content` | [`gemini_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_Gemini_Flash2.5/gemini_bbh_eval.py#L28) | — | — | — | — | — | — | `0` |
| `deepseek-reasoner` | DeepSeek, OpenAI-compatible | [`deepseek_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_Deepseek/deepseek_bbh_eval.py#L29) | `0` | — | — | — | `7200` | `False` | `0` |
| `grok-3-mini` | xAI SDK `chat.sample()` | [`xai_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_XAI/xai_bbh_eval.py#L29) | — | — | — | — | `3600` | — | `0` |
| `google/gemma-4-31B-it` | Together Chat Completions | [`gemma_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_Gemma/gemma_bbh_eval.py#L28) | `0` | `12000` | — | — | `4800` | `False` | `1` |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | Together Chat Completions | [`llama_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_Llama/llama_bbh_eval.py#L28) | `0` | — | — | — | — | `False` | `0` |
| `Qwen/Qwen3.5-9B` | Together Chat Completions | [`qwen_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_Qwen/qwen_bbh_eval.py#L28) | `0` | `12500` | — | — | — | `False` | `1` |
| `kimi-k2.5` | Moonshot, OpenAI-compatible | [`kimi_bbh_eval.py`](../../../../Tasks_benchmarks/bbh/BBH_Kimi/kimi_bbh_eval.py#L29) | `1` | — | — | — | `7200` | — | `5` |

**Two values on this page were wrong before 2026-08-29**, both because they were read from a file
that was never the one submitted. Gemma's client `timeout` is `4800` (`gemma_finish.py:19`, the file
`gemma_eval_script.sh` actually ran), not unset; Qwen's client sets **no** timeout (`qwen_finish.py:19`)
— the `18000` previously recorded came from `qwen_eval.py`, which no job ever submitted. The rewrite
preserved the submitted values, so the column above is what the numbers on disk were produced with.

Columns dropped from this table because **no runner has ever set them**: `top_p`, `top_k`,
`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `stop`, `response_format`, `tools`,
`tool_choice`, `reasoning_effort`, `verbosity`.

What the table shows:

1. **`temperature` is the only parameter under real control**, and not consistently: five models at
   `0`, Kimi at `1`, Gemini and Grok unset.
2. **Only Gemma and Qwen cap output length**, and the two caps differ.
3. **No runner sets `seed`**, and Kimi runs at `temperature=1`, so nothing here is reproducible.
   `seed` is measured working elsewhere — [../../model-parameters.md](../../model-parameters.md)
   rule 6 — so probe it per provider rather than assuming it is unavailable.
4. **A reasoning-capable model name is not a reasoning setting.** No runner passes a reasoning
   configuration, so no model's internal behaviour may be recorded as a `reasoning_effort` value.
5. **`--sleep` is per-item pacing, not a parameter of the model.** It carries over the `time.sleep`
   each old runner had, and is a CLI flag now so throttling can be changed without editing code.

## Cost control — nothing here caps anything today

**No runner sets a reasoning or output limit** beyond Gemma's and Qwen's `max_tokens`. On the
reasoning-capable models the bill is whatever the model decides to think, so all eight are out of
compliance with the standing rule. Which knob and which cap each needs — including the three that
are not reasoning models and need only an output cap — is in
[model-parameters.md](../../model-parameters.md), and why an over-tight cap returns a billed empty
response is in [reasoning-cost.md](../../reasoning-cost.md).

**This is deliberately still open.** Setting the caps changes what the models emit, which would make
new numbers incomparable with the 4,833 rows already on disk. Cap them when bbh is re-run, not
before.
