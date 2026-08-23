# Which knob each model actually has

Split out of [model-parameters.md](model-parameters.md), which holds the rules this table is read
against and the settled per-model configurations. This page is the lookup: for a given model, is
there a thinking cap, an output cap, and can either be lowered.

| Model | Called by | Reasoning? | Thinking cap | Output cap | Can it be lowered? If not → prompt |
|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | Emo | yes, at the floor | see **Settled** below | `max_tokens` / `max_output_tokens` | **No off switch; `thinking_budget` does not bind** — only `-1` changes the regime. $2.50/M |
| `gemini-3.5-flash` | not used | yes | `thinking_level`, default `medium` | `max_output_tokens` | **Yes** — `medium` → `minimal`. Measure the step |
| `gemini-2.5-flash` | Emo, bbh, NegToM, DocVQA | yes | thinking budget | `max_output_tokens` | **Not established** → treat as no. **Prompt ceiling** until a call proves otherwise |
| `grok-3-mini` | bbh, Emo, NegToM | yes | `reasoning_effort` | provider default | **No — reasoning is the model.** **Prompt ceiling on thinking and on output** |
| `deepseek-reasoner` | bbh, Emo, NegToM | yes | none exposed | `max_tokens` | **No knob at all.** **Prompt ceiling**, or change model |
| `Qwen/Qwen3.5-9B` | bbh, Emo, NegToM | hybrid | `reasoning={"enabled": False}` | `max_tokens` | **Yes** — measured off: 11/12 vs 6/12 completions, 16 vs 517 output tokens |
| `kimi-k2.5` | bbh | not established | verify before assuming | `max_tokens` | **Not established → no.** Prompt ceiling |
| `gpt-4o-mini-2024-07-18` | bbh, DocVQA | no | n/a | `max_tokens` | n/a — `max_tokens` is enforced |
| `google/gemma-4-31B-it` | bbh, Emo, NegToM | no as served | n/a | `max_tokens` | n/a **on DeepInfra: served with thinking off, and `reasoning_effort` turns it back on.** On Together it is on by default and must be disabled |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | bbh | no | n/a | `max_tokens` | n/a — enforced |

"Called by" is where the id appears in a runner today, not that the run is current.
`gemini-3.5-flash` is listed only so it is not mistaken for Flash-Lite — dearer, $1.50/M and $9.00/M.

**An unestablished knob counts as no knob.** Where the last column says *not established*, apply the
prompt ceiling until one call proves otherwise. This avoids assuming a knob exists, setting nothing,
and finding out from the bill.
