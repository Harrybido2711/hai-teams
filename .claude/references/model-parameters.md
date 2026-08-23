# Model parameters — what every runner must set

**Applies to every benchmark, not one** — keyed by model, because the same providers serve
NegotiationToM, EmoBench, bbh, mmlu and DocVQA. Read before writing or changing any runner.

## The standing rule

Set by the user on 2026-08-22:

1. **Hidden reasoning is always capped** — every reasoning-capable model, every benchmark. A
   numeric budget where one exists, a level where it does not, the prompt where neither does.
   Thinking stays on; what is bounded is what it may spend. Independent of rule 3: capping the
   hidden half never depends on whether the visible half is shown.
2. **A model that does not reason gets an explicit output cap.** `max_tokens` on every call.
3. **Whether reasoning is *shown* is the benchmark's decision, read from its own README — at run
   time, not frozen into the runner.** Record the line it came from. A literal `True`/`False` stops
   tracking the README, and the next runner copied from it carries the wrong benchmark's answer.
   EmoBench's resolver is `reasoning_visibility.py`; it raises rather than guessing. Use the upstream
   mode, keep the other branch behind a flag — different conditions do not share a results table —
   and **record what was reasoned** ([reasoning-cost.md](reasoning-cost.md)).
4. **Set the value even when it equals the default.** A default belongs to the provider and can move;
   a pinned value records what the run used.
5. **No API limit? Set it in the prompt.** The last column says which models; the wording and
   its caveats are in [prompt-ceiling.md](prompt-ceiling.md).
6. **Pin a seed wherever the provider offers one, and write it on every row.** Without one a score
   difference cannot be told from the sampler: `gemini-3.5-flash-lite` gave two different answers to
   one EmoBench item in three identical calls, byte-identical ones under `seed=42`. That noise is
   worth ~3 points at n=200 — the size of the gaps we interpret, and it already produced one.
   **Both flash-lite routes accept it**; unestablished elsewhere, so probe. Untested across an
   OpenRouter backend switch.

## Per model

| Model | Called by | Reasoning? | Thinking cap | Output cap | Can it be lowered? If not → prompt |
|---|---|---|---|---|---|
| `gemini-3.5-flash-lite` | EmoBench (planned) | yes, at the floor | `thinking_level="minimal"` | `max_output_tokens` | **No, but `thinking_budget` bounds it in tokens** — 256 spent 127. `minimal` produces zero on EmoBench, which is thinking off in all but name. Bills at $2.50/M |
| `gemini-3.5-flash` | not used | yes | `thinking_level`, default `medium` | `max_output_tokens` | **Yes** — `medium` → `minimal`. Measure the step; do not assume it is free |
| `gemini-2.5-flash` | EmoBench, bbh, NegotiationToM, DocVQA | yes | thinking budget | `max_output_tokens` | **Not established** for this generation → treat as no. **Prompt ceiling** until a call proves otherwise |
| `grok-3-mini` | bbh, EmoBench, NegotiationToM | yes | `reasoning_effort` | provider default | **No — reasoning is the model.** **Prompt ceiling on thinking and on output** |
| `deepseek-reasoner` | bbh, EmoBench, NegotiationToM | yes | none exposed | `max_tokens` | **No knob at all.** **Prompt ceiling**, or change model |
| `Qwen/Qwen3.5-9B` | bbh, EmoBench, NegotiationToM | hybrid | `reasoning={"enabled": False}` | `max_tokens` | **Yes** — measured off: 11/12 vs 6/12 completions, 16 vs 517 output tokens |
| `kimi-k2.5` | bbh | not established | verify before assuming | `max_tokens` | **Not established → treat as no.** Prompt ceiling until verified |
| `gpt-4o-mini-2024-07-18` | bbh, DocVQA | no | n/a | `max_tokens` | n/a. `max_tokens` is enforced, so no prompt fallback is needed |
| `google/gemma-4-31B-it` | bbh, EmoBench, NegotiationToM | no as served | n/a | `max_tokens` | n/a **on DeepInfra, which serves it with thinking already off — passing `reasoning_effort` turns it back on.** On Together it is on by default and must be disabled |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | bbh | no | n/a | `max_tokens` | n/a. `max_tokens` is enforced |

"Called by" lists where the model id appears in a runner today, not that the run is current.
`gemini-3.5-flash` is listed only so it is not mistaken for Flash-Lite — a dearer model at $1.50/M
in and $9.00/M out.

**An unestablished knob counts as no knob.** Where the last column says *not established*, apply the
prompt ceiling and keep it until one call proves the API can do it. The failure this avoids is
assuming a knob exists, setting nothing, and finding out from the bill.

## Choosing the number

**Measure, do not reason from the parameter name.** One small slice at two or three settings,
accuracy and cost together. Measured budgets and the four ways capping backfires:
[reasoning-cost.md](reasoning-cost.md).

**Applied so far only in EmoBench's two flash-lite runners.** What every other runner currently
sets is on its benchmark's page; this file is what they must be changed to.
