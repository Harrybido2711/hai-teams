# Model parameters — what every runner must set

<!-- size-budget: 7000 -->

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

## Choosing the number

**Measure, do not reason from the parameter name** — one slice at two or three settings, accuracy
and cost together ([reasoning-cost.md](reasoning-cost.md)). Applied so far only in EmoBench's two
flash-lite runners; what the others set is on their benchmark pages.

## Settled — `gemini-3.5-flash-lite` on OpenRouter

Measured over EmoBench, 200 EU items an arm, adopted 2026-08-23.

```python
max_tokens=2048, seed=42, extra_body={"reasoning": {"effort": "minimal"}}   # temperature unset
```

- **`effort: "minimal"`** — 0 thinking tokens over 400 items. Dynamic thinking was measured and
  rejected: 30.5 thinking tokens per token of answer, 31× cost, 4× wall clock, for +6 points that
  never reached significance (p=0.21).
- **`seed=42`** — without it 22.5% of items change between runs. **Not sufficient alone here:** the
  answer follows the serving backend and OpenRouter fails over mid-run. `--provider "Google AI
  Studio"` makes four seeded calls identical; going without is a deliberate choice, so this route is
  reproducible only as far as the routing holds.
- **no `temperature`** — unset, `0.0` and `0.6` scored the same and agreed on all 200 items.
- The native route is the same, except the cap is `thinking_budget=128` (`thinking_level` is absent
  from Quest's SDK) and its seed reproduces exactly, 120/120.
