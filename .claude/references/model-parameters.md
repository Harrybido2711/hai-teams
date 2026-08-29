# Model parameters — what every runner must set

<!-- size-budget: 12000 -->
<!-- Deliberately one file: the standing rules, the per-model capability table read against
     them, and the settled config for each model in use. Splitting it was tried and reverted —
     the halves are always read together. It grows by one block per model adopted. -->

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
6. **Probe the parameter surface; the model page is a hypothesis.** For `gpt-5.6-luna` the page
   listed `top_p`, `frequency_penalty`, `presence_penalty`, `stop` and `logprobs` as supported and
   named a `max` reasoning level — **six claims, all refused by the API.** One probe call before a
   run costs nothing; discovering it per item across 400 items costs the run.
7. **A refused *value* is not a refused *parameter*, and conflating them silently removes a cap.**
   `gpt-5.6-luna` refuses `reasoning_effort="minimal"` while supporting the parameter perfectly
   well. A negotiator that saw the name in the error and dropped the parameter would have run every
   item at the model's default `medium` — uncapped, breaching rule 1, with nothing in the log saying
   so. Read the supported list out of the error and pick from it; drop a parameter only when the
   error says the *parameter* is unsupported.
8. **Pin a seed wherever the provider offers one, and write it on every row.** Without one a score
   difference cannot be told from the sampler: `gemini-3.5-flash-lite` gave two different answers to
   one EmoBench item in three identical calls, byte-identical ones under `seed=42`. That noise is
   worth ~3 points at n=200 — the size of the gaps we interpret, and it already produced one.
   **Both flash-lite routes accept it**; unestablished elsewhere, so probe. Untested across an
   OpenRouter backend switch.

## Per model

| Model                                                 | Called by                | Reasoning?        | Thinking cap                           | Output cap                             | Can it be lowered? If not → prompt                                                                                                                  |
| ----------------------------------------------------- | ------------------------ | ----------------- | -------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gemini-3.5-flash-lite` / `google/gemini-3.5-flash-lite` **← the project's Gemini** | Emo; OpenRouter route | yes, at the floor | `{"effort": "minimal"}` — see **Settled** | `max_tokens` / `max_output_tokens` | **No off switch; `thinking_budget` does not bind** — only `-1` changes the regime. $2.50/M                                                |
| `gemini-3.5-flash`                                  | not used                 | yes               | `thinking_level`, default `medium` | `max_output_tokens`                  | **Yes** — `medium` → `minimal`. Measure the step                                                                                         |
| `gemini-2.5-flash` — **superseded** | still in the bbh, NegToM and DocVQA runners | yes | thinking budget | `max_output_tokens`                  | **Not established** → treat as no. **Prompt ceiling** until a call proves otherwise                                                     |
| `grok-3-mini`                                       | bbh, Emo, NegToM         | yes               | `reasoning_effort`                   | provider default                       | **No — reasoning is the model.** **Prompt ceiling on thinking and on output**                                                           |
| `deepseek-reasoner`                                 | bbh, Emo, NegToM         | yes               | none exposed                           | `max_tokens`                         | **No knob at all.** **Prompt ceiling**, or change model                                                                                  |
| `Qwen/Qwen3.5-9B`                                   | bbh, Emo, NegToM         | hybrid            | `reasoning={"enabled": False}`       | `max_tokens`                         | **Yes** — measured off: 11/12 vs 6/12 completions, 16 vs 517 output tokens                                                                    |
| `kimi-k2.5`                                         | bbh                      | not established   | verify before assuming                 | `max_tokens`                         | **Not established → no.** Prompt ceiling                                                                                                      |
| `gpt-5.6-luna` **← the project's GPT** | Emo | **yes** | `reasoning_effort="low"` | **`max_completion_tokens`** | **Yes, but do not set it to `none`** — that is removal, not a cap, and costs 9–11 points (p≤0.003). `minimal` and `max` are refused. $0.20/M in, $1.20/M out |
| `gpt-4o-mini-2024-07-18` — **superseded** | still in the bbh and DocVQA runners | no | n/a | `max_tokens` | n/a — `max_tokens` is enforced |
| `google/gemma-4-31B-it`                             | bbh, Emo, NegToM         | no as served      | n/a                                    | `max_tokens`                         | n/a**on DeepInfra: served with thinking off, and `reasoning_effort` turns it back on.** On Together it is on by default and must be disabled |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | bbh                      | no                | n/a                                    | `max_tokens`                         | n/a — enforced                                                                                                                                      |

"Called by" is where the id appears in a runner today, not that the run is current.
`gemini-3.5-flash` is listed only so it is not mistaken for Flash-Lite — dearer, $1.50/M and $9.00/M.

**An unestablished knob counts as no knob.** Where the last column says *not established*, apply the
prompt ceiling until one call proves otherwise. This avoids assuming a knob exists, setting nothing,
and finding out from the bill.

## Choosing the number

**Measure, do not reason from the parameter name** — one slice at two or three settings, accuracy
and cost together ([reasoning-cost.md](reasoning-cost.md)). Applied so far only in EmoBench's two
flash-lite runners; what the others set is on their benchmark pages.

## Settled — the project's GPT

**`gpt-5.6-luna` on the OpenAI platform, `reasoning_effort="low"`.** Set 2026-08-26; it replaces
`gpt-4o-mini` wherever a GPT is called. A reasoning model: 1,050,000 context, 128,000 max output,
$0.20/M in ($0.02/M cached) and $1.20/M out.

```python
max_completion_tokens=2048, reasoning_effort="low", seed=42   # temperature unsettable
```

**`low`, not `none`, and the difference is measured.** EU, 200 items, seed pinned, paired McNemar:

| effort | accuracy | thinking/row | cost / 200 items |
|---|---|---|---|
| `none` | 0.565 | 0 | $0.0189 |
| **`low`** | **0.650** | median 40 | **$0.0366** |
| `medium` (the model's default) | 0.675 | median 48 | $0.0430 |

`none` vs `low` **p=0.0033**, `none` vs `medium` **p=0.0003** — both significant. `low` vs `medium`
**p=0.332**, not significant. So switching thinking off costs 9–11 accuracy points to save two cents,
while `low` takes the entire significant gain at 85% of `medium`'s output tokens.

**This is where rule 1 gets misread.** The rule bounds *what thinking may spend*; it does not say
switch thinking off. `none` is removal, not a cap. On `gemini-3.5-flash-lite` the distinction did not
matter — capping cost nothing measurable — and copying that shape onto this model is exactly how
`none` got adopted here before the sweep ran.

**Its parameter surface is narrower than 4o-mini's, and the published page overstates it.** Probed
against 4o-mini, which accepted everything it was given:

| | `gpt-5.6-luna` | `gpt-4o-mini` |
|---|---|---|
| token cap | **`max_completion_tokens`** (`max_tokens` refused) | either |
| `temperature` | **only the default 1** | any |
| `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `logprobs` | **all refused** | all accepted |
| `seed`, `n`, `service_tier` | accepted | accepted |
| `verbosity` | accepted (`low`) | refused |
| `reasoning_effort` | `none`/`low`/`medium`/`high`/`xhigh` — **not** `minimal`, **not** `max` | not a parameter |

The model page lists `top_p`, `frequency_penalty`, `presence_penalty`, `stop` and `logprobs` as
supported and names a `max` effort level. **Five of those are refused in practice and `max` is
refused too** — the probe is what a runner should be built on, not the page.

Measured reasoning tokens on one EmoBench-shaped item: `none` 0, `medium` 129, `low` 203, `high`
512, `xhigh` 512. **A refused value is not a refused parameter** — dropping `reasoning_effort`
because it rejected `minimal` would have run the benchmark at the default `medium`, uncapped.

## Settled — the project's Gemini

**`gemini-3.5-flash-lite`, served by OpenRouter, thinking at `minimal`.** Set by the user on
2026-08-23; it replaces `gemini-2.5-flash` wherever a Gemini is called. Two decisions inside it:
**OpenRouter, not the native SDK** — the routes measured the same speed (2.92 vs 2.76 s an item) and
Quest's google-genai is a version behind, so the OpenAI-compatible client is the one to keep working.
And **thinking at `minimal`, dynamic thinking off** — see the numbers below.

Measured over EmoBench, 200 EU items an arm.

```python
max_tokens=2048, seed=42, extra_body={"reasoning": {"effort": "minimal"}}   # temperature unset
```

- **`effort: "minimal"`** — 0 thinking tokens over 400 items. Dynamic thinking was measured and
  rejected: 30.5 thinking tokens per token of answer, 31× cost, 4× wall clock, for +6 points that
  never reached significance (p=0.21).
- **`seed=42`** — without it 22.5% of items change between runs. **Not sufficient alone here:** the
  answer follows the serving backend and OpenRouter fails over mid-run. `--provider "Google AI Studio"` makes four seeded calls identical; going without is a deliberate choice, so this route is
  reproducible only as far as the routing holds.
- **no `temperature`** — unset, `0.0` and `0.6` scored the same and agreed on all 200 items.
- The native route is the same, except the cap is `thinking_budget=128` (`thinking_level` is absent
  from Quest's SDK) and its seed reproduces exactly, 120/120.
