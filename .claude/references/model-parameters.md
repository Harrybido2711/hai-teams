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

## Choosing the number

Which knob each model has: [model-capabilities.md](model-capabilities.md).

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
