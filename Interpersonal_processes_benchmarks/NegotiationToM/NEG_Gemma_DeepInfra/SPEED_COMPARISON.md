# google/gemma-4-31B-it: Together AI vs DeepInfra

Two Quest jobs, 2026-08-06, submitted within seconds of each other. Same checkpoint name, same 493
NegotiationToM rows, same harness, same sequential shape. The provider client is the only
difference the experiment intends to have.

| | Together AI | DeepInfra |
|---|---|---|
| Job | 8687844 | 8687845 |
| Serving precision | **FP8** | **bf16** |
| Rows | 493 | 493 |

## Speed

| | Together | DeepInfra | ratio |
|---|---|---|---|
| **Wall clock, 493 rows** | **74.4 min** | **35.8 min** | **2.08x** |
| effective s/row | 8.9 | 4.1 | 2.17x |
| rows/min | 6.75 | 14.64 | 2.17x |
| Projected, 14,140 rows, 1 process | 35.5 h | 17.1 h | |
| Projected, 14,140 rows, 5 shards | 7.1 h | 3.4 h | |

## Latency — the median says the opposite of the total

| percentile | Together | DeepInfra | ratio |
|---|---|---|---|
| median | **2.5 s** | 2.9 s | **0.86x — Together is FASTER here** |
| p90 | 23.9 s | 6.6 s | 3.6x |
| p99 | 48.7 s | 28.3 s | 1.7x |
| max | 98.1 s | 75.8 s | 1.3x |

**Together wins the median and loses the run.** Half its calls come back in 2.5 s, faster than
DeepInfra's 2.9 s, and it still takes twice as long overall. Anyone benchmarking these two providers
on average latency would pick the slower one.

`p90 = 24 s` is not a measurement, it is a constant. Four in-run samples (88, 139, 211, 306 calls)
and all fifteen shards of the earlier 14,138-row production run report 23.9-24.0 s. A latency
distribution does not do that on its own; something on the Together path is releasing at a fixed
boundary.

## Where Together's extra 38 minutes went

| | Together | DeepInfra |
|---|---|---|
| Empty responses | 0 | 0 |
| JSON parse failures | 0 | 0 |
| Truncated at max_tokens | 0 of 493 (measured) | 0 of 493 (measured) |
| **Calls that never returned** | **9 of 502 (1.8%)** | **0 of 493** |
| **Wall clock at the 120 s ceiling** | **18 min** | **0 min** |

18 of the 74.4 minutes — **24% of the run** — was spent inside calls that produced nothing and were
killed by the watchdog. That is most of the gap.

This overhead is invisible in the mean. A hung call contributes no latency sample, so as hangs
accumulated the mean latency *improved* (7.4 s -> 6.9 s) while throughput *degraded* (7.47 ->
6.75 rows/min). Only `effective s/row`, which divides real wall clock by rows produced, shows it:

```
mean 6.9 s + (9 hangs x 120 s / 493 rows = 2.0 s) = 8.9 s = effective s/row
```

Report `effective s/row` for this pair. `mean latency` understates Together's cost by 29%.

## Accuracy — not a tiebreaker, a finding

Same 493 items, verified: the uid sets are identical in all three tasks (166/166/161, zero
one-sided). Scored with the project's own `belief_em` / `desire_em` / intent F1, unannotated rows
excluded exactly as `run_belief` excludes them.

| Task | Together | DeepInfra | paired discordance | McNemar p |
|---|---|---|---|---|
| Desire_EM | 0.6562 | 0.6750 | 6 DeepInfra-only vs 3 Together-only | 0.51 — no difference |
| **Belief_EM** | **0.5125** | **0.6125** | **18 DeepInfra-only vs 2 Together-only** | **0.0004** |
| Intent_Micro_F1 | 0.4790 | 0.4856 | 18 of 161 predictions differ | not tested |
| Intent_Macro_F1 | 0.4727 | 0.5026 | | |

**Belief is a real gap, not sampling noise.** Of the 20 items where the two providers disagreed
about the answer, 18 went DeepInfra's way. The paired (McNemar) test is what makes 160 items enough
to say that; an unpaired comparison of 0.5125 against 0.6125 would have returned a shrug.

Raw prediction agreement, which is the number that decides whether the two result sets can be
merged:

| Task | identical predictions |
|---|---|
| belief | 124/166 = **74.7%** |
| desire | 139/166 = 83.7% |
| intention | 143/161 = 88.8% |

**An FP8 row and a bf16 row are not interchangeable.** A quarter of belief answers differ, and the
difference is systematic rather than symmetric (18:2, not the ~10:10 that noise would give).

Not "two different models", which is how an earlier draft of this put it and is too strong: the
weights are presumably the same `gemma-4-31b-it`, served at different numerical precision. The
plausible mechanism is that greedy decoding takes an argmax over a narrow label set
(Food / Water / Firewood / Not Given), so when two candidates sit close together FP8's 3-4 mantissa
bits are enough to reorder them where bf16's 8 are not.

That mechanism is inferred, not measured. What was measured is only that the outputs differ. Other
explanations are live and were not ruled out: different inference engines (different kernels and
summation orders differ at the bit level even at equal precision), different chat-template
application, different argmax tie-breaking, or simply different checkpoint uploads behind the same
name. One observation argues against the tidy version of the quantization story — if near-ties
flipped more often on harder items, agreement should track difficulty, and it does not:

| task | agreement | Together score |
|---|---|---|
| belief | 74.7% | 0.5125 |
| desire | 83.7% | 0.6562 |
| intention | 88.8% | 0.4790 |

intention has the lowest score and the highest agreement.

Separating precision from provider needs FP8 and bf16 endpoints at the *same* provider, or the
weights loaded on Quest's own GPUs and run at both precisions from one script. Neither was done
here. The merge decision below does not depend on resolving it.

## Cost

Gemma is billed per token on both. Extrapolated to the full 14,140 rows:

| | Together | DeepInfra |
|---|---|---|
| Input tokens | 4,383,673 | 4,223,554 |
| Output tokens | 207,978 | 209,153 |
| Rate (in / out per 1M) | $0.10 / $0.34 | $0.13 / $0.38 |
| **Full-run cost** | **~$0.51** | **~$0.63** |

Twelve cents apart. Cost is not a decision input here; time and accuracy are.

## What this does not establish

- **One run each, no repeat.** No variance estimate for the timing. The speed ratio held between
  1.47x and 2.17x across five in-run pulses as samples accumulated, so the direction is not in
  doubt, but the exact multiple is one observation.
- **Both ran sequentially.** Neither number reflects DeepInfra's documented 200-concurrent ceiling
  nor Together's per-shard sharding. A concurrent harness was written, reviewed, found to
  misreport throughput and hang rate above one worker, and reverted; it is parked at
  `.claude/patches/concurrency-wip.patch`.
- **Precision is confounded with provider.** FP8-on-Together vs bf16-on-DeepInfra is one comparison,
  not two. The accuracy gap cannot be attributed to quantization alone without a bf16 Together
  endpoint or an FP8 DeepInfra one.
- **Together's hang rate here (1.8%) is above its own production history (1%)** on a 502-attempt
  sample. Same order, not the same number.
- **DeepInfra's "reasoning is off by default" rests on three probe calls**, not on a committed
  measurement script like the Together side's `measure_gemma_reasoning.py`. It is corroborated
  here only indirectly: output tokens matched (median 15 both sides, max 34 both sides), which
  hidden reasoning would have broken.

## Conclusions

1. **DeepInfra is 2.08x faster end to end**, and the reason is hangs, not per-call speed. Together
   is faster at the median.
2. **The two providers do not produce the same answers.** 74.7% agreement on belief, with DeepInfra
   significantly more accurate (p=0.0004). The existing 14,138-row Together result set cannot be
   extended with DeepInfra rows, and the two cannot share a row in a results table.
3. **If NEG_Gemma is re-run on DeepInfra, it must be re-run whole** — 3.4 h across 5 shards, ~$0.63.
