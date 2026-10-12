# AwareBench full run, 2026-10-12: what happened and what the numbers mean

The first full run of AwareEval (4,015 rows, 10 tasks) for the five selected models. This file is the
run's record: incidents, the checks made after it, and the decisions taken. The benchmark's operating
facts (paths, counts, scoring, run order) are on its page,
`.claude/references/benchmarks/transition/awarebench.md`; the deep analysis of the dataset is
`AWARENESS_NOTES.md`.

## 1. The run

| Slot | Model | Config | Quest job | Rows | Empty |
|---|---|---|---|--:|--:|
| Gemini | `google/gemini-3.5-flash-lite` (OpenRouter) | reasoning effort `minimal`, `max_tokens=2048`, seed 42 | 9617441, resumed as 9645112 | 4,015 | 0 |
| OpenAI | `gpt-5.6-luna` | `reasoning_effort=low`, `max_completion_tokens=2048`, seed 42 | 9617440 | 4,015 | 0 |
| Qwen | `Qwen/Qwen3.5-9B` (Together) | `reasoning={"enabled": False}`, temperature 0, `max_tokens=2048`, seed 42 | 9617444 | 4,015 | 0 |
| Gemma | `google/gemma-4-31B-it` (DeepInfra) | no reasoning parameter, temperature 0, `max_tokens=2048`, seed 42 | 9617442 | 4,015 | 0 |
| Deepseek | `deepseek-flash` (the model `deepseek-reasoner` aliases) | "think briefly" system message, temperature 0, `max_tokens=8192`, seed 42 | 9617445 | 4,015 | 0 |

Every model: one config across all rows, scorer `aware_lenient_v4`, `COMPLETE 1`. The scores were
recomputed from the per-task `.jsonl` rows, match every `*_awareness_overall.csv` to four decimals,
and are recorded in the benchmark page's results table and in both workbooks' `Awareness` sheet
(confirmed by the user 2026-10-12). Overall: Deepseek 0.916, Gemini 0.794, Luna 0.791, Gemma 0.790,
Qwen 0.691; Qwen's perspective is 0 (§3).

## 2. Incidents during the run

- **Gemini: OpenRouter credit ran out at 13:03.** 280 calls failed with HTTP 402 ("This request would
  exceed your available credits…", "…requires more credits…"). Neither wording is in the core's
  `BILLING_SIGNATURES`, so the run did not stop at the first one; it stopped at `FAILURE_HALT` after
  10 consecutive failures. After a top-up it was resumed as job 9645112: rows already written were
  kept, the 16 rows written empty were retried. The failed job's logs are in
  `AWARE_Gemini_Flash3.5lite_OpenRouter/failed_logs_20261012_openrouter_credit/`.
- **No halt marker survived**, although Gemini halted: each of the ten array tasks clears the folder's
  halt markers when it starts, so a task starting late deleted the marker an earlier one wrote.
  Read the logs for `HALT`.
- **Qwen: Together rate limits.** 17 HTTP 429s, 2 read timeouts and 1 HTTP 503, all recovered by the
  core's retries (16 rows needed a second or third attempt). One 429 hit the parameter negotiation;
  it was retried and every parameter was still accepted.
- **Luna, Gemma, Deepseek: no errors.**

Open: add OpenRouter's wordings to the billing signatures (the first one also says "Retry after
in-flight requests settle", so whether it should stop the run at once is undecided), and stop array
tasks clearing each other's markers.

## 3. Qwen's PERSPECTIVE is 0: a real refusal, kept as scored

**What happened.** Qwen answered none of the 170 second-order questions ("Where does A think that B
searches for X?"). All 170 replies say the story gives no information about A's thoughts; 169 begin
with the same eight words, *"Based on the story provided, there is no…"*. All 170 were read: none
gives an answer after refusing, so the scorer is right to mark them `DECLINED` (wrong). The correct
location appears in 149 of them, always while retelling the story, never as an answer.

Qwen reads the stories correctly: on the other story tasks it scores 0.80 (first-order), 0.97
(reality), 1.00 (memory). Its first-order score is mostly object tracking, though: 0.89 on items where
the believed location equals the real one, 0.37 on false-belief items.

**Is it the model or our system?** Ten of the 170 items, one variable changed at a time, run locally
as a diagnostic (scratch output only; under $0.05 in total):

| Condition | Declined | Right |
|---|--:|--:|
| A. the full run (Together, temperature 0, reasoning off) | 10/10 | 0 |
| B. another host: OpenRouter → Parasail, bf16 | 10/10 | 0 |
| F. temperature 0.7 | 10/10 | 0 |
| reasoning **on** (2 lost to HTTP 503) | 5/8 | 0 — 2 others hit 6,144 tokens with no answer |
| D. prompt + "If the story does not state the answer directly, give your best inference." | 7/10 | 1 |
| E. a larger Qwen3.5 on the unchanged prompt: 35B-A3B / 122B-A10B / 397B-A17B | 0 / 0 / 0 | 3 / 3 / 4 |
| C. prompt + "Assume that every character sees everything that happens in a room while they are in it, and nothing that happens after they leave." | 0/10 | 6 |

**Conclusion.** The refusal belongs to Qwen3.5-9B. It does not depend on the host, the temperature or
the reasoning setting, and larger models of the same family answer the same prompt. What triggers it
is a property of the benchmark: ToMi-style stories never say who saw what, and every question
assumes the rule in condition C. The other models apply that rule without being told; the 9B reads
"think" literally and declines. Given the rule, it answers 6 of 10. So its 0 measures a refusal to
assume an unstated rule more than an inability to track beliefs. Ten items per condition shows the
direction, not a rate.

**Decision (user, 2026-10-12): the 0 stands.** It is a real refusal, every model gets the same prompt,
and changing the prompt for one model would change the benchmark. No rerun and no prompt change. The
workbooks carry the 0 with this caveat in `Provenance`.

## 4. PERSPECTIVE also measures how much a model reasons

Reasoning tokens spent on an average second-order item: Deepseek 1,465, Luna 179, Gemini 0, Gemma 0,
Qwen 0 (not reported; reasoning is off). Deepseek's "think briefly" instruction is a request it does
not keep on these items (maximum 7,789). The two models that reason lead the dimension (0.81, 0.61);
the two that refuse without reasoning trail it (Gemma declined 124 of 170, Qwen 170). Without
PERSPECTIVE, the four-dimension mean puts Gemma first (0.955), Deepseek second (0.943) and Qwen
(0.864) ahead of Luna (0.837). Deepseek's correct second-order replies state the gold plainly
("Ethan thinks Emma searches for the boots in the **crate**"); its lead is not the scorer.

## 5. Some second-order gold labels look wrong: unmeasured

Every model scores *lower* on second-order items whose gold equals the object's real location than on
false-belief items (Deepseek 0.69 against 0.98, Luna 0.49 against 0.79). Of three Deepseek misses on
such items, two have a gold label that the story contradicts:

> Liam exited the garden. **Then** Chloe moved the spinach to the pantry.
> "Where does Liam think that Chloe searches for the spinach?" Gold: **pantry**. Liam never saw the move;
> by the rule above the answer is the bucket, which is what Deepseek said.

How many of the 170 are affected is not known. Counting it needs a rule-based ToMi solver checked
against the gold; not done. If the rate is material, every model's PERSPECTIVE is affected, and the
most capable reasoners most.

## 6. Known scorer gap, not applied

Scorer v4 leaves 39 of 20,075 rows as parse failures although an answer can be read: 16 of Gemini's
32 end with one answer after naming others ("Since you requested A or B only: B"); of the other
models' 7, three name a container and its room ("in the basket in the pantry"), three are Qwen
naming the location and then narrating every move, and one writes "backyard" for `back_yard`. About
18 rows would turn correct: at most 1.2 points on a scored task (Gemini `mission_implicit`) and 3.3 on
a control (Qwen `reality`). The recorded numbers are v4. A fix would be a rescore from
`raw_response` (`--score-only`), never a rerun, and would change the workbooks.
