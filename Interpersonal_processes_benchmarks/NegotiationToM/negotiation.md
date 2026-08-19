# NegotiationToM — key findings

The things that changed our numbers or our costs, and would not be obvious to someone reading the
code. Companion files: `DATA_NOTES.md` (dataset structure), `ISSUES.md` (problem log with rejected
attempts), `Negotiation_script.md` (task and metric definitions).

Every figure here was measured, not estimated. Where a claim was later found wrong, the correction
is kept rather than the original quietly replaced.

---

## 1. Results so far

Only two full runs are trustworthy. Gemini and Grok completed with the right row counts but had
1,963 and 12,040 rows returned empty by quota and credit exhaustion, and an empty response scores
as a wrong answer.

| Metric | GPT-4o-mini | DeepSeek-Reasoner |
|---|---|---|
| Desire_EM | 0.5046 | **0.6234** |
| Belief_EM | 0.4522 | **0.6197** |
| Intent_Micro_F1 | 0.4194 | **0.4980** |
| Intent_Macro_F1 | 0.3376 | **0.4742** |
| All_EM | 0.0025 | **0.0496** |

Both: 14,138 rows, 0 empty responses, 0 duplicate uids, all invariants holding.

**`All_EM` is tiny by construction, not by error.** It requires every one of a dialogue's 5–6 rows
to be right simultaneously. The bottleneck is intention, which is multi-label and gets no partial
credit here even though its F1 does:

| | GPT | DeepSeek |
|---|---|---|
| desire correct for a whole dialogue | 0.289 | 0.388 |
| belief correct for a whole dialogue | 0.245 | 0.389 |
| **intention correct for a whole dialogue** | **0.011** | **0.094** |
| product, if the three were independent | 0.0008 | 0.0142 |
| **actual All_EM** | **0.0025** | **0.0496** |

Actual exceeds the independent-product prediction by ~3x, so the three tasks are positively
correlated — a model that understands a dialogue tends to get all three right together. A strict
metric also *amplifies* differences: DeepSeek leads GPT by 20–50% on the individual tasks but by
**20x** on All_EM.

---

## 2. Three dataset traps that silently change scores

Full detail in `DATA_NOTES.md`; these are the ones that actually bit.

**`"None"` is a stringified null, not an answer.** It marks a sample the annotators never filled in.
Evidence: it never appears mixed with a real label (all three priority slots together, 0 mixed cases
across four field groups); desire and belief are `"None"` on the *same* samples; and
`agent{N}_desire` still holds real values for those samples. 156 rows each in desire and belief have
no correct answer, so they are excluded from the metrics — `scored_rows` reports 4,604 of 4,760.
`Desire_EM_all` / `Belief_EM_all` report the other convention for comparison; the spread is
0.009–0.019.

**Intention has 4,618 rows, not 4,760.** Odd-length dialogues annotate only the last utterance and
set `utterance2_intent` to the string `"None"`. Emitting a second row anyway produced 142 phantom
rows with an all-zero gold bitmask — every prediction against them counted as a false positive —
plus 142 rows pairing the wrong utterance with the wrong label. **A count of 4,760 means the bug is
back.**

**Desire must be scored against the cutoff-aware flat fields.** `agent{N}_desire` is a dict that is
always a complete Food/Water/Firewood permutation, so it can never express `Not Given`; using it
asks the model for the full true ordering at a cutoff where nothing has been revealed.
`Output_template/openai_desire.csv` stores the dict but was produced by the same buggy path, so it
is not independent evidence.

---

## 3. Reasoning tokens are most of the bill, and were invisible

A $32.76 Gemini invoice against a ~$2 projection. The cause was our own accounting: `usage_from()`
read only the visible answer and ignored `thoughts_token_count`, which **bills at the output rate**.

| Gemini call | prompt | visible | thinking | undercount |
|---|---|---|---|---|
| #0 | 155 | 16 | 69 | 5.3x |
| #400 | 202 | 16 | 140 | 9.8x |
| #900 | 325 | 14 | **1,165** | **84.2x** |

The error grows with dialogue length, so a small sample understates it badly — which is exactly how
it survived so long.

**Measured properly (n=36, production prompts):**

| Setting | out/call mean | median | max | parse OK |
|---|---|---|---|---|
| uncapped | 1,222 | 984 | 5,356 | 36/36 |
| `thinking_budget=256` | 230 | 238 | 269 | 36/36 |

**76–81% saving, worst call 20x smaller, no accuracy cost.** On 15 belief items, thinking-off and
thinking-512 both scored 7/15 — the benefit saturates far below 256. Input tokens are unaffected.

**Cap reasoning; do not disable it.** This benchmark measures theory-of-mind inference and the two
reasoning-native models are the top performers, so switching reasoning off changes what is reported
rather than making it cheaper to report. The budget is written as
`ANSWER_TOKENS(32) * THINKING_MULTIPLE(8)` so the judgement stays legible when the task changes.

**Confirmed the hard way on Gemma, 2026-08-05.** Together's `reasoning={"enabled": False}` fixed a
real hang problem (see `ISSUES.md`, "A hung SDK call stalled a whole job") but was validated only on
desire — the one task where gold is self-stated, not inferred — and then applied to belief too,
which needs the model to infer the *opponent's* state. Paired on 463 identical uids against an
archived reasoning-ON pilot: Belief_EM 0.6177 (on) vs 0.4903 (off), McNemar exact p=1.2e-08. That is
a 3-rank swing (5th of 6 vs a near-tie for 2nd) from a config choice alone. Full detail, including the
mechanism (reasoning-off floods the `Not Given` slot: predicted 1,601 times vs 843 in gold) and what
is still needed to fix it, in `ISSUES.md`, "Gemma full run finished clean but belief is not usable as
published."

**Counter-intuitive, measured on Qwen: a bigger budget makes things worse.** Item #80 finished
cleanly at `max_tokens=8192` using 4,665 tokens, then burned all 16,384 on the identical prompt at
`temperature=0`. Success was 2/5 at 8192 and **0/3 at 16384**. The model spends what it is given, so
the budget shapes how much work it does rather than providing headroom for a fixed amount.

**Note on double-counting:** OpenAI-compatible APIs (DeepSeek, Together) already include reasoning
inside `completion_tokens`, so adding `reasoning_tokens` on top inflates the figure. Only Gemini
reports thoughts separately. `usage_from()` handles both shapes.

---

## 4. Silent failures — the dominant bug class here

Every one of these produced a job that reported success.

**Quota exhaustion looks like a finished run.** Grok's credits ran out mid-run; the job continued
for five more hours, wrote 9,378 empty rows, and all five shards exited `COMPLETED` with `0:0`.
Row counts were a perfect 14,138. `Belief_EM = 0.0` was an unpaid invoice, not a result.
Three guards now abort such a run, and they cover different shapes:
`halt_on_billing` stops at the *first* billing refusal or exhausted daily cap and writes a marker
file; `CONSECUTIVE_FAILURE_LIMIT = 10` catches a provider failing continuously; and a rolling window
(`FAILURE_WINDOW = 50`, `FAILURE_WINDOW_LIMIT = 0.5`) catches one failing *intermittently* — the
consecutive counter resets on any success, so at a 50% failure rate it never fires and the run still
ends with full row counts and half the scores zero.

**Billing patterns must match how providers really word it.** The original set
(`insufficient_quota`, `requests per day`, `billing`, `rate_limit`) missed both real failures: xAI
says *"used all available credits or reached its monthly spending limit"* — matching nothing, so
36,124 occurrences went unreported — and Gemini says `generate_requests_per_model_per_day` with
underscores, caught only by luck. The patterns must also **not** match timeouts, 503s or
`TypeError`, or a code bug gets misdiagnosed as a billing problem.

**A resume skips failed rows.** `load_checkpoint` adds *every* uid to the done set, including rows
stored with `raw_response: ""`. A plain re-run skips them, finishes in seconds, and rewrites the
identical broken numbers. `prune_failed_rows.py` must run first — its Quest dry run found 14,068
such rows.

**An SDK's own `timeout=` cannot be trusted.** Together ignored it twice: Gemma sat in one request
for over 2 hours at `timeout=18000`, then produced nothing for ~70 minutes at `timeout=300`, both
times with SLURM reporting RUNNING and an empty log. `guarded_call()` uses SIGALRM instead.
`CallTimeout` **must derive from `BaseException`** — as an `Exception` it was caught by each
runner's own `except Exception`, and since the alarm had already fired and been cleared, the rest of
that `call_api` ran unprotected. The guard evaporated after one use.

**Monitoring must watch progress, not job state.** A row count only moves when a checkpoint is
written, so a slow model looks frozen for a whole checkpoint interval. A stall detector reading row
counts cancelled Qwen three times mid-interval, discarding ~19 completed items each time and
livelocking the run at 160 rows. Liveness needs a second, continuous signal — log mtime — and a
short `--save-every` for slow models.

---

## 5. Conventions that must not drift

- **`deepseek-reasoner`** is the model this project uses, matching `bbh/deepseek_eval.py` and
  `EmoBench-master/EMO_Deepseek/`. An uncommitted change to `deepseek-v4-flash` with thinking
  disabled was reverted: nothing motivated it, and it would have discarded 14,138 rows from the
  best-scoring model and broken cross-benchmark comparability. The swap had touched five places,
  including the preflight registry.
- **Model-specific prompt tweaks stay in that model's runner**, never in the shared builders in
  `neg_eval_core.py`, so the other five keep answering an identical prompt and the comparability
  caveat stays confined and reportable.
- **Normalise model output only, never gold.** Gold is canonical; rewriting it changes the answer
  key.
- **Verify with the data, not with a convenience tool.** Counting uids with `cut -d, -f1` reported
  *more* unique uids than rows for DeepSeek — impossible — because embedded newlines in reasoning
  text break naive CSV parsing. Re-checking from the jsonl showed 0 duplicates. GPT's single-line
  JSON did not trigger it, so the false alarm looked model-specific and nearly caused a re-run.
