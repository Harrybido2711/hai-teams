# NegotiationToM — problem log

What broke, what was tried, what was rejected, what shipped. Maintained by the `tracker` agent.
Rejected attempts are kept on purpose: they are what stops the same road being walked twice.

---

### Reasoning tokens caused disproportionate cost   2026-07-29  fixed locally

**Symptom** — GPT, Gemini and xAI pilots used roughly 14–15 output tokens per successful call, but
DeepSeek Reasoner averaged about 466. Qwen repeatedly consumed all 8,192 tokens and returned empty
content, then retried. Empty responses were not included in the token totals, so the Qwen report
substantially understated its actual cost.

**Root cause** — reasoning was enabled for simple fixed-label classification. Qwen3.5-9B is a
hybrid model but was treated as reasoning-only; Gemini retained a 256-token thinking allowance;
the retired `deepseek-reasoner` alias selected V4 Flash thinking mode. Usage was recorded only after
non-empty content, excluding billed failed attempts.

**Fix** — disable reasoning through provider controls: Together
`reasoning={"enabled": False}` for Qwen, Gemini `thinking_budget=0`, and DeepSeek V4 Flash
`thinking: disabled`. Keep the visible output ceiling at 8,192 where already requested so valid JSON
is not truncated. Record usage from empty HTTP-success responses separately from successful calls.
GPT, xAI and Gemma remain unchanged because their pilots did not show reasoning-token inflation.

**Gate before full run** — archive stale checkpoints, run `preflight.py`, then rerun equal-size
pilots. Verify Qwen has no `finish=Length tokens=8192` loop and compare output tokens per API
response, not only per successful row.

---

### Intention rows paired with the wrong utterance   2026-07-27  fixed

**Symptom** — 4,760 intention rows where 4,618 were expected. 142 rows had an all-zero gold bitmask,
so every prediction against them counted as a false positive. Micro F1 0.4315 with them, 0.4385
without.

**Root cause** — the dataset annotates one target utterance for odd-length dialogues, not two, and
marks the absent second one with the string `"None"`. `run_intention()` appended the utt2 row
unconditionally, producing one row whose text was `turns[-2]` but whose gold belonged to
`turns[-1]`, plus a phantom row with empty gold. `"None"` is not one of the 9 labels, so
`intent_bitmask` mapped it to all zeros without raising.

**Fix** — branch on `utterance2_intent == "None"` and emit a single row targeting `turns[-1]`
(`neg_eval_core.py::run_intention`).

**Verified** — 4,618 rows on the real data, 0 speaker mispairings, 0 all-zero gold rows; confirmed
again in the pilots (464/464 for a 238-dialogue subset).

---

### Desire scored against the wrong gold field   2026-07-27  fixed

**Symptom** — desire gold contained only Food/Water/Firewood; not a single `Not Given` in 4,760
rows, though the spec's label set includes it.

**Root cause** — `run_desire()` used the `agent{N}_desire` dict, which is always a complete
permutation, instead of the cutoff-aware `agent{N}_desire_{high,medium,low}` fields. The model was
being asked for a full preference ordering at a cutoff where the dialogue had revealed nothing.

**Tried and rejected**
- Treating `Output_template/openai_desire.csv` as evidence for the dict → it was produced by this
  same buggy path, so it is not independent evidence.

**Fix** — build gold from the flat fields, and update `desire_messages()` to offer `Not Given`,
without which the model can never match those labels.

---

### `"None"` counted as a wrong answer   2026-07-28  fixed

**Symptom** — 156 desire and 156 belief rows the model could never get right.

**Root cause** — `"None"` marks a *missing annotation*, not "wants nothing". Evidence: never mixed
with a real label (all three slots together, 0 mixed cases); desire and belief are `"None"` on the
same samples; and `agent{N}_desire` still holds real values for them.

**Fix** — `is_unannotated()` / `scorable()` exclude those rows from the metrics while still writing
them to CSV, plus a `{task}_scored_rows` audit metric. `Desire_EM_all` / `Belief_EM_all` report the
other convention too, since `Negotiation_script.md` lists `None` in the label set and a published
table may have scored them. Spread between conventions: 0.009–0.019.

---

### Qwen returned empty on nearly every call   2026-07-28  superseded 2026-07-29

**Symptom** — 60 rows in 7 hours. `content` empty, HTTP 200, no exception.

**Root cause** — Qwen3.5 is a thinking model whose reasoning length is *unstable*: ~700 to past
32,768 output tokens for the same prompt at `temperature=0`. When the budget runs out mid-thought
Together returns `finish_reason=Length` with empty `content`, the reasoning sitting in a separate
field, so the answer is never emitted.

**Tried and rejected**
- `max_tokens=8192` *without* a brevity hint → truncated almost every call.
- `max_tokens=32768` alone → 3 of 4 prompts fine, the fourth burned the whole budget over 429s and
  still returned empty.
- `max_tokens=32768` **with** the brevity hint, plus a 150s watchdog → the worst combination. At
  ~90 tok/s a full 32,768-token generation needs ~364s, so every runaway was killed by the clock
  before it could report `finish_reason=Length`. The diagnostic signal was destroyed and each retry
  walked into the same wall; the pilot sat at 160 rows for over two hours.
- Tightening the watchdog further → wrong lever. A wall-clock kill bounds nothing usefully and
  explains nothing; `max_tokens` is server-enforced, returns promptly, and reports why.

**Fix** — brevity hint **plus a small budget**, `max_tokens=8192`, with the watchdog at 200s acting
only as a backstop for hung connections (above the ~110s worst case of a legitimate full-budget
generation).

**The counter-intuitive part, measured:** a bigger budget makes things *worse*. Item #80 finished
cleanly at 8192 using 4,665 tokens, then burned all 16,384 on the identical prompt at
`temperature=0`. Success rate was 2/5 at 8192 and **0/3 at 16384**. The model spends what it is
given, so the budget shapes how much work it does rather than providing headroom for a fixed
amount.

**Status** — superseded. Together documents Qwen3.5-9B as a hybrid model, so the 2026-07-29 fix
uses `reasoning={"enabled": False}` instead. The brevity-plus-budget workaround is retained here
only as rejected history.

---

### Quest ran code that had never been transferred   2026-07-29  fixed

**Symptom** — the Qwen pilot behaved exactly as it had *before* the 2026-07-29 fix: 3h10m elapsed,
desire 315/476, belief 0, intention 0, 105 empty responses all `finish_reason=Length` at 8,192
tokens, 47 timeouts. At 1.66 rows/min the 1,416-row pilot needed ~14h against a 16h wall.

**Root cause** — `reasoning={"enabled": False}` existed only on the laptop. Quest was still running
the superseded brevity-hint version. The transfer had never happened, and nothing in the loop
checked: the queue, the logs and the row counts all look normal when the code itself is old.

**Scope was wider than the model under suspicion.** A full `md5sum` comparison found **6 of 32
files** stale on Quest — `neg_eval_core.py` plus every runner except DeepSeek. DeepSeek matched only
because a previous session's revert happened to touch it. The gap opened when one session edited
several runners and a later session, reverting only the DeepSeek change, assumed the rest were in
sync.

**The dependency that makes partial sync fatal** — the runners import `record_usage` from
`neg_eval_core.py`, added 2026-07-29. Transferring a runner without the core fails at import;
transferring the core without the runners breaks whichever runner used a changed signature. Core and
runners must move together.

**Checkpoints are not always resumable.** The 315 existing Qwen rows were produced with
`BREVITY_HINT` appended to the user turn, which the new code removes. Resuming would mix two prompts
in one result set, so the checkpoint must be archived rather than resumed. A config change
invalidates a checkpoint even when the schema is unchanged.

**Fix** — a `md5sum` diff of all `*.py` and `*.sh` against Quest before every submit, recorded in
`CLAUDE.md` and `.claude/agents/executor.md`, plus standing authorisation for the planner to
`scancel` a job found to be running wrong code instead of letting it reach the wall.

**Two ways the check itself lied while being written**, both worth guarding against:
- zsh does not word-split an unquoted `$FILES`, so `md5 -r $FILES` treated the whole list as one
  filename. Both sides produced empty files and `diff` reported "in sync" — a false pass of exactly
  the check meant to prevent false passes. Use a quoted array expansion.
- `join` requires input sorted on the join field. Sorted by hash instead of filename it reported
  every file as simultaneously missing from both sides. Sort with `-k1,1` on the name, and print
  both row counts before believing the output.

**Also found** — `ssh quest` was never a real alias. `~/.ssh/config` contained only an unrelated
host, so every documented `ssh quest "..."` command failed with `Host key verification failed`; the
config block existed only as a recommendation inside a note. Installed and verified 2026-07-29.

---

### A hung SDK call stalled a whole job   2026-07-28  fixed

**Symptom** — Gemma reported RUNNING by SLURM for over 2 hours with an empty log, no stderr and no
new rows. Recurred for ~70 minutes after the first mitigation.

**Root cause** — Together's client did not honour its own `timeout=` argument. A timeout the library
can ignore is not a timeout.

**Tried and rejected**
- `timeout=18000` inherited from the EmoBench scripts → five hours of stall per hung call.
- Lowering it to `timeout=300` → still hung for ~70 minutes, so the parameter was not being applied.

**Fix** — `neg_eval_core.py::guarded_call` wraps every `call_api` in a SIGALRM alarm (420s), which
interrupts the blocking read regardless of the SDK. `CallTimeout` derives from **`BaseException`**:
as an `Exception` it was caught by each runner's own `except Exception`, and since the alarm had
already fired and been cleared, the remaining attempts ran unprotected — the guard evaporated after
one use.

**Verified** — a hung call is interrupted at the limit and retried, a normal call is untouched, the
alarm is always cleared, and the watchdog survives a runner's own `except Exception`.

---

### Stale results silently accepted as complete   2026-07-27  fixed

**Symptom** — NEG_GPT held 15 shard files (14,280 rows) from a pre-fix run, with uids identical to
what the new code generates.

**Root cause** — resume logic skips any uid already present, so a full run would have "succeeded"
in seconds while emitting the old, buggy data.

**Fix** — archived to `results_archive_<timestamp>/` before submitting. Checking for stale
checkpoints is now a standing pre-submission step.

---

### Hourly backups that backed up nothing   2026-07-28  fixed

**Symptom** — the monitor pushed to git every hour for eight hours and produced two commits; every
push reported "Everything up-to-date".

**Root cause** — the commit was conditional on local changes, but all output lived on Quest and was
never pulled down. The results had no backup at all.

**Fix** — each cycle now pulls `results/pilot/` and the logs from Quest first, then commits and
pushes. Six pilots are ~3 MB, a full run ~9 MB per model — small enough for git.

---

### False alarms, recorded so they are not re-investigated

- **xai_sdk `chat.create` rejecting `max_tokens`/`temperature`** — it accepts both (verified against
  `xai_sdk 1.4.0` on Quest). BBH and EmoBench pass only `model=` out of caution, not necessity.
- **Missing `NegotiationToM/.env`** — absent locally by design (gitignored) but present on Quest,
  which is the only place jobs run. A Globus mirror sync did delete it once;
  `cp ../EmoBench-master/.env .env` restores it.
- **grok rejecting a health-check prompt with `SAFETY_CHECK_TYPE_BIO`** — a false positive on a
  synthetic probe (`"You are a JSON API."`); the genuine eval prompts pass. `preflight.py` now
  probes with prompts built by the real prompt builders.
