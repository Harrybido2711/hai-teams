# NegotiationToM — problem log

What broke, what was tried, what was rejected, what shipped. Maintained by the `tracker` agent.
Rejected attempts are kept on purpose: they are what stops the same road being walked twice.

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

### Qwen returned empty on nearly every call   2026-07-28  fixed (partial)

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

**Status** — partial. Per-attempt success is ~40%, which across `call_api`'s 5 attempts gives ~92%
per item. Qwen remains the slowest of the six; watch the empty-response rate.

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
