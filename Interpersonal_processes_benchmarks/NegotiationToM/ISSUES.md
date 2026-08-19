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
`CLAUDE.md` and `.claude/references/quest-cluster.md`, plus standing authorisation for the planner to
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

**Outcome** — once the transfer was fixed and the sync check in place, xAI resumed after a top-up
(see "Two different refusals, both mistaken for retryable errors" below) and Gemini restarted with
thinking disabled, both as sharded full runs. The Qwen pilot itself went from 320 rows with 105
empties in 3h37m to 476 desire rows with 0 empties in about 12 minutes.

---

### An unattended watcher fought the operator   2026-07-29  open

**Symptom** — `watch.sh`, left running by an earlier session (23h11m uptime), resubmitted jobs
whenever one left the queue and auto-committed and pushed to both remotes every hour. When a
correction workflow cancelled the Qwen pilot, the watcher cancelled and resubmitted both Qwen and
Gemma. Killing the process was not enough: the originating Claude Code session relaunched it, and
the relaunched copy cancelled both jobs again 13 minutes later. It also committed unreviewed work as
`NegotiationToM: watcher checkpoint 2026-07-29 13:43` (commit `39dace6`, containing CLAUDE.md and
ISSUES.md) and pushed it to origin and backup.

**Root cause** — the watcher both *observed* queue state and *acted* on it (resubmit-on-exit), with
no way to distinguish "job left the queue because an operator cancelled it on purpose" from "job
left the queue because it finished or crashed". Its restart lived inside the session that spawned
it rather than anywhere independent of it, so killing the process removed the symptom but not the
thing that kept bringing it back. Its hourly commit staged whatever was dirty in the working tree,
with no check for whether the watcher itself had produced it.

**Tried and rejected**
  - `kill <pid>` on the watcher process → stopped the loop for a moment → the parent Claude Code
    session relaunched it, and the new copy repeated the identical cancel/resubmit within 13
    minutes, so the process, not the session, was the wrong thing to kill.

**Fix** — not yet shipped; contained only by manually killing both the watcher and the session that
owned it. Still needed: a watcher stoppable by a signal external to its own session (e.g. a
kill-file it checks every loop iteration, so a relaunch inherits the stop state), and an auto-commit
restricted to a fixed allow-list of the paths it itself writes, never a blanket `git add`.

**How it was verified** — not verified; this is an incident record, not a shipped fix. Re-open when
a watcher is (re)designed, and confirm an external stop signal survives a session restart and that
its commits touch only its own output paths.

---

### Two different refusals, both mistaken for retryable errors   2026-07-29  fixed

**Symptom** — two independent full runs went largely empty for two different unretryable reasons,
both of which the failure classifier treated as ordinary transient errors.
- **xAI**: credits ran out mid-run. The error text — "has either used all available credits or
  reached its monthly spending limit" — matched neither string the runners checked
  (`insufficient_quota`, `requests per day`), so the run kept retrying to exhaustion: 12,040 of
  15,554 rows were written empty (belief and intention 100% empty, desire 55.9%).
- **Gemini**: `gemini-2.5-flash`'s daily cap of 10,000 requests/day was 41% below the 14,138 rows the
  run needed — mathematically impossible to finish in a day at Paid Tier 1 (the tier upgrade needs
  $250 cumulative spend; the account was at ~$36). Retries count against the same daily cap, so the
  roughly 5,900 failed attempts alone consumed about 60% of that day's allowance, leaving 1,963 rows
  empty. `retry_delay` could not even parse Google's own backoff hint, "Please retry in 18h7m",
  because its regex looked for the string "try again in".

**Root cause** — the halt guard matched only the wording seen in earlier incidents (the original
`billing`/`rate_limit`/`insufficient_quota`/`requests per day` set), not either of these two exact
strings. The `CONSECUTIVE_FAILURE_LIMIT` guard that would otherwise have caught the runaway xAI
retries was written 2026-07-29 11:01 — about 30 hours *after* that xAI run had already ended.

**Tried and rejected**
  - Matching the bare word "billing" → it also appears inside Gemini's ordinary rate-limit
    boilerplate → would have hard-stopped healthy, merely rate-limited Gemini runs, not just genuine
    billing failures.
  - Expecting `thinking_budget=0` to help Gemini's cap → thinking tokens dropped but request volume
    did not → the 10,000/day limit counts requests, not tokens, so it does nothing to the quota math.

**Fix** — a three-way classification in `neg_eval_core.py`: DAILY QUOTA and BILLING each halt on
their first occurrence and write a `QUOTA_HALT.txt` / `BILLING_HALT.txt` marker; a set of transient
signatures vetoes the billing match so Gemini's ordinary 429 boilerplate (which happens to contain
the word "billing") stays retryable, while xAI/OpenAI's `insufficient_quota` — also delivered as a
429 — still halts. `CONSECUTIVE_FAILURE_LIMIT` lowered from 40 to 10.

**How it was verified** — both providers' captured error strings were replayed against the new
classifier and landed in the correct bucket: xAI's spending-limit message and Gemini's daily-quota
message each trip BILLING/DAILY QUOTA and halt on first occurrence, while Gemini's ordinary 429
rate-limit text, despite containing "billing", is vetoed back to transient and keeps retrying. Both
runs resumed after the fix shipped — see the outcome appended to "Quest ran code that had never been
transferred" above.

**Confirmed the guard is sufficient across a multi-day quota reset — 2026-08-04, job 8527057.**
Since a same-day finish was already established as mathematically impossible (14,138 rows needed,
10,000/day cap), the correct behaviour is a clean halt followed by a resume once the cap resets, not
a retry-around-it. Full run 8178454 (2026-07-29) hit the cap again with the fix in place and halted
exactly as designed: belief 4,760/4,760 and desire 4,760/4,760 written with 0 empty rows, intention
stopped at 840/4,618 with 0 empty rows — no runaway empty-row retries burning quota, unlike the xAI
incident above. The checkpoint was left alone rather than archived, since the committed runner
(`THINKING_BUDGET=0`) had not changed since that halt. Resubmitting as job 8527057 on 2026-08-04 (six
daily resets later) skipped the 9,520 belief/desire uids and the 840 good intention uids, wrote the
remaining 3,778 intention rows in about 30 minutes, and reached the full 14,138 rows with 0 empty
responses. A cohort check comparing the two halves found no discontinuity: the 840 rows written
before the halt score Micro F1 0.5278 / Macro F1 0.5183, the 3,778 written after resume score
0.5389 / 0.5119, and none of the 840 pre-existing uids were missing or changed by the resume.
**The halt guard's job is not to work around the cap — it is to stop cleanly enough that a resume
days later is safe to trust**, and this run is the first end-to-end confirmation of that for Gemini.

---

### Local halt markers and summary tables don't get refreshed after a fix lands   2026-08-04  open

**Symptom** — found while auditing the 2026-08-04 Gemini resume (job 8527057) above: two files on
the laptop still showed 2026-07-29 state after the run had finished cleanly on Quest.
`NegotiationToM/NEG_Gemini/QUOTA_HALT.txt` is dated Jul 29 16:23 and describes a halt that a run six
days later resolved. `NegotiationToM/negotiation_results.csv` is dated Jul 29 13:12 and lists only
GPT-4o-mini and DeepSeek-Reasoner, though Qwen3.5-9B, grok-3-mini and gemini-2.5-flash have since
published full runs — 2 of 5 models represented.

**Root cause** — two gaps of the same shape. The halt marker is cleared on Quest before
resubmission, because that is where the guard reads it, but nothing clears the identical local copy,
so a reader checking the laptop sees a job as still halted long after it finished. `negotiation_results.csv`
has no producer script — a `grep` for its filename across every `NegotiationToM/*.py` and `*.sh`
returns nothing — so it is a one-time hand-built table (one commit, `7306710`, the GPT/DeepSeek
publish) that nothing re-runs when another model's results are published; it silently drifts every
time. Neither file is covered by the `*.py`/`*.sh` sync check in `CLAUDE.md`, which only compares
code, not state or summary artefacts. Same failure shape as "Stale results silently accepted as
complete" below, one level up: there it was a stale checkpoint mistaken for current data, here it is
a stale marker/summary mistaken for current state.

**Fix** — not yet shipped. Needed: (1) delete or timestamp-rotate a halt marker on the laptop
whenever it is cleared on Quest, so the two copies cannot disagree; (2) regenerate
`negotiation_results.csv` from the per-model `*_overall.csv` files as a step in the publish/record
workflow, or drop it in favor of `negotiation.md`'s results table, which has been kept current.

**How it was verified** — not yet fixed; this is an open finding, not a shipped fix. Re-open check
once a regenerate/clear step exists and confirm both files' mtimes track the run they describe.

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

**The ceiling stopped an infinite hang but was still ruinous at sharded full-run scale — 2026-08-04,
job 8526978.** Relaunching Gemma's full run (14,138 rows, 5 shards) hit the same hung-Together-
connection mode this fix targets, just no longer fatal to one call — instead it dominated the whole
job. Cancelled by the operator (SIGTERM, `sacct` batch exit `0:15`) after 4h20m with 660/14,138 rows
(4.7%): desire 660/4,760, belief 0, intention 0 — the run never left the first task. The five shard
logs show 314 `call exceeded 200s hard limit` lines (59–66 per shard); 314 x 200s = 62,800s, 80.4% of
the 1,302.5 shard-minutes of wall clock. Aggregate throughput was 2.53 rows/min across all 5 shards
(0.507/shard) versus the pilot's 2.41 rows/min on a **single** shard — sharding delivered 1.05x, not
5x, because every extra worker pays the same 200s tax on its own hangs, so wall clock scales with the
hang rate almost independently of shard count. The 19.6h projection was dead; the observed rate
implied 93h against a 7-day walltime. 32 of the 660 rows (4.85%) were empty (`raw_response=""`,
`pred=null`, `neg_eval_core.py:531-544` exhausting its 3-attempt retry budget on repeated timeouts) —
the pilot had 0 empties. All 32 scored `desire_em=0`, pulling the figure from 0.6099 (628 non-empty
rows) to 0.5803 (all 660); no `_overall.csv` exists for a cancelled run, so neither number is the
pipeline's real metric and must not be quoted against the pilot's 0.650.

**Root cause** — the 200s ceiling bounds one call, but was never sized against the hang *rate* at
sharded scale. Together's ~9% hang rate on this job, multiplied by 5 parallel shards each paying the
tax independently, turned a bounded per-call cost into a throughput collapse.

**Fix, written but not yet deployed** — `NEG_Gemma/gemma_neg_eval.py:50-51` (uncommitted) drops
`max_tokens` from 8,192 to `MAX_TOKENS = 2048` and calls `set_call_timeout(90)` — 2.5x the ~36s
worst legitimate call at that budget — instead of the shared 200s default, sized from this job's own
logs, not another estimate. `neg_eval_core.py` gained `budget_report()` (`:980`) and a 600s
`_pulse()` (`:711`) driven from `guarded_call`, not from checkpoint saves, so a 100%-hang run still
reports on a clock instead of going silent. **As of this entry both files are uncommitted locally and
Quest still runs the pre-fix code** (`sync` check: 2 of 32 files differ — `neg_eval_core.py`,
`NEG_Gemma/gemma_neg_eval.py`). Do not resubmit until that diff is committed, pushed, and confirmed
clean on Quest per "Quest ran code that had never been transferred" above, and archive the 660-row
checkpoint rather than resuming — it was written at `max_tokens=8192` and would mix two configs with
whatever the corrected run produces.

**How it was verified** — not yet; the fix has not run. Re-check by looking for the diagnosis line
`budget_report()` prints (`HUNG CONNECTIONS, not slow generation`) versus `calls are genuinely
running to the ceiling`, and confirm `rows_per_min_per_shard` recovers toward the pilot's per-shard
rate once resubmitted on synced code.

**Fix shipped and confirmed — 2026-08-05, commit `9a5c734`, jobs 8625800/8625801/8625810.** The
config above (`reasoning={'enabled': False}`, `max_tokens 8192`, `set_call_timeout(120)`, backoff
that disarms the watchdog across its sleep) went in as committed, and the resubmitted run reached all
14,138 rows with 0 empty responses. Per-call hang rate fell to 0.88% (125/14,223 attempts) from the
~9% that had dominated job 8526978. Latency p99 held at 45.4-54.2s across shards against the 120s
ceiling (2.2x margin at the worst shard), so the ceiling is sized correctly and not manufacturing
hangs. **This did not come for free**: disabling reasoning entirely — beyond what this entry's fix
called for — cost 12.7 points on belief specifically and makes that column not cross-model
comparable; see "Gemma full run finished clean but belief is not usable as published" below, which
is the direct continuation of this incident.

---

### Per-task arrays beat a bigger shard count for parallelism   2026-08-05  fixed

**Symptom** — Gemma was the last of six models with no full run. Job 8625688 (`--task all`,
`--total-shards 5`) was healthy — each of the 5 workers runs desire (952 rows), then belief (952),
then intention (924) in sequence — but only ~3 minutes in with 0 rows checkpointed, so the whole
14,138-row run was still entirely ahead of it at 5-way parallelism.

**Root cause** — every one of the six models' `run_negotiation.sh` inherited `--array=0-4` from an
untested belief, stated in `NEG_GPT/run_negotiation.sh:14` as "Quest allows at most 5 parallel array
jobs, so all 5 fit in one submission." `--total-shards 5` is a data-splitting choice, not a queue
ceiling, and conflating the two capped every model's parallelism at 5 workers regardless of how many
independent tasks there were to spread across them.

**Tried and rejected**
  - Raising `--total-shards` on the running job to add workers → not attempted, ruled out before
    testing: the checkpoint filename embeds the shard count (`_shard<N>of5.jsonl`), so changing it
    orphans every row already on disk and forces them to be re-paid for. The earlier Qwen shard-ladder
    incident already established that changing the count is destructive, so this was rejected on that
    precedent alone.

**Fix** — cancelled 8625688 (6m35s elapsed, 0 rows lost, nothing to reconcile) and replaced it with
three per-task arrays, `--total-shards 5` unchanged in each: `run_task_desire.sh`,
`run_task_belief.sh`, `run_task_intention.sh` (`NegotiationToM/NEG_Gemma/`), submitted as jobs 8625800
(desire), 8625801 (belief), 8625810 (intention) — 15 workers total. Output paths are
`results/<task>/<stem>_shard<N>of5.jsonl`, one directory per task, so the three arrays cannot collide
on disk, and because the shard count never changes, a fallback to a single sequential array would
have kept every row already written. Also corrected the stale claim it was based on:
`NEG_GPT/run_negotiation.sh:13-14`, inherited by all five other models' scripts, which asserted the
5-job ceiling without ever having tested a larger submission.

**How it was verified** — `sacct` shows all 15 array tasks (3 submissions x 5 shards each) started
within 7 seconds of one another (04:14:26-04:14:33) on 8 distinct compute nodes, none queued behind
another, and all 15 finished `COMPLETED` with exit `0:0` in 1:52:28-2:39:06. Row counts landed exact:
desire 4,760/4,760, belief 4,760/4,760, intention 4,618/4,618, 0 empty responses across all 14,138.
Together's hang rate (calls killed at the 120s ceiling) was 0.88% (125/14,223 attempts) against the
~8-10% single-stream reference for this config, so putting 15 concurrent streams on the provider did
not provoke extra throttling. Latency p99 held at 45.4-54.2s across all 15 shards (2.2x margin at the
worst shard), confirming the ceiling was not manufacturing the hangs it was measuring.

**Caveat on that hang-rate number, found on closer inspection of the printed p99**: the runner's
`[pulse]`/`[budget]` p99 is computed only over calls that *returned*, excluding the 125 killed at
120s. Recomputed over all 14,223 attempts, the 99th percentile rises to roughly the observed max
(67-119s), and for 2 of the 15 shards (desire shard 1, belief shard 3) the all-attempts p99 lands
exactly on the 120s ceiling itself — one call that did return measured 119.4s, 0.6s inside the limit.
120s is still defensible for this run, but sizing the *next* one from the all-attempts tail
(~150-180s) rather than the returned-only p99 would recover more of that 0.88%. Separately, the
canned diagnostic line all 15 shards printed ("calls are genuinely running to the ceiling — cap
`max_tokens` first") is misfiring and should not be acted on: truncation is 0/14,099 and output-token
p99 is 17-22 against a cap of 8,192, so at ~2.2 output tok/s the 24-120s tail is server-side
queuing/TTFT, not generation length, and capping `max_tokens` would change nothing about it.

**The reusable lesson, for the next model** — split by *task*, not by raising the shard count.
Splitting by task multiplies the worker count without touching `--total-shards`, so no existing
checkpoint is orphaned and a fallback to fewer workers costs nothing; raising the shard count changes
the checkpoint filename itself and re-pays for every row already on disk. Use this whenever a single
array works through desire/belief/intention in sequence per worker — which, per the correction above,
was every model in this project until now.

---

### Gemma full run finished clean but belief is not usable as published   2026-08-05  open

**Symptom** — Gemma's full run (jobs 8625800/8625801/8625810, previous entry) landed with exact row
counts and 0 empty responses, but the post-run audit marked it `usable=false`. Headline
`Belief_EM = 0.5007` (`NEG_Gemma/results/gemma-4-31B-it_negotiation_overall.csv`) ranks Gemma 5th of
6 models on that task.

**Root cause** — the run shipped with Together's `reasoning={"enabled": False}` (commit `9a5c734`,
see previous entry's addendum). The A/B behind that commit was run on desire only — 138 paired
items, 0.7101 (on) vs 0.6884 (off), McNemar p=0.678, not significant — and the result was then applied
to all three tasks, including belief, which the A/B never tested. Paired on 463 uids identical
between this run and the archived reasoning-ON pilot
(`NEG_Gemma/pilot_archive_reasoning_on_20260804/`), belief scores 0.6177 with reasoning on vs 0.4903
with it off: a 12.7-point drop, McNemar exact two-sided p=1.2e-08 (84 ON-only vs 25 OFF-only
discordant pairs). The reasoning-ON estimate would rank Gemma 2nd (~0.6177), effectively tied with
DeepSeek's 0.6197, instead of last — a three-rank swing driven by a config choice, not model
capability. This is exactly the case `negotiation.md` section 3 already warns about ("Cap reasoning;
do not disable it... switching reasoning off changes what is reported rather than making it cheaper
to report"), confirmed here with a paired significance test rather than argued from principle.

**Mechanism** — without reasoning the model floods the `Not Given` slot on belief: predicted
all-three-`Not-Given` 1,601 times against 843 in gold, giving `Not Given` slot precision 0.583 against
recall 0.916 (3,056 false positives). Desire is unaffected by the same setting (P 0.788 / R 0.842)
because desire extracts self-stated content, while belief requires inferring the opponent's state —
exactly the inference step reasoning was disabled to skip.

**Tried and rejected**
  - Generalising the desire-only A/B result to belief and intention → looked justified because
    desire is the one task where reasoning provably does not matter (p=0.678) → wrong, because that
    is also the one task where gold is self-stated rather than inferred; belief was never tested by
    the A/B and turned out to be the task most sensitive to disabling reasoning.

**Not yet fixed** — the run is the first complete Gemma dataset (0 empty responses, exact row counts
on all three tasks) and desire/intention are unaffected by this finding, but the belief column must
not be published or cross-model compared as it stands: DeepSeek and Grok both ran with reasoning ON,
so a reasoning-OFF belief score is not measuring the same thing they are. Needed: either a belief-only
rerun with reasoning enabled, or the whole run redone with reasoning capped (not disabled) per the
existing convention — a small fixed budget, the way Gemini's `thinking_budget=256` was sized, rather
than the on/off choice this run made.

**Also found in the same audit, unrelated to reasoning** — intention's headline `Intent_Micro_F1
0.5345` hides a total failure on the largest gold class: `No-Intention` has 1,457 gold instances,
predicted only 39 times, recall 0.015, F1 0.029, under *both* reasoning settings (recall 0.012 with
reasoning on too), so it is model-intrinsic rather than a reasoning or prompt artifact —
`intention_messages` already lists all nine labels, so it is not a missing-option gap either. Two
further minor findings from the same audit, recorded here rather than re-investigated later: (1) 40
of the 14,138 rows (desire shards 0 and 3, 20 each) were carried over from the cancelled job 8625688
rather than freshly called — verified benign, since 8625688 started ~30s after commit 9a5c734, its
logs already show the reasoning-off signature (ceiling 120s, tokens p99 ~15-17), and no
`raw_response` among all 14,138 rows exceeds 113 characters; (2) `max_tokens=8192` is roughly 450x
larger than the measured need (output token median 16, p99 18) — harmless here but worth lowering for
the next run, and the run's own `[budget]` block already suggests 25-33.

**How it was verified** — McNemar exact test on the 463-uid subset paired between this run and the
reasoning-ON archive; row counts confirmed against the merged jsonl (`wc -l` on
`results/{desire,belief,intention}/gemma-4-31B-it_all.jsonl`: 4,760/4,760/4,618, matching the expected
counts from "Intention rows paired with the wrong utterance" and "`\"None\"` counted as a wrong
answer" above); slot-level precision/recall for `Not Given` computed from gold vs. predicted counts on
the same merged files. Re-check once a belief-only reasoning-on rerun exists: compare its `Belief_EM`
against 0.6177 rather than against this run's 0.5007.

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
