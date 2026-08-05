export const meta = {
  name: 'scale-shards',
  description: 'Find the shard count a provider actually sustains, then run the model at it',
  whenToUse: 'A model is throughput-bound on the provider rather than on Quest, and the right amount of parallelism is unknown. Pass {model, ladder, reason} plus optional {callTimeout, gateMinutes, syncIntervalMin, maxHours, push}. It CLIMBS the ladder from the lowest shard count upward, measuring at each rung whether Quest actually started the tasks AND whether the provider degraded, and keeps the highest rung that stayed healthy. It climbs rather than descends because a dynamic rate limiter shaped by recent traffic would let one oversized burst depress the quota for every rung measured after it. Use run-model instead when the shard count is already settled.',
  phases: [
    { title: 'Prepare',   detail: 'executor: stop anything running for this model, set the ceiling, clear the checkpoint' },
    { title: 'Review',    detail: 'reviewer: is the change fit to run, and is the checkpoint really clean' },
    { title: 'Sync',      detail: 'executor: transfer core+runner together, prove md5, preflight' },
    { title: 'Ladder',    detail: 'per rung: submit at N shards, then gate on Quest concurrency AND provider hang rate' },
    { title: 'Supervise', detail: 'per cycle: pull results, commit and push, re-check health' },
    { title: 'Audit',     detail: 'evaluator: are the numbers usable' },
    { title: 'Record',    detail: 'tracker: outcome and the sustained shard count into ISSUES.md' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   model:           "NEG_Gemma"     required
//   reason:          "..."           required
//   ladder:          [5, 10, 20]     shard counts to try; sorted ascending and CLIMBED, whatever
//                                    order the caller writes them in. The highest healthy rung wins.
//   callTimeout:     60              per-model SIGALRM ceiling, seconds; omit to leave the default
//   gateMinutes:     15              how long to watch each rung before judging it
//   syncIntervalMin: 180
//   maxHours:        20
//   push:            true
// }
//
// `args` can arrive as a JSON *string* rather than an object — the caller writes an object and the
// tool hands the script the serialised form, so args.model reads as undefined and the guard below
// blames a missing argument that was in fact supplied. Three launches were lost to this on
// 2026-08-04. Normalise first.
// ---------------------------------------------------------------------------

let A = args || {}
if (typeof A === 'string') {
  try {
    A = JSON.parse(A)
  } catch (error) {
    return { aborted: `args arrived as a string that is not JSON: ${error.message}`, raw: String(args).slice(0, 300) }
  }
}

const MODEL = A.model || ''
const REASON = A.reason || 'not stated'
const LADDER = Array.isArray(A.ladder) && A.ladder.length ? A.ladder.slice() : [20, 10, 5]
const CALL_TIMEOUT = A.callTimeout || null
const GATE_MIN = A.gateMinutes || 15
const SYNC_MIN = A.syncIntervalMin || 180
const MAX_HOURS = A.maxHours || 20
const DO_PUSH = A.push !== false

if (!MODEL) {
  return { aborted: 'args.model is required, e.g. {model: "NEG_Gemma", reason: "..."}' }
}
if (LADDER.some(n => !Number.isInteger(n) || n < 1)) {
  return { aborted: `args.ladder must be positive integers, got ${JSON.stringify(LADDER)}` }
}

const REPO = '/Users/harrychen/SONIC/hai-teams'
const LOCAL = REPO + '/NegotiationToM'
const QDIR = '/gpfs/projects/p32983/NegotiationToM'
const MAX_CYCLES = Math.max(1, Math.ceil((MAX_HOURS * 60) / SYNC_MIN))

const RULES = `
Target: ${MODEL}, full run. Why: ${REASON}

HARD RULES — breaking one is a failed task, not a judgement call:
- Touch ${MODEL} and nothing else. Confirm in your report that every other job is untouched.
- Do NOT call a model provider API directly. preflight.py is the one sanctioned exception, and
  only in the phase that asks for it. A reviewer once wrote four probe scripts and spent real
  quota because its prompt did not forbid this.
- sbatch and scancel belong ONLY to the phase that says so. Never submit "to check something".
- Never read, copy or overwrite .env. Under /projects/p32983 only uwr0681's directories are in scope.
- Quest is \`ssh quest\`; ignore libcrypto warnings. Remote ${QDIR}, local ${LOCAL}, repo ${REPO}.
- Staging for git means explicit paths. NEVER \`git add -A\` — an unattended loop that did swept
  unreviewed work into commits and pushed them to both remotes.
- neg_eval_core.py is shared by six runners. If you transfer it, transfer the runner in the same
  step; a runner without its core dies at import.
`

// ---------- 1. Prepare ----------
phase('Prepare')
const prep = await agent(
  `Put ${MODEL} into a state where a fresh full run can start. This phase owns scancel.
${RULES}

1. squeue -u uwr0681. If any job belongs to ${MODEL} — including a pilot — scancel it and say
   which. Leave every other model's jobs running and confirm you did.
2. Pull anything it produced down first so evidence survives:
   \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\`
${CALL_TIMEOUT ? `3. In ${LOCAL}/${MODEL}/, set the per-model watchdog ceiling to ${CALL_TIMEOUT}s by calling
   set_call_timeout(${CALL_TIMEOUT}) (imported from neg_eval_core) at module scope in the runner.
   Justify the number in a comment from measured latency: it must sit ABOVE the slowest call that
   legitimately returned, or it silently truncates good work. State the slowest observed success
   you based it on.` : '3. Leave the watchdog ceiling as it is.'}
4. The checkpoint MUST be empty for a fresh run. Report row counts under
   ${MODEL}/results/{desire,belief,intention} on BOTH sides. If non-empty, archive to a timestamped
   directory on both sides and clear it — never merge rows from a different configuration.
   Note: zsh aborts a whole command line when a glob matches nothing, and GNU find on Quest refuses
   -prune together with -delete. Both have already broken this cleanup once.
5. Report the local↔Quest md5 status of neg_eval_core.py and ${MODEL}'s runner. Do not transfer yet.

Return ready=false if anything is left in a state a fresh run cannot start from.`,
  { agentType: 'executor', phase: 'Prepare', schema: {
    type: 'object', additionalProperties: false,
    required: ['ready', 'cancelled_jobs', 'checkpoint_rows', 'detail'],
    properties: {
      ready: { type: 'boolean' },
      cancelled_jobs: { type: 'array', items: { type: 'string' } },
      checkpoint_rows: { type: 'integer' },
      archived_to: { type: 'string' },
      other_jobs_untouched: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

if (!prep || !prep.ready) {
  return { outcome: 'BLOCKED in prepare', blockers: prep?.detail || 'no verdict', prep }
}

// ---------- 2. Review ----------
phase('Review')
const review = await agent(
  `Decide whether ${MODEL}'s code is fit to run before anything is transferred or submitted.
${RULES}

The run will be submitted at up to ${LADDER[0]} shards, which is higher than this project has ever
used. Check hardest the things that only break above 5:

1. shard_slice(rows, shard, total_shards) in neg_eval_core.py: for EVERY N in ${JSON.stringify(LADDER)},
   and for row counts 4760 (desire, belief) and 4618 (intention), prove by execution that the union
   of all shards is exactly the input, with no row dropped, duplicated, or landing in two shards.
   Report the per-shard sizes. A silent gap here is unrecoverable — it looks like a complete run.
2. Output paths must carry the shard tag for every N, so no two shards overwrite each other's
   .jsonl or _overall.csv.
3. py_compile the core and ${MODEL}'s runner; bash -n the sbatch. Confirm every name the runner
   imports from neg_eval_core exists.
4. Backward compatibility: the other five runners call record_call(*usage_from(response)) with
   three positional args. Confirm that still works and that any per-model setting cannot leak to
   them — each shard is its own process, but say so from the code, not from assumption.
5. git status --porcelain — say what is uncommitted; do not commit it.

Return ok=false if anything would waste job slots or produce wrong data. Fail cheaply here.`,
  { agentType: 'reviewer', phase: 'Review', schema: {
    type: 'object', additionalProperties: false,
    required: ['ok', 'blockers', 'sharding_verified', 'checks', 'detail'],
    properties: {
      ok: { type: 'boolean' },
      blockers: { type: 'array', items: { type: 'string' } },
      sharding_verified: { type: 'boolean' },
      checks: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } }
)

if (!review || !review.ok || !review.sharding_verified) {
  return {
    outcome: 'BLOCKED in review',
    blockers: review?.blockers || ['no verdict'],
    note: review && !review.sharding_verified
      ? 'sharding was not proven lossless at the requested shard counts'
      : undefined,
    prep, review,
  }
}

// ---------- 3. Sync ----------
phase('Sync')
const synced = await agent(
  `Make Quest match local for every code file, then prove it. Submit nothing.
${RULES}

1. \`cd ${REPO} && python3 .claude/scripts/check_quest_sync.py\`.
2. Transfer each file it names with \`ssh quest "cat > ${QDIR}/<path>" < ${LOCAL}/<path>\`, core and
   runner together, and verify each with md5sum on both sides.
3. Re-run the checker, show it exiting 0, and print the file count for BOTH sides — a comparison
   over two empty lists passes trivially, and that exact false pass has happened here.
4. On Quest: py_compile the transferred files and confirm ${MODEL}'s imports resolve.
5. \`python3 preflight.py --only <provider>\` on Quest for this model only. One real API call.

Return in_sync=false if any step fails.`,
  { agentType: 'executor', phase: 'Sync', schema: {
    type: 'object', additionalProperties: false,
    required: ['in_sync', 'files_compared', 'preflight_passed', 'detail'],
    properties: {
      in_sync: { type: 'boolean' },
      transferred: { type: 'array', items: { type: 'string' } },
      files_compared: { type: 'integer' },
      preflight_passed: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

if (!synced || !synced.in_sync || !synced.preflight_passed) {
  return {
    outcome: 'BLOCKED before launch',
    reason: !synced?.in_sync ? 'code did not reach Quest' : 'preflight failed',
    prep, review, synced,
  }
}

// ---------- 4. The ladder ----------
// CLIMB, do not descend. The ladder is walked from the LOWEST rung upward, keeping the last rung
// that stayed healthy.
//
// The direction is the whole design. Together states its rate limits are dynamic and "shift with
// live model capacity and your traffic shape". Against a limiter like that, opening at the highest
// rung is self-defeating: if 20 shards depress the allowance, every lower rung measured afterwards
// inherits a degraded quota, the gate sees them fail too, and the run concludes "concurrency does
// not help" when what actually happened is that the first burst poisoned the experiment. Climbing
// cannot do this — each rung is measured against a quota that only the gentler rungs before it
// have touched.
//
// So a failing rung ends the climb rather than continuing it, and the previous rung wins. That
// also means LADDER must be sorted ascending regardless of how the caller wrote it.
phase('Ladder')
let running = null
let best = null
const attempts = []
const RUNGS = LADDER.slice().sort((a, b) => a - b)

for (const N of RUNGS) {
  const launched = await agent(
    `Submit ${MODEL} as a ${N}-shard array. This phase owns sbatch.
${RULES}

1. Edit ${LOCAL}/${MODEL}/run_negotiation.sh so it is --array=0-${N - 1} and --total-shards ${N},
   leaving every other SBATCH field alone. Transfer it to Quest and md5-verify.
2. Clear any stale halt marker in ${MODEL}/ so an old verdict is not read as a new one.
3. Report the row counts under ${MODEL}/results/{desire,belief,intention}, split by the shard
   count embedded in each filename. **This rung starts from zero unless a previous rung used this
   exact shard count.** The checkpoint filename carries it (_shard0of5.jsonl vs _shard0of10.jsonl)
   and each task loads only its own shard's file, so rows written at a different --total-shards are
   invisible here: every item is re-called and re-paid, and merge_neg_results reads only
   _shard{i}of{N}. Say plainly how many rows this rung will start from and how many earlier rows
   are now orphaned — the cost of measuring must be visible, not assumed away.
4. sbatch from ${MODEL}'s own directory. Submit NOTHING else.
5. Immediately report squeue: how many array tasks are RUNNING, how many PENDING, and the reason
   code for anything pending.`,
    { agentType: 'executor', phase: 'Ladder', label: `submit:${N}`, schema: {
      type: 'object', additionalProperties: false,
      required: ['job_id', 'requested', 'running_now', 'pending_now', 'detail'],
      properties: {
        job_id: { type: 'string' },
        requested: { type: 'integer' },
        running_now: { type: 'integer' },
        pending_now: { type: 'integer' },
        pending_reason: { type: 'string' },
        resumed_rows: { type: 'integer' },
        detail: { type: 'string' },
      },
    } }
  )

  if (!launched || !launched.job_id) {
    attempts.push({ shards: N, outcome: 'submit failed' })
    continue
  }
  log(`${MODEL} submitted at ${N} shards as ${launched.job_id}; gating ${GATE_MIN} min`)

  const gate = await agent(
    `Watch ${MODEL} job ${launched.job_id}, submitted as ${N} shards, for about ${GATE_MIN} minutes.
Observe only — you do not own scancel.
${RULES}

Report these separately, because they call for opposite responses:

A. **Did Quest actually run them?** How many array tasks reached RUNNING, sustained, not just at
   submit. If some stayed PENDING, give the reason code. This answers whether the cluster caps our
   parallelism — a belief written in NEG_GPT/run_negotiation.sh ("Quest allows at most 5 parallel
   array jobs") that the measured limits contradict, so settle it with evidence.

B. **Did the provider degrade?** The runner prints a throttled [pulse] line carrying calls, hang
   rate, trunc, empty, err, mean latency, effective s/row and rows/min. Collect it from EVERY shard
   log and report the AGGREGATE rows/min across shards and the MEDIAN per-shard hang rate. Judge on
   these, not on SLURM state — a process hung inside an API call looks identical to a working one.

C. Aggregate rows written, empty rows, and any BILLING_HALT / QUOTA_HALT / FAILURE_HALT quoted in full.

Verdict rules — and note carefully WHAT the hang rate is compared against:

**Compare each rung to the rung below it, never to a historical figure.** Any hang rate quoted
from an earlier run was measured under a different watchdog ceiling, and the ceiling alone moves
that number: lowering it mechanically reclassifies slow-but-legitimate calls as hangs. Judging this
run against such a number would attribute a ceiling change to the provider — the exact confound
this phase exists to avoid. The LOWEST rung establishes the baseline for this configuration; every
rung above it is judged against the rung immediately below.

- healthy=true when rows are accumulating, no halt marker, empty rate <= 2%, and — for any rung
  above the lowest — aggregate rows/min improved over the rung below while the median per-shard
  hang rate did not roughly double against it.
- healthy=false when a halt marker appears, or the empty rate exceeds 2%, or the hang rate roughly
  doubled versus the rung below, or aggregate rows/min FAILED TO IMPROVE. That last one matters on
  its own: more shards that buy no throughput are pure added load on a rate limiter, so the climb
  should end even when nothing looks broken.
- The lowest rung has nothing to compare against, so judge it on absolutes only: rows accumulating,
  no halt marker, empty rate <= 2%. Report its hang rate as the baseline for the rungs above.
- Also report the latency p90/p99/max the [pulse] line now prints, and say whether the ceiling of
  ${CALL_TIMEOUT || 'the shared default'}s sits comfortably above p99. If it does not, hangs are
  being manufactured by the ceiling and every rate in this report is suspect — say so plainly.
- Fewer tasks RUNNING than requested is NOT by itself a failure. Report it as quest_capped and
  judge health on B and C. If Quest granted fewer but the run is healthy, that granted number is
  the useful answer.`,
    { agentType: 'watcher', phase: 'Ladder', label: `gate:${N}`, schema: {
      type: 'object', additionalProperties: false,
      required: ['healthy', 'shards_running', 'aggregate_rows_per_min', 'median_hang_rate', 'rows', 'empty_count', 'detail'],
      properties: {
        healthy: { type: 'boolean' },
        quest_capped: { type: 'boolean' },
        shards_running: { type: 'integer' },
        aggregate_rows_per_min: { type: 'number' },
        median_hang_rate: { type: 'number' },
        rows: { type: 'integer' },
        empty_count: { type: 'integer' },
        halt_markers: { type: 'array', items: { type: 'string' } },
        failure_reason: { type: 'string' },
        detail: { type: 'string' },
      },
    } }
  )

  attempts.push({
    shards: N, job: launched.job_id, requested: launched.requested,
    running: gate?.shards_running, quest_capped: gate?.quest_capped,
    rows_per_min: gate?.aggregate_rows_per_min, hang_rate: gate?.median_hang_rate,
    healthy: gate?.healthy,
  })

  const isTopRung = N === RUNGS[RUNGS.length - 1]

  if (gate && gate.healthy) {
    best = { N, job: launched.job_id, gate }
    log(`rung ${N} holds: ${gate.shards_running} running, ${gate.aggregate_rows_per_min} rows/min, hang ${gate.median_hang_rate}`)
    if (isTopRung) {
      running = best                      // nothing higher to try; keep it running
      break
    }
    // Healthy but not the top rung: stop it before testing a higher one. Two arrays must not run
    // at once — their filenames differ (_shard0of5 vs _shard0of10) so they would not collide on
    // disk, which is exactly why this is worth stating: they would silently duplicate work and
    // both count against the same rate limit, corrupting the very measurement being taken.
    await agent(
      `${MODEL} job ${launched.job_id} passed its gate at ${N} shards. Stop it so a higher rung can
be measured against an uncontended quota. This phase owns scancel.
${RULES}

scancel ${launched.job_id} and that job only; confirm every other job still runs. Pull results down
with \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\` so its rows survive. Those rows stay
valid for any later rung: the configuration is identical and only --total-shards differs, and uids
are global, so a resume skips them correctly whatever the next shard count is. Report the row count
you are handing forward.`,
      { agentType: 'executor', phase: 'Ladder', label: `pause:${N}`, schema: {
        type: 'object', additionalProperties: false,
        required: ['cancelled', 'rows_kept', 'detail'],
        properties: {
          cancelled: { type: 'boolean' },
          rows_kept: { type: 'integer' },
          detail: { type: 'string' },
        },
      } }
    )
    continue                              // climb
  }

  // Unhealthy: the climb ends here. Everything above this rung is worse, so do not test it.
  log(`rung ${N} failed (${gate?.failure_reason || 'no verdict'}); climb ends, falling back`)
  await agent(
    `${MODEL} job ${launched.job_id} failed its gate at ${N} shards: ${gate?.failure_reason || 'no verdict'}.
Stop it. This phase owns scancel.
${RULES}

scancel ${launched.job_id} and that job only; confirm every other job still runs. Pull results down
with \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\` so the evidence survives.

Report how many rows it wrote and state that they are ORPHANED: the shard count is part of the
checkpoint filename, so nothing written at ${N} shards can be read by a rung at a different count,
and merge_neg_results reads only _shard{i}of{N}. That is the price of measuring this rung, and it
should appear in the report rather than being discovered later. Note separately any rows with an
empty raw_response — those matter only if the winning rung turns out to be ${N} as well, in which
case they must be pruned first, because a plain resume treats them as done forever.`,
    { agentType: 'executor', phase: 'Ladder', label: `stop:${N}`, schema: {
      type: 'object', additionalProperties: false,
      required: ['cancelled', 'rows_reusable', 'detail'],
      properties: {
        cancelled: { type: 'boolean' },
        rows_reusable: { type: 'boolean' },
        needs_prune: { type: 'boolean' },
        detail: { type: 'string' },
      },
    } }
  )
  break
}

// The climb stopped below the top. Put the highest healthy rung back on the cluster to finish.
if (!running && best) {
  log(`falling back to the last healthy rung: ${best.N} shards`)
  const resub = await agent(
    `Resubmit ${MODEL} at ${best.N} shards — the highest rung that stayed healthy. Higher rungs were
measured and rejected, so do not try them again. This phase owns sbatch.
${RULES}

1. Set run_negotiation.sh to --array=0-${best.N - 1} and --total-shards ${best.N}; transfer and
   md5-verify.
2. Clear any stale halt marker.
3. Report how many rows already exist under _shard*of\${best.N}.jsonl specifically. This is NOT a
   resume onto the other rungs' output: those files carry a different shard count in their names
   and are invisible to this run. Expect to start from zero unless an earlier rung happened to use
   this same shard count, in which case prune any row with an empty raw_response first, because a
   plain resume treats it as done and it would stay empty forever.
4. sbatch from ${MODEL}'s directory, nothing else, then report squeue.`,
    { agentType: 'executor', phase: 'Ladder', label: `resubmit:${best.N}`, schema: {
      type: 'object', additionalProperties: false,
      required: ['job_id', 'running_now', 'resumed_rows', 'detail'],
      properties: {
        job_id: { type: 'string' },
        running_now: { type: 'integer' },
        resumed_rows: { type: 'integer' },
        pruned_rows: { type: 'integer' },
        detail: { type: 'string' },
      },
    } }
  )
  if (resub && resub.job_id) {
    running = { N: best.N, job: resub.job_id, gate: best.gate }
  }
}

if (!running) {
  return {
    outcome: 'NO RUNG HELD',
    model: MODEL,
    attempts,
    needs_rerun: true,
    next: 'Every shard count on the ladder failed its gate. Read the per-rung hang rates before choosing a new ladder — if the rate was already bad at the lowest rung, the problem is not concurrency.',
    prep, review, synced,
  }
}

// ---------- 5. Supervise ----------
phase('Supervise')
let finished = false
let lastCycle = null
const cycles = []

for (let i = 1; i <= MAX_CYCLES && !finished; i++) {
  const cycle = await agent(
    `Supervision cycle ${i} of at most ${MAX_CYCLES} for ${MODEL}, job ${running.job} at ${running.N} shards.
Wait about ${SYNC_MIN} minutes from when you start, checking every few minutes, then sync.
${RULES}

A. **Quest -> local.** \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\`; report the row delta.
   Results flow down only; never push results upward.
${DO_PUSH ? `B. **local -> git.** Stage ONLY this model's results and logs, by explicit path:
     git add NegotiationToM/${MODEL}/results NegotiationToM/${MODEL}/log_*.txt
   Never \`git add -A\`. If nothing changed, skip the commit rather than making an empty one.
   Commit naming the model, job id and row counts, then push to \`backup\` and \`origin\`.
   Never push to \`upstream\`; it is the shared repo.` : 'B. Skipping git (push=false). Say so explicitly.'}
C. **Health.** Rows per task, aggregate rows/min, the [pulse] hang rate per shard, empty and error
   counts, any halt marker quoted in full, and how many array tasks are still in the queue.

Set finished=true when the job has left the queue. Set stop_now=true if it is still running but
should be killed — a halt marker, empty rate above 2%, or no progress at all across this cycle.`,
    { agentType: 'executor', phase: 'Supervise', schema: {
      type: 'object', additionalProperties: false,
      required: ['finished', 'stop_now', 'rows_total', 'empty_count', 'pulled', 'detail'],
      properties: {
        finished: { type: 'boolean' },
        stop_now: { type: 'boolean' },
        rows_total: { type: 'integer' },
        rows_delta: { type: 'integer' },
        rows_per_min: { type: 'number' },
        hang_rate: { type: 'number' },
        empty_count: { type: 'integer' },
        pulled: { type: 'boolean' },
        pushed: { type: 'boolean' },
        commit: { type: 'string' },
        halt_markers: { type: 'array', items: { type: 'string' } },
        detail: { type: 'string' },
      },
    } }
  )

  if (!cycle) {
    log(`cycle ${i} returned nothing — stopping supervision`)
    break
  }
  cycles.push({ cycle: i, rows: cycle.rows_total, delta: cycle.rows_delta,
                rate: cycle.rows_per_min, hang: cycle.hang_rate,
                empty: cycle.empty_count, pushed: cycle.pushed })
  lastCycle = cycle
  log(`cycle ${i}: ${cycle.rows_total} rows (+${cycle.rows_delta ?? '?'}), ${cycle.rows_per_min ?? '?'} rows/min, empty=${cycle.empty_count}`)

  if (cycle.stop_now) {
    const killed = await agent(
      `${MODEL} job ${running.job} must be stopped mid-run: ${JSON.stringify(cycle.halt_markers || [])} ${String(cycle.detail || '').slice(0, 300)}
${RULES}

scancel that job only, confirm the others still run, pull results down one final time, and
diagnose. Say whether the checkpoint needs pruning before any resubmit. Do not resubmit.`,
      { agentType: 'executor', phase: 'Supervise', schema: {
        type: 'object', additionalProperties: false,
        required: ['cancelled', 'diagnosis', 'needs_prune', 'detail'],
        properties: {
          cancelled: { type: 'boolean' },
          diagnosis: { type: 'string' },
          needs_prune: { type: 'boolean' },
          proposed_fix: { type: 'string' },
          detail: { type: 'string' },
        },
      } }
    )
    return {
      outcome: 'STOPPED MID-RUN', model: MODEL, shards: running.N, job: running.job,
      diagnosis: killed?.diagnosis, needs_prune: killed?.needs_prune,
      proposed_fix: killed?.proposed_fix, needs_rerun: true,
      attempts, cycles, prep, review, synced,
    }
  }
  finished = !!cycle.finished
}

if (!finished) {
  return {
    outcome: 'STILL RUNNING — supervision window ended',
    model: MODEL, shards: running.N, job: running.job,
    note: `Supervised ${MAX_HOURS}h without finishing. The job is unaffected by this workflow ending.`,
    attempts, cycles, last: lastCycle,
  }
}

// ---------- 6. Audit ----------
phase('Audit')
const audit = await agent(
  `${MODEL} finished as job ${running.job} at ${running.N} shards. Decide whether the numbers can be
used. Recommend only — change nothing, submit nothing.
${RULES}

1. Row counts per task against the expected 4760 / 4760 / 4618. **4760 intention rows means the
   odd-length-dialogue bug is back** — that task has 4618 rows, not 4760.
2. Duplicate uids, and rows with an empty raw_response. Count them from the jsonl, not from CSV:
   embedded newlines in reasoning text break naive CSV parsing and once produced more unique uids
   than rows, which nearly caused a needless re-run.
3. The [budget] block each shard printed at the end: output-token percentiles, truncation count,
   latency, hang rate. If truncation is non-zero the token tail is censored and the suggested
   budget is only a lower bound. If the runner never passed finish_reason it reports UNKNOWN, which
   is not the same as zero.
4. The scores, against this model's own archived pilot where one exists, and against the other
   models. Say plainly whether any configuration difference makes the comparison unfair and must be
   reported as a caveat rather than buried.
5. Cost and tokens for this run.

Return usable=false if the numbers should not be published.`,
  { agentType: 'evaluator', phase: 'Audit', schema: {
    type: 'object', additionalProperties: false,
    required: ['usable', 'row_counts', 'empty_rate', 'problems', 'detail'],
    properties: {
      usable: { type: 'boolean' },
      row_counts: { type: 'object', additionalProperties: true },
      empty_rate: { type: 'number' },
      metrics: { type: 'object', additionalProperties: true },
      comparability_caveats: { type: 'array', items: { type: 'string' } },
      problems: { type: 'array', items: { type: 'string' } },
      needs_prune: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

// ---------- 7. Record ----------
phase('Record')
const recorded = await agent(
  `Record this run in ${LOCAL}/ISSUES.md, in the file's existing style.
${RULES}

The finding worth keeping is the sustained shard count and how it was established:
  ladder tried      ${JSON.stringify(attempts)}
  rung that held    ${running.N} shards, ${running.gate?.shards_running} actually running
  provider health   ${running.gate?.median_hang_rate} median hang rate, ${running.gate?.aggregate_rows_per_min} rows/min aggregate
  audit             usable=${audit?.usable}, problems=${JSON.stringify(audit?.problems || [])}

Also correct the stale claim in NEG_GPT/run_negotiation.sh that "Quest allows at most 5 parallel
array jobs" if this run contradicts it — cite the measured limits and what actually ran, and note
that all six sbatch scripts inherited --array=0-4 from that belief. A wrong fact in a comment
propagates; leaving it costs the next session the same investigation.

Record what was rejected and why, not only what shipped.`,
  { agentType: 'tracker', phase: 'Record', schema: {
    type: 'object', additionalProperties: false,
    required: ['written', 'detail'],
    properties: {
      written: { type: 'boolean' },
      corrected_stale_claim: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

return {
  outcome: audit?.usable ? 'COMPLETE' : 'COMPLETE BUT NOT USABLE',
  model: MODEL,
  sustained_shards: running.N,
  shards_actually_running: running.gate?.shards_running,
  quest_capped: !!running.gate?.quest_capped,
  ladder_attempts: attempts,
  job: running.job,
  usable: audit?.usable,
  metrics: audit?.metrics,
  row_counts: audit?.row_counts,
  empty_rate: audit?.empty_rate,
  problems: audit?.problems,
  comparability_caveats: audit?.comparability_caveats,
  needs_prune: audit?.needs_prune,
  issues_updated: recorded?.written,
  cycles,
  next: audit?.usable
    ? 'Results are in git. Fold the sustained shard count into the other models\' sbatch scripts if it generalises.'
    : 'Read problems before republishing anything from this run.',
}
