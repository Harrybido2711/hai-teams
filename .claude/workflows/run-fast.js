export const meta = {
  name: 'run-fast',
  description: 'Run one model as three parallel per-task jobs, supervise it, and repair problems without stopping',
  whenToUse: 'A model needs to finish as fast as safely possible and the run must survive problems unattended. Pass {model, reason} plus optional {gateMinutes, syncIntervalMin, maxHours, push, tasks}. It replaces the usual single --task all array with one array per task, tripling concurrency WITHOUT changing --total-shards, so every row already written stays valid and a fallback costs nothing. Unlike run-model, the supervise phase may repair: prune rows written empty, resubmit a shard that died, and re-merge. Use run-model when a single sequential array is wanted, or scale-shards when the shard count itself is the question.',
  phases: [
    { title: 'Launch',    detail: 'executor: replace the sequential array with one array per task' },
    { title: 'Gate',      detail: 'watcher: are all workers alive and is the provider coping' },
    { title: 'Supervise', detail: 'per cycle: pull, push, judge health, and repair what can be repaired' },
    { title: 'Merge',     detail: 'executor: merge shards once every task is complete' },
    { title: 'Audit',     detail: 'evaluator: are the numbers usable' },
    { title: 'Record',    detail: 'tracker: outcome into ISSUES.md' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   model:           "NEG_Gemma"   required
//   reason:          "..."         required
//   tasks:           ["desire","belief","intention"]
//   gateMinutes:     20
//   syncIntervalMin: 120
//   maxHours:        30
//   push:            true
// }
//
// `args` may arrive as a JSON string rather than an object; normalise before reading a field, or
// the guard below blames a missing argument the caller did supply.
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
const TASKS = Array.isArray(A.tasks) && A.tasks.length ? A.tasks.slice() : ['desire', 'belief', 'intention']
const GATE_MIN = A.gateMinutes || 20
const SYNC_MIN = A.syncIntervalMin || 120
const MAX_HOURS = A.maxHours || 30
const DO_PUSH = A.push !== false

if (!MODEL) {
  return { aborted: 'args.model is required, e.g. {model: "NEG_Gemma", reason: "..."}' }
}

const REPO = '/Users/harrychen/SONIC/hai-teams'
const LOCAL = REPO + '/Interpersonal_processes_benchmarks/NegotiationToM'
const QDIR = '/gpfs/projects/p32983/NegotiationToM'
const MAX_CYCLES = Math.max(1, Math.ceil((MAX_HOURS * 60) / SYNC_MIN))
const EXPECTED = { desire: 4760, belief: 4760, intention: 4618 }

const RULES = `
Target: ${MODEL}, full run, split into one array per task. Why: ${REASON}

HARD RULES — breaking one is a failed task, not a judgement call:
- Touch ${MODEL} and nothing else. Confirm in your report that every other job is untouched.
- Do NOT call a model provider API directly. preflight.py is the one sanctioned exception, and
  only where a phase asks for it.
- Never read, copy or overwrite .env. Under /projects/p32983 only uwr0681's directories are in scope.
- Quest is \`ssh quest\`; ignore libcrypto warnings. Remote ${QDIR}, local ${LOCAL}, repo ${REPO}.
- Staging for git means explicit paths. NEVER \`git add -A\`.
- **--total-shards stays 5 for every task.** That is what makes this safe: the checkpoint filename
  embeds the shard count (_shard0of5.jsonl), so rows already written stay readable and a fallback
  loses nothing. Changing it orphans every existing row and forces them to be re-paid for.
- zsh aborts a whole command line when a glob matches nothing, and GNU find on Quest refuses
  -prune together with -delete. Both have already broken a cleanup here.
`

// ---------- 1. Launch ----------
phase('Launch')
const launched = await agent(
  `Replace ${MODEL}'s single sequential array with one array per task. This phase owns sbatch and scancel.
${RULES}

Why this is faster and why it is safe: the current job runs \`--task all\`, so each of 5 workers
does desire, then belief, then intention in sequence — 2,828 rows each. Running one array per task
puts ${TASKS.length * 5} workers on it at once. Output paths are results/<task>/<stem>_shard<N>of5.jsonl,
so tasks write to different directories and cannot collide, and because --total-shards is unchanged
every row already on disk stays valid.

1. squeue. If a ${MODEL} job is running \`--task all\`, scancel it — its rows are kept, not
   discarded, because the filenames are the ones the new jobs will resume into. Report how many
   rows it had written. Leave every other model's jobs alone and confirm you did.
2. Pull first so nothing is lost: \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\`
3. For each of ${JSON.stringify(TASKS)} create ${MODEL}/run_task_<task>.sh from run_negotiation.sh,
   changing ONLY: --job-name, --output/--error to log_<task>_shard%a, and the python line to
   \`--task <task>\`. Keep --array=0-4, --total-shards 5, the account, the partition and the
   walltime exactly as they are.
4. Before submitting, count rows already present per task and report them — those are resumed, not
   redone. Also check for rows with an empty raw_response; if any exist, run prune_failed_rows.py
   first, because a plain resume treats them as done and they stay empty forever.
5. Transfer the three scripts, md5-verify, then sbatch all three from ${MODEL}'s directory.
6. Report squeue: how many array tasks are RUNNING and how many PENDING, per job.

Return launched=false if any submission fails.`,
  { agentType: 'executor', phase: 'Launch', schema: {
    type: 'object', additionalProperties: false,
    required: ['launched', 'job_ids', 'running_now', 'resumed_rows', 'detail'],
    properties: {
      launched: { type: 'boolean' },
      job_ids: { type: 'object', additionalProperties: true },
      cancelled: { type: 'string' },
      running_now: { type: 'integer' },
      pending_now: { type: 'integer' },
      resumed_rows: { type: 'object', additionalProperties: true },
      pruned_rows: { type: 'integer' },
      detail: { type: 'string' },
    },
  } }
)

if (!launched || !launched.launched) {
  return { outcome: 'LAUNCH FAILED', detail: launched?.detail, launched }
}
log(`${MODEL}: ${JSON.stringify(launched.job_ids)}, ${launched.running_now} workers running`)

// ---------- 2. Gate ----------
phase('Gate')
const gate = await agent(
  `Watch ${MODEL}'s ${TASKS.length} jobs ${JSON.stringify(launched.job_ids)} for about ${GATE_MIN} minutes.
Observe only — you do not own scancel here.
${RULES}

The question is whether tripling concurrency broke anything. Report:

A. Workers RUNNING vs requested, per job, sustained rather than at submit. Any PENDING reason code.
B. From the [pulse] line in every shard log: hang rate, trunc, empty, err, mean latency,
   latency p90/p99/max, and rows/min. Note that the FIRST pulse of each shard reports n=1 and is
   meaningless — wait for a pulse with a double-digit call count before quoting a rate.
   The single-stream reference under this same configuration is roughly 8-10% hangs. A materially
   higher rate here is the provider objecting to ${TASKS.length * 5} concurrent streams.
C. Rows landing per task, and any BILLING_HALT / QUOTA_HALT / FAILURE_HALT quoted in full.
D. Whether latency p99 sits comfortably under the 120s ceiling. If p99 approaches it, hangs are
   being manufactured by the ceiling and every rate in the report is suspect — say so plainly.

healthy=false for: a halt marker, an empty rate above 2%, a hang rate roughly double the
single-stream reference, or no rows at all after the whole window with logs that are not advancing.
Fewer workers RUNNING than requested is NOT a failure on its own — report it and judge on B and C.`,
  { agentType: 'watcher', phase: 'Gate', schema: {
    type: 'object', additionalProperties: false,
    required: ['healthy', 'workers_running', 'hang_rate', 'rows', 'empty_count', 'detail'],
    properties: {
      healthy: { type: 'boolean' },
      workers_running: { type: 'integer' },
      hang_rate: { type: 'number' },
      rows_per_min_aggregate: { type: 'number' },
      latency_p99: { type: 'number' },
      rows: { type: 'integer' },
      empty_count: { type: 'integer' },
      halt_markers: { type: 'array', items: { type: 'string' } },
      failure_reason: { type: 'string' },
      detail: { type: 'string' },
    },
  } }
)

if (!gate || !gate.healthy) {
  // Falling back is cheap here precisely because --total-shards never changed: every row the
  // parallel attempt wrote is still readable by a sequential run.
  const fallback = await agent(
    `${MODEL}'s per-task split failed its gate: ${gate?.failure_reason || 'no verdict'}.
Fall back to a single sequential array. This phase owns scancel and sbatch.
${RULES}

1. scancel the three per-task jobs ${JSON.stringify(launched.job_ids)}, those only.
2. Pull results down. Report the row count per task — ALL of it is kept: --total-shards was never
   changed, so run_negotiation.sh (--task all) resumes straight into these same files.
3. Prune any row with an empty raw_response before resubmitting.
4. sbatch run_negotiation.sh, then report squeue.

Say plainly what the evidence was that ${TASKS.length * 5} concurrent workers were too many, so
the next attempt does not repeat it.`,
    { agentType: 'executor', phase: 'Gate', schema: {
      type: 'object', additionalProperties: false,
      required: ['job_id', 'rows_kept', 'evidence', 'detail'],
      properties: {
        job_id: { type: 'string' },
        rows_kept: { type: 'integer' },
        pruned_rows: { type: 'integer' },
        evidence: { type: 'string' },
        detail: { type: 'string' },
      },
    } }
  )
  return {
    outcome: 'FELL BACK TO SEQUENTIAL',
    model: MODEL,
    concurrency_evidence: fallback?.evidence,
    job: fallback?.job_id,
    rows_kept: fallback?.rows_kept,
    note: 'The parallel attempt cost nothing in data: --total-shards never changed, so its rows carried straight into the sequential run.',
    launched, gate, fallback,
  }
}

// ---------- 3. Supervise, and repair ----------
phase('Supervise')
let finished = false
const cycles = []
const repairs = []

for (let i = 1; i <= MAX_CYCLES && !finished; i++) {
  const cycle = await agent(
    `Supervision cycle ${i} of at most ${MAX_CYCLES} for ${MODEL}, jobs ${JSON.stringify(launched.job_ids)}.
Wait about ${SYNC_MIN} minutes from when you start, checking every few minutes, then sync and judge.
${RULES}

A. **Quest -> local.** \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\`; report the row delta.
${DO_PUSH ? `B. **local -> git.** Stage ONLY this model's results and logs, by explicit path:
     git add Interpersonal_processes_benchmarks/NegotiationToM/${MODEL}/results Interpersonal_processes_benchmarks/NegotiationToM/${MODEL}/log_*.txt
   Never \`git add -A\`. Skip the commit if nothing changed. Commit naming the model, the job ids
   and the row counts, then push to \`backup\` and \`origin\`. Never push to \`upstream\`.` : 'B. Skipping git (push=false). Say so explicitly.'}
C. **Health**, per task: rows against the expected ${JSON.stringify(EXPECTED)}, rows/min, the
   [pulse] hang rate, empty and error counts, and any halt marker quoted in full.

**This phase may repair. Three things you SHOULD fix without asking, and nothing else:**

 1. **A dead shard.** If an array task has left the queue while its task is short of its expected
    row count and no halt marker explains it, resubmit that ONE array index
    (\`sbatch --array=<N> run_task_<task>.sh\`). A resume skips what it already wrote. Say which
    index and why. Do not resubmit a task that finished legitimately.
 2. **Rows written empty.** If a task carries rows with an empty raw_response, run
    prune_failed_rows.py for that task and resubmit the affected shards, because a plain resume
    treats those uids as done and they would stay empty for the rest of the run.
 3. **A completed task.** When a task reaches its expected count with no empty rows, say so; the
    Merge phase will handle it. Do not merge here.

**Do NOT** change --total-shards, the model, the prompts, max_tokens, the ceiling, or the reasoning
setting. Those are decisions for the planner, and changing one mid-run would mix two configurations
into a single result set. If you believe one of them is wrong, set stop_now and explain.

Set finished=true when every task has reached its expected row count OR all three jobs have left
the queue. Set stop_now=true if the run should be killed — a halt marker, an empty rate above 2%
that pruning cannot fix, or no progress at all across this whole cycle.`,
    { agentType: 'executor', phase: 'Supervise', schema: {
      type: 'object', additionalProperties: false,
      required: ['finished', 'stop_now', 'rows_by_task', 'empty_count', 'pulled', 'detail'],
      properties: {
        finished: { type: 'boolean' },
        stop_now: { type: 'boolean' },
        rows_by_task: { type: 'object', additionalProperties: true },
        rows_delta: { type: 'integer' },
        rows_per_min: { type: 'number' },
        hang_rate: { type: 'number' },
        empty_count: { type: 'integer' },
        repairs_made: { type: 'array', items: { type: 'string' } },
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
  cycles.push({ cycle: i, rows: cycle.rows_by_task, delta: cycle.rows_delta,
                rate: cycle.rows_per_min, hang: cycle.hang_rate,
                empty: cycle.empty_count, pushed: cycle.pushed })
  if (cycle.repairs_made && cycle.repairs_made.length) {
    repairs.push(...cycle.repairs_made)
    log(`cycle ${i} repaired: ${cycle.repairs_made.join('; ')}`)
  }
  log(`cycle ${i}: ${JSON.stringify(cycle.rows_by_task)} (+${cycle.rows_delta ?? '?'}), empty=${cycle.empty_count}`)

  if (cycle.stop_now) {
    const killed = await agent(
      `${MODEL} must be stopped: ${JSON.stringify(cycle.halt_markers || [])} ${String(cycle.detail || '').slice(0, 300)}
${RULES}

scancel ${JSON.stringify(launched.job_ids)}, those only; confirm the others still run; pull results
down one final time; diagnose. Say whether the checkpoint needs pruning before any resubmit.
Do not resubmit.`,
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
      outcome: 'STOPPED MID-RUN', model: MODEL, jobs: launched.job_ids,
      diagnosis: killed?.diagnosis, needs_prune: killed?.needs_prune,
      proposed_fix: killed?.proposed_fix, needs_rerun: true,
      cycles, repairs, launched, gate,
    }
  }
  finished = !!cycle.finished
}

if (!finished) {
  return {
    outcome: 'STILL RUNNING — supervision window ended',
    model: MODEL, jobs: launched.job_ids,
    note: `Supervised ${MAX_HOURS}h without finishing. The jobs are unaffected by this workflow ending.`,
    cycles, repairs,
  }
}

// ---------- 4. Merge ----------
phase('Merge')
const merged = await agent(
  `Every task for ${MODEL} has finished. Merge the shards and produce the combined summary.
${RULES}

The per-task jobs never wrote the combined <stem>_negotiation_overall.csv, because run_cli only
writes it for --task all. That is expected and is this phase's job.

1. Verify row counts per task against ${JSON.stringify(EXPECTED)} BEFORE merging.
   **4,760 intention rows means the odd-length-dialogue bug is back** — intention has 4,618.
2. Count rows with an empty raw_response and duplicate uids, from the .jsonl and not from CSV:
   embedded newlines break naive CSV parsing and once reported more unique uids than rows.
3. Run merge_neg_results.py, then report the resulting metrics.
4. Report the [budget] block each shard printed at exit: latency percentiles, token percentiles,
   truncation count, hang rate. Say whether the 120s ceiling now looks right against the measured
   p99, since the next run should size it from this rather than from a probe.`,
  { agentType: 'executor', phase: 'Merge', schema: {
    type: 'object', additionalProperties: false,
    required: ['merged', 'row_counts', 'empty_rows', 'detail'],
    properties: {
      merged: { type: 'boolean' },
      row_counts: { type: 'object', additionalProperties: true },
      empty_rows: { type: 'integer' },
      duplicate_uids: { type: 'integer' },
      metrics: { type: 'object', additionalProperties: true },
      measured_latency_p99: { type: 'number' },
      ceiling_verdict: { type: 'string' },
      detail: { type: 'string' },
    },
  } }
)

// ---------- 5. Audit ----------
phase('Audit')
const audit = await agent(
  `${MODEL} has finished and merged. Decide whether the numbers can be published. Recommend only —
change nothing, submit nothing.
${RULES}

1. Row counts against ${JSON.stringify(EXPECTED)}; duplicate uids; empty raw_response, counted from
   the .jsonl.
2. The scores, against this model's own archived reasoning-ON pilot
   (${MODEL}/pilot_archive_reasoning_on_20260804) and against the other five models.
3. **The comparability caveat, stated plainly rather than buried:** this model ran with Together's
   reasoning path disabled while DeepSeek and Grok ran with reasoning ON. Say how that must be
   reported alongside the cross-model table. A paired check on 138 desire items gave 0.7101 with
   reasoning against 0.6884 without, McNemar p=0.678 — quote it, and say whether the full run's
   numbers are consistent with it now that the sample is 30x larger.
4. Cost and tokens for this run.

Return usable=false if the numbers should not be published.`,
  { agentType: 'evaluator', phase: 'Audit', schema: {
    type: 'object', additionalProperties: false,
    required: ['usable', 'row_counts', 'empty_rate', 'problems', 'detail'],
    properties: {
      usable: { type: 'boolean' },
      row_counts: { type: 'object', additionalProperties: true },
      empty_rate: { type: 'number' },
      metrics: { type: 'object', additionalProperties: true },
      vs_reasoning_on_pilot: { type: 'object', additionalProperties: true },
      comparability_caveats: { type: 'array', items: { type: 'string' } },
      problems: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } }
)

// ---------- 6. Record ----------
phase('Record')
const recorded = await agent(
  `Record this run in ${LOCAL}/ISSUES.md, in the file's existing style.
${RULES}

What is worth keeping, beyond the outcome:
  per-task parallelism  ${TASKS.length * 5} workers at --total-shards 5, and whether it held
  repairs made          ${JSON.stringify(repairs)}
  audit                 usable=${audit?.usable}, problems=${JSON.stringify(audit?.problems || [])}
  measured p99 latency  ${merged?.measured_latency_p99}, ceiling verdict ${merged?.ceiling_verdict}

The reusable lesson is the one about shape, not about Gemma: splitting by TASK multiplies workers
without touching --total-shards, so no checkpoint is orphaned and a fallback is free, whereas
raising the shard count re-pays for every row already written. Say that in a way the next model can
apply.

Also correct the claim in NEG_GPT/run_negotiation.sh that "Quest allows at most 5 parallel array
jobs" if this run contradicts it — cite the measured limits and what actually ran. All six sbatch
scripts inherited --array=0-4 from that belief.

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
  jobs: launched.job_ids,
  workers: gate?.workers_running,
  usable: audit?.usable,
  metrics: audit?.metrics,
  row_counts: audit?.row_counts,
  empty_rate: audit?.empty_rate,
  vs_reasoning_on_pilot: audit?.vs_reasoning_on_pilot,
  comparability_caveats: audit?.comparability_caveats,
  problems: audit?.problems,
  repairs,
  measured_latency_p99: merged?.measured_latency_p99,
  ceiling_verdict: merged?.ceiling_verdict,
  issues_updated: recorded?.written,
  cycles,
  next: audit?.usable
    ? 'Results are in git. All six models are now complete; report the reasoning-off caveat with the cross-model table.'
    : 'Read problems before publishing anything from this run.',
}
