export const meta = {
  name: 'check-status',
  description: 'Read-only snapshot of how a run is actually going, with a finish estimate and anything that needs a decision',
  whenToUse: 'Answer "how is it going" for a model that is currently running, at any time and as often as wanted. Pass {model} plus optional {expected, sinceMinutes}. Strictly read-only: it never submits, cancels, edits or transfers, so it is safe to run alongside a supervising workflow without disturbing it. Two agents, so it is cheap enough to repeat. Use run-fast or run-model when the run needs driving; this only reports.',
  phases: [
    { title: 'Observe', detail: 'watcher: queue, rows, pulses, logs, markers — facts only' },
    { title: 'Judge',   detail: 'evaluator: healthy or not, when it finishes, what needs a decision' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   model:         "NEG_Gemma"   required
//   expected:      {desire: 4760, belief: 4760, intention: 4618}
//   sinceMinutes:  0             optional: only consider progress in the last N minutes, for a
//                                rate that reflects now rather than an average dragged down by a
//                                bad first hour
// }
// `args` may arrive as a JSON string; normalise before reading a field.
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
const EXPECTED = A.expected || { desire: 4760, belief: 4760, intention: 4618 }
const SINCE = A.sinceMinutes || 0

if (!MODEL) {
  return { aborted: 'args.model is required, e.g. {model: "NEG_Gemma"}' }
}

const REPO = '/Users/harrychen/SONIC/hai-teams'
const LOCAL = REPO + '/Interpersonal_processes_benchmarks/NegotiationToM'
const QDIR = '/gpfs/projects/p32983/NegotiationToM'
const TOTAL = Object.values(EXPECTED).reduce((a, b) => a + b, 0)

const RULES = `
Target: ${MODEL}. Expected rows: ${JSON.stringify(EXPECTED)} (total ${TOTAL}).

THIS WORKFLOW IS READ-ONLY. Breaking that is a failed task, not a judgement call:
- NO sbatch, NO scancel, NO scontrol, NO file edits, NO transfers, NO git commits or pushes, NO
  provider API calls of any kind. Not even "just to check". A supervising workflow may be driving
  this same run right now, and a stray cancel or resubmit would corrupt it.
- Reading is unrestricted: squeue, sacct, cat, grep, wc, find, and the local repo.
- Do not run pull_quest_results.sh either — it writes locally. Read Quest directly over ssh.
- Quest is \`ssh quest\`; ignore libcrypto warnings. Remote ${QDIR}, local ${LOCAL}.
`

// ---------- 1. Observe ----------
phase('Observe')
const obs = await agent(
  `Report exactly what ${MODEL}'s run is doing right now. Facts only — no interpretation, no advice.
${RULES}

1. **Queue.** squeue for uwr0681: every job, its name, state, elapsed and node. Count array tasks
   RUNNING and PENDING per job name. If a job has left the queue, get its State, ExitCode and
   Elapsed from sacct — a job can exit while its rows look fine.
2. **Rows, per task**, counted on Quest from the .jsonl (not from CSV — embedded newlines in model
   output break naive CSV parsing and have already produced a false duplicate-uid alarm here):
   ${QDIR}/${MODEL}/results/<task>/*shard*.jsonl. Give the count per task and per shard, so a
   single stalled shard is visible rather than averaged away.
3. **Pulses.** The runner prints a throttled [pulse] line carrying calls, hang rate, trunc, empty,
   err, mean latency, effective s/row, rows/min, token p99 and latency p90/p99/max.
   **Ignore any pulse reporting fewer than 10 calls — it is a single-sample artifact and its
   rows/min is meaningless.** Quote the most recent pulse with a double-digit call count from every
   shard log. If no shard has one yet, say so explicitly rather than quoting an n=1 line.
4. **Rate.** ${SINCE ? `Measure progress over the LAST ${SINCE} MINUTES: record row counts, wait, and record again, so the rate reflects now.`
   : 'Compute rows/min two ways and give both: (a) total rows divided by elapsed since the job started, and (b) from the [pulse] effective s/row. If they disagree by more than ~20%, say so — the pulse figure is per-shard and the first is aggregate, and conflating them has caused wrong estimates here.'}
5. **Trouble.** Any BILLING_HALT / QUOTA_HALT / FAILURE_HALT under ${MODEL}/, quoted in full. Count
   rows with an empty raw_response, per task. The tail of any shard log that has not been written to
   in the last 15 minutes.
6. **Empty-handed is a finding.** If nothing is running and no rows exist, say that plainly rather
   than reporting an absence of problems.`,
  { agentType: 'watcher', phase: 'Observe', schema: {
    type: 'object', additionalProperties: false,
    required: ['anything_running', 'workers_running', 'rows_by_task', 'rows_total', 'detail'],
    properties: {
      anything_running: { type: 'boolean' },
      workers_running: { type: 'integer' },
      workers_pending: { type: 'integer' },
      jobs: { type: 'array', items: { type: 'string' } },
      rows_by_task: { type: 'object', additionalProperties: true },
      rows_by_shard: { type: 'object', additionalProperties: true },
      rows_total: { type: 'integer' },
      rows_per_min: { type: 'number' },
      rate_basis: { type: 'string' },
      hang_rate: { type: 'number' },
      latency_p99: { type: 'number' },
      empty_rows: { type: 'integer' },
      pulses_usable: { type: 'boolean', description: 'true only if a pulse with >=10 calls exists' },
      halt_markers: { type: 'array', items: { type: 'string' } },
      stalled_shards: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } }
)

if (!obs) {
  return { outcome: 'NO OBSERVATION', note: 'the watcher returned nothing; re-run before drawing conclusions' }
}

// ---------- 2. Judge ----------
phase('Judge')
const verdict = await agent(
  `Judge ${MODEL}'s run from these observations and say what, if anything, needs a decision.
Recommend only — you change nothing.
${RULES}

OBSERVED: ${JSON.stringify(obs).slice(0, 6000)}

Answer four questions, briefly and with the number that supports each:

1. **Is it healthy?** Rows accumulating, hang rate, empty rate, halt markers, stalled shards. The
   single-stream reference for this configuration is roughly 8-10% hangs; well above that means the
   provider is objecting to the concurrency.
2. **When does it finish?** Remaining rows divided by the measured rate. Give a range, and say
   which rate you used and why. **If no pulse has 10 or more calls yet, refuse to give an estimate**
   and say what would settle it — a premature estimate from a single sample has been wrong here
   repeatedly, by factors of 3 to 5.
3. **Is the ceiling right?** Compare latency p99 against the watchdog ceiling. If p99 is anywhere
   near it, legitimate slow calls are being recorded as hangs, and every rate above is suspect.
4. **Does anything need the planner?** Only genuine decisions: a halt marker, an empty rate above
   2%, a shard dead with no supervisor to restart it, or a finish estimate that overruns the
   walltime. If a supervising workflow is already driving this run it will handle a dead shard and
   empty rows itself, so do not escalate those unless they are growing.

Be blunt about uncertainty. "Cannot tell yet, need N more minutes" is a better answer than a number
that will be revised twice.`,
  { agentType: 'evaluator', phase: 'Judge', schema: {
    type: 'object', additionalProperties: false,
    required: ['healthy', 'summary', 'needs_decision', 'detail'],
    properties: {
      healthy: { type: 'string', enum: ['yes', 'no', 'too_early'] },
      progress_pct: { type: 'number' },
      eta_hours: { type: 'number' },
      eta_confidence: { type: 'string', enum: ['measured', 'provisional', 'refused'] },
      eta_basis: { type: 'string' },
      ceiling_ok: { type: 'boolean' },
      needs_decision: { type: 'array', items: { type: 'string' } },
      summary: { type: 'string' },
      detail: { type: 'string' },
    },
  } }
)

return {
  model: MODEL,
  running: obs.anything_running,
  workers: obs.workers_running,
  rows: obs.rows_by_task,
  rows_total: obs.rows_total,
  progress_pct: verdict?.progress_pct ?? Math.round((obs.rows_total / TOTAL) * 1000) / 10,
  rows_per_min: obs.rows_per_min,
  hang_rate: obs.hang_rate,
  empty_rows: obs.empty_rows,
  healthy: verdict?.healthy,
  eta_hours: verdict?.eta_hours,
  eta_confidence: verdict?.eta_confidence,
  eta_basis: verdict?.eta_basis,
  ceiling_ok: verdict?.ceiling_ok,
  halt_markers: obs.halt_markers,
  stalled_shards: obs.stalled_shards,
  needs_decision: verdict?.needs_decision || [],
  summary: verdict?.summary,
}
