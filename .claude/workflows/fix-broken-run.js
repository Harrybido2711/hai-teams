export const meta = {
  name: 'fix-broken-run',
  description: 'Stop a job that is producing bad data, fix and resync the code, then resubmit it cleanly',
  whenToUse: 'A job is running but its output cannot be used — stale code, a config that was never transferred, a provider refusing every call, or rows arriving empty. Pass {model, reason} and optionally {pilot, jobId}. Do NOT use it to start a run that has never been submitted, and do NOT use it on a job that is merely slow.',
  phases: [
    { title: 'Observe',  detail: 'watcher measures what the job is actually producing' },
    { title: 'Stop',     detail: 'executor cancels it and decides archive vs prune vs keep' },
    { title: 'Sync',     detail: 'executor proves every code file matches Quest' },
    { title: 'Gate',     detail: 'reviewer tries to find the reason the resubmit will also fail' },
    { title: 'Resubmit', detail: 'executor submits, and nothing else' },
    { title: 'Confirm',  detail: 'watcher checks the new run is healthy, not merely RUNNING' },
    { title: 'Record',   detail: 'tracker writes the outcome into ISSUES.md' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   model:  "NEG_Qwen"      required — the model directory on Quest
//   reason: "..."           required — why this run is being killed, in one sentence
//   pilot:  true|false      optional — pilot (run_pilot.sh) or full run (run_negotiation.sh)
//   jobId:  "8167589"       optional — skip discovery when you already know it
// }
// ---------------------------------------------------------------------------

// `args` can arrive as a JSON *string* rather than an object — see the note in run-model.js. Left
// unhandled, every field reads as undefined and the guard below blames a missing model.
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
const IS_PILOT = !!A.pilot
const KNOWN_JOB = A.jobId || null

if (!MODEL) {
  return { aborted: 'args.model is required, e.g. {model: "NEG_Qwen", reason: "..."}' }
}

const QDIR = '/gpfs/projects/p32983/NegotiationToM'
const LOCAL = '/Users/harrychen/SONIC/hai-teams/Interpersonal_processes_benchmarks/NegotiationToM'
const SUBMIT = IS_PILOT ? 'run_pilot.sh' : 'run_negotiation.sh'
const RESULTS = IS_PILOT ? 'results/pilot' : 'results'

const RULES = `
Target: ${MODEL}  (${IS_PILOT ? 'pilot' : 'full run'}, submitted with ${SUBMIT})
Reason this run is being stopped: ${REASON}

HARD RULES — breaking one of these is a failed task, not a judgement call:
- Touch ${MODEL} and nothing else. Other models may be running; leave every one of them alone and
  confirm in your report that you did.
- Do NOT call a model provider API. No probe scripts, no test completions. Read files and logs.
- Never overwrite .env on Quest and never copy it off.
- Under /projects/p32983 only uwr0681's directories are in scope.
- Quest is \`ssh quest\`; drop \`libcrypto\` lines from its output. Remote ${QDIR}, local ${LOCAL}.
`

// ---------- 1. Observe ----------
phase('Observe')
const state = await agent(
  `Measure what ${MODEL} is actually producing. Change nothing.
${RULES}

Report:
1. squeue — every job of uwr0681, so the report also proves which ones you must not disturb.
   ${KNOWN_JOB ? `The target is believed to be ${KNOWN_JOB}; verify that.` : `Identify ${MODEL}'s job id.`}
2. Rows per task under ${MODEL}/${RESULTS}, each file's mtime, and a rows/minute rate from two
   samples several minutes apart. A file that has not grown is not proof of a stall — check the log
   mtime too, since a slow model looks frozen between checkpoint writes.
3. The empty-\`raw_response\` and null-\`pred\` rate in the rows already written. This is the number
   that says whether the data is usable, and it is not visible in the queue or the row count.
4. Any ${MODEL}/BILLING_HALT.txt, QUOTA_HALT.txt or FAILURE_HALT.txt, quoted in full.
5. Whether ${MODEL}'s code on Quest matches local — compare with md5sum yourself.`,
  { agentType: 'watcher', phase: 'Observe', schema: {
    type: 'object', additionalProperties: false,
    required: ['job_id', 'rows_by_task', 'empty_rate', 'other_jobs', 'code_in_sync', 'detail'],
    properties: {
      job_id: { type: 'string', description: 'the target job id, or "none" if it is not running' },
      rows_by_task: { type: 'object', additionalProperties: true },
      empty_rate: { type: 'number', description: 'fraction of written rows with an empty raw_response' },
      rate_per_min: { type: 'number' },
      halt_markers: { type: 'array', items: { type: 'string' } },
      other_jobs: { type: 'array', items: { type: 'string' }, description: 'jobs that must not be touched' },
      code_in_sync: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

if (!state) return { aborted: 'the observer returned nothing; stopping before any change' }
log(`${MODEL}: job=${state.job_id} empty_rate=${state.empty_rate} in_sync=${state.code_in_sync}`)

// ---------- 2. Stop ----------
phase('Stop')
const stopped = await agent(
  `Stop ${MODEL} and put its existing rows in the right place. This has been decided; carry it out.
${RULES}

Observed: job ${state.job_id}, rows ${JSON.stringify(state.rows_by_task)}, empty rate
${state.empty_rate}, halt markers ${JSON.stringify(state.halt_markers || [])}.

1. scancel ${state.job_id} — that job only. Then run squeue and confirm every job in
   ${JSON.stringify(state.other_jobs || [])} is still RUNNING.

2. Decide what happens to the rows already written, and say which rule you applied:
   * The fix changes the PROMPT or the DECODING CONFIG -> archive the whole checkpoint to
     ${MODEL}/results_archive_<what changed>_<UTC timestamp> and start clean. Rows produced under a
     different configuration must not share a result set with new ones.
   * The rows are fine but some are empty (a provider refused mid-run) -> keep the good ones and
     run \`python3 prune_failed_rows.py --apply --only ${MODEL}\`. A plain resume would skip the
     empty rows forever, because load_checkpoint adds every uid to the done set.
   * Nothing was written, or every row is valid and the config is unchanged -> leave it and resume.
   Never delete. Archive or prune, both of which are reversible.

3. Show the row counts before and after so the decision is auditable.`,
  { agentType: 'executor', phase: 'Stop', schema: {
    type: 'object', additionalProperties: false,
    required: ['cancelled', 'others_untouched', 'disposition', 'detail'],
    properties: {
      cancelled: { type: 'boolean' },
      others_untouched: { type: 'boolean' },
      disposition: { type: 'string', enum: ['archived', 'pruned', 'kept'] },
      archive_path: { type: 'string' },
      rows_before: { type: 'integer' },
      rows_after: { type: 'integer' },
      detail: { type: 'string' },
    },
  } }
)

if (!stopped || !stopped.cancelled) {
  return { aborted: 'could not confirm the job was cancelled; nothing else was changed', state, stopped }
}
if (stopped.others_untouched === false) {
  return { aborted: 'another model appears to have been affected — stopping for a human', stopped }
}

// ---------- 3. Sync ----------
phase('Sync')
const synced = await agent(
  `Make Quest match local, then prove it.
${RULES}

${MODEL} is cancelled and its rows are ${stopped.disposition}${stopped.archive_path ? ` at ${stopped.archive_path}` : ''}.

1. Run \`python3 .claude/scripts/check_quest_sync.py\` from the repo root. It compares every *.py
   and NEG_*/*.sh and exits 1 on drift.
2. Transfer every file it names, local -> Quest, with
   \`ssh quest "cat > ${QDIR}/<path>" < ${LOCAL}/<path>\`, and verify each with md5sum.
   **Send neg_eval_core.py together with the runners.** The runners import from it, so a runner
   without the core dies at import and a core without the runners breaks changed signatures.
3. Re-run the checker and show it exiting 0.
4. On Quest: py_compile every transferred file, and confirm the names each runner imports from
   neg_eval_core actually resolve there.

Note that a sync check can pass for the wrong reason. Print the file count on both sides — a
comparison over two empty lists is not a pass.`,
  { agentType: 'executor', phase: 'Sync', schema: {
    type: 'object', additionalProperties: false,
    required: ['all_in_sync', 'files_compared', 'compile_ok', 'import_ok', 'detail'],
    properties: {
      transferred: { type: 'array', items: { type: 'string' } },
      all_in_sync: { type: 'boolean' },
      files_compared: { type: 'integer' },
      compile_ok: { type: 'boolean' },
      import_ok: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

// ---------- 4. Gate ----------
phase('Gate')
const gate = await agent(
  `Decide whether ${MODEL} is safe to resubmit. Assume the previous agent's report is optimistic and
look for the reason this run will fail too. Verify with your own commands.
${RULES}

Claimed: in_sync=${synced?.all_in_sync}, files=${synced?.files_compared}, compile=${synced?.compile_ok},
import=${synced?.import_ok}, rows ${stopped.disposition}${stopped.archive_path ? ` -> ${stopped.archive_path}` : ''}.

Check at minimum:
- Every code file really is byte-identical between local and Quest. Print both file counts.
- The change that motivated this rerun is actually present in the file ON QUEST. Name the line.
- ${MODEL}/${RESULTS} holds exactly what the disposition implies — empty after an archive, pruned of
  empty rows after a prune, untouched after a keep.
- Any stale halt marker from the previous run is gone, or will be cleared at startup.
- ${MODEL}/${SUBMIT} points at the right script, partition and walltime, and .env is intact.
- Every job in ${JSON.stringify(state.other_jobs || [])} is still RUNNING.

Return safe_to_submit=false if any of these fails, and say exactly which.`,
  { agentType: 'reviewer', phase: 'Gate', schema: {
    type: 'object', additionalProperties: false,
    required: ['safe_to_submit', 'blockers', 'fix_present_on_quest', 'others_still_running', 'detail'],
    properties: {
      safe_to_submit: { type: 'boolean' },
      blockers: { type: 'array', items: { type: 'string' } },
      fix_present_on_quest: { type: 'boolean' },
      others_still_running: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

if (!gate || !gate.safe_to_submit) {
  log('gate refused — not resubmitting')
  return {
    outcome: 'BLOCKED before resubmit',
    blockers: gate?.blockers || ['the reviewer returned no verdict'],
    state, stopped, synced, gate,
  }
}

// ---------- 5. Resubmit ----------
phase('Resubmit')
const submitted = await agent(
  `Submit ${QDIR}/${MODEL}/${SUBMIT} with sbatch, from ${MODEL}'s own directory. The reviewer cleared it.
${RULES}

Submit NOTHING ELSE — not another model, not a pilot alongside a full run. Afterwards run squeue and
confirm it lists exactly the new job plus ${JSON.stringify(state.other_jobs || [])}. Anything extra
is a failure; report it as one.`,
  { agentType: 'executor', phase: 'Resubmit', schema: {
    type: 'object', additionalProperties: false,
    required: ['job_id', 'queue_as_expected', 'detail'],
    properties: {
      job_id: { type: 'string' },
      state: { type: 'string' },
      queue_as_expected: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

// ---------- 6. Confirm ----------
phase('Confirm')
const confirmed = await agent(
  `Verify ${MODEL}'s new job ${submitted?.job_id} is healthy, not merely RUNNING. Observe only.
${RULES}

Wait for output, then report rows written and an observed rows/minute; the empty-response count,
which is the whole point of the rerun; whether any halt marker appeared; and a projection to
completion against the walltime. Judge by rows and log content, never by SLURM state. If it is too
early to tell, say so and say what observation would settle it — do not guess.

For comparison, the run that was just stopped had: rows ${JSON.stringify(state.rows_by_task)},
empty rate ${state.empty_rate}, ${state.rate_per_min || '?'} rows/min.`,
  { agentType: 'watcher', phase: 'Confirm', schema: {
    type: 'object', additionalProperties: false,
    required: ['verdict', 'rows_so_far', 'empty_count', 'detail'],
    properties: {
      verdict: { type: 'string', enum: ['healthy', 'too-early', 'degraded', 'failed'] },
      rows_so_far: { type: 'integer' },
      empty_count: { type: 'integer' },
      rate_per_min: { type: 'number' },
      projection: { type: 'string' },
      detail: { type: 'string' },
    },
  } }
)

// ---------- 7. Record ----------
phase('Record')
const recorded = await agent(
  `Record this in Interpersonal_processes_benchmarks/NegotiationToM/ISSUES.md. If an entry already covers the same root cause, extend it
rather than adding a near-duplicate. Keep the house style: symptom, root cause, what was rejected,
fix, measured evidence. Be brief — this is for someone who was not here.

  model            ${MODEL} (${IS_PILOT ? 'pilot' : 'full run'})
  why stopped      ${REASON}
  before           rows ${JSON.stringify(state.rows_by_task)}, empty rate ${state.empty_rate}
  disposition      ${stopped.disposition}${stopped.archive_path ? ` -> ${stopped.archive_path}` : ''}
  resubmitted as   ${submitted?.job_id}
  first check      ${confirmed?.verdict}, ${confirmed?.rows_so_far} rows,
                   ${confirmed?.empty_count} empty, ${confirmed?.rate_per_min || '?'} rows/min
  ${confirmed?.projection || ''}

State the before/after numbers plainly — the comparison is the evidence that the fix worked, and it
is what a reader six months from now will actually want.`,
  { agentType: 'tracker', phase: 'Record', schema: {
    type: 'object', additionalProperties: false,
    required: ['updated', 'detail'],
    properties: {
      updated: { type: 'boolean' },
      entry_title: { type: 'string' },
      detail: { type: 'string' },
    },
  } }
)

return {
  model: MODEL,
  reason: REASON,
  cancelled_job: state.job_id,
  disposition: stopped.disposition,
  archive: stopped.archive_path,
  sync: { in_sync: synced?.all_in_sync, files: synced?.files_compared },
  gate: { safe: gate.safe_to_submit, blockers: gate.blockers },
  new_job: submitted?.job_id,
  health: confirmed,
  issues_updated: recorded?.updated,
}
