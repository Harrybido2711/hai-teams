export const meta = {
  name: 'fix-run',
  description: 'Baseline: kill a job producing bad data, put the decided fix on Quest, gate, resubmit, confirm, record',
  whenToUse: 'Phase 4, under the standing kill-and-resync authorisation. A job is running and its output is unusable — stale code, a config error on every call, empty rows — and the fix is already decided and in the local tree. Pass {benchmark, model, reason, fix} and optionally {jobId, script}. Not for a job that is merely slow (monitor-run), and not for a first submit (launch-run). A baseline — adapt the phases to the task.',
  phases: [
    { title: 'Observe', detail: 'watcher — measure what the job is producing (skill: check-run)' },
    { title: 'Stop', detail: 'executor — scancel, then archive / prune / resume (skills: kill-and-resync, submit-run step 2)' },
    { title: 'Sync', detail: 'executor — the whole change set to Quest, md5 proof (skill: quest-sync)' },
    { title: 'Gate', detail: 'reviewer — the fix diff and the resubmit preconditions; may refuse' },
    { title: 'Resubmit', detail: 'executor — sbatch, nothing else (skill: submit-run step 4)' },
    { title: 'Confirm', detail: 'watcher — the new job is healthy, not merely RUNNING (skill: check-run)' },
    { title: 'Record', detail: 'tracker — before / after into the problem log' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   benchmark: "negotiationtom"   required — the page stem
//   model:     "NEG_Qwen"         required — the model folder
//   reason:    "..."              required — why this job is being killed, one sentence
//   fix:       "..."              required — what changed locally, and in which files
//   jobId:     "8167589"          optional — skip discovery
//   script:    "run_pilot.sh"     optional — the resubmit script; otherwise the one the job used
// }
// ---------------------------------------------------------------------------

let A = args || {}
if (typeof A === 'string') {
  try {
    A = JSON.parse(A)
  } catch (error) {
    return { outcome: 'aborted', reason: `args arrived as a string that is not JSON: ${error.message}` }
  }
}
const BENCH = A.benchmark || ''
const MODEL = A.model || ''
const REASON = A.reason || ''
const FIX = A.fix || ''
if (!BENCH || !MODEL || !REASON || !FIX) {
  return { outcome: 'aborted', reason: 'args.benchmark, args.model, args.reason and args.fix are all required — decide the fix before killing the job' }
}

const skill = (name) => `.claude/skills/${name}/SKILL.md`
const RULES = `
Benchmark: ${BENCH}. Target: ${MODEL}${A.jobId ? ` (job ${A.jobId})` : ''}.
Why it is being stopped: ${REASON}
The decided fix, already in the local tree: ${FIX}
Paths, task names, expected counts and the run order are on the benchmark's page: the one file
matching \`ls .claude/references/benchmarks/*/${BENCH}.md\`. Never infer a Quest path.

HARD RULES — breaking one is a failed task:
- Touch ${MODEL} and nothing else; confirm every other job is untouched.
- No provider API calls. Never overwrite .env on Quest, never copy it off.
`

// ---------- 1. Observe ----------
phase('Observe')
const state = await agent(
  `Measure what ${MODEL} is producing, following ${skill('check-run')}. Change nothing.
${RULES}
Also list every job of uwr0681 in squeue, so the later phases know what must not be disturbed, and
compare ${MODEL}'s code on Quest with local (md5).`,
  { agentType: 'watcher', phase: 'Observe', schema: {
    type: 'object', additionalProperties: false,
    required: ['job_id', 'rows_by_task', 'empty_count', 'other_jobs', 'code_in_sync', 'detail'],
    properties: {
      job_id: { type: 'string', description: 'the target job id, or "none" if it is not running' },
      submit_script: { type: 'string' },
      rows_by_task: { type: 'object', additionalProperties: true },
      empty_count: { type: 'integer' },
      rate_per_min: { type: 'number' },
      halt_markers: { type: 'array', items: { type: 'string' } },
      other_jobs: { type: 'array', items: { type: 'string' } },
      code_in_sync: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } },
)
if (!state) return { outcome: 'aborted', reason: 'the observer returned nothing; nothing was changed' }
log(`${MODEL}: job=${state.job_id} empty=${state.empty_count} in_sync=${state.code_in_sync}`)

// ---------- 2. Stop ----------
phase('Stop')
const stopped = await agent(
  `Stop ${MODEL} and put its existing rows in the right place, following ${skill('kill-and-resync')}
step 2 and ${skill('submit-run')} step 2. This has been decided; carry it out.
${RULES}
Observed: job ${state.job_id}, rows ${JSON.stringify(state.rows_by_task)}, ${state.empty_count} empty,
halt markers ${JSON.stringify(state.halt_markers || [])}.
scancel ${state.job_id} only, then show that ${JSON.stringify(state.other_jobs)} are still RUNNING.
Choose archive / prune / resume by the skill's rule — the fix above tells you whether the prompt or
decoding config changed — and show row counts before and after. Never delete.`,
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
  } },
)
if (!stopped || !stopped.cancelled) {
  return { outcome: 'aborted', reason: 'could not confirm the job was cancelled; nothing else was changed', state, stopped }
}
if (stopped.others_untouched === false) {
  return { outcome: 'aborted', reason: 'another job appears to have been affected — stopping for a human', stopped }
}

// ---------- 3. Sync ----------
phase('Sync')
const synced = await agent(
  `Make Quest match local for ${BENCH}, following ${skill('quest-sync')}. The whole change set, the
shared core with its runners. Report both file counts and any drift remaining (must be none).
${RULES}`,
  { agentType: 'executor', phase: 'Sync', schema: {
    type: 'object', additionalProperties: false,
    required: ['in_sync', 'local_count', 'quest_count', 'transferred', 'detail'],
    properties: {
      in_sync: { type: 'boolean' },
      local_count: { type: 'integer' },
      quest_count: { type: 'integer' },
      transferred: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } },
)

// ---------- 4. Gate ----------
phase('Gate')
const gate = await agent(
  `Decide whether ${MODEL} is safe to resubmit. Assume every earlier report is optimistic; look for
the reason this run will fail too, with your own commands.
${RULES}
Claimed: in_sync=${synced?.in_sync} over ${synced?.local_count}/${synced?.quest_count} files,
rows ${stopped.disposition}${stopped.archive_path ? ` -> ${stopped.archive_path}` : ''}.
1. The fix itself: review its diff (\`git diff\` / \`git log -p -1\` on the files it names) for silent
   failures — the usual reviewer pass.
2. The fix is present in the file ON QUEST — name the line. md5 counts non-zero and equal.
3. ${MODEL}'s results hold what the disposition implies; stale halt markers will be cleared.
4. The submit script is right (runner, partition, walltime, shards); .env is intact.
5. ${JSON.stringify(state.other_jobs)} are still RUNNING.
Return safe_to_submit=false if any of these fails, and say which.`,
  { agentType: 'reviewer', phase: 'Gate', schema: {
    type: 'object', additionalProperties: false,
    required: ['safe_to_submit', 'blockers', 'fix_present_on_quest', 'detail'],
    properties: {
      safe_to_submit: { type: 'boolean' },
      blockers: { type: 'array', items: { type: 'string' } },
      fix_present_on_quest: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } },
)
if (!gate || !gate.safe_to_submit) {
  return { outcome: 'blocked', phase: 'Gate', blockers: gate?.blockers || ['the reviewer returned no verdict'], state, stopped, synced, gate }
}

// ---------- 5. Resubmit ----------
phase('Resubmit')
const SCRIPT = A.script || state.submit_script || '(the script the stopped job used — find it from sacct)'
const submitted = await agent(
  `Submit ${MODEL}/${SCRIPT} with sbatch from ${MODEL}'s own directory (${skill('submit-run')} step 4).
The reviewer cleared it. Submit nothing else. Afterwards squeue must list exactly the new job plus
${JSON.stringify(state.other_jobs)}.
${RULES}`,
  { agentType: 'executor', phase: 'Resubmit', schema: {
    type: 'object', additionalProperties: false,
    required: ['job_id', 'queue_as_expected', 'detail'],
    properties: {
      job_id: { type: 'string' },
      queue_as_expected: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } },
)

// ---------- 6. Confirm ----------
phase('Confirm')
const confirmed = await agent(
  `Verify ${MODEL}'s new job ${submitted?.job_id} is healthy, not merely RUNNING, following
${skill('check-run')}. Two samples several minutes apart (sleep on the Quest side). Observe only.
${RULES}
The stopped job had: rows ${JSON.stringify(state.rows_by_task)}, ${state.empty_count} empty,
${state.rate_per_min ?? '?'} rows/min. Compare against it.`,
  { agentType: 'watcher', phase: 'Confirm', schema: {
    type: 'object', additionalProperties: false,
    required: ['verdict', 'rows_so_far', 'empty_count', 'detail'],
    properties: {
      verdict: { type: 'string', enum: ['healthy', 'too-early', 'degraded', 'failed', 'stalled', 'quota-blocked', 'stale-code', 'cannot-tell'] },
      rows_so_far: { type: 'integer' },
      empty_count: { type: 'integer' },
      rate_per_min: { type: 'number' },
      projection: { type: 'string' },
      detail: { type: 'string' },
    },
  } },
)

// ---------- 7. Record ----------
phase('Record')
const recorded = await agent(
  `Record this kill-and-resync. Name the benchmark (${BENCH}) in the entry; extend an entry with the
same root cause rather than adding a near-duplicate.

  model          ${MODEL}
  why stopped    ${REASON}
  fix            ${FIX}
  before         rows ${JSON.stringify(state.rows_by_task)}, ${state.empty_count} empty
  disposition    ${stopped.disposition}${stopped.archive_path ? ` -> ${stopped.archive_path}` : ''}
  resubmitted    ${submitted?.job_id}
  first check    ${confirmed?.verdict}, ${confirmed?.rows_so_far} rows, ${confirmed?.empty_count} empty,
                 ${confirmed?.rate_per_min ?? '?'} rows/min. ${confirmed?.projection || ''}

The before/after numbers are the evidence that the fix worked; state them plainly.`,
  { agentType: 'tracker', phase: 'Record', schema: {
    type: 'object', additionalProperties: false,
    required: ['updated', 'detail'],
    properties: {
      updated: { type: 'boolean' },
      entry_title: { type: 'string' },
      detail: { type: 'string' },
    },
  } },
)

return {
  outcome: confirmed && confirmed.verdict === 'healthy' ? 'resubmitted-healthy' : 'resubmitted-check-again',
  model: MODEL,
  cancelled_job: state.job_id,
  disposition: stopped.disposition,
  archive: stopped.archive_path,
  sync: { in_sync: synced?.in_sync, files: synced?.local_count },
  new_job: submitted?.job_id,
  health: confirmed,
  recorded: recorded?.entry_title,
}
