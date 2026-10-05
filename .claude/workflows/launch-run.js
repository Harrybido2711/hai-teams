export const meta = {
  name: 'launch-run',
  description: 'Baseline: sync one model to Quest with md5 proof, gate, submit a pilot or full run, and gate on its first rows',
  whenToUse: 'Phase 3, only after the user has verified the scripts. Pass {benchmark, model, stage: "pilot"|"full", userVerified: true, reason} and optionally {script, gateMinutes}. Use fix-run instead when a job of this model is already running and producing bad data. A baseline — adapt the phases to the task.',
  phases: [
    { title: 'Sync', detail: 'executor — whole change set to Quest, md5 both sides (skill: quest-sync)' },
    { title: 'Gate', detail: 'reviewer — independently re-verifies; may refuse' },
    { title: 'Submit', detail: 'executor — checkpoint disposition, sbatch (skill: submit-run)' },
    { title: 'First rows', detail: 'watcher — two samples minutes apart (skill: check-run)' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   benchmark:    "emobench"          required — the page stem
//   model:        "EMO_GPT_5.6_Luna"  required — the model folder
//   stage:        "pilot" | "full"    required
//   userVerified: true                required — CLAUDE.md phase 3: the user has verified the scripts
//   reason:       "..."               required — one sentence
//   script:       "run_pilot.sh"      optional — otherwise the page's run order decides
//   gateMinutes:  5                   optional — spacing of the two first-rows samples
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
const STAGE = A.stage || ''
const REASON = A.reason || 'not stated'
const GATE_MIN = A.gateMinutes || 5
if (!BENCH || !MODEL || !['pilot', 'full'].includes(STAGE)) {
  return { outcome: 'aborted', reason: 'args.benchmark, args.model and args.stage ("pilot" or "full") are required' }
}
if (A.userVerified !== true) {
  return { outcome: 'aborted', reason: 'phase 3 starts only after the user has verified the scripts — pass userVerified: true once they have' }
}

const skill = (name) => `.claude/skills/${name}/SKILL.md`
const CONTEXT = `
Benchmark: ${BENCH}. Model folder: ${MODEL}. Stage: ${STAGE}${A.script ? ` (submit script: ${A.script})` : ''}. Why: ${REASON}
Local and Quest paths, task names, expected counts and the run order are on the benchmark's page: the
one file matching \`ls .claude/references/benchmarks/*/${BENCH}.md\`. Never infer a Quest path.
Touch ${MODEL} and nothing else; other jobs may be running. No provider API calls outside the runner.
`

// ---------- 1. Sync ----------
phase('Sync')
const synced = await agent(
  `Make Quest match local for ${BENCH}, following ${skill('quest-sync')}. Read the skill first.
${CONTEXT}
Report the file counts on both sides, every file transferred, and the drift remaining after the
transfer (it must be none). List every job of uwr0681 currently in squeue.`,
  { agentType: 'executor', phase: 'Sync', schema: {
    type: 'object', additionalProperties: false,
    required: ['in_sync', 'local_count', 'quest_count', 'transferred', 'running_jobs', 'detail'],
    properties: {
      in_sync: { type: 'boolean' },
      local_count: { type: 'integer' },
      quest_count: { type: 'integer' },
      transferred: { type: 'array', items: { type: 'string' } },
      drift_after: { type: 'array', items: { type: 'string' } },
      running_jobs: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } },
)

if (!synced || !synced.in_sync || !synced.local_count || synced.local_count !== synced.quest_count) {
  return { outcome: 'blocked', phase: 'Sync', reason: 'Quest could not be shown to match local', synced }
}

// ---------- 2. Gate ----------
phase('Gate')
const gate = await agent(
  `Decide whether ${MODEL} is safe to submit. Assume the sync report is optimistic; verify with your
own commands.
${CONTEXT}
Claimed: in sync over ${synced.local_count} files; transferred ${JSON.stringify(synced.transferred)}.

Check at minimum:
- the md5 comparison again (${skill('quest-sync')} step 2) — both counts non-zero and equal, no drift;
- the change that motivated this run is present in the file ON QUEST — name the line;
- .env exists in the Quest directory; the submit script points at the right runner, partition,
  walltime and shard count (5 unless the page says otherwise);
- what is already in ${MODEL}'s results on Quest, and which disposition ${skill('submit-run')} step 2
  requires: archive (config changed since those rows), prune (some rows empty) or resume.

Return safe_to_submit=false if any check fails, and say which.`,
  { agentType: 'reviewer', phase: 'Gate', schema: {
    type: 'object', additionalProperties: false,
    required: ['safe_to_submit', 'blockers', 'disposition', 'detail'],
    properties: {
      safe_to_submit: { type: 'boolean' },
      blockers: { type: 'array', items: { type: 'string' } },
      disposition: { type: 'string', enum: ['archive', 'prune', 'resume'] },
      existing_rows: { type: 'integer' },
      detail: { type: 'string' },
    },
  } },
)

if (!gate || !gate.safe_to_submit) {
  return { outcome: 'blocked', phase: 'Gate', blockers: gate?.blockers || ['the reviewer returned no verdict'], synced, gate }
}

// ---------- 3. Submit ----------
phase('Submit')
const submitted = await agent(
  `Submit ${MODEL}'s ${STAGE} run, following ${skill('submit-run')} steps 2-4. The reviewer cleared it.
${CONTEXT}
Checkpoint disposition decided by the gate: ${gate.disposition} (${gate.existing_rows ?? '?'} rows on Quest).
Carry it out, showing row counts before and after; never delete. Then sbatch from ${MODEL}'s own
directory and nothing else. Afterwards squeue must list exactly the new job(s) plus
${JSON.stringify(synced.running_jobs)}.`,
  { agentType: 'executor', phase: 'Submit', schema: {
    type: 'object', additionalProperties: false,
    required: ['job_ids', 'disposition_done', 'queue_as_expected', 'detail'],
    properties: {
      job_ids: { type: 'array', items: { type: 'string' } },
      disposition_done: { type: 'string' },
      archive_path: { type: 'string' },
      queue_as_expected: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } },
)

if (!submitted || !submitted.job_ids || submitted.job_ids.length === 0) {
  return { outcome: 'launch-failed', reason: 'sbatch returned no job id; the sync already happened, only the submit needs redoing', synced, gate, submitted }
}

// ---------- 4. First rows ----------
phase('First rows')
const first = await agent(
  `Judge ${MODEL}'s new job(s) ${JSON.stringify(submitted.job_ids)} by their first rows, following
${skill('check-run')}. Observe only.
${CONTEXT}
Take two samples about ${GATE_MIN} minutes apart (sleep on the Quest side: \`ssh quest 'sleep ${GATE_MIN * 60}; ...'\`).
Report rows written, rows/min, empty responses, halt markers and errors. Fewer than 20 rows after
both samples is too-early, not a stall.`,
  { agentType: 'watcher', phase: 'First rows', schema: {
    type: 'object', additionalProperties: false,
    required: ['verdict', 'rows', 'empty_count', 'detail'],
    properties: {
      verdict: { type: 'string', enum: ['healthy', 'too-early', 'degraded', 'failed', 'stalled', 'quota-blocked', 'stale-code', 'cannot-tell'] },
      rows: { type: 'integer' },
      empty_count: { type: 'integer' },
      rate_per_min: { type: 'number' },
      eta: { type: 'string' },
      detail: { type: 'string' },
    },
  } },
)

const healthy = first && (first.verdict === 'healthy' || first.verdict === 'too-early')
return {
  outcome: healthy ? 'launched' : 'launched-unhealthy',
  model: MODEL,
  stage: STAGE,
  jobs: submitted.job_ids,
  disposition: submitted.disposition_done,
  archive: submitted.archive_path,
  sync: { files: synced.local_count, transferred: synced.transferred },
  first_rows: first,
  next: !healthy
    ? 'fix-run — the first rows are not usable'
    : STAGE === 'pilot'
      ? 'when the pilot finishes: finish-run to audit it, then launch-run with stage "full"'
      : 'monitor-run on a timer; finish-run when it completes',
}
