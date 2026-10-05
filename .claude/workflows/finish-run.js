export const meta = {
  name: 'finish-run',
  description: 'Baseline: pull a finished run down, audit it, and — only once the user confirms — record it in the page and both workbooks and sync',
  whenToUse: 'A pilot or full run has finished. Pass {benchmark, model}. The first call pulls and audits, then stops for the user. Call again with confirmed: true (and resumeFromRunId, so the pull and audit come from cache) to record and sync. A baseline — adapt the phases to the task.',
  phases: [
    { title: 'Pull', detail: 'executor — results and logs down, committed (skill: pull-results)' },
    { title: 'Audit', detail: 'evaluator — can the numbers be believed (skill: audit-results)' },
    { title: 'Record', detail: 'executor — page, both workbooks, Provenance, one edit (skills: record-results, sync-pass)' },
    { title: 'Log', detail: 'tracker — only when the audit found a problem' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   benchmark: "docvqa"                       required — the page stem
//   model:     "DocVQA_GPT_5.6_Luna"          required — the model folder
//   stage:     "pilot" | "full"               optional, default "full" — a pilot is audited, never recorded
//   confirmed: true                           optional — the user has confirmed the run (sync layer 4)
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
const STAGE = A.stage || 'full'
if (!BENCH || !MODEL) return { outcome: 'aborted', reason: 'args.benchmark and args.model are both required' }

const skill = (name) => `.claude/skills/${name}/SKILL.md`
// Kept free of `confirmed`, so a resumed run with confirmed: true reuses the cached pull and audit.
const CONTEXT = `
Benchmark: ${BENCH}. Model folder: ${MODEL}. Stage: ${STAGE}.
Paths, task names, expected counts, the scorer and the authoritative result file are on the
benchmark's page: the one file matching \`ls .claude/references/benchmarks/*/${BENCH}.md\`.
No provider API calls. Code never flows down from Quest.
`

// ---------- 1. Pull ----------
phase('Pull')
const pulled = await agent(
  `Bring ${MODEL}'s results and logs down from Quest and commit them, following ${skill('pull-results')}.
${CONTEXT}
First confirm with sacct that ${MODEL}'s job has finished; if it has not, pull anyway but say so.
Report rows per task on Quest and locally — they must match.`,
  { agentType: 'executor', phase: 'Pull', schema: {
    type: 'object', additionalProperties: false,
    required: ['job_finished', 'rows_quest', 'rows_local', 'match', 'commit', 'detail'],
    properties: {
      job_finished: { type: 'boolean' },
      rows_quest: { type: 'object', additionalProperties: true },
      rows_local: { type: 'object', additionalProperties: true },
      match: { type: 'boolean' },
      commit: { type: 'string' },
      detail: { type: 'string' },
    },
  } },
)
if (!pulled || !pulled.match) {
  return { outcome: 'blocked', phase: 'Pull', reason: 'local rows do not match Quest', pulled }
}

// ---------- 2. Audit ----------
phase('Audit')
const audit = await agent(
  `Decide whether ${MODEL}'s ${STAGE} results can be believed, following ${skill('audit-results')}.
Work from the local copy just pulled.
${CONTEXT}
Job finished: ${pulled.job_finished}. Rows: ${JSON.stringify(pulled.rows_local)}.
Give the headline score recomputed from the per-task files, the denominator per task, and every
unusable-row count (empty, null prediction, no answer marker, off-label).`,
  { agentType: 'evaluator', phase: 'Audit', schema: {
    type: 'object', additionalProperties: false,
    required: ['trust', 'recommendation', 'headline', 'problems', 'detail'],
    properties: {
      trust: { type: 'string', enum: ['trustworthy', 'partial', 'untrustworthy', 'cannot-tell'] },
      recommendation: { type: 'string', enum: ['continue', 'kill', 'kill-and-archive', 'prune-and-resume', 'publish', 'needs-human'] },
      headline: { type: 'string' },
      per_task: { type: 'object', additionalProperties: true },
      unusable: { type: 'object', additionalProperties: true },
      problems: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } },
)
if (!audit) return { outcome: 'blocked', phase: 'Audit', reason: 'the audit returned nothing', pulled }

if (STAGE === 'pilot') {
  return { outcome: 'pilot-audited', audit, next: audit.trust === 'trustworthy' ? 'launch-run with stage "full"' : 'fix the runner first (new-runner or fix-run)' }
}
if (A.confirmed !== true) {
  return {
    outcome: 'awaiting-confirmation',
    audit,
    pulled: pulled.commit,
    next: 'show the audit to the user; once they confirm the run, re-run with confirmed: true and resumeFromRunId',
  }
}
if (audit.trust !== 'trustworthy' && audit.trust !== 'partial') {
  return { outcome: 'blocked', phase: 'Audit', reason: `audit says ${audit.trust}; not recording even with confirmation`, audit }
}

// ---------- 3. Record ----------
phase('Record')
const recorded = await agent(
  `The user has confirmed ${MODEL}'s run. Record it following ${skill('record-results')}, then close
with ${skill('sync-pass')}.
${CONTEXT}
Audit: ${audit.trust}; headline ${audit.headline}; per task ${JSON.stringify(audit.per_task || {})};
unusable ${JSON.stringify(audit.unusable || {})}; problems ${JSON.stringify(audit.problems)}.
Recompute every cell from the per-task files on disk rather than copying these numbers. Copy each
workbook first and show the cell diff: only the intended cells changed.`,
  { agentType: 'executor', phase: 'Record', schema: {
    type: 'object', additionalProperties: false,
    required: ['page_updated', 'workbooks_updated', 'cell_diff_ok', 'commit', 'pushed', 'detail'],
    properties: {
      page_updated: { type: 'boolean' },
      workbooks_updated: { type: 'array', items: { type: 'string' } },
      cell_diff_ok: { type: 'boolean' },
      stale_claims_fixed: { type: 'array', items: { type: 'string' } },
      commit: { type: 'string' },
      pushed: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } },
)

// ---------- 4. Log ----------
let logged = null
if (audit.problems && audit.problems.length) {
  phase('Log')
  logged = await agent(
    `Record what the audit of ${MODEL} on ${BENCH} found. Name the benchmark in the entry; extend an
existing entry with the same root cause instead of adding a near-duplicate.
Problems: ${audit.problems.map((p, i) => `\n  ${i + 1}. ${p}`).join('')}
Unusable rows: ${JSON.stringify(audit.unusable || {})}.`,
    { agentType: 'tracker', phase: 'Log', schema: {
      type: 'object', additionalProperties: false,
      required: ['updated', 'detail'],
      properties: {
        updated: { type: 'boolean' },
        entry_title: { type: 'string' },
        detail: { type: 'string' },
      },
    } },
  )
}

return {
  outcome: recorded && recorded.cell_diff_ok && recorded.pushed ? 'recorded' : 'record-incomplete',
  audit: { trust: audit.trust, headline: audit.headline },
  record: recorded,
  logged: logged?.entry_title,
}
