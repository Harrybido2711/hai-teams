export const meta = {
  name: 'new-runner',
  description: 'Baseline: write a runner for one model on one benchmark, smoke-test it locally, get an independent review, commit — stops before Quest',
  whenToUse: 'Phases 1-2. A model is being added to a benchmark, moved to another provider, or its runner changed. Pass {benchmark, model, modelId, reason} and optionally {copyFrom}. Never transfers or submits: it ends with a reviewed diff for the user to verify. A baseline — adapt the phases to the task.',
  phases: [
    { title: 'Brief', detail: 'summarizer — page, closest folder, recipe gaps (skills: write-runner step 1, analyse-benchmark)' },
    { title: 'Write', detail: 'executor — build and smoke-test the runner (skill: write-runner)' },
    { title: 'Review', detail: 'reviewer gates the diff; executor fixes; at most two rounds' },
    { title: 'Commit', detail: 'executor — layers 1 and 3 only (skill: sync-pass)' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   benchmark: "bbh"                  required — the page stem: .claude/references/benchmarks/*/<benchmark>.md
//   model:     "BBH_GPT_5.6_Luna"     required — the model folder to create or change
//   modelId:   "gpt-5.6-luna"         required — the provider's model id
//   reason:    "..."                  required — one sentence
//   copyFrom:  "BBH_GPT_4o_mini"      optional — the folder to copy; otherwise the brief picks one
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
const MODEL_ID = A.modelId || ''
const REASON = A.reason || 'not stated'
if (!BENCH || !MODEL || !MODEL_ID) {
  return { outcome: 'aborted', reason: 'args.benchmark, args.model and args.modelId are all required' }
}

const skill = (name) => `.claude/skills/${name}/SKILL.md`
const CONTEXT = `
Benchmark: ${BENCH}. Model folder: ${MODEL}. Model id: ${MODEL_ID}. Why: ${REASON}
Everything specific to this benchmark — paths, task names, expected counts, output layout, run order,
its shared core — is on its page: the one file matching \`ls .claude/references/benchmarks/*/${BENCH}.md\`.
Never infer a path or a count, and never carry one over from another benchmark.
This workflow is LOCAL ONLY: no ssh to Quest, no transfer, no sbatch. Real runs are not started here.
`

// ---------- 1. Brief ----------
phase('Brief')
const brief = await agent(
  `Prepare the brief for writing ${MODEL}'s runner. Read; change nothing.
${CONTEXT}
Run \`python3 .claude/scripts/check_docs.py --model ${MODEL_ID}\` and
\`python3 .claude/scripts/check_docs.py --brief ${BENCH}\`, then report:
1. Whether the benchmark's page exists, and its path. If it does not, stop there — the page comes
   first (${skill('analyse-benchmark')}).
2. The local benchmark directory, its shared core module and the scorer function runners import.
3. The closest existing model folder to copy${A.copyFrom ? ` (the caller suggests ${A.copyFrom}; confirm or say why not)` : ''},
   chosen by provider family first, then by recency.
4. The invocation recipe for ${MODEL_ID} (client, base_url, key variable) and the limits the runner
   must set (thinking cap, output cap, seed) — and any GAP the --model report shows.
5. The exact local smoke-test command (a --limit flag or the page's equivalent).
6. Whether reasoning visibility for this benchmark is resolved from its README, and how.`,
  { agentType: 'summarizer', phase: 'Brief', schema: {
    type: 'object', additionalProperties: false,
    required: ['page_exists', 'local_dir', 'copy_from', 'recipe_gap', 'smoke_command', 'detail'],
    properties: {
      page_exists: { type: 'boolean' },
      page_path: { type: 'string' },
      local_dir: { type: 'string' },
      core_module: { type: 'string' },
      scorer: { type: 'string' },
      copy_from: { type: 'string' },
      recipe_gap: { type: 'string', description: 'what is missing from model-calls.md / model-parameters.md, or "none"' },
      limits_to_set: { type: 'array', items: { type: 'string' } },
      smoke_command: { type: 'string' },
      reasoning_visibility: { type: 'string' },
      detail: { type: 'string' },
    },
  } },
)

if (!brief) return { outcome: 'aborted', reason: 'the brief returned nothing; nothing was changed' }
if (!brief.page_exists) {
  return { outcome: 'needs-analysis', next: `write the benchmark page first — ${skill('analyse-benchmark')}`, brief }
}

// ---------- 2. Write ----------
phase('Write')
const BRIEF_TEXT = JSON.stringify(brief, null, 2)
let written = await agent(
  `Write ${MODEL}'s runner by following ${skill('write-runner')} steps 1-6. Read the skill first.
${CONTEXT}
The brief (verify anything you rely on):
${BRIEF_TEXT}

Copy ${brief.copy_from} rather than starting from nothing. If the brief names a recipe gap
(${brief.recipe_gap}), fill it in model-calls.md / model-parameters.md as part of this change.
Touch only ${MODEL}/ and those references — the shared core only if the brief says the scorer is
not yet in it, and then say so loudly, because it changes every model of ${BENCH}.
Finish with the smoke test and paste its rows' key fields.`,
  { agentType: 'executor', phase: 'Write', schema: {
    type: 'object', additionalProperties: false,
    required: ['files_changed', 'compiled', 'smoke_rows', 'smoke_empty', 'smoke_scored', 'detail'],
    properties: {
      files_changed: { type: 'array', items: { type: 'string' } },
      compiled: { type: 'boolean' },
      smoke_rows: { type: 'integer' },
      smoke_empty: { type: 'integer' },
      smoke_scored: { type: 'boolean' },
      limits_set: { type: 'array', items: { type: 'string' } },
      shared_core_changed: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } },
)

if (!written || !written.compiled || written.smoke_rows === 0) {
  return { outcome: 'blocked', reason: 'the runner did not compile or the smoke test wrote no rows', brief, written }
}

// ---------- 3. Review ----------
phase('Review')
let review = null
for (let round = 1; round <= 2; round++) {
  review = await agent(
    `Review the change to ${MODEL} before anyone spends cluster time on it. Assume the author's report
is optimistic. Use \`git diff\` and \`git status\` on: ${JSON.stringify(written.files_changed)}.
${CONTEXT}
The author reports: compiled=${written.compiled}, smoke rows=${written.smoke_rows},
empty=${written.smoke_empty}, scored=${written.smoke_scored}, shared core changed=${!!written.shared_core_changed}.

Check against .claude/references/script-skeleton.md (the invariants table) and the parameter rules in
.claude/references/model-parameters.md — every limit set, the scorer imported from the core and not
copied, empties retried, timeouts deriving from BaseException, a shard tag on every artefact. Run the
smoke test yourself if you doubt it; never call a provider any other way.`,
    { agentType: 'reviewer', phase: 'Review', label: `review:round${round}`, schema: {
      type: 'object', additionalProperties: false,
      required: ['status', 'blockers', 'detail'],
      properties: {
        status: { type: 'string', enum: ['safe-to-run', 'needs-change', 'unsafe'] },
        blockers: { type: 'array', items: { type: 'string' } },
        detail: { type: 'string' },
      },
    } },
  )
  if (!review || review.status !== 'needs-change' || round === 2) break
  written = await agent(
    `The reviewer found problems in ${MODEL}'s runner. Fix exactly these, re-compile and re-run the
smoke test (${skill('write-runner')} step 6). Do not widen the change.
${CONTEXT}
Blockers:
${review.blockers.map((b, i) => `${i + 1}. ${b}`).join('\n')}`,
    { agentType: 'executor', phase: 'Review', label: `fix:round${round}`, schema: {
      type: 'object', additionalProperties: false,
      required: ['files_changed', 'compiled', 'smoke_rows', 'smoke_empty', 'smoke_scored', 'detail'],
      properties: {
        files_changed: { type: 'array', items: { type: 'string' } },
        compiled: { type: 'boolean' },
        smoke_rows: { type: 'integer' },
        smoke_empty: { type: 'integer' },
        smoke_scored: { type: 'boolean' },
        detail: { type: 'string' },
      },
    } },
  ) || written
}

if (!review || review.status !== 'safe-to-run') {
  return { outcome: 'needs-change', blockers: review?.blockers || ['the reviewer returned no verdict'], brief, written, review }
}

// ---------- 4. Commit ----------
phase('Commit')
const committed = await agent(
  `Close this change with ${skill('sync-pass')} — layers 1 and 3 only. Layer 2 does NOT apply yet: the
user verifies the scripts before anything reaches Quest.
${CONTEXT}
Stage exactly these paths and nothing else: ${JSON.stringify(written.files_changed)}.
Commit message: what was added for ${MODEL_ID} on ${BENCH}, the smoke result, and "reviewer: safe-to-run".`,
  { agentType: 'executor', phase: 'Commit', schema: {
    type: 'object', additionalProperties: false,
    required: ['check_passed', 'commit', 'pushed', 'detail'],
    properties: {
      check_passed: { type: 'boolean' },
      commit: { type: 'string' },
      pushed: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } },
)

return {
  outcome: 'ready-for-user',
  model: MODEL,
  files: written.files_changed,
  smoke: { rows: written.smoke_rows, empty: written.smoke_empty, scored: written.smoke_scored },
  review: review.status,
  commit: committed?.commit,
  next: 'the user verifies the scripts; then launch-run with userVerified: true',
}
