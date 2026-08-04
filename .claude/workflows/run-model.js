export const meta = {
  name: 'run-model',
  description: 'Full lifecycle for one model: check local, sync to Quest, launch, gate on the first minutes, supervise with hourly local+git sync, audit the result',
  whenToUse: 'Starting or restarting a pilot or full run for one model, end to end. Pass {model, reason} plus optional {pilot, gateMinutes, syncIntervalMin, maxHours, push}. It owns both sync directions — local<->Quest and local<->git — because a run whose code never reached Quest and a result that lives only on the cluster are the two ways this project has actually lost work. Use fix-broken-run instead when a job is already running and needs killing.',
  phases: [
    { title: 'Check local',  detail: 'reviewer: is the code fit to run at all' },
    { title: 'Sync to Quest', detail: 'executor: transfer and prove Quest matches local' },
    { title: 'Launch',        detail: 'executor: submit, nothing else' },
    { title: 'Gate',          detail: 'watcher: is it actually working, judged on rows not on RUNNING' },
    { title: 'Supervise',     detail: 'per cycle: pull results to local, commit and push, re-check health' },
    { title: 'Audit',         detail: 'evaluator: are the finished numbers usable, and what must change if not' },
    { title: 'Record',        detail: 'tracker: outcome into ISSUES.md' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   model:           "NEG_Gemma"   required
//   reason:          "..."         required — why this run is happening
//   pilot:           true|false    default false
//   gateMinutes:     5             how long to watch before trusting the launch
//   syncIntervalMin: 60            how often to pull results down and push them to git
//   maxHours:        24            give up supervising after this; the job keeps running
//   push:            true          commit and push each sync cycle; false pulls only
// }
//
// `args` can arrive as a JSON *string* rather than an object — the caller writes an object but the
// tool hands the script the serialised form, and `args.model` is then silently undefined, so the
// guard below aborts a launch that looked correct. Three launches were lost to this before it was
// understood. Normalise first, and treat a string that will not parse as a hard error rather than
// letting it fall through to the "model is required" message, which points at the wrong problem.
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
const IS_PILOT = !!A.pilot
const GATE_MIN = A.gateMinutes || 5
const SYNC_MIN = A.syncIntervalMin || 60
const MAX_HOURS = A.maxHours || 24
const DO_PUSH = A.push !== false

if (!MODEL) {
  return { aborted: 'args.model is required, e.g. {model: "NEG_Gemma", reason: "..."}' }
}

const REPO = '/Users/harrychen/SONIC/hai-teams'
const LOCAL = REPO + '/NegotiationToM'
const QDIR = '/gpfs/projects/p32983/NegotiationToM'
const SUBMIT = IS_PILOT ? 'run_pilot.sh' : 'run_negotiation.sh'
const RESULTS = IS_PILOT ? 'results/pilot' : 'results'
const MAX_CYCLES = Math.max(1, Math.ceil((MAX_HOURS * 60) / SYNC_MIN))

const RULES = `
Target: ${MODEL} (${IS_PILOT ? 'pilot' : 'full run'}, submitted with ${SUBMIT})
Why this run: ${REASON}

HARD RULES — breaking one is a failed task, not a judgement call:
- Touch ${MODEL} and nothing else. Other models may be running. Confirm in your report that every
  other job is untouched.
- Do NOT call a model provider API directly. preflight.py is the one sanctioned exception, and only
  where a phase asks for it.
- Never overwrite .env on Quest, never copy it off.
- Under /projects/p32983 only uwr0681's directories are in scope.
- Quest is \`ssh quest\`; drop \`libcrypto\` lines. Remote ${QDIR}, local ${LOCAL}, repo ${REPO}.
- When staging for git, stage explicit paths. NEVER \`git add -A\`. An unattended loop that did
  swept unreviewed work into commits named "watcher checkpoint" and pushed them to both remotes.
`

// ---------- 1. Check local ----------
phase('Check local')
const local = await agent(
  `Decide whether ${MODEL}'s code is fit to run before anything is transferred or submitted.
${RULES}

Check and report a verdict for each:
1. \`python3 -m py_compile\` on ${LOCAL}/neg_eval_core.py and ${LOCAL}/${MODEL}/*.py, and
   \`bash -n\` on ${LOCAL}/${MODEL}/${SUBMIT}. A syntax error found on Quest costs a job slot.
2. Every name ${MODEL}'s runner imports from neg_eval_core actually exists there. This is the
   dependency that makes a partial sync fatal — the runner dies at import.
3. \`git status --porcelain\` — is there uncommitted work that this run should be using? Say what,
   do not commit it.
4. ${MODEL}/${SUBMIT}: account, partition, walltime, --shard/--total-shards, and that it invokes the
   right script. For a full run the shard tag must appear in the output filenames, or shards
   overwrite each other.
5. ${LOCAL}/${MODEL}/${RESULTS} — is there a checkpoint already? If so, how many rows, and were they
   produced by the current code? A stale checkpoint makes a run "succeed" in seconds while emitting
   old data.

Return ok=false if anything would waste a job slot or produce wrong data. This phase exists to fail
cheaply; do not wave something through because it is probably fine.`,
  { agentType: 'reviewer', phase: 'Check local', schema: {
    type: 'object', additionalProperties: false,
    required: ['ok', 'blockers', 'checks', 'detail'],
    properties: {
      ok: { type: 'boolean' },
      blockers: { type: 'array', items: { type: 'string' } },
      checks: { type: 'array', items: { type: 'string' } },
      existing_checkpoint_rows: { type: 'integer' },
      uncommitted: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } }
)

if (!local || !local.ok) {
  return { outcome: 'BLOCKED in local check', blockers: local?.blockers || ['no verdict'], local }
}

// ---------- 2. Sync to Quest ----------
phase('Sync to Quest')
const synced = await agent(
  `Make Quest match local for every code file, then prove it. Nothing is submitted in this phase.
${RULES}

1. \`cd ${REPO} && python3 .claude/scripts/check_quest_sync.py\`. It compares every *.py and
   NEG_*/*.sh and exits 1 on drift.
2. Transfer each file it names with \`ssh quest "cat > ${QDIR}/<path>" < ${LOCAL}/<path>\` and verify
   each with md5sum. **Send neg_eval_core.py together with the runners** — they import from it.
3. Re-run the checker and show it exiting 0, with the file count printed for both sides. A
   comparison over two empty file lists passes trivially; that exact false pass has happened here.
4. On Quest: py_compile the transferred files and confirm ${MODEL}'s imports resolve.
5. \`python3 preflight.py --only <provider>\` on Quest for this model only. One real API call proves
   the key, the model id and the call signature before a job slot is spent.

Return in_sync=false if any step fails. Do not submit anything.`,
  { agentType: 'executor', phase: 'Sync to Quest', schema: {
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
    local, synced,
  }
}

// ---------- 3. Launch ----------
phase('Launch')
const launched = await agent(
  `Submit ${QDIR}/${MODEL}/${SUBMIT} with sbatch from ${MODEL}'s own directory.
${RULES}

Before submitting, clear any stale halt marker in ${MODEL}/ so an old verdict is not mistaken for a
new one. Submit NOTHING ELSE — not another model, not a pilot alongside a full run. Then run squeue
and report the full queue, so the report itself proves nothing extra was started.`,
  { agentType: 'executor', phase: 'Launch', schema: {
    type: 'object', additionalProperties: false,
    required: ['job_id', 'queue_as_expected', 'detail'],
    properties: {
      job_id: { type: 'string' },
      queue_as_expected: { type: 'boolean' },
      other_jobs: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } }
)

if (!launched || !launched.job_id) {
  return { outcome: 'launch failed', local, synced, launched }
}
log(`${MODEL} launched as ${launched.job_id}; gating for ${GATE_MIN} min`)

// ---------- 4. Gate on the first minutes ----------
phase('Gate')
const gate = await agent(
  `Watch ${MODEL}'s job ${launched.job_id} for about ${GATE_MIN} minutes and decide whether it is
actually working. Observe only.
${RULES}

Poll every 60 seconds or so. SLURM reporting RUNNING proves nothing — a process hung inside an API
call looks identical. Judge on:
- rows appearing under ${MODEL}/${RESULTS}, and the log's mtime advancing (either one advancing means
  alive; a slow model can sit between checkpoint writes without being stuck)
- the empty-response and API-error counts in the log — a run can produce rows that are all failures
- any ${MODEL}/BILLING_HALT.txt, QUOTA_HALT.txt or FAILURE_HALT.txt, quoted in full
- whether the job left the queue already, and what sacct says its State and ExitCode were

Return healthy=false for: a halt marker, an early exit, zero rows AND a log that has not been
written for the whole window, or an empty rate above 20%. If ${GATE_MIN} minutes genuinely is not
long enough to tell for this model, say healthy="unclear" and state what would settle it.`,
  { agentType: 'watcher', phase: 'Gate', schema: {
    type: 'object', additionalProperties: false,
    required: ['healthy', 'rows', 'empty_count', 'detail'],
    properties: {
      healthy: { type: 'string', enum: ['yes', 'no', 'unclear'] },
      rows: { type: 'integer' },
      empty_count: { type: 'integer' },
      rate_per_min: { type: 'number' },
      halt_markers: { type: 'array', items: { type: 'string' } },
      failure_reason: { type: 'string' },
      detail: { type: 'string' },
    },
  } }
)

// 2.2 — the launch is bad. Stop it now rather than letting it burn to the wall.
if (!gate || gate.healthy === 'no') {
  phase('Gate')
  const killed = await agent(
    `${MODEL}'s job ${launched.job_id} failed its launch gate: ${gate?.failure_reason || 'no verdict returned'}.
Stop it and diagnose. Do not resubmit — that decision is not yours.
${RULES}

1. scancel ${launched.job_id}, that job only, and confirm every other job still runs.
2. Pull whatever it produced down to local so the evidence survives:
   \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\`
3. Diagnose from the logs and the rows: what exactly went wrong, quoting the error text. Say whether
   the checkpoint now holds rows that must be pruned before any resubmit — rows written empty are
   skipped forever by a plain resume, because load_checkpoint adds every uid to the done set.
4. Say what would have to change in the code or the configuration for the next attempt to work.`,
    { agentType: 'executor', phase: 'Gate', schema: {
      type: 'object', additionalProperties: false,
      required: ['cancelled', 'diagnosis', 'needs_prune', 'proposed_fix', 'detail'],
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
    outcome: 'FAILED THE LAUNCH GATE',
    model: MODEL,
    job: launched.job_id,
    diagnosis: killed?.diagnosis,
    needs_prune: killed?.needs_prune,
    proposed_fix: killed?.proposed_fix,
    needs_rerun: true,
    next: 'Apply the fix, then run this workflow again from phase 1.',
    local, synced, gate, killed,
  }
}

// ---------- 5. Supervise: hourly local sync + git sync ----------
phase('Supervise')
let finished = false
let lastCycle = null
const cycles = []

for (let i = 1; i <= MAX_CYCLES && !finished; i++) {
  const cycle = await agent(
    `Supervision cycle ${i} of at most ${MAX_CYCLES} for ${MODEL}, job ${launched.job_id}.
Wait about ${SYNC_MIN} minutes from when you start, checking every few minutes, then do the sync.
${RULES}

**This phase owns both sync directions. Neither is optional — each has already cost this project.**

A. **Quest -> local.** \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\` and report the row
   delta it prints. Results are generated on Quest and pulled down; never push results upward.
   ${DO_PUSH ? `
B. **local -> git.** Stage ONLY this model's results and logs by explicit path:
     git add NegotiationToM/${MODEL}/results NegotiationToM/${MODEL}/log_*.txt
   Never \`git add -A\` — an unattended loop that did swept unreviewed work into commits and pushed
   them. If nothing changed, say so and skip the commit rather than making an empty one. Commit with
   a message naming the model, the job id and the row counts, then push to \`backup\` and \`origin\`.
   Never push to \`upstream\`; it is the shared repo.` : `
B. Skipping git (push=false was requested). Say so explicitly in your report.`}

C. **Health.** Rows per task, rows/minute, empty and error counts, any halt marker quoted in full,
   and whether the job is still in the queue. If it has left, get its State and ExitCode from sacct.
   Judge by rows and log content, never by SLURM state alone.

Set finished=true when the job is no longer in the queue. Set stop_now=true if it is still running
but should be killed — a halt marker, an empty rate above 20%, or no progress at all across this
whole cycle.`,
    { agentType: 'executor', phase: 'Supervise', schema: {
      type: 'object', additionalProperties: false,
      required: ['finished', 'stop_now', 'rows_total', 'empty_count', 'pulled', 'detail'],
      properties: {
        finished: { type: 'boolean' },
        stop_now: { type: 'boolean' },
        rows_total: { type: 'integer' },
        rows_delta: { type: 'integer' },
        empty_count: { type: 'integer' },
        pulled: { type: 'boolean', description: 'results copied Quest -> local this cycle' },
        pushed: { type: 'boolean', description: 'committed and pushed to both remotes this cycle' },
        commit: { type: 'string' },
        halt_markers: { type: 'array', items: { type: 'string' } },
        job_state: { type: 'string' },
        detail: { type: 'string' },
      },
    } }
  )

  if (!cycle) {
    log(`cycle ${i} returned nothing — treating as a supervision failure and stopping the loop`)
    break
  }
  cycles.push({ cycle: i, rows: cycle.rows_total, delta: cycle.rows_delta,
                empty: cycle.empty_count, pulled: cycle.pulled, pushed: cycle.pushed })
  lastCycle = cycle
  log(`cycle ${i}: ${cycle.rows_total} rows (+${cycle.rows_delta ?? '?'}), empty=${cycle.empty_count}, pulled=${cycle.pulled}, pushed=${cycle.pushed}`)

  if (cycle.stop_now) {
    const killed = await agent(
      `${MODEL}'s job ${launched.job_id} must be stopped mid-run: ${JSON.stringify(cycle.halt_markers || [])}, ${cycle.detail?.slice(0, 300)}
${RULES}

scancel that job only, confirm the others still run, pull the results down one final time with
\`bash ${REPO}/.claude/scripts/pull_quest_results.sh\`, and diagnose. Say whether the checkpoint
needs pruning before any resubmit. Do not resubmit.`,
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
      outcome: 'STOPPED MID-RUN',
      model: MODEL, job: launched.job_id,
      diagnosis: killed?.diagnosis, needs_prune: killed?.needs_prune,
      proposed_fix: killed?.proposed_fix,
      needs_rerun: true,
      next: 'Apply the fix, then run this workflow again from phase 1.',
      cycles, local, synced, gate,
    }
  }
  finished = !!cycle.finished
}

if (!finished) {
  return {
    outcome: 'STILL RUNNING — supervision window ended',
    model: MODEL, job: launched.job_id,
    note: `Supervised for ${MAX_HOURS}h without the job finishing. It is still running on Quest and is unaffected by this workflow ending. Resume supervision by running this workflow's Supervise phase again, or rely on the background pull loop.`,
    cycles, last: lastCycle,
  }
}

// ---------- 6. Audit the finished run ----------
phase('Audit')
const audit = await agent(
  `${MODEL}'s job ${launched.job_id} has finished. Decide whether its numbers can be used.
${RULES}

Before interpreting any score, run the trustworthiness checks — every one of these has been wrong
here at least once while the job reported success:
- Row counts against expectation. Full run: desire 4,760, belief 4,760, intention 4,618. A count of
  4,760 for intention means the odd-length-dialogue bug is back. Pilot: 476 / 476 / ~440-462.
- Empty \`raw_response\` and null \`pred\` rate. These should be ~0. A run can exit COMPLETED with
  every row empty — that is exactly how eight hours were lost to an unpaid xAI invoice.
- sacct State and ExitCode. COMPLETED 0:0 is not proof; check it against the row and empty counts.
- Off-label predictions — values outside the canonical label set. Say whether it is a model problem
  or a normalisation gap.
- Whether the output could have come from a stale checkpoint: compare file mtimes against the code's.

Then:
1. If the data is usable, merge the shards with merge_neg_results.py (it now asserts the expected
   row count and will refuse a short result set) and report the headline metrics.
2. If it is not usable, say exactly what is wrong, whether the checkpoint needs pruning, and what
   must change before the next attempt. Set needs_rerun=true.

Also confirm the final results reached local, and say whether anything is still only on Quest.`,
  { agentType: 'evaluator', phase: 'Audit', schema: {
    type: 'object', additionalProperties: false,
    required: ['usable', 'needs_rerun', 'row_counts', 'empty_rate', 'detail'],
    properties: {
      usable: { type: 'boolean' },
      needs_rerun: { type: 'boolean' },
      row_counts: { type: 'object', additionalProperties: true },
      empty_rate: { type: 'number' },
      metrics: { type: 'object', additionalProperties: true },
      problems: { type: 'array', items: { type: 'string' } },
      needs_prune: { type: 'boolean' },
      proposed_fix: { type: 'string' },
      synced_to_local: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

// Step 5 of the loop: get the evidence home before anything else, usable or not.
phase('Audit')
const finalSync = await agent(
  `Get ${MODEL}'s finished output home and into git. Do this whether or not the numbers turned out
usable — a failed run's rows are the evidence for diagnosing it, and they exist only on Quest until
someone pulls them.
${RULES}

1. \`bash ${REPO}/.claude/scripts/pull_quest_results.sh\`, then verify: for each task, the local row
   count equals Quest's. Report both numbers; do not assert success without them.
${DO_PUSH ? `2. Stage ONLY this model's paths — \`git add NegotiationToM/${MODEL}/results NegotiationToM/${MODEL}/log_*.txt\` — never \`git add -A\`. Commit with a message naming the model, job ${launched.job_id}, the row counts, the empty rate, and whether the run is usable. Push to \`backup\` and \`origin\`, never \`upstream\`.` : '2. Skipping git (push=false). Say so explicitly.'}
3. State plainly whether anything of this run still exists only on Quest.`,
  { agentType: 'executor', phase: 'Audit', schema: {
    type: 'object', additionalProperties: false,
    required: ['local_matches_quest', 'anything_only_on_quest', 'detail'],
    properties: {
      local_matches_quest: { type: 'boolean' },
      pushed: { type: 'boolean' },
      commit: { type: 'string' },
      anything_only_on_quest: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)

// ---------- 7. Record ----------
phase('Record')
const recorded = await agent(
  `Record this run in NegotiationToM/ISSUES.md. Extend an existing entry if one covers the same root
cause; otherwise add one. House style: symptom, root cause, what was rejected, fix, measured
evidence. Be brief.

  model        ${MODEL} (${IS_PILOT ? 'pilot' : 'full run'}), job ${launched.job_id}
  why          ${REASON}
  gate         ${gate?.healthy} after ${GATE_MIN} min, ${gate?.rows} rows, ${gate?.empty_count} empty
  cycles       ${cycles.length} supervision cycles, ${cycles.filter(c => c.pushed).length} pushed to git
  result       ${audit?.usable ? 'usable' : 'NOT usable'}, rows ${JSON.stringify(audit?.row_counts)}, empty rate ${audit?.empty_rate}
  ${audit?.metrics ? 'metrics     ' + JSON.stringify(audit.metrics) : ''}
  ${audit?.problems?.length ? 'problems    ' + audit.problems.join('; ') : ''}
  sync         local matches Quest: ${finalSync?.local_matches_quest}; pushed: ${finalSync?.pushed}

If the run failed, the useful record is the root cause and what was ruled out — not the fact that it
failed. If it succeeded after an earlier attempt did not, say what the difference was, with numbers.`,
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
  mode: IS_PILOT ? 'pilot' : 'full',
  job: launched.job_id,
  gate: { healthy: gate?.healthy, rows: gate?.rows, empty: gate?.empty_count },
  cycles,
  usable: audit?.usable,
  metrics: audit?.metrics,
  row_counts: audit?.row_counts,
  empty_rate: audit?.empty_rate,
  problems: audit?.problems,
  needs_rerun: !!audit?.needs_rerun,
  needs_prune: audit?.needs_prune,
  proposed_fix: audit?.proposed_fix,
  sync: {
    local_matches_quest: finalSync?.local_matches_quest,
    pushed: finalSync?.pushed,
    anything_only_on_quest: finalSync?.anything_only_on_quest,
  },
  issues_updated: recorded?.updated,
  next: audit?.needs_rerun
    ? 'Apply proposed_fix (prune first if needs_prune), then run this workflow again from phase 1.'
    : 'Done. Results are in git.',
}
