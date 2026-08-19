export const meta = {
  name: 'compare-providers',
  description: 'Run the same NegotiationToM pilot on two providers on Quest and table the speed difference',
  whenToUse: 'The same model is available from two providers and the question is which is actually faster on real data. Pass {a, b} as two {dir, script, label} objects plus {rows}. Verifies the shared core before transferring, gates on that verification, then launches both jobs and compares them from their own logs.',
  phases: [
    { title: 'Verify' },
    { title: 'Sync' },
    { title: 'Launch' },
    { title: 'Supervise' },
    { title: 'Compare' },
  ],
}

// args arrives as the serialised form when the caller passes an object; three launches were lost to
// this on 2026-08-04 before the cause was clear.
let a
try {
  a = typeof args === 'string' ? JSON.parse(args) : args
} catch (e) {
  log(`args was a string but not JSON: ${e.message}`)
  return { error: 'unparseable args' }
}
if (!a || !a.a || !a.b) {
  return { error: 'need {a:{dir,script,label}, b:{dir,script,label}}; got ' + JSON.stringify(a) }
}
for (const side of [a.a, a.b]) {
  if (!side.dir || !side.script || !side.label) {
    return { error: 'each side needs dir, script and label; got ' + JSON.stringify(side) }
  }
}

const REPO = '/Users/harrychen/SONIC/hai-teams'
const BENCH = `${REPO}/Interpersonal_processes_benchmarks/NegotiationToM`
const QUEST = '/gpfs/projects/p32983/NegotiationToM'
const ROWS = a.rows || 500
const MAX_WAIT_MIN = a.maxWaitMin || 240

// Repeated into every prompt. An agent that has not been told these will helpfully do them.
const RULES = `
HARD RULES — breaking any of these invalidates the run:
- Do NOT call any model provider API directly (no curl/python against Together, DeepInfra, Google,
  OpenAI). The only provider traffic in this workflow comes from the SLURM jobs themselves.
- Do NOT run sbatch or scancel unless your task explicitly says to. Only the Launch phase submits.
- Do NOT edit any file outside the ones your task names.
- Do NOT 'git add -A' and do NOT commit or push anything.
- Only touch uwr0681's own folders under /projects/p32983.
- Report what you actually observed. If a command failed, say so and paste the output; never
  describe an intended result as an achieved one.`

const SIDES = [a.a, a.b]

// ---------------------------------------------------------------- Verify
phase('Verify')

const VERDICT = {
  type: 'object',
  required: ['safe_to_transfer', 'summary', 'findings'],
  properties: {
    safe_to_transfer: { type: 'boolean' },
    summary: { type: 'string' },
    findings: { type: 'array', items: { type: 'string' } },
  },
}

const checks = await parallel([
  () => agent(`${RULES}

Review ${BENCH}/${a.b.dir}/gemma_di_neg_eval.py — a NEW runner that has never run on Quest — for the
ways it fails silently. Do NOT call any provider API. Reading, grepping, git, and local python that
does no network IO are all fine.

Check it against the contract the other five runners follow, which is written up in
${REPO}/.claude/references/script-skeleton.md. Read that first. In particular:
  - Is an HTTP 200 with an empty body retried rather than written as a scored zero?
  - Does every except block log the exception AND call halt_on_billing?
  - Is finish_reason passed to record_call, so truncation is measured rather than assumed absent?
  - Is max_tokens sized from something measured, and can a truncated answer be told from a short one?
  - The <think>-tag stripping regex: can it eat part of a legitimate answer, or leave a partial tag
    that breaks parse_json? What happens on an unclosed tag?
  - It claims DeepInfra needs no reasoning-disable parameter. Compare against how NEG_Gemma and
    NEG_Gemini disable theirs. If the claim is wrong the run silently pays for hidden reasoning.
  - set_call_timeout(120) versus the client timeout=300: which fires first, and is that the
    intended order?

Then check the two SLURM scripts, ${BENCH}/${a.a.dir}/${a.a.script} and
${BENCH}/${a.b.dir}/${a.b.script}. They exist to be compared, so any asymmetry between them is a
defect: same --pilot-frac, same --task, same --save-every, same partition and resources. Confirm
that run_cli's fixed pilot seed really does make both select the SAME dialogues — read the pilot
selection code rather than trusting the comment.

safe_to_transfer=false for anything that would corrupt data, hide a failure, or make the two sides
non-comparable. Cosmetic issues go in findings without blocking.`,
    { label: 'review-runner', phase: 'Verify', schema: VERDICT }),

  () => agent(`${RULES}

Confirm the shared core is clean and that this experiment cannot damage the other five runners.

A concurrency refactor of ${BENCH}/neg_eval_core.py was written, reviewed, found to have defects
that only bite above concurrency=1, and then REVERTED. The work is parked at
${REPO}/.claude/patches/concurrency-wip.patch and is NOT part of this run. Your job is to prove the
revert was complete and that nothing else got swept along.

  1. cd ${REPO} && git status --porcelain and git diff --stat. neg_eval_core.py must show NO
     uncommitted changes. If it does, that is a blocking finding — say exactly what differs.
  2. Confirm neg_eval_core.py has no _run_items, no _pending, no --concurrency flag, no threading
     import, and that PRICE_PER_1M is back to empty (a stray "google/gemma-4-31B-it" entry there
     would make Together pilots print DeepInfra's prices as their own).
  3. Confirm ${BENCH}/${a.b.dir}/gemma_di_neg_eval.py imports ONLY names that exist in the current
     core — import it in a python that has pandas and sklearn and check it does not raise. Do not
     execute a run; importing the module is enough. If no local python has pandas/sklearn, say so
     and check the imports statically instead of skipping the check.
  4. List every file this experiment intends to send to Quest and confirm none of them is a shared
     file with unrelated local edits. NEG_GPT/run_negotiation.sh is known to differ from Quest for
     unrelated reasons — it must NOT be transferred, and its presence in git status is expected.
  5. Verify the parked patch is a real, complete diff (git apply --check it against HEAD) so the
     concurrency work is genuinely recoverable rather than lost.

safe_to_transfer=false if the core is not clean, if the runner cannot import, or if the patch does
not apply.`,
    { label: 'core-clean', phase: 'Verify', schema: VERDICT }),
])

// A checker that dies (agent() returns null on a terminal API error after retries) is NOT the same
// as a checker that refuses, and conflating them cost a diagnosis: run 4 returned
// blocked_at_verify with checks:[] after both agents hit a 529, which reads as "the code has
// problems" when in fact nothing was ever inspected. Separate statuses, because the responses
// differ — a refusal means fix the code, an infrastructure failure means run it again unchanged.
const returned = checks.filter(Boolean)
if (returned.length < 2) {
  log(`Verify could not run: ${2 - returned.length} of 2 checkers returned nothing (agent error, ` +
      `not a verdict). Nothing transferred, nothing submitted. Safe to re-run unchanged.`)
  return {
    status: 'verify_did_not_run',
    checks: returned,
    note: 'Checkers failed to return — an infrastructure error, not a verification refusal. ' +
          'No code defect is implied. Re-run the workflow as-is.',
  }
}

const bad = returned.filter(c => !c.safe_to_transfer)
if (bad.length) {
  log(`Verify refused: ${bad.length} blocking verdict(s). Nothing transferred, nothing submitted.`)
  return {
    status: 'blocked_at_verify',
    checks: returned,
    blocking: bad.map(c => c.summary),
    note: 'A checker judged the change unsafe. Nothing was transferred and no job was submitted.',
  }
}
log('Verify passed on both checks — proceeding to sync.')

// ---------------------------------------------------------------- Sync
phase('Sync')

const SYNC = {
  type: 'object',
  required: ['in_sync', 'summary', 'transferred', 'key_present'],
  properties: {
    in_sync: { type: 'boolean' },
    key_present: { type: 'boolean' },
    summary: { type: 'string' },
    transferred: { type: 'array', items: { type: 'string' } },
  },
}

const sync = await agent(`${RULES}

Bring Quest into sync with local for this comparison, then prove it.

1. Run: cd ${REPO} && python3 .claude/scripts/check_quest_sync.py
   Record what it reports before you change anything.

2. Transfer these local -> Quest (${QUEST}), creating remote directories as needed:
     - Interpersonal_processes_benchmarks/NegotiationToM/${a.b.dir}/gemma_di_neg_eval.py   (new; the directory does not exist on Quest)
     - Interpersonal_processes_benchmarks/NegotiationToM/${a.b.dir}/${a.b.script}
     - Interpersonal_processes_benchmarks/NegotiationToM/${a.a.dir}/${a.a.script}          (new pilot script for the Together side)
   Do NOT transfer neg_eval_core.py: its local copy was just reverted to HEAD, so the two sides
   should already match. VERIFY that by md5 and report it — if the core differs, stop and say so
   rather than transferring, because a core mismatch means something unexpected changed.
   Do NOT transfer NEG_GPT/run_negotiation.sh — it is out of sync for unrelated reasons and is not
   part of this experiment. Leave it alone and say so.

3. Verify every transferred file with md5 on both sides and show the comparison. Print the row
   count of both sides before trusting any diff — an empty-vs-empty comparison reports "in sync".

4. DEEPINFRA_API_KEY was appended to ${QUEST}/.env by hand before this run and verified to load.
   Re-confirm it rather than trusting that, and confirm TOGETHER_API_KEY too, by loading them
   through the Quest interpreter rather than by grepping the file:
     ssh quest '/projects/p32983/pythonenvs/hai-teams/bin/python -c "import os; from dotenv import load_dotenv; load_dotenv(\\"${QUEST}/.env\\"); print([k for k in (\\"DEEPINFRA_API_KEY\\",\\"TOGETHER_API_KEY\\") if os.getenv(k)])"'
   Report presence only — never print a key value, not even a prefix, into your report.
   If either is missing, set key_present=false and stop; do not try to repair it yourself.

5. Confirm the Quest interpreter can import what both runners need:
     ssh quest '/projects/p32983/pythonenvs/hai-teams/bin/python -c "import openai, together, pandas, sklearn, dotenv; print(\\"ok\\")"'
   If anything is missing the runner cannot start — report in_sync=false rather than installing it.

in_sync=true only if every intended file matches by md5 AND both API keys are present AND the
imports succeed.`, { label: 'sync', phase: 'Sync', schema: SYNC })

if (!sync || !sync.in_sync) {
  log('Sync failed — not submitting.')
  return { status: 'blocked_at_sync', sync, checks }
}

// ---------------------------------------------------------------- Launch
phase('Launch')

const LAUNCH = {
  type: 'object',
  required: ['submitted', 'jobs'],
  properties: {
    submitted: { type: 'boolean' },
    summary: { type: 'string' },
    jobs: {
      type: 'array',
      items: {
        type: 'object',
        required: ['label', 'job_id', 'dir'],
        properties: {
          label: { type: 'string' }, job_id: { type: 'string' }, dir: { type: 'string' },
          archived: { type: 'string' },
        },
      },
    },
  },
}

const launch = await agent(`${RULES}
EXCEPTION: this phase is the one that may run sbatch. It may NOT run scancel.

Submit both pilots on Quest. Both write ~${ROWS} rows total across desire/belief/intention using
--pilot-frac 0.035, and run_cli's fixed seed means both see the SAME dialogues.

For each side:
${SIDES.map(s => `  - ${s.label}: cd ${QUEST}/${s.dir} && sbatch ${s.script}`).join('\n')}

BEFORE submitting each one, archive any existing results/pilot/ directory ON QUEST in that folder to
results_archive_pilot500_<UTC timestamp>/, then PROVE it is gone.

This is the single most likely way this experiment produces a confident wrong answer, so treat it as
a gate rather than a chore. ${a.a.dir} is known to hold an earlier 10% pilot. The 3.5% selection is
a strict SUBSET of it (run_cli shuffles with a fixed seed and takes a prefix), so a resume would
skip most items: measured locally, 142 of 166 desire, 50 of 166 belief and ALL 161 of 161 intention
uids are already present. That job would issue a fraction of the calls, finish far too fast, and its
elapsed time would be compared against a full DeepInfra run. Worse, write_task_outputs scores
everything load_checkpoint returned, not just new rows, so the accuracy columns would compare 464
rows against 161.

After archiving, verify and report for BOTH sides:
    ssh quest 'ls -la ${QUEST}/<dir>/results/pilot/ 2>&1; find ${QUEST}/<dir>/results/pilot -name "*.jsonl" 2>/dev/null | xargs wc -l 2>/dev/null'
Every side must show NO .jsonl files under results/pilot/ before its sbatch runs. If any remain,
set submitted=false and stop — do not submit and then mention it.

Also delete any stale *_HALT.txt markers in both directories before submitting.

After submitting, run squeue for the account and paste it, so the job ids are corroborated by
something other than the sbatch output.

Report submitted=false if either sbatch failed.`, { label: 'launch', phase: 'Launch', schema: LAUNCH })

if (!launch || !launch.submitted || !launch.jobs || launch.jobs.length < 2) {
  return { status: 'launch_failed', launch, sync, checks }
}
log(`Submitted: ${launch.jobs.map(j => `${j.label}=${j.job_id}`).join('  ')}`)

// ---------------------------------------------------------------- Supervise
phase('Supervise')

const WATCH = {
  type: 'object',
  required: ['state', 'summary'],
  properties: {
    state: { type: 'string', enum: ['both_done', 'partial', 'stalled', 'failed', 'timeout'] },
    summary: { type: 'string' },
    per_job: { type: 'array', items: { type: 'string' } },
  },
}

const watch = await agent(`${RULES}
You are READ-ONLY. Do not sbatch, do not scancel, do not edit, do not transfer.

Watch these two Quest jobs until both finish or ${MAX_WAIT_MIN} minutes elapse:
${launch.jobs.map(j => `  - ${j.label}: job ${j.job_id} in ${QUEST}/${j.dir}`).join('\n')}

Poll with squeue and by tailing each job's log. Space your polls out — every 2-5 minutes is plenty;
do not spin. Between polls, report progress by counting rows in the checkpoint .jsonl files under
each results/pilot/<task>/ directory.

Watch for the failure modes this project has actually hit:
  - a job that is RUNNING but whose row count has not moved in 15+ minutes (a hung call)
  - a *_HALT.txt marker appearing (billing, quota or failure-rate halt)
  - rows arriving with empty raw_response
  - a job that finished suspiciously fast, which means it resumed a stale checkpoint

state='both_done' only when both jobs have left the queue AND each has written its
results/pilot/<stem>_negotiation_overall.csv. Use 'stalled' if a job is alive but not progressing,
'failed' if a job died or halted, 'timeout' if the ${MAX_WAIT_MIN} minutes ran out.`,
  { label: 'watch', phase: 'Supervise', schema: WATCH })

// ---------------------------------------------------------------- Compare
phase('Compare')

const TABLE = {
  type: 'object',
  required: ['table_markdown', 'headline', 'caveats'],
  properties: {
    table_markdown: { type: 'string' },
    headline: { type: 'string' },
    caveats: { type: 'array', items: { type: 'string' } },
    per_provider: { type: 'array', items: { type: 'object' } },
  },
}

const table = await agent(`${RULES}
You are READ-ONLY apart from writing the one report file named at the end.

Build the speed comparison between ${a.a.label} and ${a.b.label} for gemma-4-31B-it, from the two
pilot runs just completed on Quest${watch && watch.state !== 'both_done' ? ` (NOTE: the watcher reported state='${watch.state}' — ${watch.summary}. Work with whatever data exists and say plainly what is missing)` : ''}.

Pull the numbers from each job's own log and result files under ${QUEST}/<dir>/results/pilot/:
the [pulse] lines carry mean latency, effective s/row, rows/min and latency p90/p99/max; the
pilot_report block at the end carries elapsed, token counts and the health counters.

The table must have one row per provider and cover, per task and overall:
  - rows completed, and whether both providers answered the SAME item set (compare the uid sets in
    the .jsonl files — if they differ, the speed comparison is not like-for-like and you must say so)
  - mean / median call latency, p90, p99, max
  - effective s/row and rows/min  (effective s/row is the honest one: it includes retries and hangs,
    which mean latency hides)
  - empty responses, API errors, timeouts, parse failures
  - total wall clock for the ~${ROWS} rows, and the extrapolation to the full 14,140-row run
  - accuracy per task (Desire_EM, Belief_EM, Intent_Micro_F1) from the *_overall.csv — NOT to rank
    the providers on quality, but because a provider that is fast because it answered badly is not
    faster. Flag any gap above a few points.

Then state the headline: which is faster, by how much, and whether the difference is in the mean or
in the tail.

Required caveats — include each only if true, and check whether it is:
  - sample size, and that these are single runs (no repeat, so no variance estimate)
  - Together serves this checkpoint FP8 and DeepInfra bf16, so the two are not numerically identical
    models and the accuracy columns are not a clean comparison
  - both ran sequentially (--concurrency 1), so neither number reflects DeepInfra's 200-concurrent
    ceiling or Together's per-shard sharding
  - any provider-side variability you can see in the logs (time of day, rate limiting, retries)

Write the finished report to ${BENCH}/NEG_Gemma_DeepInfra/SPEED_COMPARISON.md, and also return the
table as table_markdown. Do not modify any other file and do not commit.`,
  { label: 'compare', phase: 'Compare', schema: TABLE })

return {
  status: watch && watch.state === 'both_done' ? 'complete' : `finished_with_${watch ? watch.state : 'unknown'}`,
  jobs: launch.jobs,
  watch,
  headline: table && table.headline,
  table: table && table.table_markdown,
  caveats: table && table.caveats,
  report: `${BENCH}/NEG_Gemma_DeepInfra/SPEED_COMPARISON.md`,
}
