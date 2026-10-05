export const meta = {
  name: 'monitor-run',
  description: 'Baseline: read-only health, finish time and cost of one or more live runs of one benchmark, with a recommendation',
  whenToUse: 'Phase 4. A job (or several jobs of the same benchmark) is running on Quest and the question is "is it in trouble, and when will it finish" — plus "which is cheaper/faster" when there are several. Pass {benchmark, model, jobId} for one run, or {benchmark, runs:[{label, dir, jobId}]} for several. Strictly read-only: never submits, cancels, edits or transfers, so it is safe to repeat on a timer. A baseline — adapt the phases to the task.',
  phases: [
    { title: 'Observe', detail: 'one watcher per run — queue, rows, process state, empties, errors (skill: check-run)' },
    { title: 'Judge', detail: 'evaluator — verdict per run, finish time, cost, recommendation (skill: audit-results)' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   benchmark:    "emobench"                        required — the page stem
//   model, jobId                                    one run, shorthand
//   runs: [ { label, dir, jobId,                     several runs
//             resultsDir: "results_eLow",          optional — sweep arms of ONE model share a folder
//             priceIn: 0.30, priceOut: 2.50 } ]     optional — $/M tokens; omit rather than guess
//   questDir:     "/gpfs/projects/p32983/..."       optional — otherwise taken from the page
//   expected:     { EU: 200, EA: 200 }              optional — otherwise taken from the page
//   sinceMinutes: 0                                 optional — rate over the last N minutes only
// }
// ---------------------------------------------------------------------------

let A = args || {}
if (typeof A === 'string') {
  try {
    A = JSON.parse(A)
  } catch (error) {
    return { status: 'cannot-tell', aborted: `args arrived as a string that is not JSON: ${error.message}` }
  }
}

const BENCH = A.benchmark || ''
const RUNS = Array.isArray(A.runs) && A.runs.length
  ? A.runs
  : (A.model ? [{ label: A.model, dir: A.model, jobId: A.jobId }] : [])
const SINCE = A.sinceMinutes || 0

if (!BENCH) return { status: 'cannot-tell', aborted: 'args.benchmark is required' }
if (RUNS.length < 1) return { status: 'cannot-tell', aborted: 'pass {model, jobId} or a non-empty runs array' }
for (const r of RUNS) {
  if (!r || !r.label || !r.dir) {
    return { status: 'cannot-tell', aborted: `every run needs {label, dir}; got ${JSON.stringify(r)}` }
  }
}
// Two arms reading the same path would be reported as agreeing rather than as misconfigured.
const paths = RUNS.map((r) => `${r.dir}/${r.resultsDir || 'results'}`)
if (new Set(paths).size !== paths.length) {
  return { status: 'cannot-tell', aborted: `two runs point at the same results directory (${paths.join(', ')}); sweep arms of one model need distinct resultsDir values` }
}

const skill = (name) => `.claude/skills/${name}/SKILL.md`
const WHERE = A.questDir
  ? `Quest directory: ${A.questDir}.`
  : `The Quest directory is on the benchmark's page — the one file matching \`ls .claude/references/benchmarks/*/${BENCH}.md\`. Never infer it.`
const COUNTS = A.expected
  ? `Expected rows per task, per run: ${JSON.stringify(A.expected)}.`
  : `Expected rows per task are on the benchmark's page; quote the number you used.`

const READONLY = `
THIS WORKFLOW IS READ-ONLY. Breaking that is a failed task, not a judgement call:
- NO sbatch, scancel, scontrol, file edits, transfers, git commits, and NO provider API calls of any
  kind. These jobs are live and spending real quota.
- \`srun --jobid=<id> --overlap <cmd>\` is allowed, to read /proc on the compute node only.
- An empty log is not evidence of a stall: these runners do not flush stdout. Judge from rows
  written to the .jsonl, CPU time and wchan.
`

// ---------- 1. Observe ----------
phase('Observe')
const observations = await parallel(RUNS.map((run) => () => agent(
  `Report what the ${BENCH} run "${run.label}" is doing right now, following ${skill('check-run')}.
Facts only, no advice.
${WHERE} Run folder: ${run.dir}; results under ${run.dir}/${run.resultsDir || 'results'}.
SLURM job id: ${run.jobId === undefined ? '(not supplied — find it by name in squeue)' : run.jobId}
${COUNTS}
${READONLY}
Rates ${SINCE ? `over the last ${SINCE} minutes` : 'since the job started'}; report the window you
divided by. Print one real row's keys before counting any field. Also report, where the rows record
them: thinking tokens (min/median/max), distinct reasoning-visibility values (more than one is a
finding), distinct served-by / backend values, and whether per-call usage (prompt and completion
tokens, provider cost) is recorded at all.`,
  { agentType: 'watcher', label: `observe:${run.label}`, phase: 'Observe' },
)))

const seen = observations.filter(Boolean)
if (seen.length === 0) return { status: 'cannot-tell', aborted: 'every observer failed; nothing was measured' }

// ---------- 2. Judge ----------
phase('Judge')
const priceTable = RUNS.map((r) => `  ${r.label}: ` + (
  r.priceIn === undefined && r.priceOut === undefined
    ? 'prices NOT supplied — report its cost as not established'
    : `$${r.priceIn}/M input, $${r.priceOut}/M output`
)).join('\n')

const judgement = await agent(
  `${RUNS.length === 1 ? 'One run' : `${RUNS.length} runs, side by side,`} of ${BENCH} ${RUNS.length === 1 ? 'is' : 'are'} executing.
Judge each from what its observer measured, using the checks in ${skill('audit-results')} that
apply to a run still in progress.
${READONLY}
Prices, as supplied by the caller:
${priceTable}

Observations:
${seen.map((o, i) => `----- ${RUNS[i] ? RUNS[i].label : 'run ' + i} -----\n${o}`).join('\n\n')}

Produce, in this order:
1. Per-run verdict — label, rows/expected, rows/min, health, and the one fact that decides it.
   ${RUNS.length > 1 ? 'If two runs differ in throughput by more than 2x, say which is slower and what explains it — or "not established".' : ''}
2. Finish time — elapsed, rows done, rows/min, projected remaining and total, and the assumption.
3. Cost — measured where per-call usage is recorded; otherwise thinking tokens per row, and any
   derived figure labelled "derived, not measured" with its assumptions. A gap stays a gap.
4. What needs a decision — a stall, MAX_TOKENS empties, mixed conditions in one result set, a
   backend switch, a rate that cannot finish inside the walltime.
5. For each open question, the observation that would settle it.`,
  { agentType: 'evaluator', label: 'judge', phase: 'Judge' },
)

const verdicts = {}
seen.forEach((o, i) => {
  const m = String(o).match(/STATUS:\s*([a-z-]+)\s*$/im)
  verdicts[RUNS[i] ? RUNS[i].label : 'run' + i] = m ? m[1] : 'cannot-tell'
})
const overall = String(judgement || '').match(/STATUS:\s*([a-z-]+)\s*\/\s*([a-z-]+)/i)

return {
  benchmark: BENCH,
  perRun: verdicts,
  status: overall ? overall[1] : 'cannot-tell',
  recommendation: overall ? overall[2] : 'needs-human',
  report: judgement,
  next: overall && /kill|prune/.test(overall[2]) ? 'fix-run, with the fix decided first' : undefined,
}
