export const meta = {
  name: 'watch-live-runs',
  description: 'Read-only health check across several runs of one benchmark at once, with a cost and wall-clock comparison between them',
  whenToUse: 'Two or more jobs of the same benchmark are running side by side and the question is both "is either in trouble" and "which is cheaper/faster". Pass {benchmark, questDir, runs:[{label,dir,jobId}], expected}. Strictly read-only — never submits, cancels, edits or transfers, so it is safe beside a supervising workflow and cheap enough to repeat on a timer. Use check-status for a single model on NegotiationToM; use compare-providers to launch a fresh pilot rather than watch one already running.',
  phases: [
    { title: 'Observe', detail: 'one watcher per run: queue, rows, process state, empties, retries' },
    { title: 'Compare', detail: 'evaluator: health verdict per run, then cost and finish-time table' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   benchmark:  "EmoBench"                                        required
//   questDir:   "/gpfs/projects/p32983/..../EmoBench"             required
//   runs: [                                                       required, 2+
//     { label: "Google",     dir: "EMO_..._Google",     jobId: 3810331,
//       priceIn: 0.10, priceOut: 0.40 },        // $/M tokens, omit if not established
//     { label: "OpenRouter", dir: "EMO_..._OpenRouter", jobId: 3810332,
//       priceIn: 0.30, priceOut: 2.50 },
//   ],
//   // resultsDir is optional and defaults to "results". Sweep arms of ONE model share a
//   // directory and differ only by --tag, so they need it: {dir: "EMO_GPT", resultsDir:
//   // "results_eLow"}. Without it every arm is read from the same path and reported identical,
//   // which looks like agreement rather than a bug.
//   expected:     { EU: 200, EA: 200 }   rows per task, per run
//   sinceMinutes: 0                      rate over the last N minutes only, so a bad first hour
//                                        does not drag the estimate
// }
// `args` may arrive as a JSON string; normalise before reading a field.
// ---------------------------------------------------------------------------

let A = args || {}
if (typeof A === 'string') {
  try {
    A = JSON.parse(A)
  } catch (error) {
    return { status: 'cannot-tell', aborted: `args arrived as a string that is not JSON: ${error.message}`, raw: String(args).slice(0, 300) }
  }
}

const BENCH = A.benchmark || ''
const QDIR = A.questDir || ''
const RUNS = Array.isArray(A.runs) ? A.runs : []
const EXPECTED = A.expected || {}
const SINCE = A.sinceMinutes || 0

if (!BENCH || !QDIR) {
  return { status: 'cannot-tell', aborted: 'args.benchmark and args.questDir are both required' }
}
if (RUNS.length < 2) {
  return { status: 'cannot-tell', aborted: 'args.runs needs at least two entries; use check-status for a single run' }
}
for (const r of RUNS) {
  if (!r || !r.label || !r.dir) {
    return { status: 'cannot-tell', aborted: `every run needs {label, dir}; got ${JSON.stringify(r)}` }
  }
}
// Two arms reading the same path would be reported as agreeing rather than as misconfigured.
const paths = RUNS.map((r) => `${r.dir}/${r.resultsDir || 'results'}`)
if (new Set(paths).size !== paths.length) {
  return { status: 'cannot-tell',
           aborted: `two runs point at the same results directory (${paths.join(', ')}); ` +
                    `sweep arms of one model need distinct resultsDir values` }
}

const TASKS = Object.keys(EXPECTED)
const PER_RUN = Object.values(EXPECTED).reduce((a, b) => a + b, 0)

const READONLY = `
THIS WORKFLOW IS READ-ONLY. Breaking that is a failed task, not a judgement call:
- NO sbatch, NO scancel, NO scontrol, NO file edits, NO transfers, NO git commit or push, and NO
  provider API calls of any kind — not even one "just to check whether the key works". These jobs
  are live and spending real quota; a stray probe or cancel corrupts the very run being measured.
- Reading is unrestricted: squeue, sacct, ssh, cat, head, grep, wc, find, python for arithmetic.
- \`srun --jobid=<id> --overlap <cmd>\` is allowed and is read-only — use it to inspect /proc on the
  compute node. Do not use it to run anything that writes.
- Quest is \`ssh -o BatchMode=yes uwr0681@login.quest.northwestern.edu\`. Ignore the libcrypto
  host-key warning, it is cosmetic.
`

const BUFFERING = `
**An empty log is not evidence of a stall.** These runners print progress without flush=True and
SLURM buffers stdout, so log.txt can sit at 0 bytes through a completely healthy run. Never report a
stall from log size alone. Judge progress from rows written to the .jsonl, and from CPU time and
wchan on the compute node. If you want to say "no output", say "log.txt is 0 bytes, which under
buffering tells us nothing" — do not upgrade it to a finding.
`

// ---------- 1. Observe ----------
phase('Observe')

const observations = await parallel(RUNS.map((run) => () => agent(
  `Report exactly what the ${BENCH} run "${run.label}" is doing right now. Facts only, no advice,
no interpretation beyond what the numbers force.

Remote directory: ${QDIR}/${run.dir}
Results directory: ${QDIR}/${run.dir}/${run.resultsDir || 'results'}
SLURM job id: ${run.jobId === undefined ? '(not supplied — find it by name in squeue)' : run.jobId}
Expected rows per task: ${JSON.stringify(EXPECTED)} (${PER_RUN} total for this run)
${READONLY}
${BUFFERING}

Collect and report:

1. **Queue.** squeue for uwr0681: this job's state, elapsed, node. If it has left the queue, get
   State, ExitCode and Elapsed from sacct — **a job can exit COMPLETED 0:0 with every row empty**,
   so its absence from the queue is not success.
2. **Rows per task**, counted on Quest from the .jsonl under
   ${QDIR}/${run.dir}/${run.resultsDir || 'results'}/<task>/,
   never from the CSV: model output contains embedded newlines and naive CSV line-counting has
   already produced a false alarm on this project. Report count per task, and the file's mtime.
3. **Throughput.** Rows now, and rows ${SINCE ? `in the last ${SINCE} minutes` : 'since the job started'}.
   Derive rows/min and seconds/row. Report the elapsed wall-clock you divided by, so the arithmetic
   can be checked.
4. **Process state on the node** via \`srun --jobid=<id> --overlap\`: State and Threads from
   /proc/<pid>/status, utime+stime from /proc/<pid>/stat, and /proc/<pid>/wchan. Interpret only
   this far: \`hrtimer_nanosleep\` means the process is inside a sleep between calls, which is
   normal for these runners (they sleep 2.0s per item); accumulating CPU time means it is working.
   A process pinned at zero CPU growth across two checks is the thing worth flagging.
5. **Quality of the rows already written.** From the .jsonl, count:
   - rows whose model_response is empty
   - finish_reason values, especially anything containing MAX_TOKENS — that means thinking consumed
     max_output_tokens and the item was billed for nothing scoreable
   - thinking_tokens: min, median, max
   - use_cot and use_cot_source: report the distinct values. More than one value across a run means
     two conditions are mixed in one result set, which is a finding, not a curiosity.
   - for OpenRouter-style runs only, the distinct served_by values and their counts.
6. **Errors.** grep log.err and log.txt for API error, Retrying, Traceback, rate, quota, 429, 500.
   Report counts and the last two verbatim. Then state explicitly whether the log is buffered, so
   the reader knows whether "no errors" means "none happened" or "none visible yet".
7. **Halt markers** — any BILLING_HALT / QUOTA_HALT / FAILURE_HALT file in the run directory.

End with one line, exactly:
STATUS: <healthy|too-early|degraded|failed|stalled|quota-blocked|stale-code|cannot-tell>

Use \`too-early\` when fewer than 20 rows exist — these runners checkpoint every 20 items, so before
that there is legitimately no file at all and nothing can be concluded.`,
  { label: `observe:${run.label}`, phase: 'Observe' },
)))

const seen = observations.filter(Boolean)
if (seen.length === 0) {
  return { status: 'cannot-tell', aborted: 'every observer failed; nothing was measured' }
}

// ---------- 2. Compare ----------
phase('Compare')

const priceTable = RUNS.map((r) => `  ${r.label}: ` + (
  r.priceIn === undefined && r.priceOut === undefined
    ? 'prices NOT supplied — report its cost as not established rather than guessing'
    : `$${r.priceIn}/M input, $${r.priceOut}/M output`
)).join('\n')

const comparison = await agent(
  `Two or more runs of ${BENCH} are executing side by side. Below is what each observer measured.
Judge each run's health, then compare them on cost and finish time.
${READONLY}
${BUFFERING}

Prices, as supplied by the caller:
${priceTable}

Observations:
${seen.map((o, i) => `----- ${RUNS[i] ? RUNS[i].label : 'run ' + i} -----\n${o}`).join('\n\n')}

Produce, in this order:

1. **Per-run verdict** — one line each: label, rows/${PER_RUN}, rows/min, health, and the single
   fact that decides the verdict. If two runs disagree in throughput by more than 2x, say which is
   slower and what in the observations explains it. Do not attribute a difference to a cause the
   observations do not support; "not established from these logs" is an acceptable answer.

2. **Finish-time table.** Per run: elapsed so far, rows done, rows/min, projected remaining time,
   projected total wall-clock. State the assumption — a projection from the current rate assumes the
   rate holds, and a run that has hit retries will not hold it.

3. **Cost table.** This is the part to be careful with, because **neither runner records per-call
   token usage or cost** — only thinking_tokens. So:
   - Report thinking tokens per row (median and max) from the observations. That is measured.
   - Where prices were supplied, compute a cost estimate ONLY if the observations contain enough
     token data to do so, and **label every such number "derived, not measured"**, naming what you
     assumed for input and output tokens.
   - Where prices were not supplied, or token counts are missing, write "not established" in the
     cell. Do not fill a gap with a plausible number.
   - State plainly, as a recommendation, that comparing cost properly needs the runners to record
     usage per call (prompt tokens, completion tokens, and OpenRouter's per-call usage.cost, which
     its API returns and this runner currently discards).

4. **What needs a decision** — anything the planner must act on: a stalled run, MAX_TOKENS empties
   that mean raising the cap, mixed use_cot values in one result set, a provider that switched
   backend mid-run, a rate that will not finish inside the job's time limit.

5. **What would settle an open question** — for anything you could not determine, name the specific
   observation that would resolve it.

End with one line, exactly:
STATUS: <trustworthy|partial|untrustworthy|cannot-tell> / <continue|kill|kill-and-archive|prune-and-resume|publish|needs-human>`,
  { label: 'compare', phase: 'Compare' },
)

const verdicts = {}
seen.forEach((o, i) => {
  const m = String(o).match(/STATUS:\s*([a-z-]+)\s*$/im)
  verdicts[RUNS[i] ? RUNS[i].label : 'run' + i] = m ? m[1] : 'cannot-tell'
})
const overall = String(comparison || '').match(/STATUS:\s*([a-z-]+)\s*\/\s*([a-z-]+)/i)

return {
  benchmark: BENCH,
  tasks: TASKS,
  expectedPerRun: PER_RUN,
  perRun: verdicts,
  status: overall ? overall[1] : 'cannot-tell',
  recommendation: overall ? overall[2] : 'needs-human',
  report: comparison,
}
