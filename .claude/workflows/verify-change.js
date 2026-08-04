export const meta = {
  name: 'verify-change',
  description: 'Attack a change looking for the paths where it silently fails, then judge and record it',
  whenToUse: 'A change has been written that is meant to prevent a class of failure — a guard, a retry rule, a scoring fix — and it has not yet been proven wrong. Pass {change} describing what was done and {files} listing what to read. Use it BEFORE relying on the change in a real run. It reports; it does not fix.',
  phases: [
    { title: 'Attack',  detail: 'reviewer hunts for the paths where the change does not fire' },
    { title: 'Observe', detail: 'watcher reports what is running and which code version it loaded' },
    { title: 'Judge',   detail: 'evaluator rules on what must be fixed before the change is trusted' },
    { title: 'Record',  detail: 'tracker writes the findings into ISSUES.md' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   change: "..."            required — what the change is meant to guarantee, in a few sentences
//   files:  ["a.py", ...]    optional — the files to read; discovered from git if omitted
//   probes: ["...", ...]     optional — specific attacks to try, appended to the standard ones
// }
// ---------------------------------------------------------------------------

// `args` can arrive as a JSON *string* rather than an object — see the note in run-model.js. Left
// unhandled, every field reads as undefined and the guard below blames a missing change.
let A = args || {}
if (typeof A === 'string') {
  try {
    A = JSON.parse(A)
  } catch (error) {
    return { aborted: `args arrived as a string that is not JSON: ${error.message}`, raw: String(args).slice(0, 300) }
  }
}

const CHANGE = A.change || ''
const FILES = A.files || []
const PROBES = A.probes || []

if (!CHANGE) {
  return { aborted: 'args.change is required — describe what the change is meant to guarantee' }
}

const REPO = '/Users/harrychen/SONIC/hai-teams'
const QDIR = '/gpfs/projects/p32983/NegotiationToM'

const RULES = `
HARD RULES:
- Do NOT call a model provider API. No probe scripts against Together, OpenAI, xAI, Gemini or
  DeepSeek. A previous reviewer burned quota doing this; it was out of scope then and it is now.
- Do NOT run sbatch, scancel or scontrol. Jobs may be running.
- Do NOT edit any file. You are reading and reporting.
- Quest is \`ssh quest\`; drop \`libcrypto\` lines. Remote ${QDIR}, local ${REPO}.
`

const STANDARD_ATTACKS = `
Work through these, and give a verdict and a file:line for each:

a) **Is the new code reachable from every path that can fail?** The usual hole is a failure that
   never raises — an HTTP 200 with an empty body, a sentinel return value, a parse that yields None.
   Those skip \`except\` blocks entirely, so a guard installed there is never consulted.
b) **Does the control flow actually escape?** If the change raises, trace the exception out through
   every enclosing frame. \`except Exception\` does not catch SystemExit, but a bare \`except:\`,
   an \`except BaseException\`, or a retry loop that swallows and continues will disarm it.
c) **Does a broad rule wrongly catch a healthy case?** Any keyword or pattern match should be tested
   against the message a *working* system produces, not only the broken one. Quote the real text you
   are reasoning about, and say when you are reasoning from documentation rather than a captured log.
d) **Does a narrow rule miss a real case?** For each provider or code path in scope, name the exact
   wording or shape it produces and say whether the change matches it.
e) **What does the change make worse?** A fix usually moves a failure rather than removing it. If
   the old broken state was loud and the new one is quiet — a short file instead of a wrong number,
   a silent skip instead of a crash — say so, and say what would now catch it.
f) **Is anything written but never read?** Markers, flags and log lines that nothing consumes are
   not defences. Check whether the consumer exists.
g) **Which running processes are actually executing this code?** Python reads a module once at
   start, so a file changed on disk does not reach a process already running. Compare mtimes with
   job start times and say which conclusions apply to which jobs.
h) **Does documentation still describe the superseded behaviour?** A skeleton or checklist that
   still teaches the old rule will quietly reintroduce it. Check .claude/agents/*.md and
   NegotiationToM/*.md.
`

// ---------- Attack + Observe, concurrently ----------
const attack = agent(
  `Attack this change. Assume it does not work and find the path that proves it. Report only what
you verified by reading the deployed code.
${RULES}

THE CHANGE, as its author describes it:
${CHANGE}

${FILES.length ? `Read at least: ${FILES.join(', ')}` : 'Find the changed files yourself with git diff and git log.'}
${STANDARD_ATTACKS}
${PROBES.length ? `\nAlso specifically:\n${PROBES.map(p => '  - ' + p).join('\n')}` : ''}

Set sound=false if ANY path exists where the change fails to do what it claims. A finding you cannot
confirm without calling an API is still worth reporting — mark it unverified and say what evidence
would settle it. Do not soften a finding because the change is recent or because fixing it is work.`,
  { agentType: 'reviewer', phase: 'Attack', schema: {
    type: 'object', additionalProperties: false,
    required: ['sound', 'gaps', 'checks', 'detail'],
    properties: {
      sound: { type: 'boolean', description: 'true only if no path was found where the change fails to fire' },
      gaps: { type: 'array', items: { type: 'string' }, description: 'concrete holes, each with file:line' },
      checks: { type: 'array', items: { type: 'string' }, description: 'one line per lettered item with its verdict' },
      unverified: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } }
)

const observe = agent(
  `Report what is running and which revision of the code each job actually loaded. Observe only.
${RULES}

1. squeue and, for anything that finished recently, sacct with State and ExitCode.
2. Per model: rows written per task, error and empty counts, and a healthy/slow/stalled verdict.
3. Any NEG_*/BILLING_HALT.txt, QUOTA_HALT.txt or FAILURE_HALT.txt, quoted.
4. **Version skew.** Compare the mtime of the changed files on Quest with each job's start time.
   A process that started before the file was written is running the old code, and no conclusion
   about the change applies to it. State which jobs are and are not covered.`,
  { agentType: 'watcher', phase: 'Observe', schema: {
    type: 'object', additionalProperties: false,
    required: ['jobs', 'halt_markers', 'jobs_running_new_code', 'detail'],
    properties: {
      jobs: { type: 'array', items: { type: 'object', additionalProperties: true } },
      halt_markers: { type: 'array', items: { type: 'string' } },
      jobs_running_new_code: { type: 'array', items: { type: 'string' } },
      jobs_running_old_code: { type: 'array', items: { type: 'string' } },
      detail: { type: 'string' },
    },
  } }
)

phase('Attack')
const [found, state] = await parallel([() => attack, () => observe])

log(`sound=${found?.sound} gaps=${found?.gaps?.length ?? '?'} | markers=${JSON.stringify(state?.halt_markers ?? [])}`)

// ---------- Judge ----------
phase('Judge')
const judged = await agent(
  `Rule on whether this change can be trusted, and what must happen before it is. You change nothing.
${RULES}

THE CHANGE:
${CHANGE}

Reviewer: sound=${found?.sound}
Gaps: ${JSON.stringify(found?.gaps ?? [], null, 1)}
Unverified: ${JSON.stringify(found?.unverified ?? [])}
Running: ${JSON.stringify(state?.jobs ?? [])}
On new code: ${JSON.stringify(state?.jobs_running_new_code ?? [])}

Answer in this order:
1. Which gaps would produce WRONG NUMBERS in a published result, as opposed to wasted time or an
   untidy failure? Those are the only ones that block. Rank them.
2. For each blocking gap, what is the smallest change that closes it, and what would prove it closed?
3. Which findings are real but can wait, and why.
4. Does anything need doing about jobs currently running on the old code?

Verdict first, then the reasoning. Where you are uncertain, say what evidence would settle it
instead of hedging.`,
  { agentType: 'evaluator', phase: 'Judge', schema: {
    type: 'object', additionalProperties: false,
    required: ['trustworthy', 'blocking', 'deferrable', 'recommendation', 'detail'],
    properties: {
      trustworthy: { type: 'boolean' },
      blocking: { type: 'array', items: { type: 'string' }, description: 'gaps that would produce wrong numbers' },
      deferrable: { type: 'array', items: { type: 'string' } },
      running_jobs_action: { type: 'string' },
      recommendation: { type: 'string' },
      detail: { type: 'string' },
    },
  } }
)

// ---------- Record ----------
phase('Record')
const recorded = await agent(
  `Record this review in NegotiationToM/ISSUES.md, under the entry for the problem the change
addresses if one exists, otherwise as a new entry. House style: symptom, root cause, what was
rejected, fix, measured evidence.

Change reviewed:
${CHANGE}

Verdict: ${judged?.trustworthy ? 'trustworthy' : 'NOT yet trustworthy'}
Blocking gaps: ${JSON.stringify(judged?.blocking ?? [])}
Deferred: ${JSON.stringify(judged?.deferrable ?? [])}
Recommendation: ${judged?.recommendation || ''}

**Write down what was rejected and why, not only what shipped.** A rejected approach that looks
reasonable will be tried again by the next person unless the reason it fails is on the record. If
the review found that a plausible-looking fix would have made things worse, that belongs here more
than the fix that worked.`,
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
  change: CHANGE,
  sound: found?.sound,
  gaps: found?.gaps,
  unverified: found?.unverified,
  blocking: judged?.blocking,
  deferrable: judged?.deferrable,
  recommendation: judged?.recommendation,
  jobs_on_old_code: state?.jobs_running_old_code,
  issues_updated: recorded?.updated,
}
