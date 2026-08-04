export const meta = {
  name: 'harvest-patterns',
  description: 'Find GitHub repos of Claude Code workflows/agents, extract only the patterns that transfer to this project, judge them adversarially, and record both what was adopted and what was rejected',
  whenToUse: 'Looking for outside ideas to improve this project\'s agents, workflows or references. Pass optional {focus, maxRepos, minStars, activeWithinDays}. It proposes and records; it never installs a plugin, never edits an agent or workflow, and never touches Quest. Re-running it is cheap: repos already judged in .claude/references/external-patterns.md are skipped unless they have moved since.',
  phases: [
    { title: 'Discover', detail: 'three search angles, each blind to the others' },
    { title: 'Screen',   detail: 'per candidate: does this transfer to a Python/SLURM eval repo at all' },
    { title: 'Extract',  detail: 'per survivor: read the actual files, name the mechanism, cite the path' },
    { title: 'Judge',    detail: 'one skeptic sees every pattern at once and tries to refute each' },
    { title: 'Record',   detail: 'write external-patterns.md — adopted AND rejected, so the next run skips them' },
  ],
}

// ---------------------------------------------------------------------------
// args: {
//   focus:             "..."   what to look for. Default: whatever this project is weakest at
//   maxRepos:          5       how many candidates reach the expensive Extract stage
//   minStars:          80      floor for a candidate to be worth reading
//   activeWithinDays:  365     a repo not pushed to since then is archived in practice
// }
//
// `args` can arrive as a JSON *string* rather than an object — the caller writes an object but the
// tool hands the script the serialised form, and every field then reads as undefined. Normalise
// first; a string that will not parse is a hard error, not a fall-through to a default.
// ---------------------------------------------------------------------------

let A = args || {}
if (typeof A === 'string') {
  try {
    A = JSON.parse(A)
  } catch (error) {
    return { aborted: `args arrived as a string that is not JSON: ${error.message}`, raw: String(args).slice(0, 300) }
  }
}

const FOCUS = A.focus || 'anything that would make this project\'s agents, workflows or references more reliable or cheaper to run'
const MAX_REPOS = Math.min(Math.max(1, A.maxRepos || 5), 8)
const MIN_STARS = A.minStars ?? 80
const ACTIVE_DAYS = A.activeWithinDays || 365
// Repos already judged in an earlier sweep. Discovery reads the record file itself, but that file
// does not exist until Record has run once, so an explicit list lets a re-run skip what a previous
// attempt already spent agents on.
const SKIP = new Set((A.skip || []).map((s) => String(s).toLowerCase()))

const REPO = '/Users/harrychen/SONIC/hai-teams'
const RECORD = '.claude/references/external-patterns.md'

// ---------------------------------------------------------------------------
// The rules every agent gets. Two of these were learned the expensive way and are not decoration:
// a reviewer once wrote four probe scripts and spent real provider quota because its prompt did not
// forbid it, and the GitHub API here is unauthenticated — 60 calls an hour, shared by every agent
// running at once.
// ---------------------------------------------------------------------------
// The write scope differs by phase, so it is a parameter rather than a string-replace on the rules
// block. A replace that silently fails to match would hand the Record phase two contradictory
// instructions — read-only, and also write this file.
const SCOPE_READONLY = `- READ-ONLY on this repo. Do not edit or create any file. In particular do not touch
  .claude/agents/*, .claude/workflows/*, CLAUDE.md, or anything under NegotiationToM/. This
  workflow proposes; a human decides.`

const SCOPE_RECORD = `- Write ONLY ${RECORD}, plus the one routing row in .claude/references/README.md that this task
  explicitly asks for. Do not touch .claude/agents/*, .claude/workflows/*, CLAUDE.md, or anything
  under NegotiationToM/. You are recording proposals; a human decides whether to apply them.`

const rules = (scope) => `
This project is hai-teams: LLM evaluation benchmarks in Python, run as SLURM jobs on Northwestern's
Quest cluster. Results are .jsonl/.csv. It is NOT a TypeScript/React application, and advice built
for feature development in a web app is usually not transferable here — say so rather than bending
it to fit.

HARD RULES — breaking one is a failed task, not a judgement call:
${scope}
- Do NOT install anything. No \`/plugin install\`, no \`git clone\`, no npm/pip install, no adding a
  marketplace. Read repositories over HTTP.
- Do NOT call a model provider API, and do NOT touch Quest — no ssh, no sbatch, no scancel.
- Do NOT run git add, commit, push, or checkout.
- GitHub API BUDGET: the API here is unauthenticated — 60 requests/hour total and 10 searches/minute
  across every agent running at once. Spend at most 6 API calls. Use WebSearch for discovery (it
  costs no GitHub quota) and read file contents from raw.githubusercontent.com, which is a CDN and
  does not consume the budget either. Reserve API calls for confirming stars/pushed_at on finalists.
  On HTTP 403 mentioning rate limit: STOP, report what you already have, and say the budget ran out.
  Do not retry in a loop.
- EVERY factual claim about an outside repo carries the exact path you actually fetched, e.g.
  \`owner/repo:skills/foo/SKILL.md\`. A claim you did not read the file for is a guess; mark it as
  one or drop it. Inventing a repo or a path is the worst outcome this workflow can produce.
`

const RULES = rules(SCOPE_READONLY)

const INVENTORY = `
Before judging anything, read what this project already has, so you do not propose it back:
- ${REPO}/CLAUDE.md — the planner's rules, the role split, the saved-workflow table
- ${REPO}/.claude/references/README.md — the routing table into the reference layer
- \`ls ${REPO}/.claude/agents/ ${REPO}/.claude/workflows/ ${REPO}/.claude/references/\`
Already in place, so do not re-propose them: six narrow subagents with STATUS-line return contracts;
three saved JS workflows; progressive disclosure via .claude/references/ with a routing table; a
PreToolUse hook gating sbatch on a local<->Quest md5 sync check; ISSUES.md as a problem log that
records rejected fixes and false alarms.
`

// ---------------------------------------------------------------------------
// 1. Discover — three angles, each blind to the others.
// A barrier is correct here: the dedup in the next step needs every candidate at once, and the
// whole point of three angles is that they overlap.
// ---------------------------------------------------------------------------
phase('Discover')

const ANGLES = [
  {
    key: 'by-registry',
    prompt: `Search where Claude Code extensions are *published*: plugin marketplaces, the
\`claude-code-plugin\` / \`claude-code-agents\` / \`claude-code-skills\` GitHub topics, and
awesome-lists that index them. You are looking for repositories that ship reusable agents, skills,
hooks or workflow scripts.`,
  },
  {
    key: 'by-problem',
    prompt: `Search by the problems this project actually has, not by the words "claude code":
supervising long-running cluster or batch jobs with an LLM agent; detecting silent failure in
LLM evaluation pipelines; keeping code in sync between a laptop and an HPC cluster; agent harnesses
for reproducible research runs; orchestrating LLM benchmark sweeps. A repo that solves one of these
without mentioning Claude Code is more valuable here than a generic agent collection.`,
  },
  {
    key: 'by-neighbour',
    prompt: `Start from repos adjacent to what this project already resembles: LLM-evaluation
harnesses (lm-evaluation-harness, inspect_ai, HELM and their ecosystems) and research-automation
agent frameworks. Look at how they structure run configuration, provenance of results, and
resume/checkpoint semantics. Also look at what shinpr/claude-code-workflows links to or is forked
alongside, since that neighbourhood is already known to be relevant.`,
  },
]

const discovered = await parallel(ANGLES.map((angle) => () => agent(
  `Find GitHub repositories worth reading for this project. Your angle:

${angle.prompt}

Focus for this run: ${FOCUS}
${RULES}
${INVENTORY}

Return at most 8 candidates. For each, give owner/repo, the URL, star count, the date of the last
push, and one sentence on what it claims to offer. Filters: at least ${MIN_STARS} stars and pushed
within the last ${ACTIVE_DAYS} days — but keep a repo below the star floor if it is unusually close
to this project's actual problem, and say why you kept it.

Then read ${REPO}/${RECORD} if it exists. Any repo already judged there is done: skip it unless it
has been pushed to since the date recorded, in which case keep it and say what changed.

Confirm each candidate is real before returning it: fetch its README from
raw.githubusercontent.com. A candidate whose README you could not fetch does not go in the list.`,
  { agentType: 'summarizer', phase: 'Discover', label: `discover:${angle.key}`, schema: {
    type: 'object', additionalProperties: false,
    required: ['candidates', 'detail'],
    properties: {
      candidates: {
        type: 'array',
        items: {
          type: 'object', additionalProperties: false,
          required: ['repo', 'url', 'claim', 'readme_fetched'],
          properties: {
            repo: { type: 'string', description: 'owner/repo' },
            url: { type: 'string' },
            stars: { type: 'integer' },
            last_push: { type: 'string' },
            claim: { type: 'string' },
            below_star_floor_because: { type: 'string' },
            readme_fetched: { type: 'boolean' },
          },
        },
      },
      already_recorded_skipped: { type: 'array', items: { type: 'string' } },
      budget_exhausted: { type: 'boolean' },
      detail: { type: 'string' },
    },
  } }
)))

const rows = discovered.filter(Boolean).flatMap((d) => d.candidates || []).filter((c) => c && c.repo && c.readme_fetched)
const byRepo = new Map()
for (const c of rows) {
  const key = String(c.repo).toLowerCase()
  if (SKIP.has(key)) continue
  const prev = byRepo.get(key)
  // Keep the entry that carries the most metadata; a repo found by two angles is a signal, not noise.
  if (!prev || (c.stars || 0) > (prev.stars || 0)) byRepo.set(key, { ...c, found_by: (prev?.found_by || 0) + 1 })
}

// ---------------------------------------------------------------------------
// Rank by relevance, NOT by stars.
//
// The first sweep sorted by stars and it actively selected against relevance: a 51k-star
// awesome-list (a directory of links, not a mechanism) and a 38k-star graph framework crowded out
// facebookincubator/submitit (SLURM submission with automatic checkpoint-and-requeue on preemption)
// and UKGovernmentBEIS/inspect_ai (`eval-set` schedules only the work not yet complete) — the two
// candidates closest to what this project actually does. All five screened repos came back
// keep=false, and the sweep reported "nothing found" without ever looking at the good ones.
//
// The scorer is deliberately blunt and logged in full, so a bad ranking is visible in the transcript
// rather than hidden behind a number.
// ---------------------------------------------------------------------------
const SIGNALS = [
  [/\bslurm\b|\bhpc\b|\bsbatch\b|\bcluster\b|preempt|batch job/i, 5, 'cluster'],
  [/checkpoint|requeue|resume|idempot|not yet complete|incomplete|already done/i, 4, 'resume'],
  [/\beval\b|\bevaluation\b|benchmark|harness|scoring|grader/i, 3, 'eval'],
  // `\bsuite\b` was here and matched "Plugin suite" on an unrelated repo — a bare noun that common
  // is not evidence of run provenance. `run.?spec` catches the thing actually meant (HELM's RunSpecs).
  [/provenance|reproducib|run record|run.?spec|manifest|audit trail/i, 3, 'provenance'],
  [/silent|empty response|stall|\bretry\b|quota|rate.?limit|flaky/i, 3, 'silent-failure'],
  [/handoff|compaction|\bphase\b|orchestrat|subagent/i, 2, 'handoff'],
  [/\bhook\b|\bgate\b|guard|block.*before|pre.?tool/i, 2, 'gate'],
]
const PENALTIES = [
  [/awesome.?list|curated (list|index|collection)|link directory|\bindex of\b|list of/i, -6, 'is-a-list'],
  [/react|typescript|frontend|web app|\bGUI\b|status.?line|notification/i, -3, 'wrong-stack'],
]

function relevance(c) {
  const text = `${c.repo} ${c.claim || ''}`
  let score = 0
  const why = []
  for (const [re, w, tag] of SIGNALS) if (re.test(text)) { score += w; why.push(tag) }
  for (const [re, w, tag] of PENALTIES) if (re.test(text)) { score += w; why.push(tag) }
  // Discovery kept this one despite being under the star floor — that is a deliberate relevance call.
  if (c.below_star_floor_because) { score += 3; why.push('kept-below-floor') }
  if (c.found_by > 1) { score += 1; why.push('found-twice') }
  return { score, why }
}

const candidates = [...byRepo.values()]
  .map((c) => ({ ...c, ...relevance(c) }))
  .sort((a, b) => (b.score - a.score) || ((b.stars || 0) - (a.stars || 0)))

log('ranked by relevance (stars are only a tiebreak):')
for (const c of candidates) log(`  ${String(c.score).padStart(3)}  ${c.repo}  [${c.why.join(' ') || 'no signal'}]  ${c.stars || '?'}*`)

if (!candidates.length) {
  return {
    outcome: 'nothing to screen',
    reason: 'no candidate survived discovery with a README that could actually be fetched',
    budget_exhausted: discovered.filter(Boolean).some((d) => d.budget_exhausted),
    discovered,
  }
}

const shortlist = candidates.slice(0, MAX_REPOS)
const dropped = candidates.slice(MAX_REPOS).map((c) => c.repo)
log(`${rows.length} raw candidates -> ${candidates.length} unique -> screening ${shortlist.length}`)
if (dropped.length) log(`NOT examined (over maxRepos=${MAX_REPOS}): ${dropped.join(', ')}`)

// ---------------------------------------------------------------------------
// 2+3. Screen, then Extract — a pipeline, not a barrier. A repo that screens out never pays for an
// extract, and a fast repo does not wait on a slow one.
// ---------------------------------------------------------------------------
const examined = await pipeline(
  shortlist,

  // -- Screen: cheap, and allowed to say no. Most candidates should die here.
  (cand) => agent(
    `Decide whether ${cand.repo} (${cand.url}) is worth reading in depth for this project.
It claims: ${cand.claim}
${RULES}
${INVENTORY}

Read its README and its top-level layout. Then answer one question: does anything here transfer to
a Python/SLURM LLM-evaluation repo, or is it built for a problem this project does not have?

Return keep=false for: a TypeScript/React-specific toolkit; a collection of prompts with no
mechanism behind them; a repo whose ideas this project has already implemented (name the local file
that shows it); a repo you could not actually read. Most candidates should be keep=false — a
generous screen just moves the cost to the next stage.

If you keep it, name the ONE thing that made it worth keeping, with the path you read it in.`,
    { agentType: 'summarizer', phase: 'Screen', label: `screen:${cand.repo}`, schema: {
      type: 'object', additionalProperties: false,
      required: ['repo', 'keep', 'applicability', 'reason'],
      properties: {
        repo: { type: 'string' },
        keep: { type: 'boolean' },
        applicability: { type: 'string', enum: ['direct', 'adaptable', 'irrelevant', 'already-have'] },
        best_thing: { type: 'string' },
        evidence_path: { type: 'string' },
        already_have_local_file: { type: 'string' },
        reason: { type: 'string' },
      },
    } }
  ),

  // -- Extract: only for survivors. Reads real files; a pattern without a citation does not count.
  (screen, cand) => {
    if (!screen || !screen.keep) return { repo: cand.repo, screened_out: true, screen, patterns: [] }
    return agent(
      `${cand.repo} passed screening because: ${screen.best_thing || screen.reason}
Read its actual files and extract the transferable mechanisms.
${RULES}
${INVENTORY}

A README tells you what a repo claims. Read the files that implement it — the agent definitions,
the skill bodies, the hooks, the scripts. Fetch them from raw.githubusercontent.com.

For each pattern you extract, give:
- what problem it solves, stated as a failure that happens without it
- the MECHANISM, not the slogan. "Each phase writes a file the next phase reads in a fresh context"
  is a mechanism; "good context management" is a slogan and will be rejected downstream
- the exact upstream path you read it in
- a concrete adaptation to this project, naming the local file it would change and roughly how
- honestly whether this project already has it, with the local file that proves it

Extract at most 4 patterns. Fewer good ones beats a list. If the repo turns out to have nothing
once you are past the README, return an empty list and say so — that is a useful result, not a
failure.`,
      { agentType: 'summarizer', phase: 'Extract', label: `extract:${cand.repo}`, schema: {
        type: 'object', additionalProperties: false,
        required: ['repo', 'patterns', 'detail'],
        properties: {
          repo: { type: 'string' },
          patterns: {
            type: 'array',
            items: {
              type: 'object', additionalProperties: false,
              required: ['name', 'problem', 'mechanism', 'upstream_path', 'adaptation', 'already_have'],
              properties: {
                name: { type: 'string' },
                problem: { type: 'string' },
                mechanism: { type: 'string' },
                upstream_path: { type: 'string' },
                adaptation: { type: 'string' },
                local_file_affected: { type: 'string' },
                already_have: { type: 'boolean' },
                already_have_evidence: { type: 'string' },
              },
            },
          },
          detail: { type: 'string' },
        },
      } }
    )
  }
)

const results = examined.filter(Boolean)
const screenedOut = results.filter((r) => r.screened_out)
const harvested = results.filter((r) => !r.screened_out)
const allPatterns = harvested.flatMap((r) => (r.patterns || []).map((p) => ({ ...p, repo: r.repo })))
const fresh = allPatterns.filter((p) => !p.already_have)

log(`${screenedOut.length} screened out, ${harvested.length} read in depth, ${allPatterns.length} patterns (${fresh.length} not already present)`)

// ---------------------------------------------------------------------------
// Record is a function, not a phase that only the happy path reaches.
//
// The first sweep found nothing, returned `needs_record: true` with a note saying "record it so the
// next run does not re-read the same repos" — and then exited without recording anything, so the
// next run would have re-read all five. A sweep that finds nothing is the case where the record
// matters MOST: it is the only artefact proving those repos were examined.
// ---------------------------------------------------------------------------
async function recordSweep(verdicts, gap) {
  phase('Record')
  return agent(
    `Write this sweep into ${REPO}/${RECORD}.
${rules(SCOPE_RECORD)}
${INVENTORY}

Get today's date with \`date +%F\` — do not guess it.

Append to the file if it exists, preserving what is already there; create it with a short header
explaining what it is if it does not. Newest sweep at the top, under a dated heading.

Record, in this order:

1. **Adopted / to adapt** — for each survivor: the pattern, the repo and upstream path, the local
   file it would change, the smallest honest version, and the judge's reason. Mark each clearly as
   PROPOSED — nothing here has been applied. If there are no survivors, say so in one line; a sweep
   that found nothing is a valid and useful result.
2. **Rejected patterns** — name, repo, and the reject class. This is the half that saves the next
   sweep from re-deriving the same conclusion, so do not abbreviate it.
3. **Repos screened out** — repo and one-line reason, with the date they were judged. A future run
   reads this list and skips them unless they have been pushed to since. This section is the whole
   point of the file: write it even when it is the only section with content.
4. **The gap** — what none of these repos addressed: ${gap || 'not stated'}
5. **Not examined** — ${dropped.length ? dropped.join(', ') : 'none'} — candidates that ranked below
   the maxRepos=${MAX_REPOS} cutoff. Say plainly that they were never read, so this is not mistaken
   for a clean sweep.

Then add the routing row for ${RECORD} to ${REPO}/.claude/references/README.md — that file's own
rule is that a reference nothing routes to never gets read. Phrase the row as a condition an agent
would recognise in its own task. That is the one exception to editing only ${RECORD}.

Data:
${JSON.stringify({ verdicts, screened_out: screenedOut.map((r) => ({ repo: r.repo, reason: r.screen?.reason, applicability: r.screen?.applicability })) }, null, 2)}`,
    { agentType: 'tracker', phase: 'Record', schema: {
      type: 'object', additionalProperties: false,
      required: ['file_written', 'adopted_count', 'rejected_count', 'detail'],
      properties: {
        file_written: { type: 'boolean' },
        routing_row_added: { type: 'boolean' },
        date: { type: 'string' },
        adopted_count: { type: 'integer' },
        rejected_count: { type: 'integer' },
        detail: { type: 'string' },
      },
    } }
  )
}

if (!fresh.length) {
  const recordedEmpty = await recordSweep([], 'no pattern reached the judge this sweep')
  return {
    outcome: 'nothing new found',
    note: 'A normal result. What was examined is now recorded, so the next sweep skips it.',
    focus: FOCUS,
    searched: candidates.length,
    screened: shortlist.length,
    screened_out: screenedOut.map((r) => ({ repo: r.repo, reason: (r.screen?.reason || '').slice(0, 200) })),
    already_have: allPatterns.filter((p) => p.already_have).map((p) => `${p.repo}: ${p.name}`),
    not_examined: dropped,
    recorded_to: recordedEmpty?.file_written ? RECORD : 'NOT WRITTEN — record phase failed',
  }
}

// ---------------------------------------------------------------------------
// 4. Judge — one skeptic, seeing every pattern at once.
// The barrier is justified: judging requires comparing patterns against each other and against the
// whole existing setup, which no per-pattern agent can do. The judge is deliberately NOT one of the
// extractors — this project has already shipped a broken watchdog because the author of a change
// and its verifier were the same agent.
// ---------------------------------------------------------------------------
phase('Judge')

const judged = await agent(
  `Try to REFUTE each of these patterns. Default to rejecting: a pattern survives only if you can
state the concrete failure in this project that it would have prevented.

${JSON.stringify(fresh.map((p, i) => ({
    id: i, repo: p.repo, name: p.name, problem: p.problem, mechanism: p.mechanism,
    upstream_path: p.upstream_path, adaptation: p.adaptation, local_file_affected: p.local_file_affected,
  })), null, 2)}
${RULES}
${INVENTORY}

Reject a pattern when any of these holds, and name which:
- **already-have** — this project does it. Cite the local file and line that shows it.
- **wrong-shape** — it solves a web-app feature-development problem. This repo's failure modes are
  code that never reached the cluster, providers returning HTTP 200 with an empty body, results
  living only on Quest, and quota burned on a config that could not have worked. Documentation
  ceremony does not prevent an md5 drift.
- **slogan** — the mechanism is a principle, not a procedure an agent could follow.
- **unverifiable** — the upstream path is missing, or you could not confirm the file exists. Spot
  check at least the ones you are inclined to accept, from raw.githubusercontent.com.
- **cost-exceeds-benefit** — real, but the maintenance it adds outweighs the failure it prevents at
  this project's scale. Say what scale would flip that.

For each survivor, say what it changes, which local file, and what the smallest honest version of
it is. Prefer adapt over adopt: a mechanism copied whole from a repo with different failure modes
usually brings assumptions with it.

Rank the survivors. Then, separately: name what NONE of these repos addressed that this project
still needs — the gap is often the most useful output of a sweep like this.`,
  { agentType: 'evaluator', phase: 'Judge', schema: {
    type: 'object', additionalProperties: false,
    required: ['verdicts', 'gap', 'summary'],
    properties: {
      verdicts: {
        type: 'array',
        items: {
          type: 'object', additionalProperties: false,
          required: ['id', 'name', 'repo', 'verdict', 'reason'],
          properties: {
            id: { type: 'integer' },
            name: { type: 'string' },
            repo: { type: 'string' },
            verdict: { type: 'string', enum: ['adopt', 'adapt', 'reject'] },
            reject_class: { type: 'string', enum: ['already-have', 'wrong-shape', 'slogan', 'unverifiable', 'cost-exceeds-benefit'] },
            reason: { type: 'string' },
            local_file_affected: { type: 'string' },
            smallest_version: { type: 'string' },
            rank: { type: 'integer' },
          },
        },
      },
      gap: { type: 'string', description: 'what none of these repos addressed that this project still needs' },
      summary: { type: 'string' },
    },
  } }
)

const verdicts = (judged && judged.verdicts) || []
const survivors = verdicts.filter((v) => v.verdict !== 'reject')
    .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99))
log(`judge: ${survivors.length} of ${fresh.length} patterns survived refutation`)

// ---------------------------------------------------------------------------
// 5. Record — the rejections matter as much as the adoptions.
// A repo evaluated and rejected without a written record gets re-evaluated by the next sweep, which
// is the same waste ISSUES.md exists to prevent for bugs.
// ---------------------------------------------------------------------------
const recorded = await recordSweep(verdicts, judged?.gap)

return {
  outcome: survivors.length ? 'patterns proposed' : 'swept, nothing survived refutation',
  focus: FOCUS,
  searched: candidates.length,
  screened: shortlist.length,
  not_examined: dropped,
  proposed: survivors.map((v) => ({
    rank: v.rank, name: v.name, repo: v.repo, verdict: v.verdict,
    file: v.local_file_affected, smallest_version: v.smallest_version,
  })),
  rejected: verdicts.filter((v) => v.verdict === 'reject').map((v) => `${v.name} (${v.repo}): ${v.reject_class}`),
  gap: judged?.gap,
  recorded_to: recorded?.file_written ? RECORD : 'NOT WRITTEN — record phase failed',
  next: 'Nothing has been applied. Read the proposals, pick the ones worth the maintenance, and change the files yourself or hand each to executor as a decided change.',
}
