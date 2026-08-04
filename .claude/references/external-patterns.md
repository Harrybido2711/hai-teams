# External patterns

Findings from periodic sweeps of outside repositories — other agentic-coding tool suites, ML
pipeline/orchestration frameworks, SLURM job-submission libraries — screened for patterns that
might improve hai-teams' agent architecture or its SLURM/eval pipeline. Each sweep is a dated
section, newest at the top. A repo listed under "Repos screened out" was judged not to fit *this*
project on a specific, cited basis; a later sweep should read that entry before re-evaluating the
same repo, and can skip it outright unless it has been pushed to since the recorded date.

**Nothing in this file has been applied.** Every item under "Adopted / to adapt" is a proposal —
PROPOSED, not shipped — for a human to accept or reject; this file only records what the sweep
found and why the judge ruled the way it did.

hai-teams is a Python/SLURM LLM-evaluation project — results in `.jsonl`/`.csv`, jobs submitted
with `sbatch` on Northwestern's Quest cluster. It is not a TypeScript/React application. Advice
built for feature development in a web app is flagged as inapplicable here rather than bent to fit,
per the sweep's own instructions.

---

## Sweep — 2026-08-04 (follow-up)

Repos read this sweep: `kenryu42/cc-safety-net`, `UKGovernmentBEIS/inspect_ai`,
`parcadei/Continuous-Claude-v3`, `composio-community/awesome-claude-plugins` — the four names the
earlier 2026-08-04 sweep below deferred under its own "Not examined" section (§5 there).

### 1. Adopted / to adapt

**PROPOSED — nothing below has been applied. A human decides whether to build it.**

- **Argv-level, subcommand-dispatched git rule engine** (rank 1)
  - **Pattern.** Parse a `Bash` command's git subcommand past global flags (`-C`, `-c` and their
    values), then dispatch on the subcommand to block specific dangerous invocations, each paired
    with a reason string naming the safe alternative — instead of one blanket "no `git add -A`"
    regex.
  - **Repo and upstream path.** `kenryu42/cc-safety-net`. Mechanism verified in full at
    `src/core/git/index.ts` (57 lines) and `src/core/git/parse.ts` (64 lines —
    `extractGitSubcommandAndRest` at lines 34-61), plus `src/core/git/rules.ts` (401 lines, read at
    1-40 and 104-183, and grepped for the rest) — dispatch switch `analyzeGitRule` at
    `rules.ts:115-144`, `matchesGitLongOption`'s unambiguous-prefix handling at `rules.ts:98`,
    reason strings at `rules.ts:4-40`. Confirmed via one API call: 1,468 stars, pushed
    2026-08-03T19:29:45Z, not archived.
  - **Local file it would change.** Extend
    `/Users/harrychen/SONIC/hai-teams/.claude/scripts/sbatch_sync_gate.py` (or add a sibling hook
    registered in the same PreToolUse/`Bash` array in `.claude/settings.json`); the gate already
    owns an allow()/block() exit-code contract at `sbatch_sync_gate.py:33-57`. It should ship paired
    with a one-line fix to the live contradiction at
    `/Users/harrychen/SONIC/hai-teams/.claude/memory/quest-agent-workflow.md:22` — see the reason
    below — which is the cheaper half and should land first.
  - **Smallest honest version** (~25 lines, not a port of the 401-line rule engine): `shlex.split`
    the command, walk past `-C`/`-c` and their values to find the git subcommand, and block exactly
    two things:
    1. `add` with `-A` / `--all` / `--no-ignore-removal`, or a bare `.` pathspec → block, telling
       the caller to stage `NegotiationToM/<MODEL>/results NegotiationToM/<MODEL>/log_*.txt`
       explicitly.
    2. `push` with `upstream` among the positionals → block, "upstream is cpzambo/hai-teams, the
       shared repo; push to `backup` and `origin`."
    Do **not** port `reset`/`clean`/`checkout`/`branch`/`stash`/`rebase`/`merge`/`tag`/`reflog` —
    this project has not lost work to any of them, so each would be maintenance against a failure
    that has not happened here. Do **not** touch the `Bash(git add|commit|push *)` allow-globs at
    `settings.local.json:5-7` — see below.
  - **Judge's reason.** Survives, but only after being stripped to something the upstream code
    barely contributes to. The dispatch skeleton is accurately described and real — walk past
    global flags, switch on the subcommand, attach a human-readable reason — but the upstream rule
    *content* does not transfer: its 13-subcommand switch has no `add` case at all (the `default` at
    `rules.ts:142-143` returns `null` for anything unmatched) and no rule about pushing to a named
    upstream remote. Every upstream reason string assumes a developer-editing-code threat model
    ("you will lose uncommitted local work, stash first"; "you will rewrite remote history"), not
    this project's actual risk of over-inclusive staging into a shared results repo. It also would
    **not** have prevented the incident that motivates this project's own git-safety instinct in the
    first place: `NegotiationToM/ISSUES.md:170-199` records that the six "watcher checkpoint"
    commits (git log shows `39dace6` sweeping `CLAUDE.md` and `ISSUES.md` into a results commit)
    were made by `watch.sh`, a detached background shell with 23h11m uptime that never crosses the
    `Bash` tool boundary a PreToolUse hook can see — `ISSUES.md:195` already names the correct fix
    as an allow-list *inside the watcher*, not a hook, and that fix is still recorded open. What
    keeps the pattern alive anyway is evidence found this sweep, not cited before: five prose
    restatements of "never `git add -A`" (`CLAUDE.md:147`, `executor.md:41`, `run-model.js:74`,
    `:274`, `:409`, `ISSUES.md:195`) coexist right now with one live recipe saying the opposite —
    `quest-agent-workflow.md:22`'s hourly `git add -A NegotiationToM/ && git commit && git push
    backup main && git push origin main`, untracked by git but present on disk and loaded via the
    user's `MEMORY.md`. That coexistence is empirical proof that in this repo prose does not
    converge on its own, which matches this project's own stated design principle that "the gate
    must be able to say no" (`CLAUDE.md:162-163`). The adaptation's proposed second half —
    removing the `Bash(git add|commit|push *)` allow-globs at `settings.local.json:5-7` "to stop
    bypassing the hook via auto-allow" — is rejected outright: PreToolUse hooks are evaluated
    independently of permission allow-rules, so the removal is very likely unnecessary, and doing it
    anyway would put an interactive prompt in front of every hourly commit inside `run-model.js`'s
    unattended supervision loop — the same class of mechanism that already failed once (56,774 rows
    sat only on the cluster for hours when the hourly pull/push stopped, `CLAUDE.md:142-145`). A
    cheap way to settle the hook-vs-allow-rule question without touching `settings.local.json`:
    register a no-op echo hook on the `Bash` matcher in a scratch config and fire a pre-approved
    command through it.

### 2. Rejected patterns

- **Env-var override flagged as dangerous only when paired with the subcommand class that executes
  it** — `kenryu42/cc-safety-net` — reject class: **wrong-shape**.
  Upstream mechanism, verified in full rather than retold from a summary: `analyzeGit` fires only
  when `hasGitSshEnvAssignment` AND `isGitNetworkOperation` both hold
  (`kenryu42/cc-safety-net:src/core/git/index.ts:21-27,42-45`), with
  `GIT_SSH_ENV_NAMES = {GIT_SSH_COMMAND, GIT_SSH, GIT_SSH_VARIANT}` (`src/core/git/env.ts:20-24`)
  and `GIT_NETWORK_SUBCOMMANDS = {clone, fetch, pull, push, ls-remote, submodule}`
  (`index.ts:12-19`). The conjunction is genuinely elegant upstream, for a specific reason: git
  invokes the SSH transport for only 6 of its ~150 subcommands, so "will this operation actually
  execute the override" is a real discriminator there. Retargeted at this project's plain
  `ssh quest "..."` calls, that premise collapses — a literal `ssh` invocation always uses SSH, so
  the second conjunct degrades from "will this execute the override" to merely "is the host Quest,"
  which is exactly the blanket-alert-fatigue rule the pattern exists to avoid. Two of the three
  upstream triggers are also simply wrong once retargeted: `ProxyCommand` is an `ssh_config`
  directive set via `ssh -o`, not an environment variable at all, and `GIT_SSH_COMMAND` has no
  effect on a plain `ssh` invocation — only `SSH_AUTH_SOCK` survives as a real env var in that
  context, and overriding it is ordinary. No concrete incident supports adopting it either:
  `NegotiationToM/ISSUES.md` (303 lines, searched in full) records a missing host-key config block
  (`ISSUES.md:160-161`), stale code that never left the laptop, HTTP 200 with an empty body, and
  billing/quota halts as the actual Quest failures — no env-override incident anywhere, and access
  is key-based to a single known host. The upstream pattern defends against prompt-injection-driven
  exfiltration in a general-purpose safety net; this project's Quest risk is its own drift, not an
  external attacker. One citation behind the proposal also did not check out on inspection: it
  claimed `CLAUDE.md:36-39` held the sync recipe, but that range is the reference-routing rationale
  — the actual sync recipe is at `CLAUDE.md:74-83` — a reminder to verify a claim about a local file
  rather than accept it, even inside a rejection.

### 3. Repos screened out

All judged 2026-08-04.

- **UKGovernmentBEIS/inspect_ai** — reject class: *already-have*, with an architecture-mismatch
  component on the logging/viewer side. Confirmed via one API call: 2,459 stars, not archived, MIT
  license, pushed 2026-08-03T18:44:17Z (the day before this sweep) — active and well-maintained.
  Read `README.md` (pip-installed framework, `pip install -e ".[dev]"`, plus a separate
  TypeScript/React frontend as a git submodule at `src/inspect_ai/_view/ts-mono/` for the log
  viewer — the same shape this file already flagged when rejecting HELM's separate
  `helm-frontend/` package in the sweep below); `docs/eval-logs.qmd` (own binary `.eval`/`.json`
  log format, read via Inspect's own `inspect view` viewer, not the plain `.jsonl`/`.csv` every
  hai-teams runner already writes per `script-skeleton.md` item 8); `docs/checkpointing.qmd`
  (checkpointing there is for long multi-turn *agentic* runs — saves agent message state, sandbox
  filesystem, store/event history — and needs "a checkpointing-aware agent scaffold"; hai-teams'
  benchmarks are one-call-per-row classification/extraction, not multi-turn tool-using agents, and
  its functional equivalent is already simpler and at the right grain:
  `NegotiationToM/neg_eval_core.py:592`'s UID-keyed `.jsonl` `load_checkpoint()`, which survives
  full process death — scancel, crash, preemption — with zero framework-managed state);
  `docs/eval-sets.qmd` (task-level retry/resume, keyed on `@task`-decorated function
  reconstruction — more machinery for the same outcome a flat UID-keyed `.jsonl` already gives
  this project); `docs/_errors_and_retries.md` (29 lines, in full) and
  `src/inspect_ai/model/_retry.py` (in full — request-level `tenacity` retry with exponential-jitter
  backoff driven by a provider `should_retry` classifier, genuinely more sophisticated than
  hai-teams' own regex-parsed backoff, but exception-triggered: `ModelOutput.empty` at
  `_model_output.py:287` checks `len(self.choices) == 0`, "no choices returned" — not this project's
  specific and most costly recurring bug, HTTP 200 with a present-but-empty-string body per
  `script-skeleton.md` lines 24-29's invariant and `neg_eval_core.py`'s tracked "empty responses"
  failure; flagged as *inferred, not confirmed*, since the provider-specific parsers under
  `src/inspect_ai/model/_providers/` were not read); `docs/providers.qmd` (grepped — does support
  `together/...` and `grok/...` model strings with `TOGETHER_API_KEY`/`XAI_API_KEY`, real overlap
  with this project's actual provider set — the one genuine point of contact); and
  `docs/parallelism.qmd`/`providers.qmd`/`models.qmd` (grepped for slurm/cluster/multi-node/
  distributed — zero hits; no SLURM integration exists, so using it on Quest would mean wrapping
  `inspect eval` inside an `sbatch` script the same way hai-teams already wraps its own runners, no
  marginal gain on the SLURM axis and no analog to the shard-tag filename convention or
  partition-ceiling-tuned `.sh` scripts already tuned per model). Adopting anything here means
  migrating eight benchmark folders' worth of per-model runners into Inspect's
  `Task`/`Solver`/`Scorer`/`@task` abstractions and its own log format, for benefits that either
  don't match this project's shape or duplicate `neg_eval_core.py` at a coarser grain, already
  tuned to this project's specific incident history. Worth remembering without flipping the
  verdict: `src/inspect_ai/model/_providers/` is a concrete reference for Together/xAI
  provider-specific quirks if this project ever wants one shared multi-provider client instead of
  six thin runners each supplying their own `call_api` — a look-here-later note, not an action for
  this sweep.

- **parcadei/Continuous-Claude-v3** — reject class: *already-have*. Confirmed via one API call:
  3,880 stars, pushed 2026-01-26, not archived, not a fork. Read `README.md` in full (1,288 lines,
  main branch), the `.claude/hooks/src/` directory listing (34 files via the GitHub contents API),
  and `.claude/hooks/src/pre-compact-continuity.ts` in full (203 lines). Two independent reasons to
  reject, either sufficient alone. First, wrong problem domain: it is a general
  software-feature-development environment requiring Python 3.11+, `uv`, Docker (for a
  PostgreSQL+pgvector store), and a 12-step install wizard that compiles TypeScript hooks
  (`README.md:159-190,1216-1219`); its meta-skills (`/build`, `/fix`, `/tdd`, `/refactor`,
  `/review`, `/release`) chain through 32 agents such as `kraken` (TDD implementation) and
  `phoenix` (refactor planning) in a code-editing, build/test/PR loop (`README.md:485-497,618-662`)
  that hai-teams has no analog for — no build/test step, no PR workflow, no feature branches; this
  project's loop is submit→poll→pull `.jsonl`→score. Second — the specific mechanism this sweep was
  asked to weigh, YAML handoffs surviving context compaction plus a persistent memory daemon — is
  already solved here, fitted to this project, without the infrastructure cost:
  `pre-compact-continuity.ts` parses the session transcript and writes
  `thoughts/shared/handoffs/<session>/auto-handoff-<timestamp>.yaml` plus an appended brief to a
  `CONTINUITY_*.md` ledger tracking **edited files and build pass/fail counts** pulled from
  `.git/claude/branches/*/attempts.jsonl` (`pre-compact-continuity.ts:97-160`) — vocabulary for a
  code-editing session that hai-teams has no equivalent state for (job status, row counts and md5
  sync are already read live from files, not carried across a chat transcript). hai-teams solves the
  same underlying problem — decisions and procedure surviving past a single session — three
  different ways already: `.claude/workflows/*.js` persist the multi-phase procedure itself as data
  rather than a compacted chat summary, so a restart re-reads the script instead of a summarised
  transcript; `.claude/references/handoffs.md:7-20` specifies exactly what a dispatch must carry
  between fresh, memoryless subagents, as an explicit contract instead of an LLM-generated summary
  of a transcript; and `NegotiationToM/ISSUES.md` is the durable "learnings" store — git-committed
  markdown a human can diff and grep, versus a BGE-embedding archival-memory table requiring a
  running Postgres instance (`README.md:766-797`). The hard rule against installing anything (no
  Docker, no Postgres, no npm build) also forecloses actually trying this repo's version even
  experimentally.

- **composio-community/awesome-claude-plugins** — reject class: *irrelevant*. Confirmed via one
  API call: 1,857 stars, pushed 2026-07-26T09:02:02Z, not a fork, not archived — legitimate and
  maintained, just off-target. Read `README.md` in full (raw.githubusercontent.com, `master`
  branch — note: `main` 404s, the repo's default branch is `master`). It is a curated
  link-directory for Claude Code's plugin-marketplace ecosystem, not a pattern of its own: every
  entry points to a separate external repo, installed via `/plugin install` — off-limits under this
  task's rules, so the plugins themselves cannot even be inspected without a second hop of budget.
  Its 9 categories (Integrations, Frontend & Design, Git & Version Control, Code Quality & Testing,
  Backend & Architecture, DevOps & Performance, Documentation & Security, Developer Productivity,
  Companion & Personality, Image/Video Generation) are entirely general-software-feature-development
  plugins — commit-message generators, PR review, React/Next.js/TypeScript frontend tooling
  (`senior-frontend`, `frontend-developer`, `artifacts-builder`), security scanners, AWS cost
  optimization, a TikTok-posting plugin. None targets SLURM submission, checkpoint/resume for
  long-running batch jobs, provider-API failure modes (empty-body 200s, billing halts, reasoning-
  token config), or benchmark result auditing — this project's actual open problems per §4 below.
  The one structural idea that could transfer — bundling commands/subagents/hooks/skills as a
  coordinated unit dispatched with a defined brief (confirmed via `skill-bus/.claude-plugin/
  plugin.json` and `mcp-builder/.claude-plugin/plugin.json`, both read via
  raw.githubusercontent.com) — already exists here at project-internal grain: six narrow subagents
  (`.claude/agents/{watcher,evaluator,executor,reviewer,tracker,summarizer}.md`), saved multi-agent
  procedures (`.claude/workflows/*.js`), and a progressive-disclosure reference layer with a routing
  table (`.claude/references/README.md`). hai-teams doesn't need cross-project/cross-team plugin
  distribution — it's one repo with one fixed agent roster — so the marketplace packaging itself
  solves a distribution problem this project doesn't have.

### 4. The gap

Both surviving patterns from `cc-safety-net` guard the *command* an agent types. Every failure that
has actually cost this project time is either a property of file *contents*, or happened in a
process that never crossed the tool boundary in the first place — and no mechanism read this sweep
addresses either.

1. **Content, not command.** `git add NegotiationToM/NEG_Gemma/results` is a perfectly safe command
   that can still publish an unusable result set: an intention `.jsonl` with 4,760 rows instead of
   4,618 (the odd-length-dialogue bug recurring), a nonzero empty-`raw_response` rate, off-label
   predictions, or rows written under a checkpoint from a previous prompt config. An argv parser can
   never see any of that — `cc-safety-net` has no vocabulary for it, because in its world a file's
   contents are simply not the hazard. The missing piece is the mirror image of the gate this repo
   already has: `sbatch_sync_gate.py` refuses to *submit* when local and Quest disagree; nothing
   refuses to *publish* when a results directory disagrees with the expected row counts in
   `NegotiationToM/DATA_NOTES.md`. That gate would be maybe 40 lines — on a `Bash` command matching
   `git add .*results`, resolve the pathspec, read each `.jsonl` it would stage, and block on a
   row-count mismatch or an empty-response rate above zero, printing the count it saw against the
   count expected. It would fire on exactly the commands `run-model.js:273` and `:409` already
   instruct agents to run, and it defends the boundary this project cares about most: what reaches
   the shared repo.
2. **The detached-process blind spot.** The one incident that motivates the whole git-safety
   proposal happened inside `watch.sh` (`ISSUES.md:170-199`), where no PreToolUse hook runs.
   `run-model.js` supervises for `maxHours`, and this project's standing pattern is long-lived
   background loops — anything launched with `run_in_background` inherits an unconstrained shell.
   Nothing in either `cc-safety-net` candidate, nor in `inspect_ai`, `Continuous-Claude-v3`, or
   `awesome-claude-plugins`, reaches there, and `ISSUES.md:192-199` still records that fix as "not
   yet shipped," with the watcher entry left open.

The honest ordering: fix the one-line contradiction at `quest-agent-workflow.md:22` today; build the
results-integrity publish gate next, because it is the one item with no existing analog anywhere in
this project; treat the git argv guard as the cheap third item, not the headline.

### 5. Not examined

None. All four candidates the prior 2026-08-04 sweep deferred under its own "Not examined" section
were read this round: `kenryu42/cc-safety-net`, `UKGovernmentBEIS/inspect_ai`,
`parcadei/Continuous-Claude-v3`, `composio-community/awesome-claude-plugins`. No repo was skipped
for budget reasons this sweep.

---

## Sweep — 2026-08-04

Repos read this sweep: `facebookincubator/submitit`, `snakemake/snakemake`, `Netflix/metaflow`,
`stanford-crfm/helm`, `kochetkov-ma/claude-brewcode`.

### 1. Adopted / to adapt

No survivors. All five repos examined this sweep were screened out at the repo level — wrong
submission model, wrong compute target, or a mechanism this project already has at a finer grain —
before any individual pattern reached judgment. Nothing here is PROPOSED for adoption this round.

### 2. Rejected patterns

None to log separately from the repo-level reasons below. This sweep's `verdicts` list came back
empty: every candidate repo failed the fit check before a specific mechanism was isolated and
scored on its own merits, so there is no rejected *pattern* distinct from the "Repos screened out"
entries — the repo-level reason **is** the rejection reason in every case this sweep.

### 3. Repos screened out

All judged 2026-08-04.

- **facebookincubator/submitit** — reject class: *already-have*. Read `README.md`,
  `docs/checkpointing.md`, `docs/structure.md` (raw.githubusercontent.com) plus one API call for
  repo metadata (1,630 stars, not archived, pushed 2026-01-14). It is an in-process job-submission
  library — you call `executor.submit(fn, *args)` from a running Python driver, which pickles the
  call and writes the sbatch file for you; the README's own "Non-goals" section says it *replaces*
  `sbatch <script>.sh`, not wraps it. Its checkpoint/resume story requires catching a Slurm
  preemption *signal* on a stateful, picklable callable (`docs/checkpointing.md`), and needs
  `SlurmctldParameters=preempt_send_user_signal` set cluster-wide — outside what uwr0681 controls on
  Quest. hai-teams' checkpoint is already file-level and finer-grained: a UID-keyed `.jsonl`,
  `--save-every 20`, that survives any process death (scancel, preemption, crash) with zero
  in-process state to pickle. Not the failure mode this project actually hits (billing halts, quota,
  empty-body 200s, stale code, sync drift — none of which are "Slurm preempted my job").

- **snakemake/snakemake** — reject class: *already-have*. DAG-of-rules pipeline framework; rule
  completeness is judged by output file existence/hashing, coarser than hai-teams' row-level UID
  checkpoint inside a single `.jsonl`. Native Slurm support is a separate
  `snakemake-executor-plugin-slurm` package, versus this project's hand-tuned `#SBATCH` scripts with
  measured partition ceilings and a shard-tag filename convention. The core abstraction — a DAG over
  declared input/output files — doesn't fit hai-teams' shape: one long-running per-model
  load→call→checkpoint→score loop with a mandatory human review gate between pilot and full run.
  Adopting it means installing a ~20-dependency framework (pulp, jinja2, docutils, gitpython,
  multiple `snakemake-interface-*` packages) to replace orchestration already solved at a finer
  grain, with no answer for the failures actually hit (empty-body 200s, billing/quota halts, a job
  reporting RUNNING while hung inside an API call).

- **Netflix/metaflow** — reject class: *irrelevant* (infra mismatch). Read `README.md`,
  `docs.metaflow.org/scaling/failures`, `docs.metaflow.org/scaling/checkpoint/introduction`. Remote
  execution goes through exactly two backends, `@batch` (AWS Batch) or `@kubernetes`
  (`docs.metaflow.org/scaling/remote-tasks/introduction`) — no SLURM backend exists, so its scaling
  story means leaving Quest's `sbatch`, not extending it. `@checkpoint` is a separate pip package
  (`metaflow-checkpoint`), explicitly documented as "not a built-in part of core Metaflow yet... its
  APIs may change," and snapshots a whole step, coarser than hai-teams' existing per-row checkpoint.
  `@retry`/`@catch` only fire on a raised exception — this project's actual failure, HTTP 200 with an
  empty body, raises nothing, so the decorator would silently miss exactly the failure this project
  has been burned by (see `NegotiationToM/ISSUES.md` on the Qwen empty-response incident). Full
  `FlowSpec`/`@step` DAG framework for six flat scripts is architectural overkill regardless.

- **stanford-crfm/helm** — reject class: *already-have*. Read `README.md`, `docs/tutorial.md`, the
  top-level and `src/helm/common` directory listings, and `src/helm/common/cache.py`; one API call
  confirmed 2,871 stars, pushed 2026-08-01. A 130MB pip-installed benchmark-leaderboard framework
  (Scenario/Adapter/Metric classes under a declarative RunSpec, `helm-run`/`helm-summarize`, and
  `helm-server` — the last backed by a separate React/TypeScript package, `helm-frontend/`, exactly
  the kind of artifact this project's own framing says not to bend to fit). Every mechanism that
  could transfer already exists here in project-specific form: its `--suite` output tree ↔
  hai-teams' per-model `results/<TASK>/` layout; its per-provider `Client` classes ↔ six thin runners
  sharing `neg_eval_core.py`; its generic request-keyed cache+retry decorator (`cache.py`,
  `SqliteKeyValueStore`/`MongoKeyValueStore` behind `get_retry_decorator`) ↔ hai-teams' per-row UID
  checkpoint/resume, which is already tuned to this project's specific failures (empty-string on
  HTTP 200, dynamic backoff parsed from the provider's own error text, billing check called first in
  every `except`) that HELM's generic decorator doesn't address. No HELM analog exists for SLURM
  sharding, the local↔Quest md5 sync check, or the dataset traps in `DATA_NOTES.md`.

- **kochetkov-ma/claude-brewcode** — reject class: *already-have*. Read `README.md` in full and
  `brewcode/hooks/forced-eval.mjs` and `brewcode/agents/developer.md` in full; one API call
  confirmed 29 stars, pushed 2026-08-02T21:05:14Z. A plugin suite for general software
  feature-development (implement/test/review/architect, docs sync, SSH/CI admin), distributed only
  via `claude plugin marketplace add` — out of bounds to install under this task's rules regardless.
  Its real mechanism, bounded subagent units dispatched with an explicit multi-field brief and
  STATUS-shaped reports (`brewcode/agents/developer.md:11-22`, `README.md:175-184`), already exists
  here as six narrow subagents each returning a `STATUS:` line, with the same brief contents
  independently specified in `.claude/references/handoffs.md:9-19,31-37`. Its one mechanism with no
  local analog — a `UserPromptSubmit` hook that re-injects a 3-line role/delegation reminder into
  every prompt (`brewcode/hooks/forced-eval.mjs:26-31,70-77`) — targets a "main session forgets to
  delegate" failure that this project's `CLAUDE.md` is structurally built to avoid ("The planner is
  this session, not a subagent... Sequencing... stay here"); adopting it would push toward the
  always-delegate posture this repo's design deliberately rejects.

### 4. The gap

None of the five repos read this sweep proposed anything for hai-teams' actual open failure modes:
a SLURM job that reports RUNNING while hung inside a provider API call, an HTTP 200 with an empty
body, a provider billing/quota halt, or drift between the local and Quest copies of the shared
`neg_eval_core.py`. Every repo was rejected at the architecture level — wrong submission model
(submitit), wrong compute target (metaflow's AWS Batch/Kubernetes-only backends), or a mechanism
this project already has at a finer grain (snakemake, helm, claude-brewcode) — before any of their
individual patterns were extracted and scored against those specific problems. No pattern reached
the judge this sweep.

### 5. Not examined

`UKGovernmentBEIS/inspect_ai`, `parcadei/Continuous-Claude-v3`, `kenryu42/cc-safety-net`,
`composio-community/awesome-claude-plugins` ranked below this sweep's `maxRepos=5` cutoff and were
never read — no README, no file content, no API call was spent on any of them. Their absence from
"Repos screened out" above is not a verdict; a future sweep should evaluate them fresh rather than
assume they were checked and rejected.
