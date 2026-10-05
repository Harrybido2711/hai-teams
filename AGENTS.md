<!-- size-budget: 12500 -->
<!-- One job: map a situation to the fastest correct first move and to the one file that holds the
     detail. An index keyed by symptom, where references/README.md is keyed by topic. It grows by
     one row per new situation; the knowledge itself stays in .claude/references/. -->
# The fast path

For any coding agent in this repo — Claude Code, Codex, or another. It answers one question: **I am
in situation X — what do I do first, and where is the detail?** It holds no knowledge of its own.
Every row points at the file that does; if a row and its target disagree, the target wins and the
row is what to fix.

## Before anything

1. **[`CLAUDE.md`](CLAUDE.md) binds every agent, Codex included.** Its two gates above all: nothing
   is transferred to Quest or submitted until **the user has verified the scripts**, and nothing
   enters the record (benchmark page, both workbooks) until **the user has confirmed the run**.
2. **Orient in one page:** [`.claude/INDEX.md`](.claude/INDEX.md). Working on one benchmark? Its page
   and its group page under [`.claude/references/benchmarks/`](.claude/references/benchmarks/README.md).
3. **Retrieve, don't read.** The references are ~150 KB; a task uses a few sections of them.

   ```bash
   python3 .claude/scripts/check_docs.py --brief <term>…   # the matching sections, whole
   python3 .claude/scripts/check_docs.py --model <id>      # one model across every file
   python3 .claude/scripts/check_docs.py --impact <term>   # every file a change must touch
   ```

## Situation → first move → detail

Bare filenames below are in `.claude/references/`; `CLAUDE.md` and `PLAN.md` are at the repo root.
**Bold** names are skills — `.claude/skills/<name>/SKILL.md`, a numbered procedure any agent can
follow, Codex included. A workflow named beside one runs that whole stretch in Claude Code.

### Starting

| Situation | First move | Detail |
|---|---|---|
| "Where is the project, what is left to run?" | read `PLAN.md` § Two workbooks and § Open work; `ssh quest squeue -u uwr0681` for what is live. Never answer from memory | `PLAN.md` |
| A benchmark nobody has analysed | read the code before the paper; write its page from the template **and** its group-index row in one edit; mark anything unestablished as such | **`analyse-benchmark`**; `benchmarks/README.md` |
| A new model on an existing benchmark | `--model <id>` and `--brief <benchmark>`; copy the closest existing model folder and swap the client; import the scorer from the benchmark's shared core | **`write-runner`** · workflow `new-runner`; `model-calls.md`, `model-parameters.md`, `script-skeleton.md` |
| A model page lists a parameter | probe one real call before the run. A refused *value* is not a refused *parameter* — never drop a cap because its name appeared in an error | `model-parameters.md` rules 6–7 |
| Should the model show its reasoning? | the benchmark's own upstream README decides, resolved at run time. Hidden reasoning is capped regardless | `model-parameters.md` rules 1, 3 |

### Running on Quest

| Situation | First move | Detail |
|---|---|---|
| "Are local, GitHub and Quest in sync?" | `python3 .claude/scripts/sync_audit.py --fetch` — git refs, then code (md5) and results (size) for every benchmark with a Quest path. Read-only. A finding is not always a fault: a renamed folder or merged shards explain themselves | **`quest-sync`**, **`pull-results`** |
| Scripts look ready | **stop and hand them to the user.** After their OK: local `--limit` pilot → transfer *every* modified file as one set → `md5sum` both sides, printing both list lengths → `sbatch` | **`quest-sync`**, **`submit-run`** · workflow `launch-run`; `quest-cluster.md` § Transferring |
| Which Quest directory? | the Paths table on the benchmark's page. Ours live under `Interpersonal_processes_benchmarks/` and `Tasks_benchmarks/`; flat top-level folders belong to other accounts | `quest-cluster.md` |
| `ssh quest` → `Host key verification failed` | the alias is missing from `~/.ssh/config`; reinstall it, check with `ssh quest hostname` | Claude memory `quest-access`; `quest-cluster.md` |
| The pre-submit hook says *in sync* | for anything but NegotiationToM it compared the wrong files. Run `sync_audit.py <benchmark>` | **`quest-sync`**; `quest-cluster.md` § The pre-submit gate |
| A manual md5 compare says *in sync* | print both list lengths. zsh does not split `$FILES`; `join` needs input sorted by filename | **`quest-sync`** step 2; `quest-cluster.md` |
| How many shards? | 5. Lower the per-item sleep before adding shards; keep ≥ ~25 items a shard | `quest-cluster.md` § SLURM |
| `.env` missing on Quest | copy it from a sibling benchmark directory of ours on Quest. Never copy it off Quest | `quest-cluster.md` |

### Watching a run

| Situation | First move | Detail |
|---|---|---|
| "How is it going?" | halt markers → rows in the `.jsonl` and its mtime → `squeue`/`sacct`. **Rows written, never job state; never log size** (stdout is unflushed). | **`check-run`** · workflow `monitor-run`; `quest-cluster.md` § Reading the live state |
| RUNNING, rows not growing | a call is hung. `srun --jobid=<id> --overlap` and read `/proc/<pid>/wchan`. `timeout=` is not a guard; the SIGALRM watchdog is | `provider-gotchas.md` § Timeouts |
| `COMPLETED 0:0` | not evidence. Count the non-empty response rate and null predictions | `shared-context.md` § Counting rows |
| A halt marker exists | it names the cause and whether to prune before resubmitting — quote it | `quest-cluster.md` |
| Watching for hours (Claude Code) | a fixed-count loop ends silently. Use one persistent `Monitor` filtered on stalls, restarts and completion — not on the routine status line | — |

### Something is wrong

| Situation | First move | Detail |
|---|---|---|
| A job is writing bad data | **standing authorisation, do not ask:** `scancel` → fix locally → transfer the whole change set, verify `md5sum` → decide resume / prune / archive and say which → resubmit | **`kill-and-resync`** · workflow `fix-run`; `CLAUDE.md`; `quest-cluster.md` § Replacing the code |
| Fixed locally, job still wrong | the live process imported the old modules. Cancel *before* transferring | `quest-cluster.md` |
| HTTP 200, empty body | that provider's row in the gotchas table; empties are retried, never scored as zero | `provider-gotchas.md`; `script-skeleton.md` §4 |
| The same error on every call (`ValidationError`, `INVALID_ARGUMENT`, `Extra inputs`) | a permanent config error — make it fatal. Check the SDK version Quest's interpreter has, not yours | `provider-gotchas.md` § google-genai |
| The same few items fail at a fixed seed (`MALFORMED_RESPONSE`) | a property of those inputs; retrying is waste. Record them as unanswerable | `provider-gotchas.md` |
| Billing / quota / "credits" wording | the shared `halt_on_billing` classifier — never a hand-written string test | `provider-gotchas.md` § Classifying a refusal |
| 429, TPM, requests per day | shards and sleep; requests-per-day is the limit that bit hardest | `quest-cluster.md` § SLURM; `Tasks_benchmarks/DocVQA/OPENAI_EVAL_NOTES.md` |
| A provider refuses the health-check probe | probe with the real prompt builders (`NegotiationToM/preflight.py` is the pattern) | `provider-gotchas.md` |
| A resumed run "finished" in seconds | stale checkpoint. After any prompt or decoding change, archive (timestamped), never resume | **`submit-run`** step 2; `script-skeleton.md` §6 |
| Empty rows never get retried | resume marks every uid done, empty or not. Prune them first (`NegotiationToM/prune_failed_rows.py`) | **`submit-run`** step 2; `script-skeleton.md` §6 |
| A sharded summary holds one category | untagged shard outputs overwrote each other | `script-skeleton.md` §7b |

### Scores

| Situation | First move | Detail |
|---|---|---|
| Scores low, or split by output format | strict scorer. One lenient matcher per benchmark, imported from its core; **rescore the stored rows offline** — no rerun needed | **`rescore-offline`**; `script-skeleton.md` §7 |
| A pilot or run has finished — can its numbers be believed? | one real row's schema → counts against the page → empties and off-label predictions → one configuration → recompute the score | **`audit-results`** · workflow `finish-run` |
| The CSV shows more ids than rows | newlines inside responses. Count from the `.jsonl` | `shared-context.md` |
| A count of zero, or larger than the row count | a guessed field name, or shards and merged file counted together. Print one real row's keys first | Habits below |
| A score you cannot read | check the denominator: `{task}_scored_rows`, and the exclusions on the page | the benchmark's page |

### Recording and finishing

| Situation | First move | Detail |
|---|---|---|
| A model finished a whole benchmark | wait for the user's confirmation. Then **one** edit: the page, both workbooks, the `Provenance` row — rebuilt from the per-task files on disk | **`record-results`** · workflow `finish-run`; `sync-and-consistency.md` § Layer 4; `PLAN.md` § Two workbooks |
| A selected model has not run a sheet | its cell stays blank. Never borrow the number of the model that used to hold the slot | `PLAN.md` § Two workbooks |
| The commit is blocked by the doc check | read the finding first, then fix it or declare it with a reason | `doc-check.md` |
| "Have we hit this before?" | grep the benchmark's own notes (named on its page), `Interpersonal_processes_benchmarks/NegotiationToM/ISSUES.md` (false alarms at the end), `provider-gotchas.md` | — |
| A task is done | the sync pass: `check_docs.py` → stage **explicit paths** → commit → push `origin` and `backup`; plus `md5sum` on Quest if code that lives there changed. Never `git add -A`; never push `upstream` | **`sync-pass`** |

## Habits — each one cost a session

1. **Verify the intermediate result, not only the final one.** A negative finding ("0 files", "never
   run") needs a positive control first — the path exists, the command scanned something. Never
   `2>/dev/null` a search you will report, and use absolute paths: `cd` persists between shell calls,
   and a search from the wrong directory looks exactly like an empty one.
2. **Print one real row before counting anything.** Field names differ per runner.
3. **Match structured data by exact value, never substring.** The workbooks' `Provenance` column A
   holds compound values (`MMLU · DocVQA`); `"DocVQA" in cell` once cleared MMLU's row.
4. **Cut a document by explicit anchors**, section by section — never "from heading A to heading B"
   on an assumed order. Three sections were moved silently that way.
5. **Follow the entry point** — the `.sh` that submits, the config it reads — not the file whose
   name looks right.
6. **Local is for `--limit` pilots and offline rescoring.** A real run happens on Quest, even when
   the sync is awkward — solve the sync.
7. **Do not re-ask what the user has answered.** Batch independent commands. Report with the output
   attached, not with an assertion.

## Where a lesson goes

A lesson left in a transcript dies with the session. Write it where the next agent will look:

| What you learned | Where it goes |
|---|---|
| a trap, count, path or run order of one benchmark | its page under `.claude/references/benchmarks/` |
| how a provider or SDK fails | `provider-gotchas.md` |
| Quest, SLURM, transfers | `quest-cluster.md` |
| the runner shape, an invariant a diff must keep | `script-skeleton.md` |
| the history of a problem, with its dead ends | the benchmark's own notes file (NegotiationToM: `ISSUES.md`) |
| a rule the user set | `CLAUDE.md` |
| a new situation | a row here, pointing at where the detail went |

## Tools, by agent

Skills are the steps, agents the roles, workflows the baselines that compose them —
[`.claude/tools/README.md`](.claude/tools/README.md) lists all three.

- **Claude Code** — invoke a skill by name; run a workflow by path, or read it, adapt it to the task
  and pass it inline (its detail page's *Adapting* section says what may change). Wide reading goes
  to `Explore` or `summarizer` with the `check_docs.py` command, not a file list. What a session
  cost: `python3 .claude/scripts/token_report.py --top 15`.
- **Codex** — no subagents and no `Workflow` tool, and none is needed: a skill is a plain numbered
  procedure, and a workflow's *Phases* table (`.claude/tools/<workflow>.md`) is the order to run its
  skills in by hand. Where a phase names an agent, follow `.claude/agents/<agent>.md` as a checklist
  in a separate pass — for `reviewer` above all, before anything reaches Quest.
