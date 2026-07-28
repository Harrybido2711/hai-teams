# hai-teams — Evaluation Script Conventions

How we write and run benchmark evaluation scripts in this repo. This is the *engineering* side —
folder layout, retry contract, checkpointing, SLURM. For *what* each benchmark requires
(metrics, gold labels, whether an LLM judge is needed), see `benchmark_evaluation_guide.md`.

Per-benchmark design notes and post-run findings live at:

- `EmoBench-master/EMO_SCRIPT.md` — EmoBench (EA/EU), plus notes on the BBH and DocVQA scripts
- `NegotiationToM/Negotiation_script.md` — desire/belief/intention tasks, sharding, pilot workflow

Append to those after every run: new model rows, root cause → fix pairs, final accuracy tables.

---

# Agent workflow

Work on this project is split between a planner and five specialist subagents, defined in
`.claude/agents/`.

## The planner is the main session, not a subagent

**A subagent cannot spawn another subagent** — only the top-level session can. So the planner role
is how the main session behaves, not a file in `.claude/agents/`:

- Decide what needs doing and in what order; delegate the doing.
- Do not read the repo widely in the main context — that is `summarizer`'s job, and its findings
  come back as a summary instead of filling the main context with file contents.
- Hold the decisions, the constraints and the open questions. Subagents start cold every time and
  only know what they are told, so a delegation must carry the *decision*, not the problem.
- Keep the user's open questions visible and unblock them early.

## The five subagents

| Agent | Does | Never |
|---|---|---|
| `summarizer` | Reads widely, returns conclusions — layout, what a script does, how two implementations differ | Changes anything |
| `executor` | Writes and fixes scripts, transfers to Quest, submits/cancels jobs, runs verification | Re-opens a decision |
| `watcher` | Reports live job state, progress, errors, stalls, quota | Fixes or resubmits |
| `evaluator` | Judges whether results are trustworthy, what they mean, what to do next; audits token cost | Changes code or jobs |
| `tracker` | Records problems, rejected attempts and shipped fixes | Invents entries it did not verify |

## The loop

```
planner ──▶ summarizer   (what is the current state?)
        ──▶ executor     (make this decided change / submit this run)
        ──▶ watcher      (how is it going?)  ──▶ evaluator
                                                   │
            planner ◀── recommendation ────────────┘
        ──▶ tracker      (record what broke and what fixed it)
```

`watcher` observes but does not judge; `evaluator` judges but does not act; `executor` acts but does
not decide. Keeping those separate is what stops a monitoring signal turning into an unreviewed fix.

## Cost

`python3 .claude/scripts/token_report.py [--top N]` reports per-task input / output / cache tokens
and USD, parsed from Claude Code's own transcripts. A task is one user turn plus the work that
followed it. Cache reads dominate (~97% of billed tokens measured here) and grow with conversation
length — which is the concrete reason to send wide file reading to `summarizer` rather than doing it
in the main context.

---

## 1. Folder layout

Every benchmark uses **one folder per model**, containing a Python eval script, a SLURM `.sh`,
and a `results/` directory. Copy the closest existing folder and swap the client — do not invent
a new structure, because the cross-model summary CSVs depend on this shape.

```
<BENCH>_<Provider>/            e.g. EMO_Qwen/, NEG_GPT/
├── <provider>_<bench>_eval.py
├── run_<bench>.sh
└── results/
    ├── <TASK_A>/
    └── <TASK_B>/
```

Existing examples: `EmoBench-master/EMO_{Gemini,XAI,Qwen,Gemma,Deepseek}/`, `NegotiationToM/NEG_GPT/`.

---

## 2. Script skeleton

Keep these sections, in this order.

**1. Paths and client.** Resolve everything from the script's own location, never from cwd:

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # benchmark root
load_dotenv(os.path.join(ROOT, ".env"))
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=18000)
```

Results go to `os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", task)`.

**2. Prompt builders.** System prompt assembled from the benchmark's `src/configs/prompts.yaml`
and `response.yaml`; choices rendered as a lettered menu (`A) …`, `B) …`); the model is asked to
answer as JSON.

**3. `parse_json(text)`.** Accept both bare JSON and ```` ```json ```` fenced output. Return `None`
on failure (the sample then scores 0).

**4. `call_api(messages, model, max_retries=3)` — the shared retry contract.**

- up to 3 attempts, `temperature=0`, explicit `max_tokens=8192`
- **retry on empty-string responses, not just exceptions.** HTTP 200 with an empty body is the
  most common failure mode here and raises nothing:
  ```python
  content = (resp.choices[0].message.content or "").strip()
  if not content:
      time.sleep(5); continue
  ```
- dynamic backoff off the provider's own message:
  `re.search(r'try again in ([\d.]+)(ms|s)', err)`, else default 5s
- hard stops: `insufficient_quota` → `raise SystemExit` (billing, retrying is pointless);
  `requests per day` → `return None` (daily quota gone)
- `time.sleep(2.0)` after every success to stay under RPM/TPM

**5. `call_and_parse()` — second retry layer** (see `NEG_GPT/openai_neg_eval.py`). If
`parse_json()` returns `None` on a *non-empty* response, re-issue the API call, up to 3 times.
Call sites should use this rather than bare `call_api` + `parse_json`.

**6. Checkpoint / resume.** Write a `.jsonl` keyed by a stable UID — `qid` for EmoBench,
`"<dialogue_id>_<agent>_<task>"` for NegotiationToM. On start, load completed UIDs and skip them.
`--save-every 20`. Always persist the raw `model_response` so failures can be inspected.
To force a clean re-run, delete the `.jsonl` checkpoints first.

**7. Normalization before scoring.** `.strip().upper()` for letter answers; canonical maps for
free-text labels (`"food"` → `"Food"`, `"not given"` → `"Not Given"`); check both `"high"` and
`"High"` key casings in prediction dicts. **Normalize model output only — never touch gold labels.**

**8. `evaluate()`.** Writes two CSVs per task: `{model}_en.csv` (per-sample) and
`{model}_en_overall.csv` (one row per category plus an `Overall` row). Sanitize the model name with
`.replace(".", "_").replace("/", "-")`.

**9. Entry point.** `argparse` with `--model`, `--task all`, `--save-every 20`; tasks looped
sequentially.

---

## 3. Provider-specific gotchas

All of these have already cost a debugging cycle. Most fail with HTTP 200 and no exception, so a
run looks "complete" while silently scoring 0.

| Provider | Client | Must do |
|---|---|---|
| OpenAI `gpt-4o-mini` | `openai.OpenAI` | baseline pattern |
| DeepSeek `deepseek-reasoner` | `openai.OpenAI`, `base_url="https://api.deepseek.com"`, `timeout=7200` | `temperature=0` (reasoning model); when `content` is empty the answer sits in `reasoning_content` — fall back to it before retrying |
| Gemini `gemini-2.5-flash` | `google.genai.Client` | no `system` role in the messages list — pass `GenerateContentConfig(system_instruction=...)`; do **not** set `max_output_tokens` (256 truncated JSON mid-object) |
| xAI `grok-3-mini` | `xai_sdk.Client` | no message dicts — `client.chat.create(model=...)`, then `chat.append(xai_system(...))` / `chat.append(xai_user(...))`, then `chat.sample()` |
| Qwen `Qwen/Qwen3.5-9B` | `together.Together`, `timeout=18000` | thinking model: emits a long `<think>` block. With no explicit `max_tokens`, Together's default budget is consumed during thinking and the content comes back empty. **Set `max_tokens=8192`** and retry on empty |
| Gemma `google/gemma-4-31B-it` | `together.Together` | returns an empty string intermittently at HTTP 200 — retry up to 5× before skipping the sample |

Rule of thumb: for every provider, treat an empty string as a retryable failure.

---

## 4. SLURM on Quest

```bash
#SBATCH --account=p32983
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00

module purge
/projects/p32983/pythonenvs/hai-teams/bin/python <script>.py --task all --save-every 20
```

**Prefer a single job (no `--array`)** so every category lands in one output file and no merge step
is needed. Shard only when the run is genuinely too long — NegotiationToM is ~14,280 calls, which
is ~8h single-threaded. When sharding:

- Quest allows **at most 5 simultaneous array jobs** → `--array=0-4`. More shards go in batches
  (`--array=0-4`, then `--array=5-9`).
- Shard outputs **must** carry a shard tag: `{model}_shard{N}of{M}.jsonl`.
- A separate `merge_shards.py` dedupes by UID and recomputes final scores.

> Known bug to avoid: the original EmoBench scripts wrote `_en_overall.csv` with no shard tag, so
> each shard overwrote the previous one and the final file held only the last shard's single
> category. Results for GPT-4o-mini, DeepSeek-Reasoner and Grok-3-mini had to be recomputed by
> merging the four shard `.jsonl` files by hand.

---

## 5. Run order

1. **Pilot** when cost is unknown — `sbatch run_pilot.sh`: ~50 random samples (seed=42), sums
   `resp.usage` prompt/completion tokens across all attempts, prints projected total cost and wall
   time. Read `log_pilot.txt` before committing to a partition and time limit.
2. **Full run** — `sbatch run_<bench>.sh`. Monitor with `squeue -u $USER` and
   `wc -l results/*/*.jsonl`.
3. **Merge** — `bash run_merge.sh`, only if the run was sharded.
