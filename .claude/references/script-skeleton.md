# Eval script skeleton

<!-- size-budget: 8000 -->
<!-- One job: the runner shape, as a numbered checklist a diff is checked against. It grew when
     rule 7 stopped being a style note and became the fairness rule, which needs its evidence,
     and again when 7b took the sharding pattern from NegotiationToM. -->

The order below is the shape every runner in this project follows; NegotiationToM's six are the
worked example. Copy the closest existing folder and swap the client rather than starting from this
list.

**1. Paths and client.** Resolve from the script's own location, never from cwd:

```python
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # benchmark root
load_dotenv(os.path.join(ROOT, ".env"))
client = Together(api_key=os.getenv("TOGETHER_API_KEY"), timeout=180)  # see provider-gotchas.md
```

**2. Prompt builders.** System prompt from the benchmark's `src/configs/prompts.yaml` and
`response.yaml`; choices as a lettered menu; answer requested as JSON.

**3. `parse_json(text)`.** Accept bare JSON and ```` ```json ```` fenced output; return `None` on
failure.

**4. `call_api(messages, model, max_retries=3)` — the shared retry contract.**

- up to 3 attempts, `temperature=0`, and an explicit `max_tokens` chosen from what the model
  actually needs (see [provider-gotchas.md](provider-gotchas.md)) rather than habit
- **retry on empty-string responses, not just exceptions** — HTTP 200 with an empty body is the
  most common failure here and raises nothing:
  ```python
  content = (resp.choices[0].message.content or "").strip()
  if not content: time.sleep(5); continue
  ```
- dynamic backoff from the provider's own message:
  `re.search(r'try again in ([\d.]+)(ms|s)', err)`, else 5s
- **every `except` block calls `halt_on_billing(error, model, SCRIPT_DIR)` first** — see
  [provider-gotchas.md](provider-gotchas.md) for why the narrow string test is not enough
- `time.sleep(2.0)` after every success
- **log the exception** in every `except` block. Without it a `TypeError` from a bad call signature
  is retried as if it were a network fault, then scores 0, with nothing in the SLURM log

**5. `call_and_parse()` — second retry layer.** If `parse_json()` returns `None` on a *non-empty*
response, re-issue the call, up to 3 times. Call sites use this, not bare `call_api` + `parse_json`.

**6. Checkpoint / resume.** `.jsonl` keyed by a stable UID (`qid`, or
`"<dialogue_id>_<agent>_<task>"`). On start, load completed UIDs and skip them. `--save-every 20`.
Always persist the raw response so failures can be inspected.

> **Check for stale checkpoints before every full run.** Resume skips any UID already present, so
> leftovers from an older code version make a run "succeed" in seconds while emitting old, wrong
> data. NEG_GPT held 14,280 such rows. Archive to a timestamped directory; never delete outright.
>
> Rows written *empty* are skipped forever by a plain resume too, because `load_checkpoint` adds
> every uid to the done set regardless of whether its response was usable. Prune before resubmitting.

**7. Normalisation — every model in a benchmark is scored by the SAME lenient matcher.** Generous
about how an answer is written, strict about what it says. This is a standing rule from the user
(2026-08-29) and it applies to every benchmark, not just the one being written: a strict scorer and
a lenient one produce numbers that cannot be compared, and the difference is not small. bbh scored
five of its eight models strictly and three leniently; rescoring the identical stored responses with
the one lenient matcher moved deepseek from 0.448 to 0.961, kimi from 0.243 to 0.884 and
gpt-4o-mini from 0.308 to 0.834. Those models were being scored on whether they wrote `(B)` or `B`.

**Put the matcher in the shared core and import it — never copy it into a runner.** A copied scorer
is a scorer that can drift; an imported one cannot be opted out of.
`bbh/bbh_eval_core.py::score_response` is the reference implementation; the extracted, generalised
version is `neg_eval_core.py::clean_surface`:

```python
text = value.strip().strip("\"'`").strip()   # quotes and backticks are packaging
text = re.sub(r"\s+", " ", text)             # collapse whitespace runs
text = text.strip(" .,;:!")                  # trailing punctuation
return ALIASES.get(text.lower(), text.title())
```

Add aliases for what the models **actually emit** — check a pilot rather than guessing. Measured
here: GPT returned lowercase items 486 times (17% of its values), plus `Wood` for `Firewood` and
`unknown`/`N/A` for `Not Given`. Check both `"high"` and `"High"` key casings.
**Normalise model output only, never gold labels** — gold is canonical, and rewriting it changes
the answer key.

**7b. Sharding — copy NegotiationToM's, including the part that looks trivial.**
`NEG_GPT/openai_neg_eval.py` + `merge_shards.py` is the reference; bbh and mmlu now follow it.
Four details carry the weight:

- **`shard_tag` is EMPTY when `total_shards == 1`.** That is what lets sharding be added to a
  benchmark whose existing results are single untagged files — an unsharded run keeps the exact
  filename it always wrote, so nothing already on disk is invalidated and no merge is needed.
- **Tag EVERY artefact, not just the `.jsonl`.** The `.csv` and `_overall.csv` need it too, or
  shard 1 silently overwrites shard 0's summary while the row files look fine.
- **Slice after `enumerate`, not before.** `idx` must index the whole task; if each shard
  re-numbers from 0 the merged file has five rows claiming `idx 0`.
- **The merge reports a missing shard, merges what exists, and exits non-zero.** Failing outright
  throws away four good shards for one dead job; failing silently reports a partial number as a
  whole one. Name the absent file and stamp the row count actually merged.

**8. `evaluate()`.** Per-sample CSV plus an overall CSV. Sanitize model names with
`.replace(".", "_").replace("/", "-")`.

**9. Entry point.** `argparse` with `--model`, `--task all`, `--save-every 20`.

## Invariants a change is checked against

These are settled; a change that breaks one is wrong unless it argues otherwise explicitly.

| Invariant | Why |
|---|---|
| Gold labels are never rewritten — normalisation applies to model output only | rewriting gold changes the answer key |
| Shard-level and merged metrics call the **same** scoring function | two implementations drift silently |
| An empty API response is retryable, not a scored zero | HTTP 200 + empty body raises nothing |
| Every `except` block logs the exception | otherwise a `TypeError` is retried as a network fault and scores 0 |
| A timeout exception derives from `BaseException` | `except Exception` in each runner would swallow it |
| Job scripts `export PYTHONUNBUFFERED=1` | or the log is empty while the job runs |
| Shard outputs carry a shard tag | untagged `_overall.csv` files overwrite each other |
| Model-specific prompt tweaks stay in that model's runner | the shared builders must serve every model identically |
| Row totals and exclusions match the benchmark's page, not another benchmark's | `benchmarks/<group>/<name>.md` holds the counts; a total that "looks about right" is how a returning bug survives review |
