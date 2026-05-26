# Script Analysis Notes

## 1. BBH — Models and How They Are Called

All BBH eval scripts share the same structure: load an API key from `.env`, iterate over task splits, call the model with a `"Final Answer: <answer>"` prompt, and save per-split CSVs plus an overall results CSV.

| File | Model | SDK / Client |
|------|-------|-------------|
| `openai_eval.py` | `gpt-4o-mini-2024-07-18` | `openai.OpenAI` |
| `gemini_eval.py` | `gemini-2.5-flash` | `google.genai.Client` |
| `deepseek_eval.py` | `deepseek-reasoner` | `openai.OpenAI` with `base_url="https://api.deepseek.com"` and `timeout=7200` |
| `qwen_eval.py` | `Qwen/Qwen3.5-9B` | `together.Together` with `timeout=18000` |
| `xai_eval.py` | `grok-3-mini` | `xai_sdk.Client` — uses its own API: `client.chat.create(model=...)` then `chat.append(user(prompt))` then `chat.sample()` |
| `gemma_eval.py` | `google/gemma-4-31B-it` | `together.Together` |

**Key pattern:** DeepSeek reuses the OpenAI SDK by swapping `base_url` — no separate SDK needed. xAI is the only model with its own SDK and a different call pattern. Gemma and Qwen both go through Together AI.

---

## 2. DocVQA — How the Script Handles Model Non-Responses (Retry Logic)

The retry logic lives inside `get_model_response()` in `DocVQA/openai_eval.py`.

### Retry loop
- Runs up to **3 attempts** with `for attempt in range(3)`.
- On any `Exception`, the error string is inspected before deciding what to do next.

### How the wait time is determined
The script parses the rate-limit message from OpenAI using:
```python
m = re.search(r'try again in ([\d.]+)(ms|s)', err)
```
If found, it converts the value to seconds (dividing by 1000 if the unit is `ms`) and adds 1 extra second as a buffer. If no match, it defaults to a 5-second wait.

### Early-exit conditions (no retry)
| Error string | Action |
|---|---|
| `insufficient_quota` | `raise SystemExit` — billing issue, retrying is pointless |
| `requests per day` | `return None` — daily quota exhausted, further retries waste remaining quota |

### After all retries fail
Returns `None`. The caller assigns `score = 0` and `anls = 0.0` for that sample.

### Rate limiting on success
After every successful API call: `time.sleep(2.5)`. With 5 parallel shards this keeps throughput safely under the 200k TPM limit.

### Checkpoint / Resume (separate from retry)
The script also tracks completed `questionId`s in the output CSV. On restart it skips already-finished samples — this is resume logic, not retry logic.

---

## 3. EmoBench OpenAI — Multiple-Choice Handling and run_emobench.sh

### How multiple-choice questions are handled (`openai_emo_eval.py`)

**Formatting choices into the prompt**

`rank_choices()` converts a list of strings into a lettered menu:
```
A) Yes
B) No
C) Maybe
```
This string is injected into the user prompt template from `prompts.yaml`.

**Asking for structured output**

The system prompt (built by `build_system_prompt()`) instructs the model to reply in JSON. The exact JSON schema is read from `response.yaml` and differs by task:
- **EA**: model returns `{"answer": "A"}` (single letter)
- **EU**: model returns `{"answer_q1": "B", "answer_q2": "C"}` (emotion + cause, two letters)

**Parsing the model's response**

`parse_json()` handles two formats:
1. Plain JSON: `{"answer": "A"}`
2. Markdown-fenced: ` ```json\n{"answer": "A"}\n``` `

If parsing fails, it returns `None` and the sample gets score 0.

**Scoring**

Labels are also converted to letters at save time:
```python
LETTERS[sample["choices"].index(sample["label"])]  # e.g. "C"
```
Then compared string-to-string after `.strip().upper()` normalization.

- **EA**: `score = 1` if `answer == label`
- **EU**: `score = 1` only if **both** `emo_answer == emo_label` AND `cause_answer == cause_label`

**API retry in `call_api()`**

Same pattern as DocVQA:
- Up to 3 retries
- Parses `"try again in X(ms|s)"` from rate-limit errors
- Early exits on `insufficient_quota` or `requests per day`
- `time.sleep(2.0)` after every successful call

---

### `run_emobench.sh` — SLURM job array

The shell script is a SLURM batch script for Northwestern's Quest HPC cluster.

```
#SBATCH --array=0-3        → launches 4 parallel jobs (shards 0, 1, 2, 3)
#SBATCH --partition=long   → long-running partition
#SBATCH --time=24:00:00    → 24-hour wall-clock limit
#SBATCH --mem=8GB
```

Each job runs:
```bash
python openai_emo_eval.py \
    --model gpt-4o-mini \
    --task all \          # runs both EU and EA
    --lang all \          # runs both English and Chinese
    --shard $SLURM_ARRAY_TASK_ID \
    --total-shards 4 \
    --save-every 20
```

`$SLURM_ARRAY_TASK_ID` is automatically 0, 1, 2, or 3 for each job. The script slices the dataset into 4 equal chunks and each job processes its own chunk. Results are saved to separate shard files (e.g., `gpt-4o-mini_en_shard1of4.jsonl`) and then merged after all jobs finish.

---

## 4. EmoBench — New Model Evaluation Scripts

Five independent evaluation folders were created under `EmoBench-master/`, one per model. Each folder contains a Python eval script and a SLURM shell script. Results are saved inside the folder itself.

### Folder structure

```
EmoBench-master/
├── EMO_Gemini/
│   ├── gemini_emo_eval.py
│   ├── run_emobench.sh
│   └── results/
│       ├── EA/
│       └── EU/
├── EMO_XAI/
│   ├── xai_emo_eval.py
│   ├── run_emobench.sh
│   └── results/
│       ├── EA/
│       └── EU/
├── EMO_Qwen/
│   ├── qwen_emo_eval.py
│   ├── run_emobench.sh
│   └── results/
│       ├── EA/
│       └── EU/
├── EMO_Gemma/
│   ├── gemma_emo_eval.py
│   ├── run_emobench.sh
│   └── results/
│       ├── EA/
│       └── EU/
└── EMO_Deepseek/
    ├── deepseek_emo_eval.py
    ├── run_emobench.sh
    └── results/
        ├── EA/
        └── EU/
```

### Model details

| Folder | Script | Model | SDK / Client |
|--------|--------|-------|-------------|
| `EMO_Gemini` | `gemini_emo_eval.py` | `gemini-2.5-flash` | `google.genai.Client` with `GenerateContentConfig(system_instruction=...)` |
| `EMO_XAI` | `xai_emo_eval.py` | `grok-3-mini` | `xai_sdk.Client` — `xai_system()` + `xai_user()` objects |
| `EMO_Qwen` | `qwen_emo_eval.py` | `Qwen/Qwen3.5-9B` | `together.Together` with `timeout=18000` |
| `EMO_Gemma` | `gemma_emo_eval.py` | `google/gemma-4-31B-it` | `together.Together` with empty-response retry (max 5 attempts) |
| `EMO_Deepseek` | `deepseek_emo_eval.py` | `deepseek-reasoner` | `openai.OpenAI` with `base_url="https://api.deepseek.com"`, `temperature=0` |

### Design decisions

- **English only**: all scripts filter `language == "en"` from the data — no `--lang` argument.
- **Both tasks**: `--task all` (default) runs EA then EU in sequence within each shard job.
- **Path resolution**: each script uses `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` to find `EmoBench-master/` and load `data/` and `src/configs/` regardless of where it is run from.
- **Results location**: saved to `<folder>/results/<task>/` using absolute paths derived from the script's own location.
- **Gemini system prompt**: Gemini's SDK does not accept a `system` role in the messages list — system instructions are passed via `GenerateContentConfig(system_instruction=...)` instead.
- **XAI messages**: the `xai_sdk` does not use message dicts — system and user content are appended as `xai_system()` and `xai_user()` objects before calling `chat.sample()`.
- **Gemma empty responses**: Together AI occasionally returns an empty string for Gemma; the script retries up to 5 times and only moves on if the response is non-empty.
- **DeepSeek temperature**: set to `0` (not `0.6`) because `deepseek-reasoner` is a reasoning model and deterministic output is preferred.
- **SLURM**: all `.sh` files use `--array=0-3` (4 shards), `--partition=long`, `--time=24:00:00`, and the same Quest Python environment as the OpenAI script.
