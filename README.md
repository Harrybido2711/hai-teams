# hai-teams
Human-AI Teams project repo

## BBH — model call parameters

Every generation parameter set by the BBH runners in `Tasks_benchmarks/bbh/`, read from the code on
2026-08-22.

Three things needed to read the table correctly:

- **`—` means the parameter is never passed in the Python call**, so the provider or SDK default
  applies. It does not mean `0`, `False`, or unsupported.
- **Two SLURM scripts do not submit the file their name implies.** `gemma_eval_script.sh:18` runs
  `gemma_finish.py` and `qwen_eval_script.sh:18` runs `qwen_finish.py`, so for those two models the
  **effective** `max_tokens` is the second value in the cell — Gemma `12000`, Qwen `12500` — not the
  one in the main script.
- Only what the code proves is recorded. Nothing is back-filled from a provider's current defaults.

| Model (exact id in code) | API / SDK | Script | `temperature` | `top_p` | `top_k` | `max_tokens` | `frequency_penalty` | `presence_penalty` | `repetition_penalty` | `seed` | `stop` | `response_format` | `tools` | `tool_choice` | `reasoning` | `reasoning_effort` | `verbosity` | client `timeout` | `stream` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---:|---|
| `gpt-4o-mini-2024-07-18` | OpenAI Chat Completions | [`openai_eval.py`](Tasks_benchmarks/bbh/openai_eval.py#L26) | `0` | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | not set |
| `gemini-2.5-flash` | Google Gen AI `generate_content` | [`gemini_eval.py`](Tasks_benchmarks/bbh/gemini_eval.py#L26) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | not set |
| `deepseek-reasoner` | DeepSeek, OpenAI-compatible | [`deepseek_eval.py`](Tasks_benchmarks/bbh/deepseek_eval.py#L26) | `0` | — | — | — | — | — | — | — | — | — | — | — | — | — | — | `7200` | `False` |
| `grok-3-mini` | xAI SDK `chat.sample()` | [`xai_eval.py`](Tasks_benchmarks/bbh/xai_eval.py#L28) | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | `3600` | not set |
| `google/gemma-4-31B-it` | Together Chat Completions | [`gemma_eval.py`](Tasks_benchmarks/bbh/gemma_eval.py#L28) · [`gemma_finish.py`](Tasks_benchmarks/bbh/gemma_finish.py#L28) | `0` | — | — | `12500` / **`12000`** | — | — | — | — | — | — | — | — | — | — | — | — / `4800` | `False` |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | Together Chat Completions | [`llama_eval.py`](Tasks_benchmarks/bbh/llama_eval.py#L26) | `0` | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | `False` |
| `Qwen/Qwen3.5-9B` | Together Chat Completions | [`qwen_eval.py`](Tasks_benchmarks/bbh/qwen_eval.py#L28) · [`qwen_finish.py`](Tasks_benchmarks/bbh/qwen_finish.py#L28) | `0` | — | — | `8192` / **`12500`** | — | — | — | — | — | — | — | — | — | — | — | `18000` / — | `False` |
| `kimi-k2.5` | Moonshot, OpenAI-compatible | [`kimi_eval.py`](Tasks_benchmarks/bbh/kimi_eval.py#L35) | `1` | — | — | — | — | — | — | — | — | — | — | — | — | — | — | `7200` | not set |

What the table shows:

1. **`temperature` is the only parameter under real control**, and it is not controlled consistently:
   five models at `0`, Kimi at `1`, Gemini and Grok not set at all.
2. **Only Gemma and Qwen cap output length**, and the two caps differ.
3. **Twelve parameters are set by no runner anywhere** — `top_p`, `top_k`, both penalties,
   `repetition_penalty`, `seed`, `stop`, `response_format`, `tools`, `tool_choice`, `reasoning`,
   `reasoning_effort`, `verbosity`. With no `seed` and one model sampling at `temperature=1`, none of
   these runs is reproducible.
4. **A reasoning-capable model name is not a reasoning setting.** `deepseek-reasoner` and
   `grok-3-mini` may reason internally, but no runner passes a reasoning configuration, so their
   behaviour must not be recorded as a `reasoning_effort` value.
