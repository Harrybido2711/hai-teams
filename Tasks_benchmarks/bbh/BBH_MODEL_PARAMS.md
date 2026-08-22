# BBH — what every model was actually asked, and with which settings

Read from the eight `*_eval.py` scripts in this folder on 2026-08-22. Every value below carries a
`file:line`; nothing is inferred from what a script "probably" does. Where a script is inconsistent
with another, that is recorded rather than reconciled.

**The headline is not in the parameter table.** Four of the eight scripts score with strict string
equality and four score with a six-branch lenient matcher. Their accuracies are therefore not
measuring the same thing, and the gap is not a property of the models. See
[Findings](#findings-that-change-what-the-numbers-mean).

## The output contract — identical across all eight

Every script sends one user message, no system message, built from the same f-string:

```text
You are a helpful assistant.
Question: {question}

Please show your reasoning, then end your response with:
"Final Answer: <your concise answer here>"
```

· Source: `openai_eval.py:16-22` and the same block in all seven others.

So the required output is **free-form reasoning followed by a line beginning `Final Answer:`**. No
JSON, no schema, no stop sequence, no length instruction. The contract is enforced only by the
extraction regex, never by an API parameter.

Extraction is `re.search(r"Final Answer:\s*(.*)", output, re.IGNORECASE)`, taking group 1. **When the
pattern does not match, the entire response is used as the answer** rather than being recorded as a
parse failure — so a model that reasons past its own final line is scored on its whole transcript.
· Source: `openai_eval.py:37-39`

## Generation parameters

Blank means the script never sets it, so the provider default applies. No script sets `top_p`,
`top_k`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, `stop`,
`response_format`, `tools`, `tool_choice`, `reasoning`, `reasoning_effort` or `verbosity` —
those twelve columns are omitted because they are empty for all eight.

| Script | Model id | SDK / endpoint | Env var | `temperature` | `max_tokens` | client `timeout` | `stream` |
|---|---|---|---|---|---|---|---|
| `openai_eval.py:27` | `gpt-4o-mini-2024-07-18` | `openai.OpenAI` | `OPENAI_API_KEY` | **0** | — | — | — |
| `deepseek_eval.py:27` | `deepseek-reasoner` | `openai.OpenAI`, `base_url=api.deepseek.com` | `DEEPSEEK_API_KEY` | **0** | — | 7200 | `False` |
| `gemini_eval.py:27` | `gemini-2.5-flash` | `google.genai.Client` | `GEMINI_API_KEY` | — | — | — | — |
| `gemma_eval.py:29` | `google/gemma-4-31B-it` | `together.Together` | `TOGETHER_API_KEY` | **0** | **12500** | — | `False` |
| `llama_eval.py:27` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | `together.Together` | `LLAMA_API_KEY` | **0** | — | — | `False` |
| `qwen_eval.py:29` | `Qwen/Qwen3.5-9B` | `together.Together` | `TOGETHER_API_KEY` | **0** | **8192** | 18000 | `False` |
| `kimi_eval.py:36` | `kimi-k2.5` | `openai.OpenAI`, `base_url=api.moonshot.ai/v1` | `KIMI_API_KEY` | **1** | — | 7200 | — |
| `xai_eval.py:29` | `grok-3-mini` | `xai_sdk.Client` → `chat.create/append/sample` | `GROK_API_KEY` | — | — | 3600 | — |

Three things this table says that are easy to miss:

- **`kimi` runs at `temperature=1`** while every other script that sets it uses `0`
  (`kimi_eval.py:37`). Its numbers are the only ones with sampling noise in them.
- **`gemini` and `xai` set no temperature at all**, so they run at whatever the provider defaults to
  — which is not 0, and is not pinned against a provider-side change.
- **`gemma` and `qwen` are the only two with a `max_tokens` ceiling**, and the two values differ
  (12500 vs 8192). Every other model can be truncated only by the provider's own default.

## Scoring, retries, and how failure is handled

| Script | Scorer | Sees the question? | Retry | On exception |
|---|---|---|---|---|
| `openai` | strict equality | no | none | returns `None` |
| `deepseek` | strict equality | no | none | returns `None` |
| `gemini` | strict equality | no | none | returns `None` |
| `llama` | strict equality | no | none | returns `None` |
| `kimi` | strict equality | no | recursive, max 5 | returns `None` — see below |
| `gemma` | **6-branch lenient** | **yes** | 5×, empty responses only | returns `None` |
| `qwen` | **6-branch lenient** | **yes** | none | returns `None` |
| `xai` | **6-branch lenient** | **yes** | none | returns `None` |

The lenient scorer's branches, in order (`qwen_eval.py:47-79`, identical in `gemma` and `xai`):

1. exact case-insensitive equality
2. gold is `(D)`-shaped and the answer starts with the same letter, with or without parentheses
3. gold is `(D)`-shaped and the answer is the *text* of that option, parsed out of the question
4. comma-vs-space differences — `"barn, damp"` matches `"barn damp"`
5. the answer repeats the question's `Input:` prefix before the gold string
6. comma normalisation on both sides

The strict scorer is a single line: `int(final.lower().strip() == gold.lower().strip())`
· Source: `openai_eval.py:42-46`

## Task coverage

20 task JSONs are present. The hardcoded `splits` list in each script no longer matches what is on
disk:

| Script | `splits` entries | Result CSVs on disk |
|---|---|---|
| `gemini`, `llama`, `qwen`, `xai` | 20 | 21 |
| `openai` | 14 | 21 |
| `deepseek` | 5 | 21 |
| `gemma` | 4 | 21 |
| `kimi` | 1 active, 1 commented out | 11 |

(21 = 20 task CSVs + one `*_overall_results.csv`.) The short lists are leftovers from partial
re-runs: a script was narrowed to the failing tasks, run, and never restored. **A CSV on disk is
therefore not evidence that the current script would produce it**, and re-running any of these five
as they stand would silently regenerate only a subset.

## Findings that change what the numbers mean

1. **Two scorers, one leaderboard.** Four models are scored by strict equality and four by the
   lenient matcher. On multiple-choice tasks the difference is large and systematic: a model
   answering `B` where gold is `(B)` scores 0 under the strict scorer and 1 under the lenient one.
   The four strict-scored models are penalised for formatting, not for reasoning. Nothing in the
   result CSVs records which scorer produced them.

2. **A single failed call discards a whole task.** `get_model_response` returns `None` on any
   exception, and `extract_final_answer(None)` then raises `TypeError` inside `re.search`. That
   exception is caught by the per-split handler, which writes the overall CSV and **moves to the next
   split** — the current split's accumulated rows are thrown away and no CSV is written for it.
   · Source: `openai_eval.py:32-34, 37-39, 99-104`

3. **`kimi`'s retry cannot succeed.** The recursive call's return value is discarded —
   `get_model_response(question, num_tries+1, question_num)` with no `return` — so the function
   yields `None` even when the retry worked. Its backoff is also `time.sleep(2^num_tries)`, and `^`
   is XOR in Python, not exponentiation: the delays are 3, 0, 1, 6, 7 seconds, including a zero.
   · Source: `kimi_eval.py:44-46`

4. **`gemma`'s retry does not cover errors.** The 5-attempt loop only re-issues on an *empty string*;
   any exception returns `None` immediately. · Source: `gemma_eval.py:26-43`

5. **Nothing is reproducible.** No script sets `seed`, and `kimi` samples at `temperature=1`. Re-running
   any of these produces different text, and for `kimi` plausibly a different score.

6. **A missing `Final Answer:` is scored, not counted.** The extractor falls back to the full
   response, which is then compared against the gold string and almost always scores 0. Parse
   failures are therefore indistinguishable from wrong answers in every CSV here — the project's
   standing rule is that they are counted and reported as their own category.

## If these are re-run

The parameters worth deciding before, not after:

- **One scorer for all models**, applied to the stored responses. All eight CSVs keep
  `model_response`, so scoring can be redone without spending a single call.
- **`seed`, and one `temperature`** across models, or an explicit statement of why a model differs.
- **`max_tokens`** set for everyone or no one; two of eight is the worst case, because truncation
  then looks like a model difference.
- **A parse-failure category** in the output, separate from a score of 0.
