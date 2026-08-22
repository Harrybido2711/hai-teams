# Capping what reasoning costs

A reasoning model bills its internal thinking as output tokens, and on this project's own usage that
line dominates everything else — one day's bill read **$16.99 reasoning against $0.41 completion**,
a factor of 41. The visible answer is nearly free; the thinking is the entire cost.

## The knob, per client

| Client | Parameter | Values | Off entirely? |
|---|---|---|---|
| OpenAI reasoning models | `reasoning_effort` | `none` · `minimal` · `low` · `medium` · `high` · `xhigh` · `max`, **model-dependent** | `none`, where supported |
| Gemini 3.x native | `thinking_level` | `minimal` · `low` · `medium` · `high`. Replaces `thinking_budget`; the two cannot both be sent | no — `minimal` is the floor |
| OpenRouter (any model) | `reasoning: {effort, max_tokens, exclude, enabled}` | `effort` as OpenAI; `max_tokens` numeric | `effort: "none"` |
| Together hybrid models | `reasoning={"enabled": False}` | on/off | yes — this project's shipped fix for Qwen |
| xAI, DeepSeek reasoner | reasoning is the model | — | no. Choose a non-reasoning model instead |

`gpt-4o-mini`, `gemma-*` and `llama-*` are not reasoning models and contribute no reasoning tokens.
A reasoning line on the bill therefore names which model produced it.

## Four ways capping goes wrong

- **`max_output_tokens` includes reasoning.** It does bound the spend, but when thinking exhausts the
  budget the response comes back with `status: "incomplete"` and
  `incomplete_details.reason: "max_output_tokens"` — **input and reasoning are billed and there is no
  visible answer**. A cap set too low converts cost into empty rows, which is the more expensive
  failure because the run has to be repeated. OpenAI's own guidance is to reserve **at least 25,000
  tokens** for reasoning plus output while calibrating.
- **`exclude: true` is not a saving.** OpenRouter still bills excluded reasoning — "reasoning tokens
  are considered output tokens and charged accordingly". It hides the trace; it does not stop the
  work. Use `effort: "none"` to actually stop it.
- **`reasoning_effort` and a token cap can conflict.** Sending `reasoning_effort: "none"` together
  with `max_completion_tokens` is reported to make the API ignore the effort setting, so the cap
  becomes the only control and the model reasons anyway. Send one, verify with the usage numbers, and
  do not assume both applied.
- **A cap changes the measurement.** Less thinking lowers accuracy on exactly the hard items a
  benchmark exists to test, so a capped run and an uncapped run are not comparable and must not share
  a results table. Record the setting beside the score.

## The standing rule

Set by the user on 2026-08-22, and it applies to every runner in this repo:

1. **A reasoning-capable model gets an explicit thinking cap.** Never left to the model's discretion.
2. **A model that does not reason gets an explicit output cap.** `max_tokens` on every call.
3. **Every model is asked to show its reasoning, with the same prompt.** That contract does not vary
   by model, and it is not a cost lever — cutting it trims the cheap side and changes what is being
   measured.
4. **Set the value even when it equals the default.** A default belongs to the provider and can move
   without notice; a pinned value is a record of what the run actually used.

| Model | Class | Thinking cap | Output cap | Note |
|---|---|---|---|---|
| `gemini-3.5-flash-lite` | reasoning, already at the floor | `thinking_level="minimal"` | `max_output_tokens` | `minimal` is the default *and* the floor — "as close as possible to a zero budget… but still requires thought signatures". **There is no off.** Thinking bills at the $2.50/M output rate |
| `gemini-3.5-flash` | reasoning | `thinking_level`, default `medium` | `max_output_tokens` | one step down from the default is the first thing to measure |
| `gemini-2.5-flash` | reasoning | thinking budget | `max_output_tokens` | pre-`thinking_level` |
| `grok-3-mini` | reasoning | `reasoning_effort` | provider default | reasoning is the model; cannot be removed |
| `deepseek-reasoner` | reasoning, no knob | none exposed | `max_tokens` | the only lever is choosing a different model |
| `Qwen/Qwen3.5-9B` | hybrid | `reasoning={"enabled": False}` | `max_tokens` | this project's shipped fix |
| `kimi-k2.5` | not established | verify before assuming | `max_tokens` | — |
| `gpt-4o-mini` · `gemma-4-31B-it` · `llama-4-Maverick` | not reasoning | n/a | `max_tokens` | rule 2 only |

**Pick the value by measuring, not by reading the parameter name.** Run the same small slice at two
or three settings and compare **accuracy and cost together**; a setting that halves the bill and
costs two accuracy points is a decision someone makes knowingly. Read the cost from
`usage.completion_tokens_details.reasoning_tokens`, not from the response length — the expensive
tokens are the ones you never see.
