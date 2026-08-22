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

## What to set

The standing rule, and the per-model knob and cap for every model this project calls, are in
[model-parameters.md](model-parameters.md). This file is the mechanism behind those choices; that one
is the decision. When a model has no knob at all, the ceiling goes in the prompt — that fallback and
its three caveats are on the same page.

**Measure, do not reason from the parameter name.** Run one small slice at two or three settings and
compare accuracy and cost together. Read the cost from
`usage.completion_tokens_details.reasoning_tokens`, not from the response length — the expensive
tokens are the ones you never see.
