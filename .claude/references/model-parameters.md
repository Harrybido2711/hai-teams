# Model parameters — what every runner must set

**Applies to every benchmark, not one.** The table is keyed by model because the same providers are
called from NegotiationToM, EmoBench, bbh, mmlu and DocVQA; a decision made here holds wherever that
model is called. Read this before writing or changing any runner.

## The standing rule

Set by the user on 2026-08-22:

1. **A reasoning-capable model gets an explicit thinking cap.** Never left to the model's discretion.
2. **A model that does not reason gets an explicit output cap.** `max_tokens` on every call.
3. **Every model is asked to show its reasoning, with the same prompt.** That contract does not vary
   by model, and it is not a cost lever — cutting it trims the cheap side and changes what is being
   measured.
4. **Set the value even when it equals the default.** A default belongs to the provider and can move
   without notice; a pinned value records what the run actually used.
5. **When a limit cannot be set through the API, set it in the prompt** — see below. Every model ends
   up with a stated ceiling on both thinking and output, whether the provider enforces it or not.

## When the knob does not exist, the prompt is the knob

Some models cannot have reasoning switched off — `grok-3-mini` and `deepseek-reasoner` reason by
construction, and Gemini's `minimal` is a floor rather than an off switch. **These are not exempt
from rule 1.** Where the API offers nothing, the ceiling goes into the prompt:

```text
Think briefly. Use at most <N> sentences of reasoning before your final line.
Then end your response with:
"Final Answer: <your concise answer here>"
```

The same applies to output length wherever `max_tokens` is unavailable or unsafe to set tightly:
state the length in the prompt instead of relying on a hard cut.

Three things to hold onto about this fallback:

- **It is a request, not a limit.** The provider does not enforce it, so the ceiling can be exceeded
  and the bill can still surprise you. Verify from
  `usage.completion_tokens_details.reasoning_tokens`, never from how long the answer looks.
- **It is still better than nothing**, because the alternative — a hard `max_tokens` sized to what
  you *want* to pay — truncates mid-thought and returns a **billed empty response**, which costs the
  money and the row.
- **It changes the prompt, so it changes the measurement.** A prompt-limited run and an unlimited one
  are different conditions and must not share a results table. Record the wording with the score.

## Per model

| Model                                                 | Called by                             | Reasoning?        | Thinking cap                           | Output cap            | If it cannot be disabled                                                                                                                                                                                                   |
| ----------------------------------------------------- | ------------------------------------- | ----------------- | -------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gemini-3.5-flash-lite`                             | EmoBench (planned)                    | yes, at the floor | `thinking_level="minimal"`           | `max_output_tokens` | **No off switch.** `minimal` is the default *and* the floor — "as close as possible to a zero budget… but still requires thought signatures". Thinking bills at the $2.50/M output rate, so cap the prompt too |
| `gemini-3.5-flash`                                  | —                                    | yes               | `thinking_level`, default `medium` | `max_output_tokens` | one step below the default is the first thing to measure                                                                                                                                                                   |
| `gemini-2.5-flash`                                  | EmoBench, bbh, NegotiationToM, DocVQA | yes               | thinking budget                        | `max_output_tokens` | pre-`thinking_level` generation                                                                                                                                                                                          |
| `grok-3-mini`                                       | bbh, EmoBench, NegotiationToM         | yes               | `reasoning_effort`                   | provider default      | **Cannot be removed** — reasoning is the model. Use the prompt ceiling                                                                                                                                              |
| `deepseek-reasoner`                                 | bbh, EmoBench, NegotiationToM         | yes               | none exposed                           | `max_tokens`        | **No API knob at all.** Prompt ceiling, or change model                                                                                                                                                              |
| `Qwen/Qwen3.5-9B`                                   | bbh, EmoBench, NegotiationToM         | hybrid            | `reasoning={"enabled": False}`       | `max_tokens`        | can be switched off; this project's shipped fix                                                                                                                                                                            |
| `kimi-k2.5`                                         | bbh                                   | not established   | verify before assuming                 | `max_tokens`        | —                                                                                                                                                                                                                         |
| `gpt-4o-mini-2024-07-18`                            | bbh, DocVQA                           | no                | n/a                                    | `max_tokens`        | rule 2 only                                                                                                                                                                                                                |
| `google/gemma-4-31B-it`                             | bbh, EmoBench, NegotiationToM         | no                | n/a                                    | `max_tokens`        | rule 2 only                                                                                                                                                                                                                |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` | bbh                                   | no                | n/a                                    | `max_tokens`        | rule 2 only                                                                                                                                                                                                                |

"Called by" lists where the model id appears in a runner today; it is not a claim that every one of
those runs is current.

## Choosing the number

**Measure, do not reason from the parameter name.** Run the same small slice at two or three settings
and compare accuracy and cost together — a setting that halves the bill and costs two accuracy points
is a decision someone makes knowingly.

How the knobs behave per client, and the four ways capping backfires — including the cap that returns
a billed empty response — are in [reasoning-cost.md](reasoning-cost.md). Read it before choosing a
value, not after.

**Nothing in this table has been applied to a runner yet.** What each runner currently sets is
recorded on its benchmark's page; this file is what they must be changed to.
