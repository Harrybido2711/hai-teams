# The prompt ceiling — capping a model that has no knob

Split out of [model-parameters.md](model-parameters.md), whose rule 5 sends you here. Rule 1 there is
unconditional: **hidden reasoning is always capped.** This page is what to do when the API offers
nothing to cap it with.

Some models expose no thinking parameter. **They are not exempt from rule 1** — the ceiling goes in
the prompt:

```text
Think briefly. Use at most <N> sentences of reasoning before your final line.
Then end your response with:
"Final Answer: <your concise answer here>"
```

Same for output length where `max_tokens` is unavailable or unsafe to set tightly.

- **A request, not a limit.** Verify from `usage.completion_tokens_details.reasoning_tokens`, never
  from how long the answer looks.
- **It changes the prompt, so it changes the measurement.** Record the wording with the score.

Which models this applies to is the last column of the table in
[model-parameters.md](model-parameters.md) — anything reading *No* or *not established*. **An
unestablished knob counts as no knob:** apply the ceiling and keep it until one call proves the API
can do it.
