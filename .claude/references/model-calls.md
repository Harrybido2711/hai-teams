# How to call each model

Which client, `base_url`, key, model id, and the non-optional parameters. Limits a runner must set:
[model-parameters.md](model-parameters.md). How each client fails:
[provider-gotchas.md](provider-gotchas.md).

Recipes are *measured* — from a runner that has run, or from a probe. A documented shape is a
hypothesis until one call confirms it.

## `google/gemma-4-31B-it` — DeepInfra

```python
OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
       base_url="https://api.deepinfra.com/v1/openai", timeout=300)
# .chat.completions.create(model="google/gemma-4-31B-it", messages=…, temperature=0, max_tokens=…)
```

Same client shape as DeepSeek and GPT; only `base_url` and the key differ.
· Source: `NEG_Gemma_DeepInfra/gemma_di_neg_eval.py:33-36, 103-110, 171`

**Pass no reasoning parameter here.** DeepInfra serves this checkpoint with thinking already off —
measured 15–17 token answers, no `<think>` tags. It *accepts* `reasoning_effort`, and passing it
turns thinking back **on**. The one reasoning-capable model here with no knob to set; adding one
helpfully is a regression.

**The same checkpoint on Together is the opposite** — reasoning on by default, off via
`reasoning={"enabled": False}`. Measured A/B, 12 calls per arm: on gave 6/12 completions at 9.3 s and
517 output tokens; off gave 11/12 at 0.3 s and 16. Capping `max_tokens` is no substitute — every call
that returned stopped on its own. · Source: `NEG_Gemma/gemma_neg_eval.py:21-53`

Across both: **SIGALRM at 120 s is the primary guard**, below the 300 s socket timeout — Together
ignores `timeout=` outright. And **tolerate four `<think>` spellings**, not one.

## `gemini-3.5-flash-lite` — two routes, both probed

**Both run end to end, 2026-08-22**: six real EmoBench items each — prompts, JSON parse, checkpoint,
scoring, CSVs. Per item 192/18 tokens native, 194/18 via OpenRouter at **$0.0001032**; a 400-item run
is about four cents. **`minimal` produced zero thinking tokens**, and the SDK accepts
`types.ThinkingConfig(thinking_level="minimal")`.

**OpenRouter switched backend inside six calls** — `provider` returned both `Google` and `Google AI
Studio`. The routes agreed on five of six items: neither is deterministic, since no seed is offered
and temperature is deliberately unset.

**Route A · Google AI Studio, native SDK.** `client.models.generate_content(model=, contents=,
config=types.GenerateContentConfig(system_instruction=, thinking_config=types.ThinkingConfig(
thinking_level="minimal"), max_output_tokens=…))` — the shape every runner here already uses. The 3.5
docs also show a newer `client.interactions.create` surface; the older one is what was probed.

- `thinking_level`: **`minimal` is the default and the floor** — there is no off, and the response
  carries a `thoughtSignature` either way. REST accepts `thinkingConfig.thinkingLevel` in both cases
  (`MINIMAL` and `minimal` both returned 200). Set it to pin the default; on a structured task it
  already costs nothing, so the prompt ceiling is only needed where thinking actually fires.
- **Omit `temperature`, `top_p`, `top_k`** — 3.x guidance is to remove, not tune.
- The key is an `AQ.` auth key. Those are *reported* to 401 here, and **this project's key does
  not** — probed, HTTP 200. Check per key; it is not a property of AQ. keys.

**Route B · OpenRouter.** Same client shape as DeepInfra and DeepSeek:

```python
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
client.chat.completions.create(model="google/gemini-3.5-flash-lite", messages=messages,
                               extra_body={"reasoning": {"effort": "minimal"}})
```

`Authorization: Bearer sk-or-v1-…`. `reasoning.effort` maps to the thinking level. 1,048,576 context,
65,536 max output, **$0.30/M in and $2.50/M out**, thinking billed at the output rate. Not to be
confused with `gemini-3.5-flash`: a different model at $1.50/M and $9.00/M, defaulting to `medium`.

Served by Google AI Studio *and* Google Vertex (US) with failover, so **the serving path can change
mid-run** — the probe was answered by Google AI Studio. Record which one answered; the response
carries `provider` and a per-call `usage.cost`.

## `deepseek-reasoner`

```python
OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com", timeout=7200)
# .chat.completions.create(model=…, messages=…, temperature=0, max_tokens=8192)
```

· Source: `NEG_Deepseek/deepseek_neg_eval.py:16-17, 29-40`

- **The long timeout is not carelessness.** DeepSeek queues; the short timeouts used elsewhere here
  would classify a queued call as a hang.
- **The answer sometimes arrives in `reasoning_content` with `content` empty.** A runner reading only
  `message.content` scores those as failures. Fall back explicitly, and log when it fires.
- **`deepseek-reasoner` is an alias, not a model id.** What it resolves to can change under you;
  record the resolved identity with the results.

No reasoning knob — the levers are in [model-parameters.md](model-parameters.md).
