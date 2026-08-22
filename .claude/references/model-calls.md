# How to call each model

Which client, `base_url`, key, model id, and the non-optional parameters. What limits a runner must
set: [model-parameters.md](model-parameters.md). How each client fails:
[provider-gotchas.md](provider-gotchas.md).

Recipes for models already run here are taken from that runner and are *measured*; the rest come from
documentation and say so — a documented shape is a hypothesis until one call confirms it.

## `google/gemma-4-31B-it` — DeepInfra

```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("DEEPINFRA_API_KEY"),
                base_url="https://api.deepinfra.com/v1/openai",
                timeout=300)
client.chat.completions.create(model="google/gemma-4-31B-it",
                               messages=messages, temperature=0, max_tokens=MAX_TOKENS)
```

Same client shape as DeepSeek and GPT; only `base_url` and the key differ.
· Source: `NEG_Gemma_DeepInfra/gemma_di_neg_eval.py:33-36, 103-110, 171`

**Pass no reasoning parameter here.** DeepInfra serves this checkpoint with thinking already off —
measured 15–17 token answers, no `<think>` tags. It *accepts* `reasoning_effort`, and passing it
turns thinking back **on**. The one reasoning-capable model here with no knob to set; adding one
helpfully is a regression.

**The same checkpoint on Together behaves oppositely** — reasoning on by default, switched off with
`reasoning={"enabled": False}`. Measured A/B, 12 calls per arm, `max_tokens` held constant: on gave
6/12 completions at 9.3 s and 517 output tokens; off gave 11/12 at 0.3 s and 16. Capping `max_tokens`
is no substitute — every call that returned stopped on its own.
· Source: `NEG_Gemma/gemma_neg_eval.py:21-53`

Across both: **SIGALRM at 120 s is the primary guard**, deliberately below the 300 s socket timeout
so a hang is classified the same either way — Together ignores `timeout=` outright. And **tolerate
four `<think>` spellings**, not one; a missing opening tag and a pipe-delimited variant both occur.

## `gemini-3.5-flash-lite` — two routes, neither used here yet

**Not called by any runner here.** Everything below is from documentation and needs one call to
confirm before a run depends on it.

**Route A · Google AI Studio, native SDK.** Two API surfaces exist in `google-genai`; the 3.5 docs
use the newer one:

```python
client.interactions.create(model="gemini-3.5-flash-lite", input=prompt,
                           generation_config={"thinking_level": "minimal"})
```

Every existing runner here uses the older `client.models.generate_content(model=, contents=,
config=types.GenerateContentConfig(system_instruction=, …))`. **Check which the installed SDK
supports before writing.**

- `thinking_level`: **`minimal` is both the default and the floor** for Flash-Lite — "as close as
  possible to a zero budget for thinking but still requires thought signatures". There is no off, so
  set it to pin the default **and** add the prompt ceiling from
  [model-parameters.md](model-parameters.md); on this model the prompt is the only lever left.
- **Omit `temperature`, `top_p` and `top_k`** — the 3.x guidance is to remove them, not tune them.
- The key is an `AQ.` auth key, and those are reported to 401 against
  `generativelanguage.googleapis.com` with `ACCESS_TOKEN_TYPE_UNSUPPORTED`. **Probe before committing
  to this route.**

**Route B · OpenRouter, OpenAI-compatible.** The same client shape as DeepInfra and DeepSeek:

```python
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
client.chat.completions.create(model="google/gemini-3.5-flash-lite", messages=messages,
                               extra_body={"reasoning": {"effort": "minimal"}})
```

`Authorization: Bearer sk-or-v1-…`. `reasoning.effort` maps to the thinking level. 1,048,576 context,
65,536 max output, **$0.30/M in and $2.50/M out**, thinking billed at the output rate. Not to be
confused with `gemini-3.5-flash`: a different model at $1.50/M and $9.00/M, defaulting to `medium`.

Served by Google AI Studio *and* Google Vertex (US) with failover, so **the serving path can change
mid-run**. Record which one answered.

## `deepseek-reasoner`

```python
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com", timeout=7200)
client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=8192)
```

· Source: `NEG_Deepseek/deepseek_neg_eval.py:16-17, 29-40`

- **The long timeout is not carelessness.** DeepSeek queues; the short timeouts used elsewhere here
  would classify a queued call as a hang.
- **The answer sometimes arrives in `reasoning_content` with `content` empty.** A runner reading only
  `message.content` scores those as failures. Fall back explicitly, and log when it fires.
- **`deepseek-reasoner` is an alias, not a model id.** What it resolves to can change under you;
  record the resolved identity with the results.

No reasoning knob — the levers are in [model-parameters.md](model-parameters.md).
