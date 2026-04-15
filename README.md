# modelux

Python SDK for the Modelux LLM proxy. Wraps the OpenAI SDK with Modelux-specific extensions for routing, budgets, caching, and observability.

## Install

```bash
pip install modelux
```

Requires Python 3.9+. Depends on `openai`.

## Quick start

```python
from modelux import Modelux

client = Modelux(api_key="mlx_sk_...")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.choices[0].message.content)
print(response.modelux.provider_used)  # "openai"
```

## Modelux extensions

Pass Modelux-specific parameters alongside standard OpenAI params:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],

    # Modelux extensions
    user_id="user_123",                     # end-user tracking
    tags={"tier": "premium", "team": "ml"}, # routing & analytics tags
    trace_id="req-abc-123",                 # distributed tracing
    no_cache=True,                          # skip semantic cache
    dry_run=True,                           # routing evaluation only, no LLM call
)
```

These are mapped to `X-Modelux-*` request headers automatically.

## Response metadata

Every response includes a `.modelux` object with metadata extracted from response headers:

```python
response.modelux.request_id      # unique request ID
response.modelux.provider_used   # "openai", "anthropic", etc.
response.modelux.model_used      # actual model that served the request
response.modelux.cache_hit       # True if served from semantic cache
response.modelux.cache_similarity # cosine similarity (on cache hits)
response.modelux.ab_variant      # A/B test variant label
response.modelux.budget_name     # matching budget name
response.modelux.budget_remaining # USD remaining in budget period
response.modelux.budget_action   # "block", "downgrade", or "warn_only"
```

## Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    user_id="user_123",
)

# Metadata available immediately from response headers
print(stream.modelux.provider_used)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    print(delta, end="")
```

## Async

```python
from modelux import AsyncModelux

async with AsyncModelux(api_key="mlx_sk_...") as client:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.modelux.provider_used)
```

## Budget errors

When a request is blocked by budget enforcement, the SDK raises a `BudgetExceededError`:

```python
from modelux import BudgetExceededError

try:
    client.chat.completions.create(...)
except BudgetExceededError as err:
    print(err.budget.name)        # "team-ml"
    print(err.budget.spend_usd)   # 105.0
    print(err.budget.cap_usd)     # 100.0
    print(err.budget.period)      # "monthly"
    print(err.retry_after)        # seconds until budget resets
```

## Configuration

```python
client = Modelux(
    api_key="mlx_sk_...",                     # required
    base_url="https://api.modelux.ai/v1",   # default
    timeout=60.0,                             # seconds, default 60
    max_retries=2,                            # retry on 429/5xx, default 2
)
```

Use as a context manager for clean connection lifecycle:

```python
with Modelux(api_key="mlx_sk_...") as client:
    response = client.chat.completions.create(...)
```

## Routing configs

Use the `@config-name` selector to route through a named routing config:

```python
response = client.chat.completions.create(
    model="@production",  # uses the "production" routing config
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Using with AI assistants

If you're using Claude (or another agent harness that supports Skills), install the official Modelux Skill so your assistant has built-in knowledge of our APIs, MCP tools, and routing config schema:

```bash
curl -fsSL https://docs.modelux.ai/skill/install.sh | sh
```

## License

MIT — see [LICENSE](./LICENSE).
