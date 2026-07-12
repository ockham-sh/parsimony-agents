# Configuration

Everything about an [`Agent`](../reference/agent.md) is set at construction time: which model
it calls, the credentials behind that model, the connectors it can fetch from, the system
prompt that shapes its behaviour, and the guardrails that bound a single run. This page walks
through every knob, grounded in the constructor signature.

```python
from parsimony_agents import Agent

agent = Agent(model="claude-sonnet-4-6")
```

`Agent.ask`, `Agent.run`, and `Agent.resume` are all `async`, so the runnable snippets below
wrap calls in an `asyncio.run` entrypoint. See the [Quickstart](quickstart.md) for the full
end-to-end example.

## Convenience vs expert construction

The constructor exposes two front doors:

- **Convenience** — pass `model=` (a litellm model string) plus the optional `api_key=`. The
  agent builds the underlying model configuration for you.
- **Expert** — pass `model_config=`, a dict handed straight to litellm. Use this when you need
  to set `temperature`, `api_base`, or any other litellm parameter.

When both are supplied, `model_config` takes precedence and the convenience
`model` / `api_key` values are ignored. The constructor resolves the model like
this:

```python
# from parsimony_agents/agent/agent.py
if model_config is not None:
    resolved_config = model_config
elif model is not None:
    resolved_config = {"model": model, **({"api_key": api_key} if api_key else {})}
else:
    raise TypeError("Agent requires either model_config={...} or model='model-name'")
```

If you supply neither `model` nor `model_config`, construction raises
`TypeError`. When using `model_config`, fold the key into that dictionary
(`model_config={"model": ..., "api_key": ...}`).

```python
import asyncio

from parsimony_agents import Agent


async def main() -> None:
    # Convenience: just a model string.
    simple = Agent(model="claude-sonnet-4-6")

    # Expert: full litellm config, including temperature.
    expert = Agent(
        model_config={"model": "gpt-4o", "temperature": 0.7},
    )

    result = await simple.ask("Show me US GDP trends")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
```

## Models via litellm

The agent calls the LLM through [litellm](https://docs.litellm.ai/), so the `model` string (and
everything in `model_config`) follows litellm's provider conventions. The string selects the
provider and model; any extra `model_config` keys are forwarded to `litellm.acompletion`
unchanged:

```python
# from parsimony_agents/agent/llm.py
response_stream = await litellm.acompletion(
    messages=messages,
    tools=tools,
    tool_choice=tool_choice,
    request_timeout=request_timeout_s,
    stream=True,
    **model_config,
)
```

Because `model_config` is spread into the call, it is the place to set any litellm-supported
parameter:

```python
agent = Agent(
    model_config={
        "model": "gpt-4o",
        "temperature": 0.2,
        "api_base": "https://my-proxy.example.com/v1",
        "api_key": "sk-...",
    },
)
```

Anthropic-routed models (`claude-*`, `anthropic/*`, and OpenRouter/Bedrock Anthropic routes) get
prompt-cache breakpoints applied automatically before each call; this is a no-op on other
providers and needs no configuration on your side.

## API keys

You have three equivalent ways to supply a provider key:

1. **`api_key=` convenience argument** (only with `model=`):

   ```python
   agent = Agent(model="claude-sonnet-4-6", api_key="sk-ant-...")
   ```

   The key is merged into the resolved config only when truthy.

2. **Inside `model_config`** (expert path):

   ```python
   agent = Agent(model_config={"model": "gpt-4o", "api_key": "sk-..."})
   ```

3. **Environment variables**, which litellm reads directly (for example `ANTHROPIC_API_KEY`,
   `OPENAI_API_KEY`). When the key is in the environment, omit it from the constructor entirely:

   ```python
   import os

   os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
   agent = Agent(model="claude-sonnet-4-6")
   ```

Connector API keys are separate from the LLM key — those are bound on the connector bundle, not
the agent (see below).

## connectors=

Connectors are the data sources the agent can fetch from. Pass them with `connectors=`. The
argument accepts either:

- a single **`Connectors`** bundle, or
- a **`Mapping[str, Connectors]`** — multiple bundles keyed by a binding name.

Anything else raises `TypeError` at construction:

```python
# from parsimony_agents/agent/agent.py
if connectors is not None and not isinstance(connectors, (Connectors, Mapping)):
    raise TypeError(
        "connectors must be a Connectors or Mapping[str, Connectors]; "
        f"got {type(connectors).__name__}"
    )
```

A connector package exports a `CONNECTORS` bundle. Bind its credentials with `.bind(...)`, which
returns a new bundle with the key applied to every connector that accepts it:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent


async def main() -> None:
    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
    )
    result = await agent.ask("Fetch US GDP from FRED")
    print(result.datasets)


if __name__ == "__main__":
    asyncio.run(main())
```

A bare `Connectors` bundle is exposed to the kernel under the binding name `connectors`. To expose
several bundles, pass a mapping — each key becomes the binding name the agent sees in the
connector catalog:

```python
from parsimony_fred import CONNECTORS as FRED

agent = Agent(
    model="claude-sonnet-4-6",
    connectors={
        "fred": FRED.bind(api_key="..."),
        # "fmp": FMP.bind(api_key="..."),
    },
)
```

The catalog of available connectors is rendered into the conversation per turn (not baked into
the system prompt), so binding a different set of connectors between turns refreshes what the
agent can reach. For the full data-fetching model, see [Connectors](../concepts/connectors.md).

## Custom instructions

`instructions=` sets the system prompt. When you omit it, the agent uses the built-in
`DEFAULT_DATA_ANALYSIS_PROMPT`:

```python
# from parsimony_agents/agent/agent.py
resolved_instructions = (
    instructions if instructions is not None else DEFAULT_DATA_ANALYSIS_PROMPT
)
```

`DEFAULT_DATA_ANALYSIS_PROMPT` (defined in `parsimony_agents/agent/prompts.py`) is the data
terminal persona: it defines the five artifact kinds, the discover-before-fetching workflow, the
publish/refresh tool catalog, and visualization/report guidance. It is the single source of
truth shared by the OSS quickstart and the host terminal app.

Passing `instructions=` **overrides** the default entirely — you get exactly the string you
supply, nothing is appended:

```python
agent = Agent(
    model="claude-sonnet-4-6",
    instructions="You are a terse SQL assistant. Answer only with the query.",
)
```

The connector catalog is **not** part of the system prompt, so a custom `instructions=` string
does not lose access to connectors — they are advertised separately each turn.

## Guardrails

`guardrails=` takes an `AgentGuardrails` (a Pydantic model) that bounds a single run with safety
limits and timeouts. Omit it and every field uses its default:

```python
from parsimony_agents.agent.config import AgentGuardrails

agent = Agent(
    model="claude-sonnet-4-6",
    guardrails=AgentGuardrails(max_iterations=20, max_execution_time_s=600.0),
)
```

`AgentGuardrails` (importable from `parsimony_agents.agent.config`) has these fields and
defaults:

| Field | Default | What it limits |
|---|---|---|
| `max_iterations` | `50` | Maximum loop iterations (LLM call + tool execution cycles) before the run stops. |
| `max_execution_time_s` | `300.0` | Wall-clock budget for the whole run, in seconds. |
| `llm_timeout_s` | `60.0` | Per-LLM-call request timeout, in seconds. |
| `llm_max_retries` | `3` | Maximum retries for a failing LLM call. |
| `tool_timeout_s` | `600.0` | Per-tool-call timeout cap, in seconds (also caps `dry_execute_code`'s `timeout_seconds`). |
| `stall_threshold_s` | `30.0` | Phase-boundary stall detector: fires after this many seconds of silence between yielded events. |
| `stream_heartbeat_s` | `20.0` | Streaming heartbeat: max seconds of silence between LLM stream chunks before the call is treated as a transient failure. |
| `loop_soft_threshold` | `2` | Repeats of the same tool-call signature that trigger the soft (logged-only) loop warning. |
| `loop_hard_threshold` | `6` | Repeats that trigger the hard `loop_detected` failure. |

`tool_timeout_s` is the global cap for any single tool call. For `dry_execute_code`, the per-call
`timeout_seconds` argument is honoured but clamped to this cap, so it can never run longer than
`tool_timeout_s`. For how the loop reacts when one of these limits trips, see
[Failure handling & recovery](../concepts/failure-and-recovery.md).

## session_id and multi-turn carryover

`session_id=` names the conversation session. If you omit it, the agent generates a random UUID:

```python
# from parsimony_agents/agent/agent.py
self.session_id = session_id or str(uuid4())
```

The session id names the conversation and any host-provided session services,
such as the file store. To carry the *conversation* forward across turns, pass
the `AgentContext` from one result into the next call's `ctx=` argument. The
final context is returned on `AgentResult.context`:

```python
import asyncio

from parsimony_agents import Agent


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6", session_id="my-session")

    first = await agent.ask("Fetch Q1 sales")
    print(first.text)

    # Reuse the prior context to preserve the full transcript.
    second = await agent.ask("Now compare to Q2", ctx=first.context)
    print(second.text)


if __name__ == "__main__":
    asyncio.run(main())
```

Passing `ctx=` preserves the message history so the second turn sees everything from the first.
The same applies to the streaming API — pass `ctx=` to `Agent.run` and consume the events with
`async for`. For the full pattern, see [Multi-turn conversations](../guides/multi-turn.md); to
pause for user input and continue later, see [Suspend and resume](../guides/suspend-resume.md).
