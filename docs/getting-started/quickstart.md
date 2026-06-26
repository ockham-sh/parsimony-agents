# Quickstart

This page takes you from zero to a working data-analysis agent: pick a model, supply an API key, bind a connector so the agent can fetch real data, ask a question, and read the structured results.

If you have not installed the package yet, see [Installation](installation.md) first. Most of this page assumes `parsimony-agents` plus a connector package (here, FRED) are available.

## Pick a model (litellm model strings)

An `Agent` is constructed with a model string. The string is passed straight through to [litellm](https://docs.litellm.ai/), so it uses litellm's provider-prefixed format. A bare `"claude-sonnet-4-6"` resolves to the Anthropic provider; other providers use a prefix (for example `"gemini/gemini-3-flash-preview"`).

```python
from parsimony_agents import Agent

agent = Agent(model="claude-sonnet-4-6")
```

`Agent.__init__` requires *either* the convenience `model=` keyword *or* an explicit `model_config={...}` dict — passing neither raises `TypeError`. The convenience form is what you want here; `model_config` is the expert escape hatch covered in [Configuration](configuration.md).

## Supply an API key

The LLM provider key can be supplied two ways:

- **Environment variable** — the litellm default. Set the variable your provider expects (for example `ANTHROPIC_API_KEY` for Claude models) and the agent picks it up automatically.
- **Constructor argument** — pass `api_key=` directly. Internally this is folded into the resolved `model_config` as `{"model": ..., "api_key": ...}`.

```python
import os

from parsimony_agents import Agent

agent = Agent(
    model="claude-sonnet-4-6",
    api_key=os.environ["ANTHROPIC_API_KEY"],  # or omit and let litellm read the env var
)
```

You cannot pass both `model=` and `model_config=` — the constructor treats `model_config` as the explicit override and only builds one from `model`/`api_key` when `model_config` is absent.

## Bind a connector for data

On its own the agent can reason and run code, but it has no way to reach external data sources. **Connectors** are the bridge. A connector package exports a `CONNECTORS` object — a [`Connectors`](../concepts/connectors.md) collection — and you bind any required secrets onto it before passing it to the agent's constructor. Behind the scenes, under the sandbox the agent's code only ever sees a name-routed `RemoteConnector` stub for each connector — the credentialed connector stays in the trusted supervisor. Under the out-of-process sandbox (bubblewrap on Linux) the agent's code runs in a separate, no-network kernel and bound credentials never enter it; in the in-process fallback there is no process boundary. See [Code execution](../concepts/code-execution.md) for the boundary tiers.

The FRED connector (Federal Reserve Economic Data) is a good first connector because the API key is free. Get one at <https://fred.stlouisfed.org/docs/api/api_key.html>.

```python
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
)
```

`FRED.bind(api_key=...)` is the binding pattern: `Connectors.bind(**kwargs)` returns a **new** `Connectors` collection with the matching parameter fixed on every connector that accepts it. The original `CONNECTORS` is left untouched (it is immutable), so binding is safe to do inline. FRED's connectors (`fred_search`, `fred_fetch`) both declare `api_key` as a secret, so the bound key flows to whichever one the agent calls.

`connectors=` accepts a single `Connectors` bundle or a `Mapping[str, Connectors]`; anything else raises `TypeError`. To combine providers, compose collections with `+` before binding or after:

```python
from parsimony_fred import CONNECTORS as FRED
from parsimony_sdmx import CONNECTORS as SDMX

connectors = FRED.bind(api_key="...") + SDMX
```

See [Connectors](../concepts/connectors.md) for discovery helpers and the full binding model.

## `agent.ask()` and reading `AgentResult`

The simplest way to run the agent is `await agent.ask(message)`. It drives the agent loop to completion and collects every streamed event into a single [`AgentResult`](../reference/agent.md). `ask` is a coroutine, so it must be awaited from inside an async context.

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

    result = await agent.ask(
        "What is the current US unemployment rate? Fetch the data and show me."
    )

    print(result.text)                  # assistant's final text
    print(list(result.datasets))        # logical_ids of returned datasets
    print(list(result.charts))          # logical_ids of returned charts
    print(result.ok)                    # True if no error/handoff/partial-run events occurred


if __name__ == "__main__":
    asyncio.run(main())
```

`ask()` has the signature `async def ask(self, message: str | Text, *, ctx: AgentContext | None = None, **kwargs) -> AgentResult`. The `message` may be a plain `str` or a `Text` block. Pass `ctx=` to continue a previous conversation — see [Multi-turn conversations](../guides/multi-turn.md).

### `AgentResult` fields

`AgentResult` is a dataclass bundling everything produced by a single run:

| Field | Type | Meaning |
| --- | --- | --- |
| `text` | `str` | Concatenated assistant text (every `TextDelta`). |
| `datasets` | `dict[str, Dataset]` | Returned `Dataset` objects keyed by `logical_id`. |
| `charts` | `dict[str, Chart]` | Returned `Chart` objects keyed by `logical_id`. |
| `reports` | `dict[str, Report]` | Returned `Report` objects keyed by `logical_id` (via `return_report`). |
| `code` | `dict[str, Script]` | Reserved for `Script` artifacts, but **currently always empty** — see note below. |
| `context` | `AgentContext \| None` | Final conversation context — pass back as `ctx=` to continue. |
| `events` | `list[Any]` | The full event log yielded during the run. |
| `ok` | `bool` (property) | `True` when the run produced no error, handoff, or partial_run_summary event. |

`ok` is a computed property: it is `True` only if the run produced none of `error`, `handoff`, or `partial_run_summary` events — so it is `False` on an error, a handoff (the agent gave up), or a partial/incomplete run (e.g. budget exhausted), even though handoff and partial-run summaries carry no separate error event. `assert result.ok` is a quick check that the run completed cleanly.

> **`code` is not yet wired.** The `code` field is declared on `AgentResult`, but the collection step that builds the result (`AgentResult._collect`, shared by both `ask()` and `stream_to_display`) only ever populates `text`, `context`, `datasets`, and `charts`. Nothing assigns to `code`, so after a run `result.code` is **always an empty dict**. Treat it as reserved/not-yet-implemented — do not rely on it to recover the notebook source the agent ran.

## Stream events instead of waiting (preview of `run()`)

`ask()` is a convenience wrapper. Under the hood it consumes `agent.run(...)`, which is an **async generator** yielding events as they happen — ideal when you want live output (a spinner, token-by-token text, tool progress) instead of a single result at the end.

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent
from parsimony_agents.agent.events import AgentError, TextDelta, ToolEvent


async def main() -> None:
    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
    )

    async for event in agent.run("Show me US unemployment since 2020"):
        if isinstance(event, TextDelta):
            print(event.content, end="", flush=True)
        elif isinstance(event, ToolEvent) and event.completed:
            print(f"\n[tool] {event.tool_name} done")
        elif isinstance(event, AgentError):
            print(f"\n[error] {event.message}")


if __name__ == "__main__":
    asyncio.run(main())
```

`run()` has the signature `async def run(self, user_message, *, ctx=None, tool_choice="auto", cancellation=None)`. Each event carries a `type` string discriminator (`"text_delta"`, `"tool_event"`, `"state_snapshot"`, `"error"`, …), so you can branch on `isinstance` or on `event.type`.

For ready-made terminal rendering you do not have to write the loop yourself. The package ships `stream_to_display`, which wraps `run()` with a spinner, tool-progress lines, dataset tables, and syntax-highlighted code, and still returns an `AgentResult`:

```python
from parsimony_agents import Agent, stream_to_display

agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind(api_key="..."))
result = await stream_to_display(
    agent,
    "What is the current US unemployment rate? Fetch the data and show me.",
)
```

`stream_to_display` requires the `display` extra (`pip install parsimony-agents[display]`); without `rich` installed it falls back to plain text. See [Streaming and displaying results](../guides/streaming-and-displaying-results.md) and the full [Events](../concepts/events.md) catalogue.

## Where the results live (datasets, charts, reports)

When the agent fetches data and analyzes it, it does not just describe the answer in prose — it publishes typed **artifacts**, which is what populates `AgentResult`:

- **Datasets** (`result.datasets`) — each value is a `Dataset` artifact wrapping a tabular result, keyed by its content-derived `logical_id`. Returned via the agent's `return_dataset` tool.
- **Charts** (`result.charts`) — each value is a `Chart` artifact (a Vega-Lite spec), keyed by `logical_id`. Returned via `return_chart`.
- **Reports** (`result.reports`) — each value is a `Report` artifact (a Quarto `.qmd` body), keyed by `logical_id`. Returned via `return_report`.

These are the deliverable artifact types `AgentResult` surfaces today. (`result.code` is declared for `Script` artifacts but is not yet populated — see the note under [`AgentResult` fields](#agentresult-fields).)

`Dataset`, `Chart`, and `Report` are all importable from the top-level package (`from parsimony_agents import Dataset, Chart, Report`). Because the keys are content-addressed `logical_id`s, the same content always lands under the same key, which is what makes lineage and re-use automatic. The deeper model — logical identity versus content hash — is covered in [Artifacts, identity & lineage](../concepts/artifacts.md).

Results also live durably on disk, not just in memory. As each deliverable is returned, the framework persists it (and the notebook that produced it) to a content-addressed `.ockham/<kind>s/<logical_id>/` tree — `curation.json`, an append-only `log.jsonl`, and an immutable `<content_sha>.<ext>` snapshot — written through the code executor's storage seam. This happens standalone, with no host: a plain `agent.ask()` produces reusable, refreshable artifacts on disk, which is what lets a follow-up turn discover and re-use them.

## Next steps

- [Configuration](configuration.md) — `model_config`, `instructions`, guardrails, and the expert constructor parameters.
- [How it works: the agent loop](../concepts/how-it-works.md) — what happens between `ask()` and the result.
- [Connectors](../concepts/connectors.md) — binding, composing, and discovering data sources.
- [Multi-turn conversations](../guides/multi-turn.md) — reusing `result.context` across questions.
- [Streaming and displaying results](../guides/streaming-and-displaying-results.md) — building custom UIs on top of `run()`.
- [Agent, AgentResult, AgentConfig, AgentGuardrails](../reference/agent.md) — full API reference.
