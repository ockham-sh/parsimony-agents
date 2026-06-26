# Streaming and displaying results

`Agent.run(...)` is an async generator that yields a stream of typed
[events](../concepts/events.md) as the agent thinks, calls tools, fetches data,
and writes its reply. This guide shows three ways to consume that stream:

1. **Roll your own loop** — `async for event in agent.run(...)`, pattern-match on
   `event.type`, and accumulate into an `AgentResult`.
2. **Use the built-in display** — `stream_to_display(...)` for live rich terminal
   output, or `display_result(...)` to render a finished result.
3. **Pipe events elsewhere** — a websocket, a metrics counter, a log — by
   serialising each event with `event.model_dump(mode="json")`.

All three import from the top-level `parsimony_agents` package. `Agent.run`,
`Agent.ask`, and `Agent.resume` are async, so every example uses `await` /
`async for` inside an `asyncio.run(...)` entrypoint.

---

## Consuming `run()` with `match`/`case` on `event.type`

`Agent.run` yields one [`AgentEvent`](../reference/events.md) at a time. Every
event is a Pydantic model with a `type` string discriminator (`text_delta`,
`tool_event`, `error`, …) and type-specific fields you can read directly. The
most direct consumer is a `match`/`case` over `event.type`:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY environment variable to run this example.")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    async for event in agent.run("What is the current US unemployment rate?"):
        match event.type:
            case "text_delta":
                # Incremental assistant text — print without a newline.
                print(event.content, end="", flush=True)
            case "tool_event" if not event.completed:
                # A tool call just started.
                print(f"\n  -> {event.tool_name}...", end="", flush=True)
            case "tool_event" if event.completed:
                # The same tool finished.
                print(f" done ({event.ui_message_completed or 'ok'})")
            case "error":
                print(f"\n[ERROR] {event.message} (recoverable={event.recoverable})")
            case _:
                pass  # reasoning_delta, state_snapshot, etc.

    print()


if __name__ == "__main__":
    asyncio.run(main())
```

Notes:

- `TextDelta` carries a `content` chunk and a `message_id`. Concatenate the
  `content` of consecutive deltas to assemble the full reply.
- `ToolEvent` fires twice per tool call: once on start (`completed=False`) and
  once on finish (`completed=True`). `tool_name`, `tool_type` (`"code"`,
  `"utility"`, `"return"`, `"system"`), and `ui_message_completed` are useful for
  progress lines.
- The full set of event types — `TextDelta`, `ReasoningDelta`, `ToolEvent`,
  `StateSnapshot`, `AgentError`, `RunCancelled`, `LLMCallCompleted`,
  `ToolResultObserved`, `UserInputRequested`, `Handoff`, `PartialRunSummary` — is
  documented in the [Events reference](../reference/events.md). Pattern-match only
  the ones you care about and fall through (`case _`) on the rest.

If you prefer type-checked dispatch over string matching, import the event
classes and use `isinstance`:

```python
from parsimony_agents.agent.events import TextDelta, ToolEvent, AgentError

async for event in agent.run("Analyze this dataset"):
    if isinstance(event, TextDelta):
        print(event.content, end="", flush=True)
    elif isinstance(event, ToolEvent) and event.completed:
        print(f"\nTool {event.tool_name} completed.")
    elif isinstance(event, AgentError):
        print(f"\nError: {event.message}")
```

---

## Accumulating into an `AgentResult` with `_collect`

Looping over events gives you live control, but you usually also want the
finished artifacts: the full text, returned datasets and charts, executed code,
and the [context](../guides/multi-turn.md) for a follow-up turn.
[`AgentResult`](../reference/agent.md) accumulates exactly that. Create an empty
result and feed each event to `result._collect(event)` as it arrives:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, AgentResult


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY environment variable to run this example.")
        return

    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    result = AgentResult()
    async for event in agent.run("What is the current US unemployment rate?"):
        result._collect(event)  # accumulate while you process

        if event.type == "text_delta":
            print(event.content, end="", flush=True)
    print()

    # The result is now fully populated — same as agent.ask() would return.
    print("Datasets:", list(result.datasets.keys()))
    print("Charts:  ", list(result.charts.keys()))
    print("Reports: ", list(result.reports.keys()))
    print("Success: ", result.ok)

    # Reuse result.context for a multi-turn follow-up.
    follow_up = await agent.ask(
        "How has it changed since 2020?",
        ctx=result.context,
    )
    print(follow_up.text[:200])


if __name__ == "__main__":
    asyncio.run(main())
```

`_collect` is the same routine `stream_to_display` and `display_result` use
internally. It concatenates `TextDelta.content` into `result.text`, extracts
`Dataset`, `Chart`, and `Report` objects from completed `ToolEvent`s into
`result.datasets`, `result.charts`, and `result.reports` (each keyed by logical
id), and updates `result.context` from each `StateSnapshot`. (`result.code` is
declared on `AgentResult` but is not populated by `_collect` today.) After the
loop, `result.ok` is `True` only if the run produced no `error`, `handoff`, or
`partial_run_summary` events — handoff and partial-run-summary are
non-interactive terminal failures (the agent gave up or ran out of budget) and
carry no `error` event, so `ok` checks for them explicitly. `result.events`
holds the raw event log for inspection or replay.

> If you don't need per-event control at all, skip the loop and call
> `await agent.ask(message, ctx=...)` — it drives `run()` internally and returns
> the same populated `AgentResult`.

---

## `stream_to_display` for live rich terminal output

For an interactive CLI, you rarely want to hand-roll rendering.
`stream_to_display` wraps `agent.run(...)` and paints a live terminal view: a
"Thinking…" spinner, one progress line per tool call (with elapsed time and a
type icon), the streamed response text, then panels for datasets, executed code,
and charts. It returns the same fully-populated `AgentResult`:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, stream_to_display


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY environment variable to run this example.")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    # Ask a question — full display with spinner, datasets, code
    result = await stream_to_display(
        agent,
        "What is the current US unemployment rate? Fetch the data and show me.",
    )

    # Follow-up (multi-turn), reusing context
    await stream_to_display(
        agent,
        "Now show me how unemployment has changed since 2020",
        ctx=result.context,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

The full signature is:

```python
stream_to_display(
    agent,
    message,
    *,
    ctx=None,
    console=None,
    show_code=True,
    show_data=True,
    max_table_rows=5,
    max_code_lines=30,
)
```

| Parameter        | Default | Effect                                                              |
| ---------------- | ------- | ------------------------------------------------------------------- |
| `ctx`            | `None`  | An `AgentContext` for multi-turn continuation (pass `result.context`). |
| `console`        | `None`  | A custom `rich.console.Console`; defaults to a fresh, fixed-width console. |
| `show_code`      | `True`  | Render executed notebooks as syntax-highlighted code panels.        |
| `show_data`      | `True`  | Render the data-fetch log and returned datasets as tables.          |
| `max_table_rows` | `5`     | Maximum preview rows shown per dataset table.                       |
| `max_code_lines` | `30`    | Maximum lines shown per code notebook.                              |

Turn off the noisy panels for a terse run:

```python
result = await stream_to_display(
    agent,
    "Just answer in prose, no tables.",
    show_code=False,
    show_data=False,
)
```

---

## `display_result` for a finished result

When you already have an `AgentResult` — from `await agent.ask(...)`, from your
own `_collect` loop, or loaded from storage — and want to render it after the
fact, use `display_result`. It is synchronous, does not stream, and reuses the
same panels as `stream_to_display`:

```python
import asyncio

from parsimony_agents import Agent, display_result


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")

    # Run to completion without streaming.
    result = await agent.ask("Create a chart and a dataset from sample data.")

    # Render the finished result to the terminal.
    display_result(
        result,
        show_code=True,
        show_data=True,
        max_table_rows=10,
        max_code_lines=50,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

`display_result(result, ...)` takes the same `console`, `show_code`, `show_data`,
`max_table_rows`, and `max_code_lines` keyword arguments as `stream_to_display`,
but no `agent`, `message`, or `ctx` — it renders an already-finished result
rather than driving a run. Use `stream_to_display` for live runs and
`display_result` for results you compute or load elsewhere.

---

## Building a custom event handler (websocket / metrics pipe)

Because every event is a Pydantic model, you can serialise it for transport with
`event.model_dump(mode="json")`, which produces a JSON-safe `dict`. Wrap your own
loop around `agent.run(...)` to forward events to a websocket while tallying
metrics:

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.events import (
    AgentError,
    StateSnapshot,
    TextDelta,
    ToolEvent,
)


async def send_to_websocket(ws, payload: dict) -> None:
    """Stub: forward one JSON-safe event to the connected client."""
    # await ws.send_json(payload)
    ...


async def run_and_pipe(agent: Agent, message: str, ws) -> dict:
    metrics = {"text_chunks": 0, "tool_calls": 0, "errors": 0, "iterations": 0}

    async for event in agent.run(message):
        # Tally metrics with isinstance for type-checked dispatch.
        if isinstance(event, TextDelta):
            metrics["text_chunks"] += 1
        elif isinstance(event, ToolEvent) and event.completed:
            metrics["tool_calls"] += 1
        elif isinstance(event, AgentError):
            metrics["errors"] += 1
        elif isinstance(event, StateSnapshot):
            metrics["iterations"] += 1

        # Pipe the event to the client as JSON.
        await send_to_websocket(
            ws,
            {"type": event.type, "data": event.model_dump(mode="json")},
        )

    return metrics


async def main() -> None:
    agent = Agent(model="claude-sonnet-4-6")
    metrics = await run_and_pipe(agent, "Analyze the sample data.", ws=None)
    print(metrics)


if __name__ == "__main__":
    asyncio.run(main())
```

Key points:

- `event.model_dump(mode="json")` is the canonical way to put an event on the
  wire — it serialises nested models and enums to JSON-safe primitives. Pair it
  with `event.type` so the receiving end can dispatch.
- The handler stays fully streaming: you forward each event the moment it
  arrives, so the client sees text deltas and tool progress live.
- For long-running tasks you can pass a `cancellation=CancellationRequest()` to
  `agent.run(...)` and call `.set()` on it from another task to stop the run; the
  loop then emits a `RunCancelled` event. See
  [Failure handling & recovery](../concepts/failure-and-recovery.md).
- If the agent suspends to ask the user a question, you'll receive a
  `UserInputRequested` event carrying a `suspension_record`. Persist it and call
  `agent.resume(record, reply)` to continue — see
  [Suspend and resume](../guides/suspend-resume.md).

---

## Rich vs plain fallback (the `display` extra)

The polished output from `stream_to_display` and `display_result` depends on
[`rich`](https://github.com/Textualize/rich). The display module imports it
behind a `try` / `except ImportError`, and both helpers select their backend at
runtime:

- If `rich` imports successfully, they use a rich backend with a spinner,
  Markdown panels, syntax-highlighted code, and coloured tables.
- If `rich` is **absent**, they fall back to a plain backend that uses ordinary
  `print()` — no colour, no spinner, no syntax highlighting — but the same text,
  dataset tables, and code are still emitted. The plain backend renders tables
  with [`tabulate`](https://pypi.org/project/tabulate/) when it's installed, and
  falls back to `DataFrame.to_string()` otherwise.

This means `stream_to_display(agent, message)` and `display_result(result)` work
out of the box with no extra dependency; installing `rich` only upgrades the
formatting. To get the rich experience, install the `display` extra:

```bash
pip install "parsimony-agents[display]"
```

Both helpers accept a `console=` argument so you can inject a pre-configured
`rich.console.Console` (for example, to fix the width or capture output in
tests). When `rich` is not installed, the `console` argument is ignored and the
plain backend takes over.

---

## See also

- [Events](../concepts/events.md) — the event model and the agent loop that emits it.
- [Events reference](../reference/events.md) — every event class and its fields.
- [Agent, AgentResult, AgentConfig, AgentGuardrails](../reference/agent.md) — the
  result container and run/ask/resume signatures.
- [Multi-turn conversations](../guides/multi-turn.md) — reusing `result.context`.
- [Suspend and resume](../guides/suspend-resume.md) — handling `UserInputRequested`.
- [Embedding in a host application](../guides/embedding-in-a-host.md) — wiring the
  event stream into a server or UI.
