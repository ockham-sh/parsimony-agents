# Events

Everything an agent does while it works is reported as a stream of **events**. When you call
[`Agent.run(...)`](../reference/agent.md), you get an async generator that yields one
[`AgentEvent`](../reference/events.md) at a time: text as it streams, reasoning tokens, tool
calls starting and finishing, state snapshots, errors, cancellations, and the terminal events
that signal a suspended or stalled run.

This page catalogs the event model: the base class and its `type` discriminator, the eleven
concrete event types and when each fires, the tool-type taxonomy on `ToolEvent`, and how
[`AgentResult`](../reference/agent.md) folds the stream into a single result object.

If you just want a polished terminal display rather than raw events, use
[`stream_to_display`](../guides/streaming-and-displaying-results.md) — it consumes this same
stream for you. This page is for when you want to handle events yourself (a custom UI, a
websocket pipe, metrics collection, an inspector).

## `AgentEvent` base and the type discriminator

Every event is a Pydantic model that inherits from `AgentEvent`. The base contributes exactly
one field — a string `type` discriminator:

```python
class AgentEvent(BaseModel):
    type: str
```

Each concrete subclass pins `type` to a `Literal` so you can dispatch on it. You have two
equivalent ways to handle events:

- **`match`/`case` on `event.type`** — dispatch on the literal string (`"text_delta"`,
  `"tool_event"`, `"error"`, …).
- **`isinstance(event, ...)`** — dispatch on the model class. More type-safe, and your editor
  knows which fields exist on each branch.

The union of all eleven concrete types is exported as the `AgentEventUnion` alias:

```python
from parsimony_agents.agent.events import AgentEventUnion

# AgentEventUnion is:
#   TextDelta | ReasoningDelta | ToolEvent | StateSnapshot | AgentError
#   | RunCancelled | LLMCallCompleted | ToolResultObserved
#   | UserInputRequested | Handoff | PartialRunSummary
```

Use `AgentEventUnion` as a type annotation on your event handlers. The individual classes and
the alias all live in `parsimony_agents.agent.events`.

## The eleven event types and when each fires

| Event | `type` literal | Fires when |
|---|---|---|
| `TextDelta` | `"text_delta"` | An incremental chunk of the assistant's reply arrives. |
| `ReasoningDelta` | `"reasoning_delta"` | A thinking/reasoning token arrives (models with extended thinking). |
| `ToolEvent` | `"tool_event"` | A tool call starts (`completed=False`) or finishes (`completed=True`). |
| `StateSnapshot` | `"state_snapshot"` | At run start and after state changes; carries the full `AgentContext`. |
| `AgentError` | `"error"` | A failure is surfaced; carries a structured `Failure` classification. |
| `RunCancelled` | `"run_cancelled"` | The run is stopped by user request or client disconnect. |
| `LLMCallCompleted` | `"llm_call_completed"` | Once per LLM call, after the streamed chunks are assembled. |
| `ToolResultObserved` | `"tool_result_observed"` | Right after a tool result is appended to the conversation. |
| `UserInputRequested` | `"user_input_requested"` | The run suspends pending a user reply (terminal for this turn). |
| `Handoff` | `"handoff"` | The agent cannot finish and surfaces structured blockers (terminal). |
| `PartialRunSummary` | `"partial_run_summary"` | The run stops early without asking the user (e.g. budget exhaustion). |

Their fields, as declared in source:

```python
class TextDelta(AgentEvent):
    type: Literal["text_delta"] = "text_delta"
    content: str
    message_id: str
    delta: bool = True

class ReasoningDelta(AgentEvent):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    content: str
    message_id: str
    title: str | None = None
    delta: bool = True

class ToolEvent(AgentEvent):
    type: Literal["tool_event"] = "tool_event"
    tool_name: str
    tool_call_id: str
    tool_type: str                       # "code" | "utility" | "return" | "system"
    completed: bool
    result: Any | None = None            # populated on completion (e.g. Dataset, Chart)
    ui_message: str | None = None
    ui_message_completed: str | None = None
    also_executed: bool = False

class StateSnapshot(AgentEvent):
    type: Literal["state_snapshot"] = "state_snapshot"
    context: Any                         # the full AgentContext

class AgentError(AgentEvent):
    type: Literal["error"] = "error"
    message: str
    failure: Failure | None = None       # structured classification (canonical)
    recoverable: bool = False            # legacy transport field
    error_type: str | None = None        # legacy transport field

class RunCancelled(AgentEvent):
    type: Literal["run_cancelled"] = "run_cancelled"
    message: str
    reason: Literal["user_request", "client_disconnect"] = "user_request"
```

Putting the common ones together, here is a full streaming consumer:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent


async def main() -> None:
    fred_key = os.environ["FRED_API_KEY"]
    agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind(api_key=fred_key))

    async for event in agent.run("What is the current US unemployment rate?"):
        match event.type:
            case "text_delta":
                print(event.content, end="", flush=True)
            case "reasoning_delta":
                pass  # thinking tokens — hide or show in a side panel
            case "tool_event" if not event.completed:
                print(f"\n  -> {event.tool_name} ({event.tool_type})...", end="", flush=True)
            case "tool_event" if event.completed:
                print(f" done ({event.ui_message_completed or 'ok'})")
            case "state_snapshot":
                ctx = event.context            # keep for multi-turn continuation
            case "error":
                print(f"\n[ERROR] {event.message}")
            case "run_cancelled":
                print(f"\n[CANCELLED] {event.reason}")
            case _:
                pass  # terminal/recorder events — see sections below


if __name__ == "__main__":
    asyncio.run(main())
```

## Tool types: code / utility / return / system

`ToolEvent.tool_type` is a string that classifies the tool into one of four categories. Use it
to group, icon, or color tool activity in your UI:

| `tool_type` | Meaning | Example tools |
|---|---|---|
| `"code"` | Runs code in the kernel | `execute_code`, `edit_notebook`, `dry_execute_code` |
| `"utility"` | Reads/writes files and artifacts | `read_file`, `write_file`, `list_artifacts` |
| `"return"` | Publishes a result artifact | `return_dataset`, `return_chart` |
| `"system"` | Internal / framework tools | termination and control tools |

When a `return`-type tool completes, its `ToolEvent.result` carries the published framework
object — a `Dataset` for `return_dataset`, a `Chart` for `return_chart`. That is exactly what
`AgentResult` harvests (next section). You can branch on `tool_type` to treat code execution,
file utilities, and result publication differently:

```python
async for event in agent.run("Fetch and plot unemployment"):
    if event.type == "tool_event" and event.completed:
        match event.tool_type:
            case "code":
                print(f"ran code via {event.tool_name}")
            case "return":
                print(f"published {type(event.result).__name__}")
            case "utility" | "system":
                pass
```

## Streaming vs non-streaming (`run()` vs `ask()`)

There are two ways to drive an agent, both coroutines/async-generators:

- **`agent.run(user_message, *, ctx=None, tool_choice="auto", cancellation=None)`** is an
  **async generator**. It yields events as they happen — use `async for`. This is the
  streaming API: you see text and tool activity live.
- **`agent.ask(message, *, ctx=None, **kwargs)`** is a **coroutine** that runs to completion
  and returns a single [`AgentResult`](../reference/agent.md). It internally consumes `run()`
  and accumulates every event for you. Use this when you only want the final result.

```python
# Streaming: handle each event yourself.
async for event in agent.run("Show me GDP trends"):
    ...

# Non-streaming: one await, one result object.
result = await agent.ask("Show me GDP trends")
print(result.text)
print(result.ok)
```

`ask()` is the simplest entry point; `run()` is what you reach for when you need a custom UI or
want to react to individual events. For a live terminal renderer built on `run()`, see
[Streaming and displaying results](../guides/streaming-and-displaying-results.md).

## How `AgentResult` accumulates events (`_collect`)

`AgentResult` is the container `ask()` returns. It is a dataclass with these fields:

```python
@dataclass
class AgentResult:
    text: str = ""                          # concatenated TextDelta content
    datasets: dict[str, Dataset] = ...       # keyed by logical_id
    charts: dict[str, Chart] = ...           # keyed by logical_id
    reports: dict[str, Report] = ...         # keyed by logical_id
    code: dict[str, Script] = ...            # keyed by notebook path
    context: AgentContext | None = None      # final context, for multi-turn
    events: list[Any] = ...                   # full event log
```

It builds itself from the event stream through `_collect(event)`, called once per event. The
accumulation rules are exactly:

- **`text_delta`** → append `event.content` to `result.text`.
- **`state_snapshot`** → set `result.context = event.context` (the latest snapshot wins, so the
  final value is the up-to-date `AgentContext` for your next turn).
- **`tool_event`** with `completed=True` → inspect `event.result`: a `Dataset` (with a
  `logical_id`) lands in `result.datasets`, a `Chart` lands in `result.charts`, and a `Report`
  lands in `result.reports`, all keyed by `logical_id`.
- **Every** event is appended to `result.events`, so the full stream is available for
  inspection or replay.

The `ok` property is derived from that event log:

```python
@property
def ok(self) -> bool:
    """True if the run finished without an error or terminal failure.

    handoff and partial_run_summary are non-interactive terminal failures
    (the agent gave up, or ran out of budget). They carry no error event,
    so they must be checked explicitly.
    """
    failed = {"error", "handoff", "partial_run_summary"}
    return not any(getattr(e, "type", None) in failed for e in self.events)
```

So `result.ok` is `True` exactly when none of the three failure events — `AgentError`
(`type == "error"`), `Handoff` (`type == "handoff"`), or `PartialRunSummary`
(`type == "partial_run_summary"`) — was emitted during the run. The latter two are non-interactive
terminal failures that carry no error event, so they are checked explicitly.

Because `_collect` is just an accumulator, you can drive it yourself while still streaming —
get live events *and* a fully populated `AgentResult` at the end:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, AgentResult


async def main() -> None:
    fred_key = os.environ["FRED_API_KEY"]
    agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind(api_key=fred_key))

    result = AgentResult()
    async for event in agent.run("Show me unemployment since 2020"):
        result._collect(event)          # accumulate while we stream
        if event.type == "text_delta":
            print(event.content, end="", flush=True)

    print(f"\ndatasets: {list(result.datasets.keys())}")
    print(f"ok: {result.ok}")
    # result.context can be passed as ctx= to the next run for a follow-up.


if __name__ == "__main__":
    asyncio.run(main())
```

## Terminal/suspension events (`UserInputRequested`, `Handoff`, `PartialRunSummary`)

Three events mark a run that stopped before producing a normal completion. They differ in
*why* the run stopped and what you should do next.

```python
class UserInputRequested(AgentEvent):
    type: Literal["user_input_requested"] = "user_input_requested"
    question: str
    context: str | None = None
    choices: list[str] | None = None
    suspension_record: Any                  # SuspensionRecord — persist this
    originating_failure_kind: str | None = None

class Handoff(AgentEvent):
    type: Literal["handoff"] = "handoff"
    rationale: str
    blockers: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)

class PartialRunSummary(AgentEvent):
    type: Literal["partial_run_summary"] = "partial_run_summary"
    missing: list[str] = Field(default_factory=list)
    learned_facts: list[str] = Field(default_factory=list)
    next_step_plan: str | None = None
```

- **`UserInputRequested`** — the agent suspended and needs an answer to continue. It carries a
  `question`, optional `choices`, and a `suspension_record`. Persist the record, show the user
  the question, then continue with
  [`Agent.resume(suspension_record, user_reply)`](../guides/suspend-resume.md). This is the only
  one of the three that is *resumable*.
- **`Handoff`** — the agent cannot finish and is handing the task back. No question is posed;
  instead it surfaces `rationale`, `blockers`, and `suggested_next_steps`. This is terminal.
- **`PartialRunSummary`** — the run stopped early without asking the user (for example, a budget
  was exhausted). It reports `learned_facts`, what is `missing`, and an optional
  `next_step_plan`.

```python
from parsimony_agents.agent.events import (
    UserInputRequested,
    Handoff,
    PartialRunSummary,
)

async for event in agent.run("Do a multi-step analysis"):
    if isinstance(event, UserInputRequested):
        record = event.suspension_record           # persist for resume()
        print(f"Agent asks: {event.question}")
        if event.choices:
            print(f"Options: {event.choices}")
        break
    elif isinstance(event, Handoff):
        print(f"Handing off: {event.rationale}")
        print(f"Blockers: {event.blockers}")
        print(f"Try next: {event.suggested_next_steps}")
    elif isinstance(event, PartialRunSummary):
        print(f"Stopped early. Learned: {event.learned_facts}")
        print(f"Still missing: {event.missing}")
```

For the full suspend/resume lifecycle and how recovery decides to suspend, see
[Suspend and resume](../guides/suspend-resume.md) and
[Failure handling & recovery](failure-and-recovery.md).

## Recorder events (`LLMCallCompleted`, `ToolResultObserved`)

Two events exist for **inspection and recording** rather than for driving a UI. They expose the
exact inputs and outputs the model worked with, so a recorder can capture a run without
re-parsing the streamed deltas.

```python
class LLMCallCompleted(AgentEvent):
    type: Literal["llm_call_completed"] = "llm_call_completed"
    iteration: int
    response_text: str
    reasoning_text: str | None = None
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    latency_ms: int

class ToolResultObserved(AgentEvent):
    type: Literal["tool_result_observed"] = "tool_result_observed"
    tool_call_id: str
    tool_name: str
    llm_content: str | list[dict[str, Any]]   # exactly what the LLM sees
```

- **`LLMCallCompleted`** fires once per LLM call, after the streamed chunks have been
  assembled. It carries the full `response_text`, decoded `tool_calls`, token `usage`, and
  `latency_ms` — handy for cost/latency metrics without summing up `TextDelta`s yourself.
- **`ToolResultObserved`** fires right after a tool result is appended to the conversation. Its
  `llm_content` is the exact content the model reads back (a flat string, or a list of blocks
  for multi-modal results).

```python
from parsimony_agents.agent.events import LLMCallCompleted, ToolResultObserved

calls = 0
async for event in agent.run("Analyze the data"):
    if isinstance(event, LLMCallCompleted):
        calls += 1
        print(f"iter {event.iteration}: {event.latency_ms} ms, usage={event.usage}")
    elif isinstance(event, ToolResultObserved):
        print(f"{event.tool_name} returned content the model now sees")
```

---

## Related pages

- [Streaming and displaying results](../guides/streaming-and-displaying-results.md) — render
  this event stream with a spinner, tables, and syntax-highlighted code.
- [How it works: the agent loop](how-it-works.md) — where in the loop each event is emitted.
- [Failure handling & recovery](failure-and-recovery.md) — the `Failure` carried by
  `AgentError`, and when the loop suspends or hands off.
- [Suspend and resume](../guides/suspend-resume.md) — handling `UserInputRequested` and calling
  `Agent.resume`.
- [Events reference](../reference/events.md) — full field-by-field reference for every event.
- [Agent, AgentResult, AgentConfig, AgentGuardrails](../reference/agent.md) — the `AgentResult`
  container and the `run()` / `ask()` signatures.
