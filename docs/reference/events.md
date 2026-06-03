# Events reference

This page is the authoritative, field-level reference for every event the agent emits and for the message-content types that carry text into the model. It is the API contract behind [`Agent.run`](agent.md) — every object yielded by that async generator is one of the `AgentEvent` subclasses documented here.

For a narrative introduction to the event model, see [Events](../concepts/events.md). For consuming events in practice, see [Streaming and displaying results](../guides/streaming-and-displaying-results.md).

## AgentEvent base and AgentEventUnion

Every streaming event inherits from `AgentEvent`, a Pydantic model whose single shared field is a `type` discriminator. Subclasses fix `type` to a `Literal[...]` string so you can pattern-match on it with `match`/`case` or dispatch with `isinstance`.

```python
from parsimony_agents.agent.events import AgentEvent
# class AgentEvent(BaseModel):
#     type: str
```

Because each event is a Pydantic model, you can read its fields directly (`event.content`, `event.tool_name`, …) and serialize it with `event.model_dump(mode="json")`.

### AgentEventUnion

`AgentEventUnion` is the type alias unifying all eleven concrete event types. Use it to annotate consumers that handle the full stream.

```python
from parsimony_agents.agent.events import AgentEventUnion
# AgentEventUnion = (
#     TextDelta | ReasoningDelta | ToolEvent | StateSnapshot
#     | AgentError | RunCancelled | LLMCallCompleted | ToolResultObserved
#     | UserInputRequested | Handoff | PartialRunSummary
# )
```

All eleven event classes import from `parsimony_agents.agent.events`:

```python
from parsimony_agents.agent.events import (
    TextDelta, ReasoningDelta, ToolEvent, StateSnapshot,
    AgentError, RunCancelled, UserInputRequested, Handoff,
    PartialRunSummary, LLMCallCompleted, ToolResultObserved,
)
```

### Consuming the stream

[`Agent.run`](agent.md) is an async generator. Iterate with `async for` and dispatch on `event.type`:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, AgentResult


async def main() -> None:
    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
    )
    result = AgentResult()

    async for event in agent.run("What is the current US unemployment rate?"):
        result._collect(event)  # accumulate into datasets/charts/text/context

        match event.type:
            case "text_delta":
                print(event.content, end="", flush=True)
            case "tool_event" if not event.completed:
                print(f"\n  -> {event.tool_name}...", end="", flush=True)
            case "tool_event" if event.completed:
                print(f" done ({event.ui_message_completed or 'ok'})")
            case "error":
                print(f"\n[ERROR] {event.message}")
            case "state_snapshot":
                ctx = event.context  # use for multi-turn continuation
            case _:
                pass

    print(f"\nDatasets: {list(result.datasets.keys())}")
    print(f"Success: {result.ok}")


if __name__ == "__main__":
    asyncio.run(main())
```

`AgentResult._collect(event)` is the same accumulation routine used internally by [`stream_to_display`](../guides/streaming-and-displaying-results.md). See [`AgentResult`](agent.md) for the fields it populates.

## Stream events

These four event types arrive frequently during a run. They carry the assistant's text, its reasoning tokens, tool-call progress, and periodic state snapshots.

### TextDelta

An incremental chunk of the assistant's reply. Concatenate `content` across `TextDelta` events to reconstruct the full message; `message_id` groups deltas belonging to the same assistant message.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["text_delta"]` | `"text_delta"` | Discriminator. |
| `content` | `str` | — | The text fragment. |
| `message_id` | `str` | — | Identifies the assistant message this chunk belongs to. |
| `delta` | `bool` | `True` | True when `content` is an incremental fragment rather than a full message. |

```python
class TextDelta(AgentEvent):
    type: Literal["text_delta"] = "text_delta"
    content: str
    message_id: str
    delta: bool = True
```

### ReasoningDelta

An incremental reasoning/thinking token from a model with extended thinking enabled. Structurally mirrors `TextDelta` but adds an optional `title`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["reasoning_delta"]` | `"reasoning_delta"` | Discriminator. |
| `content` | `str` | — | The reasoning fragment. |
| `message_id` | `str` | — | Identifies the reasoning block this chunk belongs to. |
| `title` | `str \| None` | `None` | Optional heading for the reasoning section. |
| `delta` | `bool` | `True` | True for incremental fragments. |

```python
class ReasoningDelta(AgentEvent):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    content: str
    message_id: str
    title: str | None = None
    delta: bool = True
```

### ToolEvent

Fired twice per tool call: once when it starts (`completed=False`) and once when it finishes (`completed=True`). Carries the tool's name, ID, category, and — on completion — the produced `result`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["tool_event"]` | `"tool_event"` | Discriminator. |
| `tool_name` | `str` | — | The invoked tool's name (e.g. `execute_code`, `return_dataset`). |
| `tool_call_id` | `str` | — | Correlates the start and completion events for one call. |
| `tool_type` | `str` | — | One of `"code"`, `"utility"`, `"return"`, `"system"`. |
| `completed` | `bool` | — | `False` on start, `True` on finish. |
| `result` | `Any \| None` | `None` | The produced object (e.g. a `Dataset` or `Chart`) on completion. |
| `ui_message` | `str \| None` | `None` | Human-readable label for the in-progress state. |
| `ui_message_completed` | `str \| None` | `None` | Human-readable label for the completed state. |
| `also_executed` | `bool` | `False` | True when a non-code tool also triggered code execution. |

```python
class ToolEvent(AgentEvent):
    type: Literal["tool_event"] = "tool_event"
    tool_name: str
    tool_call_id: str
    tool_type: str
    completed: bool
    result: Any | None = None
    ui_message: str | None = None
    ui_message_completed: str | None = None
    also_executed: bool = False
```

The four `tool_type` categories distinguish the kind of work performed:

| `tool_type` | Examples | Role |
|---|---|---|
| `"code"` | `execute_code`, `edit_notebook`, `dry_execute_code` | Runs code in the executor. |
| `"utility"` | `read_file`, `write_file`, `list_artifacts` | Reads or writes artifacts. |
| `"return"` | `return_dataset`, `return_chart` | Surfaces a final artifact to the caller. |
| `"system"` | internal/framework tools | Framework-internal operations. |

See [Agent tools](agent-tools.md) for the full tool catalogue.

### StateSnapshot

A full `AgentContext` snapshot emitted at the start of each run and after state changes. Capture `event.context` and pass it back via the `ctx=` parameter of [`Agent.run`](agent.md) / [`Agent.ask`](agent.md) to continue a multi-turn session.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["state_snapshot"]` | `"state_snapshot"` | Discriminator. |
| `context` | `AgentContext` | — | The full session state. (Typed as `Any` in source to avoid a circular import.) |

```python
class StateSnapshot(AgentEvent):
    type: Literal["state_snapshot"] = "state_snapshot"
    context: Any  # AgentContext
```

`AgentContext` imports from `parsimony_agents.agent.models`. See [Multi-turn conversations](../guides/multi-turn.md) for how snapshots thread session state.

## Outcome events

These events describe how a run ended: an error, a cancellation, a suspension awaiting input, a handoff, or a partial summary. `UserInputRequested` and `Handoff` are terminal for the current iteration.

### AgentError

Carries a structured `Failure` classification plus legacy string fields retained for transport consumers. Prefer the `failure` field when present; fall back to `error_type` / `recoverable` otherwise.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["error"]` | `"error"` | Discriminator. |
| `message` | `str` | — | Human-readable error message. |
| `failure` | `Failure \| None` | `None` | Structured classification (canonical form). |
| `recoverable` | `bool` | `False` | Legacy flag: whether the error is recoverable. |
| `error_type` | `str \| None` | `None` | Legacy string error category. |

```python
class AgentError(AgentEvent):
    type: Literal["error"] = "error"
    message: str
    failure: Failure | None = None
    recoverable: bool = False
    error_type: str | None = None
```

When `failure` is set, it exposes a `kind` (a `FailureKind` enum), an `explanation`, and `blockers`:

```python
from parsimony_agents.agent.events import AgentError

# inside the async-for loop:
if isinstance(event, AgentError):
    if event.failure:
        print(f"Failure kind: {event.failure.kind.value}")
        print(f"Explanation: {event.failure.explanation}")
        if event.failure.blockers:
            print(f"Blockers: {event.failure.blockers}")
    elif event.error_type:
        print(f"Error type: {event.error_type} (recoverable={event.recoverable})")
```

See [Failure handling & recovery](../concepts/failure-and-recovery.md) for the `Failure` model.

### RunCancelled

The run was stopped by user request or because the client disconnected.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["run_cancelled"]` | `"run_cancelled"` | Discriminator. |
| `message` | `str` | — | Human-readable cancellation message. |
| `reason` | `Literal["user_request", "client_disconnect"]` | `"user_request"` | Why the run stopped. |

```python
class RunCancelled(AgentEvent):
    type: Literal["run_cancelled"] = "run_cancelled"
    message: str
    reason: Literal["user_request", "client_disconnect"] = "user_request"
```

### UserInputRequested

The agent suspended itself pending a user reply. Carries the question, optional clarifying context and choices, and the `suspension_record` needed to resume. Resume via [`Agent.resume`](agent.md). See [Suspend and resume](../guides/suspend-resume.md).

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["user_input_requested"]` | `"user_input_requested"` | Discriminator. |
| `question` | `str` | — | The question posed to the user. |
| `context` | `str \| None` | `None` | Optional background for the question. |
| `choices` | `list[str] \| None` | `None` | Optional suggested answers. |
| `suspension_record` | `Any` | — | Token to pass to `Agent.resume` to continue. |
| `originating_failure_kind` | `str \| None` | `None` | Set when synthesized by the recovery funnel. |

```python
class UserInputRequested(AgentEvent):
    type: Literal["user_input_requested"] = "user_input_requested"
    question: str
    context: str | None = None
    choices: list[str] | None = None
    suspension_record: Any
    originating_failure_kind: str | None = None
```

```python
from parsimony_agents.agent.events import UserInputRequested

if isinstance(event, UserInputRequested):
    print(f"Question: {event.question}")
    if event.choices:
        print(f"Suggested choices: {event.choices}")
    # In a host app: capture a reply and call
    #   await agent.resume(event.suspension_record, reply)
```

### Handoff

The agent cannot finish the task and surfaces structured blockers. Distinct from `UserInputRequested` because no question is posed — it is a terminal handoff back to the caller.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["handoff"]` | `"handoff"` | Discriminator. |
| `rationale` | `str` | — | Why the agent is handing off. |
| `blockers` | `list[str]` | `[]` (`Field(default_factory=list)`) | What blocked completion. |
| `suggested_next_steps` | `list[str]` | `[]` (`Field(default_factory=list)`) | Proposed actions for the caller. |

```python
class Handoff(AgentEvent):
    type: Literal["handoff"] = "handoff"
    rationale: str
    blockers: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
```

### PartialRunSummary

The run stopped before completion without requesting user action (for example, budget exhaustion). Summarises what was learned and what remains.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["partial_run_summary"]` | `"partial_run_summary"` | Discriminator. |
| `missing` | `list[str]` | `[]` (`Field(default_factory=list)`) | Items still outstanding. |
| `learned_facts` | `list[str]` | `[]` (`Field(default_factory=list)`) | Facts gathered before stopping. |
| `next_step_plan` | `str \| None` | `None` | Optional plan for continuing. |

```python
class PartialRunSummary(AgentEvent):
    type: Literal["partial_run_summary"] = "partial_run_summary"
    missing: list[str] = Field(default_factory=list)
    learned_facts: list[str] = Field(default_factory=list)
    next_step_plan: str | None = None
```

## Recorder events

These two events are emitted for inspectors and eval recorders. They expose exactly what passed between the agent and the model, so consumers can record runs without re-parsing the token stream.

### LLMCallCompleted

Emitted once per LLM call after the streamed chunks are assembled. Carries the full assembled response, decoded tool calls, usage statistics, and latency.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["llm_call_completed"]` | `"llm_call_completed"` | Discriminator. |
| `iteration` | `int` | — | The agent-loop iteration this call belongs to. |
| `response_text` | `str` | — | Full assembled assistant text. |
| `reasoning_text` | `str \| None` | `None` | Full assembled reasoning text, if any. |
| `tool_calls` | `list[dict[str, Any]]` | — | Decoded tool calls from this response. |
| `usage` | `dict[str, Any] \| None` | `None` | Token-usage statistics. |
| `latency_ms` | `int` | — | Call latency in milliseconds. |

```python
class LLMCallCompleted(AgentEvent):
    type: Literal["llm_call_completed"] = "llm_call_completed"
    iteration: int
    response_text: str
    reasoning_text: str | None = None
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any] | None = None
    latency_ms: int
```

### ToolResultObserved

Emitted right after a tool result is appended to the conversation. `llm_content` is the exact content the model will read next — a flat string, or a list of blocks for multi-modal results.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["tool_result_observed"]` | `"tool_result_observed"` | Discriminator. |
| `tool_call_id` | `str` | — | The tool call this result answers. |
| `tool_name` | `str` | — | The tool that produced the result. |
| `llm_content` | `str \| list[dict[str, Any]]` | — | Exactly what the model sees. |

```python
class ToolResultObserved(AgentEvent):
    type: Literal["tool_result_observed"] = "tool_result_observed"
    tool_call_id: str
    tool_name: str
    llm_content: str | list[dict[str, Any]]
```

## Message content

Inbound text to the agent is modeled by `MessageContent` and its subclasses. The user message you pass to [`Agent.run`](agent.md) / [`Agent.ask`](agent.md) can be a plain `str` or a `Text` instance for finer control (titles, attached file metadata, XML wrapping).

### MessageContent

The base class for all message-content types. Subclasses implement `to_llm(mode)` to produce LLM content blocks and `to_frontend_dict()` for UI serialization.

```python
from parsimony_agents.messages import MessageContent
# class MessageContent(BaseModel):
#     type: str | None = None
#     def to_frontend_dict(self) -> dict[str, Any]: ...
#     def to_llm(self, mode: str = "default") -> list[dict[str, Any]]: ...
```

### Text

The `Text` content type carries plain text plus optional presentation metadata. Import it from `parsimony_agents.messages`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `type` | `Literal["text"]` | `"text"` | Discriminator. |
| `content` | `str` | — | The text body. |
| `title` | `str \| None` | `None` | Optional title metadata. |
| `files` | `list[dict[str, Any]] \| None` | `None` | Attached-file metadata dicts (e.g. `{"file_name": ...}`). |
| `wrap_in_tags` | `str \| None` | `None` | If set, wraps the rendered content in `<tag>...</tag>`. |

```python
class Text(MessageContent):
    type: Literal["text"] = "text"
    content: str
    title: str | None = None
    files: list[dict[str, Any]] | None = None
    wrap_in_tags: str | None = None
```

#### `to_llm(mode="default")`

`Text.to_llm` renders the content into a single text block — a list containing one `{"type": "text", "text": ...}` dict. Two transformations apply, in order:

1. **Files metadata.** If `files` is non-empty, a line `\n\nFiles uploaded: [<file_name>, ...] at <timestamp>` is appended. Each file's name comes from its `"file_name"` key (falling back to `"unknown"`), and the timestamp is the current local time formatted `%Y-%m-%d %H:%M:%S`.
2. **Tag wrapping.** If `wrap_in_tags` is set, the result is wrapped as `<{wrap_in_tags}>{content}</{wrap_in_tags}>`.

```python
from parsimony_agents.messages import Text

user_msg = Text(
    content="Analyze this dataset",
    title="User Query",
    files=[{"file_name": "data.csv"}, {"file_name": "schema.json"}],
    wrap_in_tags="user_request",
)

blocks = user_msg.to_llm()
print(blocks[0]["text"])
# <user_request>Analyze this dataset
#
# Files uploaded: ['data.csv', 'schema.json'] at 2026-06-02 12:00:00</user_request>
```

### blocks_to_text

A small helper that flattens a list of content blocks back into a string by extracting each block's `"text"` field and joining with `sep` (default a newline).

```python
def blocks_to_text(blocks: list[dict[str, Any]], sep: str = "\n") -> str:
    return sep.join([block["text"] for block in blocks])
```

```python
from parsimony_agents.messages import Text, blocks_to_text

blocks = Text(content="line one").to_llm()
print(blocks_to_text(blocks))  # "line one"
```

Note that `blocks_to_text` reads `block["text"]` unconditionally, so it expects blocks that carry a `"text"` field (as produced by `Text.to_llm`).

## XML escaping helpers

Several parts of the stack build XML fragments by f-string interpolation to feed structured context to the model. Any user- or connector-controlled value placed into that XML must be escaped first, or it could close tags early or inject pseudo-instructions. Two helpers from `parsimony_agents.agent.xml_render` do this.

### escape_attr

For values placed inside XML **attributes**. Escapes `&`, `<`, `>`, `"`, and `'` to their entity references. Returns the empty string for `None`.

```python
def escape_attr(value: object) -> str:
    # returns "" for None, otherwise:
    #   s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    #    .replace('"', "&quot;").replace("'", "&apos;")
    ...
```

### escape_text

For values placed inside an XML **text node** (between tags). Escapes `&`, `<`, and `>`. Returns the empty string for `None`.

```python
def escape_text(value: object) -> str:
    # returns "" for None, otherwise:
    #   s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    ...
```

#### Usage

Always escape before interpolating into f-string XML:

```python
from parsimony_agents.agent.xml_render import escape_attr, escape_text

user_series_id = 'GDPC1" trust=""'         # injection attempt
connector_description = "S&P 500 <trending> data"

# Attribute context -> escape_attr
xml = f'<data_fetch series_id="{escape_attr(user_series_id)}">'
# <data_fetch series_id="GDPC1&quot; trust=&quot;&quot;">

# Text-node context -> escape_text
desc = f"<description>{escape_text(connector_description)}</description>"
# <description>S&amp;P 500 &lt;trending&gt; data</description>
```

Rule of thumb:

```python
# correct
f'<attr="{escape_attr(value)}">'
f"<tag>{escape_text(value)}</tag>"

# wrong — never interpolate raw values
f'<attr="{value}">'
f"<tag>{value}</tag>"
```

## See also

- [Events](../concepts/events.md) — narrative overview of the event model.
- [Streaming and displaying results](../guides/streaming-and-displaying-results.md) — consuming events with `stream_to_display`.
- [Agent, AgentResult, AgentConfig, AgentGuardrails](agent.md) — the `run` / `ask` / `resume` API and `AgentResult`.
- [Agent tools](agent-tools.md) — the tools referenced by `ToolEvent.tool_name` / `tool_type`.
- [Failure handling & recovery](../concepts/failure-and-recovery.md) — the `Failure` model behind `AgentError.failure`.
- [Suspend and resume](../guides/suspend-resume.md) — handling `UserInputRequested`.
