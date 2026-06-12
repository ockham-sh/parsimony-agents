# Agent tools

This page is a reference for the **system tools** the agent itself can call during a
run — the functions it uses to write and execute notebooks, publish datasets/charts/reports,
read files, and signal that a turn is done. It also documents the primitives those tools are
built from: the `@toolmethod` decorator, the `Tool`/`ToolMethod`/`ToolResult` classes, and the
`Tools` registry.

These tools are **internal to the agent loop** — you do not call them yourself. The agent
selects and invokes them via the LLM as it works toward an answer. You observe the calls as
[`ToolEvent`](events.md) entries in the event stream and consume their published artifacts via
[`AgentResult`](agent.md). Understanding the catalog is useful when you are embedding the agent
in a host, declaring your own tools, or interpreting tool events.

For how a single run drives these tools, see [How it works: the agent loop](../concepts/how-it-works.md).
For the published outputs, see [Artifacts, identity & lineage](../concepts/artifacts.md).

## The `@toolmethod` decorator and `tool_type` taxonomy

Every tool the agent can call is declared as a method on the `Agent` class, decorated with
`@toolmethod`. The decorator captures the tool's JSON-Schema contract and metadata, and registers
the method into the agent's `system_tools` registry so the loop can dispatch calls by name.

```python
from parsimony_agents.tools import toolmethod
```

The decorator signature (all arguments are keyword-only in practice):

```python
def toolmethod(
    *,
    name: str,
    description: str,
    parameters_schema: dict[str, Any],
    tool_type: str,                       # "code" | "utility" | "return" | "system"
    ui_message: str | None = None,
    ui_description: str | None = None,
    ui_message_completed: str | None = None,
) -> Callable
```

| Argument | Meaning |
|---|---|
| `name` | The tool name the LLM calls (must be unique within the registry). |
| `description` | Natural-language description sent to the LLM as the tool's contract. |
| `parameters_schema` | JSON Schema (`{"type": "object", "properties": {...}, "required": [...]}`) describing the tool's inputs. |
| `tool_type` | One of `"code"`, `"utility"`, `"return"`, `"system"` — categorizes the tool for dispatch, per-layer timeouts, and rendering (see below). |
| `ui_message` | Short label shown while the tool runs (e.g. `"Returning dataset"`). |
| `ui_description` | Optional longer UI description. |
| `ui_message_completed` | Optional label shown once the tool finishes. |

Under the hood, `@toolmethod` wraps the method in a `ToolMethod` descriptor (a subclass of
`Tool`). When accessed on an `Agent` instance, the descriptor's `__get__` binds the method to that
instance and returns a plain `Tool` whose `function` calls the bound method. This is why every tool
method receives `context: AgentContext` as a keyword argument — the loop supplies it at dispatch time.

### `tool_type` values

`tool_type` is one of exactly four literals, declared as
`Literal["code", "utility", "return", "system"]`:

| `tool_type` | Purpose | Tools |
|---|---|---|
| `code` | Write or run Python in the kernel/notebooks | `return_notebook`, `edit_notebook` |
| `return` | Publish a typed deliverable artifact | `return_dataset`, `return_chart`, `return_report`, `edit_report`, `refresh` |
| `system` | Termination/suspension control and workspace system reads | `return_done`, `return_unable`, `ask_user`, `read_artifact`, `list_artifacts`, `list_files`, `read_file`, `restart_kernel`, `output_read`, `output_search` |
| `utility` | Plain side-effecting helpers (file writes, dry runs) | `write_file`, `edit_file`, `dry_execute_code` |

The loop reads `tool_type` to decide per-layer timeouts, recovery handling, and how the call is
surfaced in the UI. The `tool_type` also drives the prefix string the schema exposes to the LLM
(`[CODE CELLS TOOL]`, `[UTILITY TOOL]`, and so on).

### Declaring a tool method

```python
from parsimony_agents.tools import toolmethod
from parsimony_agents.agent.models import AgentContext

@toolmethod(
    name="return_dataset",
    description="Publish a DataFrame deliverable with lineage.",
    parameters_schema={
        "type": "object",
        "properties": {
            "dataset_variable_name": {"type": "string"},
            "title": {"type": "string"},
            "live_name": {"type": "string"},
        },
        "required": ["dataset_variable_name", "title", "live_name"],
    },
    tool_type="return",
    ui_message="Returning dataset",
)
async def return_dataset(
    self,
    *,
    context: AgentContext,
    dataset_variable_name: str,
    title: str,
    description: str,
    notes: list[str],
    live_name: str,
    tags: list[str] | None = None,
) -> Dataset:
    ...
```

There is a parallel `tool` decorator for free functions (rather than `Agent` methods); it produces
a `Tool` directly instead of a `ToolMethod` descriptor.

## `Tool`, `ToolResult`, `Tools` registry primitives

These three classes are the substrate the catalog is built on. All are importable from
`parsimony_agents.tools`.

### `Tool`

`Tool` carries a callable plus its schema and structural declarations:

```python
Tool(
    function,
    name,
    description,
    parameters_schema,
    tool_type,                  # Literal["code", "utility", "return", "system"]
    method=False,
    ui_message=None,
    ui_message_completed=None,
    ui_description=None,
    idempotent=False,
    retryable_on_error=False,
    parallelizable=False,
    timeout_s=None,
)
```

Calling a `Tool` (`await tool(...)`) runs its function and wraps the outcome in a `ToolResult`
(see below). Control-flow exceptions — `SuspensionRequest`, `TerminationRequest`, and
`asyncio.CancelledError` — are **not** wrapped; they propagate so the loop can translate them
into the appropriate event. Any other exception is caught and returned as a failed `ToolResult`.

The structural flags (`idempotent`, `retryable_on_error`, `parallelizable`, `timeout_s`) are read
by the loop to serialize non-parallelizable tools, auto-retry retryable tools through the recovery
funnel, and apply per-tool timeouts that override the global `tool_timeout_s` guardrail.

`Tool.schema` renders the LLM-facing JSON (function name, description, and `parameters_schema`,
prefixed per `tool_type`).

`ToolMethod` is a `Tool` subclass used as a descriptor: its `__get__` binds the underlying method
to the `Agent` instance and returns a plain `Tool`. This is what `@toolmethod` produces.

### `ToolResult`

Tool functions return (or are wrapped into) a `ToolResult`, the structured carrier for a tool's
outcome:

```python
class ToolResult(BaseModel):
    exception_message: str | None
    data: Any | None
    failure: Failure | None = None
    partial_data: Any = None
```

| Field | Meaning |
|---|---|
| `exception_message` | Plain error text (set on a caught exception). `None` on success. |
| `data` | The tool's payload on success (the published artifact, kernel output, confirmation string, …). |
| `failure` | A structured, typed [`Failure`](../concepts/failure-and-recovery.md) (with a `kind` and `explanation`) for recovery decisions. `None` when there is no structured failure. |
| `partial_data` | Any work completed before a failure occurred. |

There are two computed properties:

- **`ok`** — `True` iff *both* `failure` and `exception_message` are `None`. The recovery funnel
  uses this to decide whether a tool call succeeded.
- **`success`** — a deprecated alias for `ok`, retained for legacy call-sites.

Convenience constructors: `ToolResult.from_data(data)`, `ToolResult.from_exception(exc)`
(redacts sensitive text), and `ToolResult.from_failure(failure, partial_data=...)` (populates
`exception_message` from `failure.explanation` so message-only consumers still see useful text).

```python
from parsimony_agents.tools import ToolResult

result = ToolResult.from_data({"rows": 120})
assert result.ok                       # True — no failure, no exception_message

if not result.ok:
    print(result.exception_message)    # plain text, or
    print(result.failure)              # typed Failure for recovery
    print(result.partial_data)         # any work done before the failure
```

### `Tools` registry

`Tools` is the dict-like container that holds an agent's tool catalog:

```python
from parsimony_agents.tools import Tools

registry = Tools([tool_a, tool_b])     # de-duplicates by name
registry["return_done"]                # lookup by name (KeyError if missing)
registry.get("ask_user")               # lookup with default
"refresh" in registry                  # membership by name
registry.pop("read_file")              # remove and return by name
combined = registry_a + registry_b     # union into a new Tools
schemas = registry.to_llm()            # list[dict] of every tool's LLM schema
clone = registry.copy()                # deep copy
```

The agent assembles all its `@toolmethod` methods plus the termination tools into
`Agent.system_tools` (a `Tools` instance), and `registry.to_llm()` is what gets sent to the model
as its available functions on each iteration.

## Code tools (`return_notebook`, `edit_notebook`, `dry_execute_code`)

These tools are how the agent writes and runs Python. Notebooks are the durable, publishable
unit of code (`return_notebook` / `edit_notebook` are `tool_type="code"`); `dry_execute_code` is
for throwaway exploration and is declared `tool_type="utility"`. See
[Code execution](../concepts/code-execution.md).

### `return_notebook`

Publish a `.py` notebook revision under `notebooks/`. Takes the notebook `path`, the full Python
`code` (with docstring and comments), and an optional `execute` flag to run it on publish.

```python
async def return_notebook(
    self, *, context: AgentContext,
    path: str, code: str, execute: bool = False,
) -> str | KernelOutput
```

Returns a confirmation `str`, or a `KernelOutput` when `execute=True`.

### `edit_notebook`

Surgical edit of an existing notebook: replace `old_str` with `new_str` (use `old_str=""` for a
full rewrite). Optional `execute` to re-run after the edit.

```python
async def edit_notebook(
    self, *, context: AgentContext,
    path: str, old_str: str, new_str: str, execute: bool = False,
) -> str | KernelOutput
```

### `dry_execute_code`

Run temporary Python in the kernel **without** modifying any notebook. Use it to inspect a value,
test an approach, or probe data before committing it to a notebook.

```python
async def dry_execute_code(
    self, *, context: AgentContext,
    code: str, timeout_seconds: float = 120.0,
) -> UtilityToolOutput
```

Returns a `UtilityToolOutput` wrapping the `KernelOutput`.

## Return/publish tools (`return_dataset`, `return_chart`, `return_report`, `edit_report`)

`tool_type="return"`. These tools turn in-kernel values into typed, lineage-tracked
[artifacts](../concepts/artifacts.md). Each is keyed by a user-facing `live_name`; reusing a
`live_name` appends a new snapshot under the same logical identity.

### `return_dataset`

Publish a pandas `DataFrame` (named by `dataset_variable_name` in the kernel) as a typed `Dataset`.

```python
async def return_dataset(
    self, *, context: AgentContext,
    dataset_variable_name: str, title: str, description: str,
    notes: list[str], live_name: str, tags: list[str] | None = None,
) -> Dataset
```

### `return_chart`

Publish an Altair chart (named by `chart_variable_name`) as a typed `Chart`.

```python
async def return_chart(
    self, *, context: AgentContext,
    chart_variable_name: str, title: str, description: str,
    notes: list[str], live_name: str, tags: list[str] | None = None,
) -> Chart
```

### `return_report`

Publish a markdown report (rendered via Quarto) as a typed `Report`. The `markdown` body may embed
artifact URIs; publishing freezes those references into a pin map. When `formats` is omitted (or
empty), the report defaults to `["html", "pdf"]`.

```python
async def return_report(
    self, *, context: AgentContext,
    title: str, markdown: str, description: str, notes: list[str],
    live_name: str, subtitle: str | None = None,
    tags: list[str] | None = None, formats: list[str] | None = None,
) -> Report
```

### `edit_report`

Surgical edit of an existing report's markdown body only (`old_str` → `new_str`), addressed by
`live_name`. It does **not** re-pin embedded artifacts — use `return_report` for title, subtitle,
or format changes.

```python
async def edit_report(
    self, *, context: AgentContext,
    live_name: str, old_str: str, new_str: str,
) -> Report
```

## Termination tools (`return_done`, `return_unable`, `ask_user`)

`tool_type="system"`. These three are the **only** valid end-of-turn signals — a text-only
response with no tool call is treated as no-progress and routed through recovery. They are
importable as `Tool` instances and bundled as `TERMINATION_TOOLS`:

```python
from parsimony_agents.agent.termination_tools import (
    return_done, return_unable, ask_user, TERMINATION_TOOLS,
)
# TERMINATION_TOOLS == [return_done, return_unable, ask_user]
```

These tools are always registered into the agent's `system_tools`. Unlike the workspace tools,
two of them signal control flow by *raising* a typed exception rather than returning a value, so
the loop can catch it and emit the matching event.

### `return_done`

Explicit **success** termination. The agent calls it with a summary string; the loop sets
`state.done = True`. Declared `idempotent=True`. This is the normal, happy-path end of a turn.

### `return_unable`

Explicit **failure** termination. The agent calls it with a `blockers` list and a `rationale`
string when it cannot complete the task. It raises `TerminationRequest`; the loop emits a
`Handoff` event (terminal — the run cannot continue). Declared `idempotent=False`.

### `ask_user`

Soft **suspension** for clarification. The agent calls it with a `question`, plus optional
`context` and `choices`. It raises `SuspensionRequest`; the loop suspends the run and emits
`UserInputRequested`, carrying a JSON-serializable, HMAC-signed `SuspensionRecord`. The host
persists the record, shows the question, and later calls `Agent.resume(record, user_reply)` to
continue. Declared `idempotent=True`. See [Suspend and resume](../guides/suspend-resume.md).

## Utility tools

The remaining tools let the agent read and write raw workspace files, inspect kernel state, and
discover existing artifacts. Several are `tool_type="system"` (workspace reads / discovery), a
couple are `tool_type="utility"` (file mutations), and `refresh` (re-running lineage) is
`tool_type="return"`. The discovery tools `read_artifact` and `list_artifacts` are always
registered: a host can inject `read_artifact_fn` / `list_artifacts_fn`, but when neither is
supplied the standalone `Agent` falls back to a local backend that scans the on-disk `.ockham/`
artifact tree (`parsimony_agents.agent.local_store`). It is all-or-nothing — a host that provides
one must provide both.

### File tools

`write_file` and `edit_file` are `tool_type="utility"`; `read_file` and `list_files` are
`tool_type="system"`.

```python
# Write or overwrite a UTF-8 text file (does not execute). Returns a confirmation string.
async def write_file(self, *, context: AgentContext, path: str, content: str) -> str

# Replace exactly one occurrence of old_str with new_str. Returns a confirmation string.
async def edit_file(self, *, context: AgentContext, path: str, old_str: str, new_str: str) -> str

# Raw UTF-8 read of any workspace text file (not parquet / typed artifacts).
async def read_file(self, *, context: AgentContext, path: str) -> SystemToolOutput

# Discover unregistered workspace files (user-dropped CSV/JSON, raw text); optional subdir prefix.
async def list_files(self, *, context: AgentContext, prefix: str = "") -> SystemToolOutput
```

### Kernel tools

`restart_kernel` is `tool_type="system"`.

```python
# Clear the kernel namespace (loses variables; workspace files persist).
async def restart_kernel(self, *, context: AgentContext) -> SystemToolOutput
```

### Artifact discovery tools

`read_artifact` and `list_artifacts` are `tool_type="system"`. They are always registered. With a
host they reach into the host's artifact index (which can span sibling-terminal artifacts) via the
supplied `read_artifact_fn` / `list_artifacts_fn`. Standalone, they are backed by the local
`.ockham/` tree the framework writes — single-terminal, summary-level reads only.

```python
# Read a typed artifact by live_name + kind. Optional view (summary|outline|page|full),
# legacy mode, and locator for pagination.
async def read_artifact(
    self, *, context: AgentContext,
    live_name: str, kind: str,
    view: str | None = None, mode: str | None = None, locator: dict | None = None,
) -> SystemToolOutput

# Discover artifacts by topical query keyword. Optional kind filter; limit 1-100 (default 20).
# With a host backend, can return sibling-terminal artifacts; standalone (local .ockham/) is single-terminal.
async def list_artifacts(
    self, *, context: AgentContext,
    query: str | None = None, kind: str | None = None, limit: int = 20,
) -> SystemToolOutput
```

### `refresh`

`tool_type="return"`. Re-run the lineage that produces an existing dataset, chart, or report
(addressed by `live_name`), appending a fresh snapshot under the same logical identity.

```python
async def refresh(self, *, context: AgentContext, live_name: str) -> Dataset | Chart | Report
```

### Output inspection tools

`output_read` and `output_search` are `tool_type="system"`. They let the agent page through and
search large kernel values without re-printing them in full.

```python
# Read a paginated kernel variable or cell reference (e.g. df[row, col]).
async def output_read(
    self, *, context: AgentContext, variable_name: str, pages: list,
) -> SystemToolOutput

# Hybrid (keyword + semantic) search within a kernel variable.
async def output_search(
    self, *, context: AgentContext,
    query: str, variable_name: str | None = None, top_k: int = 5,
) -> SystemToolOutput
```

## Putting it together

You do not invoke any of these tools directly. You run the agent and observe the tool calls as
events; the published artifacts surface on the result. The minimal end-to-end loop:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent
from parsimony_agents.agent.events import ToolEvent


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY to run this example.")
        return

    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    async for event in agent.run(
        "What is the current US unemployment rate? Fetch the data and show me."
    ):
        if isinstance(event, ToolEvent) and event.completed:
            # event.tool_type is one of "code" | "utility" | "return" | "system"
            print(f"{event.tool_name} ({event.tool_type}) -> {event.result!r}")


if __name__ == "__main__":
    asyncio.run(main())
```

`Agent.run` is an async generator and `Agent.ask`/`Agent.resume` are coroutines/async generators,
so always drive them with `async for` / `await` from an `asyncio.run` entrypoint.

## See also

- [Agent, AgentResult, AgentGuardrails](agent.md) — constructing and configuring the agent.
- [Events reference](events.md) — the `ToolEvent` and other events these tools emit.
- [Artifacts reference](artifacts.md) — the `Dataset`, `Chart`, `Report`, and notebook types the return tools publish.
- [Code execution](../concepts/code-execution.md) — how the code tools reach the kernel.
- [How it works: the agent loop](../concepts/how-it-works.md) — where in the loop tool dispatch happens.
- [Failure handling & recovery](../concepts/failure-and-recovery.md) — how `ToolResult.failure` drives recovery.
- [Suspend and resume](../guides/suspend-resume.md) — handling `ask_user` suspensions.
