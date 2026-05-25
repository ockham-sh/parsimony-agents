# parsimony-agents API Reference

## Agent

### `Agent`

Main orchestrator for data analysis workflows.

```python
from parsimony_agents import Agent
from parsimony import discover

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=discover.load_all().bind_env(),
)
```

#### Constructor

```python
Agent(
    model: str | None = None,
    api_key: str | None = None,
    model_config: dict | None = None,
    instructions: str | None = None,
    connectors: Connectors | None = None,
    code_executor: BaseCodeExecutor | None = None,
    output_factory: OutputFactory | None = None,
    guardrails: AgentGuardrails | None = None,
    session_id: str | None = None,
    file_store: FileStore | None = None,
    read_artifact_fn: Callable | None = None,
    list_artifacts_fn: Callable | None = None,
)
```

**Parameters:**
- `model` — LLM model string (e.g., `"claude-sonnet-4-6"`, `"gpt-4o"`). Passed to LiteLLM.
- `api_key` — LLM provider API key (alternative to setting via environment variable).
- `model_config` — Advanced LLM configuration: `{"model": "...", "api_key": "...", ...}`
- `instructions` — System prompt for the agent
- `connectors` — Data sources available to the agent (a `parsimony.Connectors` instance)
- `code_executor` — Python execution engine (default: new `CodeExecutor()`)
- `output_factory` — Output type dispatcher (default: new `OutputFactory()`)
- `guardrails` — Execution safety limits (`AgentGuardrails`)
- `session_id` — Workspace session identifier for file materialization
- `file_store` — Optional `FileStore` implementation for artifact persistence
- `read_artifact_fn` — Optional callable for reading stored artifacts by id
- `list_artifacts_fn` — Optional callable for listing stored artifacts

#### Methods

##### `ask(message: str, *, ctx=None, **kwargs) -> Coroutine[AgentResult]`

Simple ask-and-answer mode. Returns complete result.

```python
result = await agent.ask("Show me Apple revenue growth over 5 years")
print(result.text)       # str — natural language analysis
print(result.datasets)   # dict[str, Dataset]
print(result.charts)     # dict[str, Chart]
print(result.code)       # dict[str, Script] — named scripts
print(result.ok)         # bool — success indicator
```

Pass `ctx=result.context` on a follow-up call to continue the same conversation:

```python
result1 = await agent.ask("Fetch US unemployment data")
result2 = await agent.ask("Plot a chart", ctx=result1.context)
```

**Returns:** `AgentResult` with text, datasets, charts, code, and success status.

##### `run(user_message: str, *, ctx=None, tool_choice="auto", cancellation=None) -> AsyncGenerator`

Streaming mode. Yields typed event objects as they occur.

```python
async for event in agent.run("Analyze market trends"):
    match event.type:
        case "text_delta":
            print(event.content, end="", flush=True)
        case "tool_event":
            if event.completed:
                print(f"\n[Tool done: {event.tool_name}]")
            else:
                print(f"\n[Calling: {event.tool_name}]")
        case "error":
            print(f"Error: {event.message}")
```

**Yields:** Typed event objects — see [Events (Streaming)](#events-streaming) for the full list.

### `AgentResult`

Structured response from `agent.ask()` or accumulated from `agent.run()`.

```python
@dataclass
class AgentResult:
    text: str                          # Natural language analysis
    datasets: dict[str, Dataset]       # Returned data
    charts: dict[str, Chart]           # Returned visualizations
    code: dict[str, Script]            # Executed notebooks (path -> Script)
    context: AgentContext | None       # Conversation state for follow-up calls
    events: list[Any]                  # All events collected during the run
    ok: bool                           # True if no unrecoverable error occurred
```

**Example:**

```python
result = await agent.ask("Calculate GDP per capita")

# Access datasets
gdp = result.datasets.get("gdp_per_capita")
if gdp:
    print(f"Title: {gdp.title}")
    print(f"Description: {gdp.description}")

# Access charts
for name, chart in result.charts.items():
    print(f"Chart '{name}': {chart.title}")

# Access execution code
for path, script in result.code.items():
    print(f"Script '{path}':\n{script.code}")

# Continue conversation
result2 = await agent.ask("Now show year-over-year growth", ctx=result.context)
```

**Accumulating manually from a stream:**

```python
result = AgentResult()
async for event in agent.run("Question"):
    result._collect(event)
    # your custom per-event logic here
```

## Execution

### `CodeExecutor`

In-process Python code execution with a restricted `__builtins__` and normal imports.

```python
from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory

executor = CodeExecutor(
    cwd="/tmp/work",
    output_factory=OutputFactory(local_dir="/tmp/work"),
)
```

#### Constructor

```python
CodeExecutor(
    *,
    cwd: str,
    output_factory: OutputFactory,
    file_session_materializer=None,  # optional async hook for workspace file materialization
)
```

#### Methods

##### `execute(code: str, dry_run=False, timeout_seconds=None, producer_notebook_path=None, seen_live_names=None) -> KernelOutput` (async)

Execute Python code in the persistent kernel namespace.

```python
output = await executor.execute("x = 1 + 1")
print(output.outputs)    # list of KernelOutputType (PrimitiveObject, DataFrameObject, etc.)
print(output.fetch_log)  # list of FetchLogEntry
```

`dry_run=True` copies the namespace before execution; mutations do not persist.

### `OutputFactory`

Converts execution outputs to typed artifact objects.

```python
from parsimony_agents.execution.factory import OutputFactory

factory = OutputFactory(local_dir="/tmp/outputs")
```

## Artifacts

### `Dataset`

Curated data artifact with identity, curation, and lineage.

```python
class Dataset(_ArtifactBase):
    type: Literal["dataset"] = "dataset"
    logical_id: str
    content_sha: str
    title: str
    description: str
    tags: list[str]
    notes: list[str]
    live_name: str | None
    notebook_refs: list[ArtifactRef]
    source_refs: list[ArtifactRef]
    variable_name: str
```

The DataFrame payload is held in a private `_payload: DataFrameObject | None` attribute, not exposed as a public field. Use `dataset.payload` to access it.

**Example:**

```python
dataset = result.datasets["unemployment"]
print(f"Title: {dataset.title}")
print(f"Description: {dataset.description}")
print(f"Tags: {dataset.tags}")
```

### `Chart`

Visualization artifact with curation and lineage.

```python
class Chart(_ArtifactBase):
    type: Literal["chart"] = "chart"
    logical_id: str
    content_sha: str
    title: str
    description: str
    tags: list[str]
    notes: list[str]
    live_name: str | None
    notebook_ref: ArtifactRef | None
    source_dataset_refs: list[ArtifactRef]
    source_refs: list[ArtifactRef]
    variable_name: str
```

The Altair/Vega-Lite payload is held in `_payload: FigureObject | None`. Use `chart.payload` to access it.

**Example:**

```python
chart = result.charts["revenue_trend"]
print(f"Title: {chart.title}")
```

### `Report`

Markdown report artifact.

```python
class Report(_ArtifactBase):
    type: Literal["report"] = "report"
    logical_id: str
    content_sha: str
    title: str
    subtitle: str
    description: str
    tags: list[str]
    notes: list[str]
    live_name: str | None
    markdown: str
    formats: list[str]
    live_name_pins: dict[str, ArtifactRef]
```

## Tools

### Built-in Tools

Agents have these tools automatically available (the LLM calls them; you do not invoke them directly):

#### Notebook / code tools

| Tool | Purpose |
|------|---------|
| `return_notebook` | Write notebook cells to disk (creates or replaces) |
| `edit_notebook` | Edit individual cells within an existing notebook |
| `dry_execute_code` | Execute code in a throwaway sandbox for inspection |

#### File tools

| Tool | Purpose |
|------|---------|
| `write_file` | Write a new file to the working directory |
| `edit_file` | Apply a patch to an existing file |
| `read_file` | Read a file from the working directory |
| `list_files` | List files in the working directory |

#### Data tools

| Tool | Purpose |
|------|---------|
| `read_data` | Fetch data from a bound connector |
| `refresh` | Re-fetch connector data (e.g. for live feeds) |
| `restart_kernel` | Clear the executor namespace |

#### Return / artifact tools

| Tool | Purpose |
|------|---------|
| `return_dataset` | Finalize and publish a dataset artifact |
| `return_chart` | Finalize and publish a chart artifact |
| `return_report` | Finalize and publish a report artifact |
| `edit_report` | Edit an in-progress report artifact |

#### Utility tools

| Tool | Purpose |
|------|---------|
| `output_read` | Read a previously returned artifact by name |
| `output_search` | Semantic search over previous outputs (requires RAG extra) |

## Display & Streaming

### `stream_to_display`

Async helper that runs a full agent turn and formats events for terminal output.

```python
from parsimony_agents import stream_to_display, AgentResult

async def main():
    result = await stream_to_display(agent, "What is the current unemployment rate?")
    # result is an AgentResult
    print(result.text)

# Pass ctx for multi-turn conversations
result2 = await stream_to_display(agent, "Show a chart", ctx=result.context)
```

**Signature:**

```python
async def stream_to_display(
    agent,
    message: str,
    *,
    ctx=None,
    console=None,
    show_code: bool = True,
    show_data: bool = True,
    max_table_rows: int = 5,
    max_code_lines: int = 30,
) -> AgentResult
```

### `display_result`

Pretty-print an already-collected `AgentResult`.

```python
from parsimony_agents import display_result

result = await agent.ask("Query")
display_result(result)
```

**Signature:**

```python
def display_result(
    result: AgentResult,
    *,
    console=None,
    show_code: bool = True,
    show_data: bool = True,
    max_table_rows: int = 5,
    max_code_lines: int = 30,
) -> None
```

## Configuration

### `AgentGuardrails`

Safety limits on agent execution.

```python
from parsimony_agents.agent.config import AgentGuardrails

guardrails = AgentGuardrails(
    max_iterations=30,           # Max LLM turns
    max_execution_time_s=120.0,  # Max agent run wall time
    tool_timeout_s=600.0,        # Per-tool timeout
    llm_timeout_s=60.0,          # Per-LLM-call timeout
)

agent = Agent(guardrails=guardrails, ...)
```

## Events (Streaming)

Typed event objects yielded by `agent.run()`. Each event has a `type` attribute you can match on:

| Type | Class | Key fields |
|------|-------|------------|
| `text_delta` | `TextDelta` | `content: str`, `message_id: str`, `delta: str` |
| `reasoning_delta` | `ReasoningDelta` | `content: str`, `message_id: str`, `title: str`, `delta: str` |
| `tool_event` | `ToolEvent` | `tool_name: str`, `tool_call_id: str`, `tool_type: str`, `completed: bool`, `result: Any`, `ui_message: str`, `ui_message_completed: bool` |
| `state_snapshot` | `StateSnapshot` | `context: AgentContext` |
| `error` | `AgentError` | `message: str`, `recoverable: bool`, `error_type: str` |
| `run_cancelled` | `RunCancelled` | `message: str`, `reason: str` |
| `llm_call_completed` | `LLMCallCompleted` | _(no payload fields)_ |
| `tool_result_observed` | `ToolResultObserved` | _(no payload fields)_ |

**Example — accumulate text and track tool calls:**

```python
messages: dict[str, str] = {}
async for event in agent.run("question"):
    if event.type == "text_delta":
        messages[event.message_id] = messages.get(event.message_id, "") + event.content
    elif event.type == "tool_event" and not event.completed:
        print(f"Calling tool: {event.tool_name}")
    elif event.type == "error":
        print(f"Error: {event.message} (recoverable={event.recoverable})")
```

**Importing event types:**

```python
from parsimony_agents.agent.events import (
    TextDelta,
    ReasoningDelta,
    ToolEvent,
    StateSnapshot,
    AgentError,
    RunCancelled,
    LLMCallCompleted,
    ToolResultObserved,
)
```

## Notebooks

### `Script`

Workspace notebook file: path, code body, and (after execution) kernel output.

```python
class Script(BaseModel):
    type: Literal["script"] = "script"
    path: str         # Workspace path, e.g. "notebooks/main.py"
    code: str         # Full Python source
    output: KernelOutput
    data_objects: list[FetchLogEntry]
```

### `ScriptPreview`

Lightweight UI-oriented projection of a `Script`.

```python
class ScriptPreview(BaseModel):
    type: Literal["script_preview"] = "script_preview"
    path: str
    code: str
    error_message: str | None
    data_objects: list[FetchLogEntry]
    output: KernelOutput | None
    steps: list[ScriptStepPreview]   # computed from code structure
```

## Environment Variables

Set these to configure agents:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude models |
| `OPENAI_API_KEY` | OpenAI GPT models |
| `GEMINI_API_KEY` | Google Gemini models |
| `AZURE_API_KEY` | Azure OpenAI models |
| `FRED_API_KEY` | FRED data source |
| `FMP_API_KEY` | FMP financial data |

## See Also

- [Documentation Index](INDEX.md) — Navigation guide by user role
- [Architecture](ARCHITECTURE.md) — System design, data flow, and design patterns
- [RUNBOOK](RUNBOOK.md) — Deployment, configuration, monitoring, and troubleshooting
- [CODEMAPS](CODEMAPS.md) — Code structure and public API exports
- [COMMANDS](COMMANDS.md) — Development commands and testing
- [Contributing](../CONTRIBUTING.md) — How to contribute
- [Examples](../examples/) — Runnable code examples
- [LiteLLM docs](https://docs.litellm.ai/) — Supported LLM providers
