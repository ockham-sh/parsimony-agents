# Code Structure & API Map

Directory and module organization for parsimony-agents.

## Package Structure

```
parsimony_agents/
├── __init__.py               # Public API exports
├── agent/                    # Core Agent orchestrator
│   ├── agent.py             # Main Agent class and AgentResult
│   ├── config.py            # AgentConfig, AgentGuardrails, FileStore protocol
│   ├── events.py            # Typed event classes (streaming protocol)
│   ├── models.py            # AgentContext, AgentContextSnapshot
│   ├── prompts.py           # DEFAULT_DATA_ANALYSIS_PROMPT
│   ├── tracing.py           # OpenTelemetry spans
│   ├── helpers.py           # TurnState, parse_cell_ref, system_error
│   ├── outputs.py           # UtilityToolOutput, SystemToolOutput
│   └── __init__.py
├── execution/               # Code execution engine
│   ├── executor.py          # CodeExecutor, BaseCodeExecutor (sandboxed Python)
│   ├── factory.py           # OutputFactory (dispatch outputs to artifact types)
│   ├── outputs.py           # KernelOutput, KernelOutputType subtypes
│   ├── metadata.py          # Execution metadata structures
│   ├── fetch_log.py         # Data fetch tracking & make_fetch_logger
│   ├── dataframe_ref.py     # DataframeRef, StorageBackend
│   ├── data_objects.py      # make_data_object_persister
│   ├── connector_cache.py   # ConnectorCache, MemoizingConnectorBundle
│   ├── helpers.py           # normalize_connector_bundles
│   ├── load.py              # build_load_dataset
│   ├── pagination.py        # StringPaginator, TablePaginator
│   ├── run_scope.py         # OriginLedger, VariableOrigin, RunScope
│   └── __init__.py
├── artifacts.py             # Dataset, Chart, Report (single file, not a subpackage)
├── notebook.py              # Script, ScriptPreview, ScriptStepPreview
├── identity.py              # ArtifactRef, logical_id computation
├── messages.py              # MessageContent, to_litellm, from_litellm
├── tools.py                 # Tool, ToolMethod, Tools, @tool, @toolmethod
├── display.py               # stream_to_display, display_result
├── dataset_io.py            # read_dataset, write_dataset_bytes, (de)serialize_dataset
├── chart_io.py              # read_chart, write_chart_bytes, (de)serialize_chart
├── notebook_io.py           # read_notebook, save_notebook, state helpers
├── report_format.py         # compose_snapshot, frontmatter helpers
├── refresh.py               # embedded_refs_from_markdown
├── storage.py               # Storage helpers
├── theme.py                 # Altair theme registration
├── views.py                 # LLM view configs
├── _naming.py               # slug_from_title
└── rag/                     # RAG (Retrieval-Augmented Generation) — optional extra
    ├── __init__.py          # hybrid_search (RRF + semantic re-rank)
    └── processors/          # TextProcessor, TabularProcessor, OutputProcessor
        └── __init__.py
```

## Public API

Exposed in `parsimony_agents/__init__.py`:

```python
from parsimony_agents import (
    # Core agent
    Agent,              # Main agent class
    AgentResult,        # Structured response

    # Artifact types
    Chart,              # Altair/Vega-Lite visualization
    Dataset,            # Dataset with curation + lineage
    Report,             # Markdown report artifact

    # Notebook classes
    Script,             # Workspace notebook file (path + code + output)
    ScriptPreview,      # UI-oriented projection of a Script

    # Dataset I/O
    read_dataset,
    serialize_dataset,
    deserialize_dataset,

    # Chart I/O
    read_chart,
    serialize_chart,
    deserialize_chart,

    # Notebook I/O
    read_notebook,
    save_notebook,
    serialize_notebook,
    deserialize_notebook,
    save_notebook_state,
    load_notebook_state,
    notebook_state_cache_key,
    decode_notebook_state,

    # Display helpers
    stream_to_display,  # Run agent turn and format output for terminal
    display_result,     # Pretty-print AgentResult
)
```

## Core Classes

### Agent (`agent/agent.py`)

**Main entry point** for all agent interactions.

```python
agent = Agent(
    model="claude-sonnet-4-6",
    connectors=...,
    guardrails=AgentGuardrails(...),
)

# Simple mode
result = await agent.ask("Query")

# Streaming mode
async for event in agent.run("Query"):
    ...
```

**Key methods:**
- `ask(message: str, *, ctx=None) -> AgentResult` — Complete response (async)
- `run(user_message: str, *, ctx=None, tool_choice="auto", cancellation=None) -> AsyncGenerator` — Stream events

### AgentResult (`agent/agent.py`)

Complete response from `agent.ask()` or accumulated via `AgentResult._collect(event)`:

```python
@dataclass
class AgentResult:
    text: str                          # Natural language analysis
    datasets: dict[str, Dataset]       # Returned data
    charts: dict[str, Chart]           # Returned charts
    code: dict[str, Script]            # Executed notebooks (path -> Script)
    context: AgentContext | None       # Conversation state for follow-up calls
    events: list[Any]                  # All events collected during the run
    ok: bool                           # True if no unrecoverable error occurred
```

### CodeExecutor (`execution/executor.py`)

**Sandboxed Python execution engine** for agent-generated code.

```python
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.executor import CodeExecutor

executor = CodeExecutor(
    cwd="/tmp/work",
    output_factory=OutputFactory(local_dir="/tmp/work"),
)

output = await executor.execute(code)
```

**Key attributes:**
- `cwd` — Working directory
- `locals` — Kernel namespace (persists across `execute` calls)
- `origin_ledger` — Per-kernel variable origin ledger

**Key methods (all async):**
- `execute(code: str, dry_run=False, ...) -> KernelOutput`
- `eval(expr: str, ...) -> KernelOutput`
- `get(key: str) -> KernelOutputType | None`
- `clear_namespace() -> None`
- `set_cwd(cwd: str, session_id=None) -> None`
- `read_workspace_file(path: str) -> bytes`
- `write_workspace_file(path: str, data: bytes) -> None`
- `list_workspace_files(prefix="") -> list[tuple[str, int]]`

### Artifact Classes (`artifacts.py`)

All three artifact classes inherit from `_ArtifactBase` (a Pydantic model) and share:
- `logical_id`, `content_sha` — dual-identity fields
- `title`, `description`, `tags`, `notes`, `live_name` — curation fields

**Dataset** — data artifact:
- Payload: `_payload: DataFrameObject | None` (private, use `.payload`)
- Identity fields: `variable_name`, `notebook_refs`, `source_refs`

**Chart** — visualization artifact:
- Payload: `_payload: FigureObject | None` (private, use `.payload`)
- Identity fields: `variable_name`, `notebook_ref`, `source_dataset_refs`, `source_refs`

**Report** — markdown document:
- Content: `markdown: str`, `subtitle: str`, `formats: list[str]`
- Pins: `live_name_pins: dict[str, ArtifactRef]`

### Notebook Classes (`notebook.py`)

**Script** — workspace notebook file:

```python
class Script(BaseModel):
    type: Literal["script"] = "script"
    path: str           # Workspace path, e.g. "notebooks/main.py"
    code: str           # Full Python source
    output: KernelOutput
    data_objects: list[FetchLogEntry]
```

**ScriptPreview** — UI-oriented projection:

```python
class ScriptPreview(BaseModel):
    type: Literal["script_preview"] = "script_preview"
    path: str
    code: str
    error_message: str | None
    data_objects: list[FetchLogEntry]
    output: KernelOutput | None
    steps: list[ScriptStepPreview]   # computed field
```

## Built-in Tools

Available to agents automatically (registered in `Agent.__init__`):

| Tool | Type | Purpose |
|------|------|---------|
| `return_notebook` | `code` | Write notebook cells to disk |
| `edit_notebook` | `code` | Edit individual notebook cells |
| `dry_execute_code` | `code` | Execute code in a dry-run sandbox |
| `write_file` | `code` | Write a file to the working directory |
| `edit_file` | `code` | Patch an existing file |
| `read_file` | `utility` | Read a file from the working directory |
| `read_data` | `utility` | Read a dataset from a connector |
| `list_files` | `utility` | List files in the working directory |
| `restart_kernel` | `utility` | Clear executor namespace |
| `return_dataset` | `return` | Finalize and publish a dataset artifact |
| `return_chart` | `return` | Finalize and publish a chart artifact |
| `return_report` | `return` | Finalize and publish a report artifact |
| `edit_report` | `return` | Edit an in-progress report artifact |
| `refresh` | `utility` | Refresh connector data |
| `output_read` | `utility` | Read a previously returned artifact |
| `output_search` | `utility` | Semantic search over outputs (if RAG enabled) |

## Events (Streaming)

Typed event objects yielded by `agent.run()` — defined in `agent/events.py`:

| Type string | Class | Key fields |
|-------------|-------|------------|
| `text_delta` | `TextDelta` | `content`, `message_id`, `delta` |
| `reasoning_delta` | `ReasoningDelta` | `content`, `message_id`, `title`, `delta` |
| `tool_event` | `ToolEvent` | `tool_name`, `tool_call_id`, `tool_type`, `completed`, `result`, `ui_message`, `ui_message_completed` |
| `state_snapshot` | `StateSnapshot` | `context` |
| `error` | `AgentError` | `message`, `recoverable`, `error_type` |
| `run_cancelled` | `RunCancelled` | `message`, `reason` |
| `llm_call_completed` | `LLMCallCompleted` | _(no payload fields)_ |
| `tool_result_observed` | `ToolResultObserved` | _(no payload fields)_ |

## Configuration

### AgentGuardrails (`agent/config.py`)

Safety limits on execution:

```python
AgentGuardrails(
    max_iterations=30,         # Max LLM turns
    max_execution_time_s=120.0,  # Max agent run wall time
    tool_timeout_s=600.0,
    llm_timeout_s=60.0,
    llm_max_retries=3,
)
```

## Import Hierarchy

```
parsimony_agents
├── __init__ (public exports)
│   ├── Agent, AgentResult
│   ├── Chart, Dataset, Report
│   ├── Script, ScriptPreview
│   ├── read_dataset / serialize_dataset / deserialize_dataset
│   ├── read_chart / serialize_chart / deserialize_chart
│   ├── read_notebook / save_notebook / serialize_notebook / deserialize_notebook
│   ├── save_notebook_state / load_notebook_state / notebook_state_cache_key / decode_notebook_state
│   └── stream_to_display, display_result
├── agent.agent
│   └── Agent, AgentResult (main class)
├── execution.*
│   ├── CodeExecutor (executor.py)
│   ├── OutputFactory (factory.py)
│   └── KernelOutput (outputs.py)
├── artifacts.py
│   ├── Dataset
│   ├── Chart
│   └── Report
├── notebook.py
│   ├── Script
│   └── ScriptPreview
└── rag.*  (optional extra)
    └── hybrid_search, OutputProcessor
```

## Key Interfaces

### Connector Protocol

From `parsimony`:

```python
# Connectors are discovered and bound with:
from parsimony import discover

connectors = discover.load_all().bind_env()   # autoload + env-bind
connectors = discover.load("fred", "sdmx")   # explicit selection

# Connectors.unbound reports which names are missing env vars
for name in connectors.unbound:
    print(f"Missing env for: {name}")
```

## Testing Structure

```
tests/
├── integration/           # End-to-end tests requiring live or mock env
├── test_agent_*.py        # Agent-level unit tests
├── test_code_executor_*.py
├── test_chart_io.py
├── test_dataset_io.py
├── test_notebook_io.py
└── ... (per-module test files)
```

## See Also

- [Documentation Index](INDEX.md) — Navigation guide by user role
- [Architecture](ARCHITECTURE.md) — Design patterns and data flow
- [API Reference](API.md) — Complete method signatures and parameter details
- [RUNBOOK](RUNBOOK.md) — Deployment and operations
- [COMMANDS](COMMANDS.md) — Development commands and testing
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Development setup
