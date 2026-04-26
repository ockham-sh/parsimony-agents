# Code Structure & API Map

Directory and module organization for parsimony-agents.

## Package Structure

```
parsimony_agents/
├── __init__.py               # Public API exports
├── agent/                    # Core Agent orchestrator
│   ├── agent.py             # Main Agent class
│   ├── config.py            # AgentGuardrails configuration
│   └── __init__.py
├── execution/               # Code execution engine
│   ├── executor.py          # CodeExecutor (sandboxed Python execution)
│   ├── factory.py           # OutputFactory (dispatch outputs to artifact types)
│   ├── outputs.py           # Output models (ExecutionResult, etc.)
│   ├── metadata.py          # Execution metadata structures
│   ├── fetch_log.py         # Data fetch tracking & provenance
│   ├── dataframe_ref.py     # DataFrame reference & metadata
│   ├── helpers.py           # Utility functions
│   ├── pagination.py        # Pagination utilities
│   └── __init__.py
├── artifacts/               # Artifact definitions
│   ├── artifact.py          # Base Artifact class
│   ├── dataset.py           # Dataset artifact
│   ├── chart.py             # Chart artifact
│   └── __init__.py
├── notebook.py              # Script & Notebook classes
├── variable.py              # Variable state management
├── messages.py              # Message & Event definitions
├── tools.py                 # Built-in tool definitions
├── display.py               # Terminal output formatting
└── rag/                     # RAG (Retrieval-Augmented Generation)
    ├── __init__.py
    ├── vector_store.py      # VectorStore interface
    ├── embeddings.py        # Embedding models
    └── processors/          # Output processors
        └── __init__.py
```

## Public API

Exposed in `parsimony_agents/__init__.py`:

```python
from parsimony_agents import (
    Agent,              # Main agent class
    AgentResult,        # Structured response
    Script,             # Executed notebook
    ScriptPreview,      # Lightweight script reference
    stream_to_display,  # Format streaming events
    display_result,     # Pretty-print results
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

**Methods:**
- `ask(query: str)` → `AgentResult` — Complete response
- `run(query: str)` → `AsyncIterator[Event]` — Stream events
- `reset()` → None — Clear execution state

### CodeExecutor (`execution/executor.py`)

**Sandboxed Python execution engine** for agent-generated code.

```python
from parsimony_agents.execution.factory import OutputFactory

executor = CodeExecutor(
    cwd="/tmp/work",
    output_factory=OutputFactory(local_dir="/tmp/work"),
)

result = await executor.execute(code)
```

**Key attributes:**
- `cwd` — Working directory
- `locals` — Kernel namespace (persists across `execute` calls)

**Methods:**
- `execute(code: str, ...)` → `KernelOutput` (async)

### OutputFactory (`execution/factory.py`)

**Dispatches execution outputs to artifact types.**

```python
factory = OutputFactory(local_dir="/tmp/outputs")
dataset = factory.create("my_data", dataframe)
chart = factory.create("my_chart", vega_spec)
```

**Methods:**
- `create(name: str, value: Any)` → `Artifact | None`
- `register(artifact_class)` → None

### Artifact Classes (`artifacts/`)

Base class for all deliverables:

- **Dataset** — DataFrame with metadata and provenance
  - `data: pd.DataFrame`
  - `metadata: DatasetMetadata`
  - `provenance: list[FetchRecord]`

- **Chart** — Altair/Vega-Lite visualization
  - `spec: dict`  # Vega-Lite JSON
  - `dataset_ref: str | None`

### Variable Store (`variable.py`)

**Manages execution state across multiple code runs.**

```python
store = VariableStore()
store.set("df", dataframe)
value = store.get("df")
store.clear()
```

**Methods:**
- `set(name: str, value: Any)` → None
- `get(name: str)` → Any
- `list()` → list[str]  # Variable names
- `clear()` → None

### Notebook Classes (`notebook.py`)

**Script** — Immutable record of executed code:
- `name: str` — Notebook name
- `source: str` — Full source code
- `cells: list[Cell]` — Individual cells
- `execution_time: float`

**ScriptPreview** — Lightweight reference:
- `name: str`
- `cell_count: int`

## Built-in Tools

Available to agents automatically:

| Tool | Module | Purpose |
|------|--------|---------|
| `code_set` | `tools.py` | Create new notebook |
| `code_edit` | `tools.py` | Modify notebook cell |
| `return_dataset` | `tools.py` | Finalize dataset |
| `return_chart` | `tools.py` | Finalize chart |
| `get_context` | `tools.py` | Inspect execution state |
| `output_search` | `rag/` | Search outputs (if RAG enabled) |

## Data Models

### AgentResult (`agent/agent.py`)

Complete response from `agent.ask()`:

```python
@dataclass
class AgentResult:
    text: str                    # Natural language analysis
    datasets: dict[str, Dataset] # Returned data
    charts: dict[str, Chart]     # Returned charts
    code: dict[str, Script]      # Executed code
    artifacts: list[Artifact]    # All artifacts
    messages: list[Message]      # Full conversation
    ok: bool                     # Success flag
    error_message: str | None    # Error details
```

### Event (`messages.py`)

Emitted by `agent.run()`:

```python
@dataclass
class Event:
    type: str          # "text_delta", "tool_call", "error", etc.
    content: Any       # Event data
    timestamp: float   # Unix timestamp
    tool_name: str | None
    error_message: str | None
```

**Event types:**
- `"text_delta"` — Incremental LLM response
- `"tool_call"` — Agent invoking a tool
- `"tool_result"` — Tool completed
- `"error"` — Execution error
- `"done"` — Finished (content is AgentResult)

### ExecutionResult (`execution/outputs.py`)

Result from `CodeExecutor.execute()`:

```python
@dataclass
class ExecutionResult:
    outputs: dict[str, Any]  # Returned variables
    stdout: str              # Captured print output
    error: Exception | None  # Exception if failed
    execution_time: float    # Wall-clock time
```

### FetchRecord (`execution/fetch_log.py`)

Data provenance entry (in `Dataset.provenance`):

```python
@dataclass
class FetchRecord:
    source: str              # Connector name
    parameters: dict[str, Any]  # Query parameters
    timestamp: datetime      # When fetched
    endpoint: str            # Specific API endpoint
```

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
│   ├── Agent
│   ├── AgentResult
│   ├── Script, ScriptPreview
│   └── stream_to_display, display_result
├── agent.agent
│   └── Agent (main class)
├── execution.*
│   ├── CodeExecutor
│   ├── OutputFactory
│   └── ExecutionResult
├── artifacts.*
│   ├── Dataset
│   ├── Chart
│   └── Artifact (base)
├── variable
│   └── VariableStore
├── notebook
│   ├── Script
│   └── Notebook
└── rag.*
    ├── VectorStore
    └── OutputProcessor
```

## Key Interfaces

### Connector Protocol

From `parsimony`:

```python
class Connector(Protocol):
    def discover(self) -> list[DataSource]:
        """List available data sources"""
    
    def fetch(self, source_id: str, **params) -> Result:
        """Fetch data"""
```

### Artifact Protocol

Base class for all deliverables:

```python
class Artifact(BaseModel):
    name: str
    type: str  # Literal union of artifact types
    metadata: dict
```

## Testing Structure

```
tests/
├── unit/              # Component tests
│   ├── test_executor.py
│   ├── test_factory.py
│   └── test_variable.py
├── integration/       # End-to-end tests
│   ├── test_agent.py
│   └── test_agent_streaming.py
└── fixtures/          # Shared test data
    └── conftest.py
```

## See Also

- [Documentation Index](index.md) — Navigation guide by user role
- [Architecture](ARCHITECTURE.md) — Design patterns and data flow
- [API Reference](API.md) — Complete method signatures and parameter details
- [RUNBOOK](RUNBOOK.md) — Deployment and operations
- [COMMANDS](COMMANDS.md) — Development commands and testing
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Development setup
