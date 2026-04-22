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
    model_config: dict | None = None,
    instructions: str | None = None,
    connectors: ConnectorGroup | None = None,
    code_executor: CodeExecutor | None = None,
    output_factory: OutputFactory | None = None,
    guardrails: AgentGuardrails | None = None,
    tools: list[Callable] | None = None,
    max_retries: int = 3,
)
```

**Parameters:**
- `model` — LLM model string (e.g., `"claude-sonnet-4-6"`, `"gpt-4o"`). Passed to LiteLLM.
- `model_config` — Advanced LLM configuration: `{"model": "...", "api_key": "...", ...}`
- `instructions` — System prompt for the agent
- `connectors` — Data sources available to the agent
- `code_executor` — Python execution engine (default: new `CodeExecutor()`)
- `output_factory` — Output type dispatcher (default: new `OutputFactory()`)
- `guardrails` — Execution safety limits
- `tools` — Additional tools beyond built-in ones
- `max_retries` — Retry failed tool calls this many times

#### Methods

##### `ask(query: str, **kwargs) -> Coroutine[AgentResult]`

Simple ask-and-answer mode. Returns complete result.

```python
result = await agent.ask("Show me Apple revenue growth over 5 years")
print(result.text)           # str — natural language analysis
print(result.datasets)       # dict[str, Dataset]
print(result.charts)         # dict[str, Chart]
print(result.code)           # dict[str, Script] — named scripts
print(result.ok)             # bool — success indicator
print(result.error_message)  # str | None — error details
```

**Returns:** `AgentResult` with text, datasets, charts, code, and error status.

##### `run(query: str, **kwargs) -> AsyncIterator[Event]`

Streaming mode. Yields events as they occur.

```python
async for event in agent.run("Analyze market trends"):
    match event.type:
        case "text_delta":
            print(event.content, end="", flush=True)
        case "tool_call":
            print(f"\n[Tool: {event.tool_name}]")
        case "tool_result":
            print(f"[Result: {event.content}]")
        case "error":
            print(f"Error: {event.message}")
        case "done":
            print("\nDone.")
```

**Yields:** Event objects with type, content, and metadata.

### `AgentResult`

Structured response from `agent.ask()`.

```python
@dataclass
class AgentResult:
    text: str                          # Natural language analysis
    datasets: dict[str, Dataset]       # Returned data
    charts: dict[str, Chart]           # Returned visualizations
    code: dict[str, Script]            # Executed code (name -> Script)
    artifacts: list[Artifact]          # All artifacts produced
    messages: list[Message]            # Full conversation history
    ok: bool                           # Success indicator
    error_message: str | None          # Error details if ok=False
```

**Example:**

```python
result = await agent.ask("Calculate GDP per capita")

# Access datasets
gdp = result.datasets.get("gdp_per_capita")
print(f"Type: {gdp.type}")
print(f"Shape: {gdp.data.shape}")
print(f"Provenance: {gdp.provenance}")

# Access charts
for name, chart in result.charts.items():
    print(f"Chart '{name}': {chart.spec}")

# Access execution code
for name, script in result.code.items():
    print(f"Script '{name}':\n{script.source}")
```

## Execution

### `CodeExecutor`

Sandboxed Python code execution.

```python
from parsimony_agents.execution.executor import CodeExecutor

executor = CodeExecutor(
    cwd="/tmp/work",
    sandbox=True,
    allowed_imports=["pandas", "numpy", ...]
)
```

#### Constructor

```python
CodeExecutor(
    cwd: str = "/tmp/parsimony-work",
    sandbox: bool = True,
    allowed_imports: list[str] | None = None,
    output_factory: OutputFactory | None = None,
    timeout_s: float = 120.0,
)
```

#### Methods

##### `execute(code: str, variables: dict) -> ExecutionResult`

Execute Python code with given variables.

```python
result = executor.execute(
    code="x = df.sum()",
    variables={"df": my_dataframe}
)

print(result.outputs)      # dict of returned variables
print(result.stdout)       # Captured print output
print(result.error)        # Exception if execution failed
```

### `OutputFactory`

Converts execution outputs to typed Artifacts.

```python
from parsimony_agents.execution.factory import OutputFactory

factory = OutputFactory(local_dir="/tmp/outputs")
```

#### Methods

##### `create(name: str, value: Any) -> Artifact | None`

Convert a Python value to an Artifact.

```python
import pandas as pd
import altair as alt

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
chart = alt.Chart(df).mark_point().encode(x="x", y="y")

dataset_artifact = factory.create("my_data", df)      # Dataset
chart_artifact = factory.create("my_chart", chart)    # Chart
```

## Artifacts

### `Dataset`

Typed data artifact with metadata and provenance.

```python
@dataclass
class Dataset(Artifact):
    data: pd.DataFrame
    metadata: DatasetMetadata
    provenance: list[FetchRecord]
```

**Properties:**
- `name` — Dataset identifier
- `type` — Always `"dataset"`
- `data` — Pandas DataFrame
- `metadata.description` — Human-readable description
- `metadata.columns` — Column information
- `provenance` — List of data fetches with source, parameters, timestamp

**Example:**

```python
dataset = result.datasets["unemployment"]
print(f"Source: {dataset.provenance[0].source}")
print(f"Fetched: {dataset.provenance[0].timestamp}")
print(f"Parameters: {dataset.provenance[0].parameters}")
```

### `Chart`

Visualization artifact with Altair/Vega-Lite specification.

```python
@dataclass
class Chart(Artifact):
    spec: dict  # Vega-Lite JSON spec
    dataset_ref: str | None  # Reference to source dataset
```

**Properties:**
- `name` — Chart identifier
- `type` — Always `"chart"`
- `spec` — Vega-Lite specification (dict)

**Example:**

```python
chart = result.charts["revenue_trend"]
print(chart.spec)  # Raw Vega-Lite spec
# Use vl-convert to render: vega_to_html(chart.spec)
```

## Tools

### Built-in Tools

Agents have these tools automatically available:

#### `code_set`

Write a new code notebook.

```python
# Called automatically by agent, but signature is:
# code_set(notebook_name: str, code: str) -> ExecutionResult
```

#### `code_edit`

Modify an existing notebook cell.

```python
# code_edit(notebook_name: str, cell_index: int, new_code: str)
```

#### `return_dataset`

Finalize and return a dataset.

```python
# return_dataset(name: str, description: str = "")
```

#### `return_chart`

Finalize and return a chart.

```python
# return_chart(name: str, description: str = "")
```

#### `get_context`

Inspect current execution state.

```python
# get_context() -> dict
# Returns: {
#   "variables": list of variable names and types,
#   "notebooks": list of notebook names,
#   "artifacts": list of artifact names and types
# }
```

#### `output_search` (if RAG enabled)

Semantic search over previous outputs.

```python
# output_search(query: str, limit: int = 5) -> list[OutputMatch]
```

## Display & Streaming

### `stream_to_display`

Format streaming events for terminal output.

```python
from parsimony_agents import stream_to_display

async for event in agent.run("Query"):
    stream_to_display(event)
```

### `display_result`

Pretty-print an AgentResult.

```python
from parsimony_agents import display_result

result = await agent.ask("Query")
display_result(result)
```

## Configuration

### `AgentGuardrails`

Safety limits on agent execution.

```python
from parsimony_agents.agent.config import AgentGuardrails

guardrails = AgentGuardrails(
    max_iterations=30,              # Max LLM turns
    max_execution_time_s=120.0,     # Max execution time
    max_output_size_mb=100,         # Max output file size
    max_code_lines=500,             # Max code lines per turn
    allowed_imports=[               # Whitelist imports
        "pandas", "numpy", "altair", ...
    ]
)

agent = Agent(guardrails=guardrails, ...)
```

## Events (Streaming)

Event types yielded by `agent.run()`:

```python
@dataclass
class Event:
    type: str          # Event type
    content: Any       # Event data
    timestamp: float   # Unix timestamp
    tool_name: str | None      # Tool name (if tool_* event)
    error_message: str | None  # Error details (if error event)
```

**Event types:**

| Type | Content | Meaning |
|------|---------|---------|
| `text_delta` | str | Incremental text response |
| `tool_call` | ToolCall | Agent calling a tool |
| `tool_result` | str | Tool returned a result |
| `error` | str | Execution error |
| `done` | AgentResult | Agent finished |

## Notebooks

### `Script`

Immutable record of executed code.

```python
@dataclass
class Script:
    name: str                    # Notebook name
    source: str                  # Full source code
    language: str                # "python"
    cells: list[Cell]            # Individual cells with outputs
    execution_time: float        # Total execution time
```

### `ScriptPreview`

Lightweight reference to a script.

```python
@dataclass
class ScriptPreview:
    name: str
    language: str
    cell_count: int
```

## Environment Variables

Set these to configure agents:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude models |
| `OPENAI_API_KEY` | OpenAI GPT models |
| `GEMINI_API_KEY` | Google Gemini models |
| `FRED_API_KEY` | FRED data source |
| `FMP_API_KEY` | FMP financial data |
| `LITELLM_PROXY_URL` | LLM routing proxy |

## See Also

- [Documentation Index](index.md) — Navigation guide by user role
- [Architecture](ARCHITECTURE.md) — System design, data flow, and design patterns
- [RUNBOOK](RUNBOOK.md) — Deployment, configuration, monitoring, and troubleshooting
- [CODEMAPS](CODEMAPS.md) — Code structure and public API exports
- [COMMANDS](COMMANDS.md) — Development commands and testing
- [Contributing](../CONTRIBUTING.md) — How to contribute
- [Examples](../examples/) — Runnable code examples
- [LiteLLM docs](https://docs.litellm.ai/) — Supported LLM providers
