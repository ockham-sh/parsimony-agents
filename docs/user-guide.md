# ockham-agents User Guide

ockham-agents is a Python library that gives you an LLM-powered data analysis agent. You ask questions in plain English, the agent writes and executes Python code in a sandboxed environment, and you get back structured results: datasets, charts, and explanatory text.

---

## Installation

ockham-agents is not yet published to PyPI. Install directly from source.

### From source

```bash
git clone https://github.com/espinetandreu/ockham-agents
cd ockham-agents
pip install -e .
```

### With optional extras

```bash
# Terminal display with Rich (spinner, live progress, chart previews)
pip install -e ".[display]"

# Hybrid RAG search over variables (ChromaDB + Tantivy)
pip install -e ".[rag]"

# SQL queries against DataFrames via DuckDB
pip install -e ".[sql]"

# All extras
pip install -e ".[all]"
```

### Required dependencies installed automatically

| Package | Purpose |
|---------|---------|
| `ockham` | Data connector protocol |
| `litellm` | LLM provider abstraction (OpenAI, Anthropic, Gemini, etc.) |
| `pydantic` | Data validation |
| `pandas` | DataFrame operations |
| `altair==6.0.0` | Chart specification |
| `vl-convert-python==1.8.0` | Chart rendering to PNG |
| `httpx` | HTTP client for data fetching |
| `dateparser` | Natural language date parsing |

> **Important**: Three packages are imported at module load time but are not declared in `pyproject.toml`: `scipy`, `statsmodels`, and `opentelemetry-api`. If your environment does not have these installed transitively, you will see an `ImportError` when you first `import ockham_agents`. See the [Deployment Guide](./deployment.md) for full details and mitigations.

---

## Quick Start

### 1. Set your LLM API key

ockham-agents uses litellm, which picks up API keys from environment variables:

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google (Gemini)
export GEMINI_API_KEY="..."
```

Keys are never stored by this library. They are forwarded to litellm via the `model_config` dict you provide at construction.

### 2. Create an agent and ask a question

```python
import asyncio
from ockham_agents import Agent

async def main():
    agent = Agent(model="claude-sonnet-4-6")
    result = await agent.ask("Write a Python function that returns the Fibonacci sequence up to n=20, then call it and show the result.")
    print(result.text)

asyncio.run(main())
```

### 3. Check what the agent returned

```python
print(result.ok)         # True if no errors
print(result.text)       # agent's written explanation
print(result.datasets)   # dict[str, Dataset] — any returned data tables
print(result.charts)     # dict[str, Chart] — any returned charts
print(result.code)       # dict[str, Script] — the generated code notebooks
```

---

## Handling Streaming Events

`agent.ask()` collects all events and returns a final result. When you need real-time access to tokens and tool progress, use `agent.run()` directly.

```python
import asyncio
from ockham_agents import Agent, AgentResult

async def main():
    agent = Agent(model="claude-sonnet-4-6")

    result = AgentResult()
    async for event in agent.run("Analyze the top 10 Fibonacci numbers"):
        # Accumulate into a result object while also handling each event
        result._collect(event)

        match event.type:
            case "text_delta":
                # Stream text tokens to the terminal
                print(event.content, end="", flush=True)
            case "tool_event" if not event.completed:
                # Tool is starting — show the LLM's pre-execution hint
                print(f"\n  -> {event.tool_name}: {event.ui_message or '...'}", end="")
            case "tool_event" if event.completed:
                # Tool finished — show the LLM's post-execution summary
                print(f" ({event.ui_message_completed or 'done'})")
            case "state_snapshot":
                # Full context is available here for UI synchronization
                vars_count = len(event.context.data_context.variables)
                print(f"\n[State: {vars_count} variables in scope]")
            case "error":
                print(f"\n[ERROR] {event.message}")
            case _:
                pass  # reasoning_delta is emitted by extended-thinking models

    print(f"\n\nFinal datasets: {list(result.datasets.keys())}")
    print(f"Success: {result.ok}")

asyncio.run(main())
```

### The section field

Every event has a `section` field: `"analysis"` or `"final_response"`. The agent emits analysis-section events while it is reasoning and calling tools. It emits final-response-section events when writing its concluding explanation.

```python
async for event in agent.run("question"):
    if event.type == "text_delta":
        if event.section == "analysis":
            # Show in a collapsible "thinking" panel
            render_analysis_text(event.content)
        else:
            # Show in the main response area
            render_response_text(event.content)
```

---

## Multi-Turn Conversations

Pass `result.context` as `ctx` to continue a conversation. The agent restores all previously computed variables and notebook state, and avoids re-running code that was already executed (warm executor optimization).

```python
import asyncio
from ockham_agents import Agent, stream_to_display

async def main():
    agent = Agent(model="claude-sonnet-4-6")

    # First turn: fetch and analyze data
    result1 = await agent.ask("Load a sample dataset and summarize it")

    # Second turn: build on the first result
    result2 = await agent.ask(
        "Now plot the distribution of the first numeric column",
        ctx=result1.context,
    )

    # Third turn: refine further
    result3 = await agent.ask(
        "Change the chart color to blue and add a title",
        ctx=result2.context,
    )

asyncio.run(main())
```

Each `ask()` call receives the full prior conversation history and variable state. The `ctx` object is immutable from the caller's perspective — the agent creates a new context for each run.

---

## Terminal Display

The `stream_to_display` function provides a Rich terminal UI with live progress indicators and inline chart images. It requires the `[display]` optional extra.

```python
import asyncio
import os
from ockham_agents import Agent, stream_to_display
from ockham.connectors.fred import CONNECTORS as FRED

async def main():
    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind_deps(api_key=os.environ["FRED_API_KEY"]),
    )

    result = await stream_to_display(
        agent,
        "What is the current US unemployment rate?",
    )

    # Continue the session
    result2 = await stream_to_display(
        agent,
        "Show how it has changed since 2020",
        ctx=result.context,
    )

asyncio.run(main())
```

---

## Using CodeExecutor Directly

`CodeExecutor` provides a sandboxed Python execution environment that you can use independently of the agent.

```python
import asyncio
import tempfile
from ockham_agents.execution.executor import CodeExecutor
from ockham_agents.execution.factory import OutputFactory

async def main():
    # Create an executor with a temp working directory
    tmpdir = tempfile.mkdtemp()
    factory = OutputFactory(local_dir=tmpdir)
    executor = CodeExecutor(cwd=tmpdir, output_factory=factory)

    # Execute a code block
    output = await executor.execute("""
import pandas as pd
df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
display(df)
""")
    for item in output.outputs:
        print(item.type, item)  # "dataframe", DataFrameObject(...)

    # Evaluate an expression
    output2 = await executor.eval("df.shape")
    print(output2.outputs[0].value)  # (3, 2)

    # Dry run: does not modify sandbox state
    dry_output = await executor.execute("z = 999", dry_run=True)
    check = await executor.eval("'z' in dir()")
    print(check.outputs[0].value)  # False — z was not committed

asyncio.run(main())
```

### The execution namespace

Code runs with access to:
- `pd` (pandas), `np` (numpy), `alt` (altair)
- `datetime`, `timedelta`, `timezone`
- `display()` — captures any value as a structured output
- `print()` — captures text output (DataFrames and charts are promoted to display)
- `client` — connectors object (when connectors are attached)

Standard library and installed packages are available via `import`.

### Capturing outputs

The `display()` function inside the execution context captures any value as a `KernelOutputType`:

```python
output = await executor.execute("""
import altair as alt
chart = alt.Chart(...).mark_line()
display(chart)       # captured as FigureObject
display(df)          # captured as DataFrameObject
print("done")        # captured as PrimitiveObject
""")
```

All captured values appear in `output.outputs` in the order they were emitted.

---

## Registering Custom Tools

You can extend the agent with custom tools using the `@tool` and `@toolmethod` decorators.

### Standalone tool (free function)

```python
from ockham_agents.tools import tool, Tools

@tool(
    name="get_exchange_rate",
    description="Fetch the current exchange rate between two currencies.",
    parameters_schema={
        "type": "object",
        "properties": {
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"},
        },
        "required": ["from_currency", "to_currency"],
    },
    tool_type="utility",
    ui_message_completed="Fetched exchange rate",
)
async def get_exchange_rate(from_currency: str, to_currency: str, **kwargs) -> str:
    # kwargs absorbs _ui_message and other agent-injected fields
    rate = await fetch_rate_from_api(from_currency, to_currency)
    return f"1 {from_currency} = {rate} {to_currency}"
```

To add the tool to an agent, pass it via `system_tools`:

```python
agent = Agent(model="claude-sonnet-4-6")
agent.system_tools = agent.system_tools + Tools([get_exchange_rate])
```

### Tool method on a class

```python
from ockham_agents.tools import toolmethod

class DataAgent(Agent):
    @toolmethod(
        name="query_db",
        description="Query a local SQLite database.",
        parameters_schema={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL query to execute"},
            },
            "required": ["sql"],
        },
        tool_type="utility",
    )
    async def query_db(self, sql: str, **kwargs) -> str:
        # self is bound correctly via the descriptor protocol
        result = self._db_connection.execute(sql).fetchall()
        return str(result)
```

### Tool types and their behavior

| Type | When to use | Effect on agent loop |
|------|-------------|---------------------|
| `code` | Writing or editing notebook cells | Agent tracks changes to the active Script |
| `utility` | Data fetching, search, output inspection | Result is shown to the LLM; loop continues |
| `return` | Delivering a final Dataset or Chart | Terminates the agent loop |
| `system` | Session and context management | Used for internal agent housekeeping |

---

## Chart Generation with Altair

The agent generates charts using [Altair](https://altair-viz.github.io/) (Vega-Lite). Charts are validated against the spec at generation time using vl-convert. The Ockham theme is applied automatically: dark background, Ubuntu Mono font, 640x400 dimensions.

To produce a chart, the agent writes code like:

```python
import altair as alt
chart = alt.Chart(df).mark_line().encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("value:Q", title="Value"),
).properties(title="My Chart")
display(chart)
```

The chart is captured as a `FigureObject`, validated, and included in the `AgentResult.charts` dict when the agent calls `return_chart`.

### Extending OutputFactory for custom chart types

If you use Plotly, Matplotlib, or another charting library, register a handler:

```python
import plotly.graph_objects as go
from ockham_agents.execution.factory import OutputFactory
from ockham_agents.execution.outputs import PrimitiveObject

OutputFactory.register(
    go.Figure,
    lambda val, **kw: PrimitiveObject(value=val.to_json()),
)
```

---

## Connecting Data Sources

The `connectors` parameter accepts a `ockham.Connectors` object. When attached, the connector catalog is appended to the system prompt so the agent knows what data sources are available.

```python
import os
from ockham.connectors.fred import CONNECTORS as FRED
from ockham.connectors.sdmx import CONNECTORS as SDMX

# Single connector
agent = Agent(
    model="claude-sonnet-4-6",
    connectors=FRED.bind_deps(api_key=os.environ["FRED_API_KEY"]),
)

# Multiple connectors composed together
all_connectors = (
    FRED.bind_deps(api_key=os.environ["FRED_API_KEY"])
    + SDMX
)
agent = Agent(model="claude-sonnet-4-6", connectors=all_connectors)
```

Inside executed code, the connector is available as `client`:

```python
# Agent-generated code can call connectors by name:
result = await client["fred_fetch"](series_id="UNRATE")
df = result.data  # Returns a DataFrame directly
```

---

## Environment Variables

ockham-agents does not read any environment variables directly. LLM API keys are passed through litellm via the `model_config` dict or the `api_key` constructor parameter. litellm reads provider-specific environment variables by convention:

| Variable | Provider |
|----------|---------|
| `ANTHROPIC_API_KEY` | Anthropic (Claude models) |
| `OPENAI_API_KEY` | OpenAI (GPT models) |
| `GEMINI_API_KEY` | Google (Gemini models) |
| `COHERE_API_KEY` | Cohere |
| `AZURE_API_KEY` + `AZURE_API_BASE` | Azure OpenAI |

Data connector API keys are not managed by this library. Pass them when constructing connector objects (e.g., `FRED.bind_deps(api_key="...")`).

---

## Learning from the Examples

The repository includes four example files in `examples/`:

| File | What it demonstrates |
|------|---------------------|
| `examples/quickstart.py` | Minimal `Agent` construction, `stream_to_display`, multi-turn reuse with `ctx=` |
| `examples/event_stream.py` | Direct `agent.run()` loop, match-case event handling, `AgentResult._collect()` pattern |
| `examples/terminal_chat.py` | Interactive REPL session with persistent context across multiple user inputs |
| `examples/usage_patterns.py` | Additional configuration patterns: custom guardrails, custom instructions, model_config |

All examples require a FRED API key (free: https://fred.stlouisfed.org/docs/api/api_key.html) and a supported LLM API key.

---

## Troubleshooting

### ImportError on `import ockham_agents`

The most common cause is missing undeclared dependencies. Check whether `scipy`, `statsmodels`, or `opentelemetry-api` are installed:

```bash
python -c "import scipy; import statsmodels; import opentelemetry"
```

If any fail, install them manually:

```bash
pip install scipy statsmodels opentelemetry-api
```

See the [Deployment Guide](./deployment.md) for a complete workaround.

### Agent stops with AgentError

Check `result.ok` and inspect `result.events` for the `AgentError` event:

```python
for event in result.events:
    if event.type == "error":
        print(f"Error: {event.message}")
        print(f"Type: {event.error_type}")
        print(f"Recoverable: {event.recoverable}")
```

Common causes: LLM rate limits (handled automatically up to `llm_max_retries`), execution timeouts (increase `AgentGuardrails.max_execution_time_s`), or guardrail iteration limits (increase `max_iterations`).

### Chart validation fails

Charts fail validation when the Altair spec is invalid for the pinned Vega-Lite version. The `FigureObject` is replaced with an `ExceptionObject` whose `value` describes the spec error. Check the chart variable in the agent's output:

```python
output = await executor.get("my_chart")
if output.type == "exception":
    print(output.value)  # Vega-Lite error message
```

### Multi-turn context grows too large

Very long sessions accumulate message history that can exceed the LLM's context window. Start a new `Agent` instance and pass only the relevant context if this occurs.
