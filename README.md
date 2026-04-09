# ockham-agents

[![PyPI version](https://img.shields.io/pypi/v/ockham-agents)](https://pypi.org/project/ockham-agents/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/ockham-agents)](https://pypi.org/project/ockham-agents/)
[![CI](https://github.com/espinetandreu/ockham-agents/actions/workflows/test.yml/badge.svg)](https://github.com/espinetandreu/ockham-agents/actions)

Build AI agents that discover, fetch, and analyze data.

### Why ockham-agents?

LLM frameworks are generic by design. `ockham-agents` is purpose-built for data analysis: agents write and execute Python code against typed data connectors, track provenance for every data fetch, and produce reproducible datasets and Altair visualizations. Works with any LLM provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers) (OpenAI, Anthropic, Google, Azure, local models, and more).

## Quick Start

```python
from ockham_agents import Agent
from ockham.connectors.fred import CONNECTORS as FRED

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=FRED.bind_deps(api_key="your-fred-key"),
)

result = await agent.ask("Show me US GDP trends over the last 20 years")

print(result.text)        # Natural language response
print(result.datasets)    # {"us_gdp": <DataFrame>}
print(result.code)        # {"main": Script(...), ...} — named scripts keyed by notebook name
```

## Installation

```bash
pip install ockham-agents
```

Requires [ockham](../ockham) (installed automatically as a dependency).

## Features

### Code execution with provenance

Agents write Python code that runs in a sandboxed executor. Every data fetch is tracked with full provenance — source, parameters, timestamps.

```python
# Agent writes this code automatically:
result = await client["fred_fetch"](series_id="GDPC1", observation_start="2005-01-01")
gdp = result.data  # pandas DataFrame with provenance attached
```

### Composable data sources

Plug in any combination of data sources via [ockham](../ockham) connectors:

```python
from ockham.connectors.fred import CONNECTORS as FRED
from ockham.connectors.sdmx import CONNECTORS as SDMX
from ockham.connectors.fmp import CONNECTORS as FMP

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=(
        FRED.bind_deps(api_key="...")
        + SDMX
        + FMP.bind_deps(api_key="...")
    ),
)
```

### Two consumption modes

**Simple mode** — ask a question, get a structured result:

```python
result = await agent.ask("Compare Apple and Microsoft revenue growth")
result.text        # str — natural language analysis
result.datasets    # dict[str, Dataset] — returned datasets
result.charts      # dict[str, Chart] — returned charts
result.code        # dict[str, Script] — named scripts (order preserved)
result.ok          # bool — True if no errors
```

**Streaming mode** — consume events as they arrive (see [examples/event_stream.py](examples/event_stream.py) for a runnable version):

```python
async for event in agent.run("Analyze S&P 500 returns"):
    match event.type:
        case "text_delta":
            print(event.content, end="", flush=True)
        case "tool_event" if event.completed:
            print(f"\n[Tool: {event.tool_name}]")
        case "error":
            print(f"\nError: {event.message}")
```

### Multi-turn conversations

State persists across calls — the agent remembers previous data and code:

```python
await agent.ask("Fetch quarterly US GDP since 2010")
await agent.ask("Now calculate year-over-year growth rates")
result = await agent.ask("Plot the growth rates as a bar chart")
```

### Notebooks and artifacts

Agents organize code into notebooks (editable, re-executable cells) and produce typed artifacts:

- **Dataset** — a curated dataset with metadata, provenance, and version tracking
- **Chart** — an Altair/Vega-Lite visualization linked to its source dataset

### Built-in tools

| Tool | Description |
|------|-------------|
| `code_set` | Write a new code notebook |
| `code_edit` | Modify an existing notebook |
| `dry_execute_code` | Preview code output without committing |
| `return_dataset` | Finalize a dataset as a deliverable |
| `return_chart` | Finalize a chart as a deliverable |
| `output_read` | Read a specific execution output |
| `output_search` | Search across all outputs |
| `get_context` | Inspect current variables and notebooks |

## Architecture

```
ockham (connectors, catalog, Result model)
     |
ockham-agents (this package)
     |
     +-- Agent                  — LLM loop, tool orchestration
     +-- CodeExecutor           — in-process Python execution
     +-- Variable / VariableStore — execution state tracking
     +-- Notebooks              — editable, re-executable code cells
     +-- Artifacts              — typed deliverables (datasets, charts)
     +-- OutputFactory          — value -> typed output dispatch
     +-- RAG (optional)         — semantic + keyword search over outputs
```

## Power Usage

For full control, use `Agent` directly with explicit configuration:

```python
from ockham_agents import Agent
from ockham_agents.agent.config import AgentGuardrails
from ockham_agents.execution.executor import CodeExecutor
from ockham_agents.execution.factory import OutputFactory

agent = Agent(
    model_config={"model": "claude-sonnet-4-6", "api_key": "..."},
    instructions="You are a specialized economic research agent...",
    code_executor=CodeExecutor(cwd="/tmp/work", output_factory=OutputFactory(local_dir="/tmp/work")),
    output_factory=OutputFactory(local_dir="/tmp/work"),
    guardrails=AgentGuardrails(max_iterations=30, max_execution_time_s=120.0),
    connectors=my_connectors,
)
```

## Optional extras

```bash
pip install ockham-agents[rag]       # ChromaDB + Tantivy for semantic search
pip install ockham-agents[sql]       # DuckDB for SQL over DataFrames
pip install ockham-agents[display]   # Rich terminal output for streaming events
pip install ockham-agents[all]       # Everything
```

## Supported LLM Providers

`ockham-agents` uses [LiteLLM](https://docs.litellm.ai/docs/providers) for LLM access, which supports 100+ providers:

| Provider | Model example |
|----------|--------------|
| Anthropic | `claude-sonnet-4-6` |
| OpenAI | `gpt-4o` |
| Google | `gemini/gemini-2.0-flash` |
| Azure | `azure/gpt-4o` |
| Local (Ollama) | `ollama/llama3` |

Pass any LiteLLM-compatible model string to `Agent(model="...")`.

## Troubleshooting

**Missing LLM API key**: Set the appropriate environment variable for your provider (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) or pass `api_key=` to the Agent constructor.

**Code execution errors**: The agent executes Python code in-process. If you see import errors, ensure the required packages are installed in your environment (e.g., `pandas`, `numpy`).

**Timeout errors**: For long-running analyses, increase the guardrails: `Agent(guardrails=AgentGuardrails(max_execution_time_s=600.0))`.

**Streaming not printing**: Use `stream_to_display()` for formatted terminal output, or iterate `agent.run()` events manually.

## License

Apache 2.0
