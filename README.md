# parsimony-agents

[![PyPI version](https://img.shields.io/pypi/v/parsimony-agents)](https://pypi.org/project/parsimony-agents/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/parsimony-agents)](https://pypi.org/project/parsimony-agents/)
[![CI](https://github.com/ockham-sh/parsimony-agents/actions/workflows/test.yml/badge.svg)](https://github.com/ockham-sh/parsimony-agents/actions)

Build AI agents that discover, fetch, and analyze data.

### Why parsimony-agents?

LLM frameworks are generic by design. `parsimony-agents` is purpose-built for data analysis: agents write and execute Python code against typed data connectors, track provenance for every data fetch, and produce reproducible datasets and Altair visualizations. Works with any LLM provider supported by [LiteLLM](https://docs.litellm.ai/docs/providers) (OpenAI, Anthropic, Google, Azure, local models, and more).

## Quick Start

```python
from parsimony_agents import Agent
from parsimony import discover

connectors = discover.load_all()

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=connectors,
)

result = await agent.ask("Show me US GDP trends over the last 20 years")

print(result.text)        # Natural language response
print(result.datasets)    # {"us_gdp": <DataFrame>}
print(result.code)        # {"main": Script(...), ...} — named scripts keyed by notebook name
```

## Installation

```bash
pip install parsimony-agents
```

Requires [parsimony](../parsimony) (installed automatically as a dependency).

## Features

### Code execution with provenance

Agents write Python code that runs in a sandboxed executor. Every data fetch is tracked with full provenance — source, parameters, timestamps.

```python
# Agent writes this code automatically:
result = await client["fred_fetch"](series_id="GDPC1", observation_start="2005-01-01")
gdp = result.data  # pandas DataFrame with provenance attached
```

### Composable data sources

Plug in any combination of data sources via [parsimony](../parsimony) connectors:

```python
from parsimony import Connectors, discover
from parsimony_fred import CONNECTORS as FRED
from parsimony_sdmx import CONNECTORS as SDMX
from parsimony_fmp import CONNECTORS as FMP

# Either compose explicitly...
connectors = Connectors.merge(
    FRED.bind(api_key="..."),
    SDMX,
    FMP.bind(api_key="..."),
)

# ...or autodiscover everything installed and bind from env vars.
connectors = discover.load_all()

agent = Agent(
    model="claude-sonnet-4-6",
    connectors=connectors,
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
| `return_notebook` | Write notebook cells to disk |
| `edit_notebook` | Edit individual cells within an existing notebook |
| `dry_execute_code` | Preview code output without committing to state |
| `write_file` | Write a file to the working directory |
| `edit_file` | Apply a patch to an existing file |
| `read_file` | Read a file from the working directory |
| `read_data` | Fetch data from a bound connector |
| `list_files` | List files in the working directory |
| `restart_kernel` | Clear the executor namespace |
| `return_dataset` | Finalize a dataset as a deliverable |
| `return_chart` | Finalize a chart as a deliverable |
| `return_report` | Finalize a report document as a deliverable |
| `edit_report` | Edit an in-progress report |
| `refresh` | Re-fetch connector data |
| `output_read` | Read a previously returned artifact |
| `output_search` | Semantic search across outputs (requires `[rag]` extra) |

## Architecture

```
parsimony (connectors, catalog, Result model)
     |
parsimony-agents (this package)
     |
     +-- Agent                  — LLM loop, tool orchestration
     +-- CodeExecutor           — in-process Python execution; workspace files are the notebook source of truth
     +-- Notebooks              — editable, re-executable code cells
     +-- Artifacts              — typed deliverables (datasets, charts, reports)
     +-- OutputFactory          — value -> typed output dispatch
     +-- RAG (optional)         — semantic + keyword search over outputs
```

## Power Usage

For full control, use `Agent` directly with explicit configuration:

```python
from parsimony_agents import Agent
from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory

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
pip install parsimony-agents[rag]       # ChromaDB + Tantivy for semantic search
pip install parsimony-agents[sql]       # DuckDB for SQL over DataFrames
pip install parsimony-agents[display]   # Rich terminal output for streaming events
pip install parsimony-agents[all]       # Everything
```

## Supported LLM Providers

`parsimony-agents` uses [LiteLLM](https://docs.litellm.ai/docs/providers) for LLM access, which supports 100+ providers:

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

## Documentation

Comprehensive guides for developing, deploying, and operating parsimony-agents:

**[Start with Documentation Index →](docs/index.md)** — Choose your path by role (API developer, operations, architect, contributor)

| Guide | Purpose |
|-------|---------|
| [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) | System design, components, data flow, extension points |
| [**API.md**](docs/API.md) | Complete API reference for Agent, CodeExecutor, artifacts, and tools |
| [**RUNBOOK.md**](docs/RUNBOOK.md) | Deployment, monitoring, performance tuning, and troubleshooting |
| [**COMMANDS.md**](docs/COMMANDS.md) | Development commands: testing, linting, building, packaging |
| [**CODEMAPS.md**](docs/CODEMAPS.md) | Code structure, module organization, and public API exports |

## License

Apache 2.0
