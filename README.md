<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/parsimony-agents-brand-dark.png" />
  <img src="docs/assets/parsimony-agents-brand-light.png" alt="parsimony-agents" width="460" />
</picture>

**Extensible agent for data discovery, analysis, and visualization**

[![PyPI](https://img.shields.io/pypi/v/parsimony-agents.svg)](https://pypi.org/project/parsimony-agents/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/pypi/pyversions/parsimony-agents.svg)](https://pypi.org/project/parsimony-agents/)

</div>


<p align="center">
  <img src="docs/assets/parsimony-agents-hero.gif" alt="parsimony-agents: an Agent bound to FRED answers &quot;How has US unemployment changed since 2020?&quot; by fetching the UNRATE series, running code, and publishing a Dataset and a Chart artifact, ending in result.ok = True." width="900" />
</p>

---

Parsimony Agents finds data through [Parsimony connectors](https://github.com/ockham-sh/parsimony-connectors), analyzes it by writing and executing Python, and returns the work as reusable artifacts. The producing notebooks and source lineage are persisted with each dataset, chart, and report so the result can be inspected, reused, or refreshed.

## Quickstart

```bash
pip install parsimony-agents parsimony-fred
export ANTHROPIC_API_KEY="..."
export FRED_API_KEY="..."
```

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED
from parsimony_agents import Agent


async def main() -> None:
    agent = Agent(
        model="anthropic/claude-sonnet-5",
        connectors=FRED.bind(api_key=os.environ["FRED_API_KEY"]),
    )
    result = await agent.ask("Show me how US unemployment has changed since 2020")

    print(result.text)
    print(list(result.datasets.keys()))  # published dataset logical IDs
    print(list(result.charts.keys()))    # published chart logical IDs
    assert result.ok


asyncio.run(main())
```

`Agent.ask()` returns an `AgentResult` containing the datasets, charts and reports, the event log, and the context required for a follow-up turn.

## Follow-up turns

Pass the previous result's `context` back in to continue the conversation — the agent keeps the full transcript, so it can resolve references like "that chart" or "the same period":

```python
follow_up = await agent.ask("Now do the same for the Eurozone.", ctx=result.context)
print(follow_up.text)
```

See [Multi-turn conversations](docs/guides/multi-turn.md) for streaming, suspension, and resume.

## How it works

The agent works in the same sequence as an analyst:

1. It searches and fetches data through a consistent connector interface.
2. It writes Python in a stateful execution environment and observes the result.
3. It publishes useful outputs rather than leaving them inside the conversation.

Published artifacts use open formats:

- datasets are Parquet;
- charts are Vega-Lite;
- reports are Quarto Markdown;
- notebooks are plain Python.

Each artifact has a stable identity, a content-addressed snapshot, and links to the code and data that produced it. Refresh walks that lineage and appends a snapshot only when the resulting bytes change.

For different integration levels:

- `Agent.ask()` collects a run into one structured result;
- `Agent.run()` streams typed events for applications and custom interfaces;
- `Agent.resume()` continues a run that paused to ask the user a question.



## Adding data sources

Connector packages expose immutable `CONNECTORS` collections. Bind credentials, compose the sources you need, and pass the collection to the agent:

```python
from parsimony_fred import CONNECTORS as FRED
from parsimony_sdmx import CONNECTORS as SDMX
from parsimony_agents import Agent

connectors = FRED.bind(api_key="...") + SDMX
agent = Agent(model="anthropic/claude-sonnet-5", connectors=connectors)
```

See [parsimony-connectors](https://github.com/ockham-sh/parsimony-connectors) for available financial and economic data sources, or use `parsimony-core` to define connectors for internal data.

## Code execution

A standalone `Agent` runs Python in-process by default. This is convenient for local scripts, but it is not a security boundary.

Hosts can pass `create_executor(cwd=...)` as `code_executor=` to select bubblewrap isolation on supported Linux systems. The factory falls back to in-process execution with a warning when that boundary is unavailable. Check `executor.capability_tier` before running untrusted code.

See [Code execution](docs/concepts/code-execution.md) and the [Security policy](SECURITY.md).

## Documentation

- [Quickstart](docs/getting-started/quickstart.md)
- [Connectors](docs/concepts/connectors.md)
- [Artifacts and lineage](docs/concepts/artifacts.md)
- [Streaming results](docs/guides/streaming-and-displaying-results.md)
- [Embedding in a host](docs/guides/embedding-in-a-host.md)
- [Public API](docs/reference/public-api.md)

The complete documentation is published at [docs.parsimony.dev](https://docs.parsimony.dev).

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run mypy parsimony_agents/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).