

**A code-writing agent for data analysis that returns datasets, charts, and reports — not just answers.**

[PyPI](https://pypi.org/project/parsimony-agents/)
[License: Apache-2.0](LICENSE)
[Python](https://pypi.org/project/parsimony-agents/)





---

Parsimony Agents finds data through [Parsimony connectors](https://github.com/ockham-sh/parsimony-connectors), analyzes it by writing and executing Python, and publishes the work as reusable artifacts. The producing notebooks and source lineage are persisted with each dataset, chart, and report so the result can be inspected, reused, or refreshed.

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

`Agent.ask()` returns an `AgentResult` containing the narrative, published datasets, charts and reports, the event log, and the context required for a follow-up turn.

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

Hosts can pass `create_executor(cwd=...)` as `code_executor=` to select bubblewrap isolation on supported Linux systems. The factory falls back in-process with a warning when that boundary is unavailable. Check `executor.capability_tier` before running untrusted code.

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