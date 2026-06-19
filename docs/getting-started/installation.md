# Installation

`parsimony-agents` is an AI agent framework for data analysis: it writes and
executes Python code to answer questions about data. This page gets the package
installed with the right optional extras for your use case and explains what each
extra unlocks.

## Requirements (Python 3.11 / 3.12)

The package declares `requires-python = ">=3.11,<3.13"`, so you need **Python 3.11
or 3.12**. Python 3.10 and below, and Python 3.13 and above, are not supported.

Check your interpreter:

```bash
python --version
# Python 3.11.x  or  3.12.x
```

A core dependency, `parsimony-core` (pinned `>=0.7,<0.8`), is installed
automatically — it provides the connector model (`parsimony.connector`) and plugin
discovery (`parsimony.discover`) used throughout the framework.

## pip install parsimony-agents

The base install gives you the full agent loop, code execution (with optional
out-of-process sandboxing when deployed), charts (Altair / Vega-Lite), and the streaming event API:

```bash
pip install parsimony-agents
```

This pulls in the runtime dependencies declared in `pyproject.toml` — `pandas`,
`numpy`, `altair`, `litellm` (the LLM gateway), and `parsimony-core`, among others.

The public API is imported from the top-level `parsimony_agents` package:

```python
from parsimony_agents import Agent, AgentResult, stream_to_display
from parsimony_agents import Dataset, Chart, Script
```

> You also need an LLM provider key (for example `ANTHROPIC_API_KEY`) for the
> `Agent` to call a model. That is environment configuration, not an install step —
> see [Configuration](configuration.md).

## Optional extras: sql, display, documents, all

Several capabilities are gated behind optional extras so the base install stays
lean. Install them with the standard `pip install "parsimony-agents[extra]"`
syntax (quote the brackets in most shells):

| Extra | Pulls in | Unlocks |
|---|---|---|
| `sql` | `duckdb` | In-kernel SQL over your data |
| `display` | `rich` | Polished terminal rendering via `stream_to_display` / `display_result` |
| `documents` | `pypdf`, `openpyxl`, `python-pptx` | Reading PDF, Excel, and PowerPoint files |
| `all` | `sql` + `display` + `documents` | Everything above in one install |

To search a large output, a result is a kernel variable, so an agent searches a
DataFrame in code with the core catalog
(`auto_catalog(df).search(...)`, BM25 — `parsimony-core`'s `catalog` extra,
which the agent runtime already ships).

Examples:

```bash
# Just the rich terminal UI
pip install "parsimony-agents[display]"

# Document parsing
pip install "parsimony-agents[documents]"

# Everything (sql + display + documents)
pip install "parsimony-agents[all]"
```

### `display` — required for rich terminal output

`stream_to_display` (and `display_result`) render a live spinner, numbered tool
progress, streamed text, dataset tables, and syntax-highlighted code. That polished
rendering is powered by `rich`, which only ships with the **`display`** extra. The
canonical quickstart uses it:

```python
import asyncio
import os

from parsimony_fred import CONNECTORS as FRED

from parsimony_agents import Agent, stream_to_display


async def main() -> None:
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Set FRED_API_KEY environment variable to run this example.")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    agent = Agent(
        model="claude-sonnet-4-6",
        connectors=FRED.bind(api_key=fred_key),
    )

    # Ask a question — full display with spinner, datasets, code
    result = await stream_to_display(
        agent,
        "What is the current US unemployment rate? Fetch the data and show me.",
    )

    # Follow-up (multi-turn), reusing context
    await stream_to_display(
        agent,
        "Now show me how unemployment has changed since 2020",
        ctx=result.context,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

If you do not install `display`, prefer the non-streaming `Agent.ask` or the raw
`Agent.run` event generator — both work without `rich`. See
[Streaming and displaying results](../guides/streaming-and-displaying-results.md).

### `documents` — required for PDF / Excel / PowerPoint inputs

The document readers the agent's kernel exposes — `read_pdf_text`, `read_excel`,
and `read_pptx_text` — import their backing libraries **at call time**. With the
base install those calls raise a `RuntimeError` telling you to install the
`documents` extra:

- `read_pdf_text(path, max_pages=...)` needs `pypdf`
- `read_excel(path, sheet_name=0, **kwargs)` needs `openpyxl`
- `read_pptx_text(path)` needs `python-pptx`

Install the extra to enable them:

```bash
pip install "parsimony-agents[documents]"
```

See [SQL and document inputs](../guides/sql-and-documents.md) for usage.

### `sql` — DuckDB

The `sql` extra adds `duckdb` for running SQL inside the execution kernel. See
[SQL and document inputs](../guides/sql-and-documents.md).

## Installing connectors (parsimony-fred, parsimony-sdmx, parsimony-fmp)

`parsimony-agents` ships **no data connectors of its own**. Connectors are separate
packages that register against the `parsimony.providers` entry-point group; each one
exports a `CONNECTORS` object (a `parsimony.connector.Connectors` collection).

Install the providers you need directly:

```bash
pip install parsimony-fred parsimony-sdmx parsimony-fmp
```

Each provider exposes a `CONNECTORS` constant you bind your API key onto:

```python
from parsimony_fred import CONNECTORS as FRED
from parsimony_sdmx import CONNECTORS as SDMX
from parsimony_fmp import CONNECTORS as FMP

# Fix the api_key parameter across every connector that accepts it
fred = FRED.bind(api_key="your-fred-key")
fmp = FMP.bind(api_key="your-fmp-key")

# Compose collections with the + operator
combined = fred + fmp
```

Alternatively, discover every installed provider at once with
`parsimony.discover` — no explicit imports needed:

```python
from parsimony import discover

connectors = discover.load_all()   # forgiving: logs and skips load failures
# or, strict — raises if a name is not installed:
connectors = discover.load("fred", "fmp")
```

> **Tip:** the framework's `[examples]` extra
> (`pip install "parsimony-agents[examples]"`) bundles `parsimony-fred`,
> `parsimony-sdmx`, `parsimony-fmp`, and `python-dotenv` together so the runnable
> `examples/` scripts work out of the box.

For the full connector model — binding, composition, and discovery — see
[Connectors](../concepts/connectors.md).

## Verifying the install

Confirm the interpreter and core import resolve:

```bash
python -c "import parsimony_agents; print('parsimony-agents OK')"
```

Check the top-level API symbols are importable:

```python
from parsimony_agents import Agent, AgentResult, stream_to_display, Dataset, Chart, Script
print("imports OK")
```

Verify the **`display`** extra is present (only succeeds when `rich` is installed):

```bash
python -c "import rich; print('display extra OK')"
```

Verify the **`documents`** extra by exercising a reader's import guard — with the
extra installed this prints nothing and exits cleanly; without it you get a clear
`RuntimeError` pointing you back here:

```bash
python -c "import pypdf, openpyxl, pptx; print('documents extra OK')"
```

List the connectors visible to plugin discovery:

```python
from parsimony import discover

for p in discover.iter_providers():
    print(p.name, p.version)
```

If `iter_providers()` prints `fred`, `sdmx`, and/or `fmp`, those connector packages
are installed and ready to bind.

## Next steps

- [Quickstart](quickstart.md) — your first agent run end to end.
- [Configuration](configuration.md) — LLM provider keys and per-connector keys.
- [Connectors](../concepts/connectors.md) — binding, composing, and discovering data sources.
- [Streaming and displaying results](../guides/streaming-and-displaying-results.md) — `Agent.run`, `Agent.ask`, and `stream_to_display`.
