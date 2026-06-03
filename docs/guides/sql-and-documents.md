# SQL and document inputs

The agent does its work by writing and running Python in a stateful kernel (see
[Code execution](../concepts/code-execution.md)). That kernel comes pre-loaded with the
usual data-analysis primitives — `pd`, `np`, `alt`, `datetime`, `load_dataset` — plus three
**document helpers** for pulling text and tables out of PDF, Excel, and PowerPoint files.

Two capabilities in this guide depend on optional install extras:

- The document helpers (`read_pdf_text`, `read_excel`, `read_pptx_text`) need the
  **`[documents]`** extra (`pypdf`, `openpyxl`, `python-pptx`).
- SQL over your data needs the **`[sql]`** extra, which pulls in **`duckdb`**.

Both are off by default so the base package stays lean. Install what you need (see
[Installation](../getting-started/installation.md)):

```bash
pip install "parsimony-agents[documents]"   # PDF / Excel / PPTX helpers
pip install "parsimony-agents[sql]"          # duckdb for SQL queries
pip install "parsimony-agents[documents,sql]"
pip install "parsimony-agents[all]"          # rag + sql + display + documents
```

## The documents extra: `read_pdf_text`, `read_excel`, `read_pptx_text`

The three helpers live in `parsimony_agents.execution.documents`. Their exact signatures:

```python
read_pdf_text(path: str, *, max_pages: int | None = None) -> str
read_excel(path: str, *, sheet_name: int | str = 0, **kwargs) -> pd.DataFrame
read_pptx_text(path: str) -> list[dict[str, Any]]
```

What each returns:

- **`read_pdf_text(path, max_pages=...)`** — extracts plain text from a PDF and joins it into
  one string (pages separated by blank lines). Pass `max_pages` to stop after the first *N*
  pages; omit it (the default `None`) to read the whole document. Backed by `pypdf`.
- **`read_excel(path, sheet_name=0, **kwargs)`** — reads an `.xlsx` workbook into a
  `pandas.DataFrame` using the `openpyxl` engine. `sheet_name` accepts a sheet index (`0` for
  the first sheet) or a sheet name; any extra keyword arguments pass straight through to
  `pandas.read_excel` (e.g. `header`, `usecols`, `skiprows`).
- **`read_pptx_text(path)`** — returns a list of per-slide dicts. Each dict has `index`
  (0-based slide number) and `text` (the slide's shape text, joined with newlines). Backed by
  `python-pptx`.

Because these are plain Python functions, you can also call them directly in your own host
code if you want to pre-process a file before handing the result to the agent:

```python
from parsimony_agents.execution.documents import (
    read_pdf_text,
    read_excel,
    read_pptx_text,
)

# First page only
summary = read_pdf_text("reports/q1-summary.pdf", max_pages=1)

# Second sheet of a workbook, skip a title row
df = read_excel("data/figures.xlsx", sheet_name=1, skiprows=1)

# Per-slide text from a deck
slides = read_pptx_text("decks/board-update.pptx")
for slide in slides:
    print(slide["index"], slide["text"][:80])
```

The helpers import their third-party dependency **at call time**, not at module import. That
is deliberate: `parsimony-agents` installs and runs fine without the `[documents]` extra, and
you only hit the dependency the moment a helper is actually invoked. If the extra is missing,
the call raises a `RuntimeError` telling you to install it:

```
PDF support requires the optional documents stack: install parsimony-agents
with the ``documents`` extra (included in ``[all]``).
```

The same pattern applies to `read_excel` (needs `openpyxl`) and `read_pptx_text` (needs
`python-pptx`).

## How document helpers appear in the kernel

You rarely call these functions yourself in an agent workflow — the executor **injects them
into the kernel's `locals`** so the agent's code can call them by bare name. The same names
are restored whenever the kernel resets: `clear_namespace()` re-seeds the base namespace with
`pd`, `np`, `alt`, `datetime`, `read_pdf_text`, `read_excel`, `read_pptx_text`, and
`load_dataset`.

The practical consequence: agent-written code uses the helpers directly, with **no import
statement**. In fact, importing framework helpers is flagged by the `FrameworkImportLinter`
(see below) — they are pre-injected, so `from parsimony_agents... import` is both wrong and
unnecessary. A cell the agent might write looks like this:

```python
# Agent-authored cell — note: no imports
text = read_pdf_text("filings/10-k.pdf", max_pages=3)
tables = read_excel("filings/financials.xlsx", sheet_name="Income")
display(tables.head())
```

When you give the agent a question that references a file you've put in front of it, the agent
chooses the right helper, runs it in the kernel, and the result (text or a `DataFrame`) becomes
part of the conversation it reasons over. DataFrames returned by `read_excel` flow through the
same typed-output path as any other kernel DataFrame, so they are previewed and persisted like
fetched data.

## The sql extra (duckdb) for querying data

The `[sql]` extra installs **DuckDB**, an in-process SQL engine. DuckDB is the canonical way to
run SQL over the DataFrames already living in the kernel (and over Parquet/CSV files on disk)
without standing up a database server. Once the extra is installed, the agent can write a cell
that queries data with SQL:

```python
# Agent-authored cell using duckdb over an in-kernel DataFrame
import duckdb

result = duckdb.query("""
    SELECT region, SUM(revenue) AS total
    FROM sales
    GROUP BY region
    ORDER BY total DESC
""").to_df()
display(result)
```

DuckDB reads pandas DataFrames in the local namespace directly (here `sales`), and `.to_df()`
hands you a pandas `DataFrame` back — which then flows through the kernel's typed-output and
lineage machinery exactly like any other table.

A note on how SQL results are handled downstream: the framework prefers **typed dataset I/O**
over raw Parquet round-trips. The `RawParquetIOLinter` softly discourages
`df.to_parquet(...)` and `pd.read_parquet(...)` in agent code, steering toward `return_dataset()`
for writes (which embeds curation metadata) and the typed read path for reads. So SQL is the
right tool for *querying*, while persisting a result as an artifact goes through the dataset
return tools rather than a bare Parquet dump. See
[Artifacts, identity & lineage](../concepts/artifacts.md).

## Data-quality helpers (`inspect_object`, `series_na_report`, `check_code`)

Reading from files and querying with SQL both invite data-quality problems — stray nulls,
unexpected gaps, index surprises. `parsimony_agents.quality` exposes three helpers the agent
uses to catch these early. All three are pure functions you can also call yourself.

```python
from parsimony_agents.quality import (
    check_code,
    inspect_object,
    series_na_report,
)
```

### `inspect_object(obj)` — quick NA report for a DataFrame or Series

```python
inspect_object(obj: pd.DataFrame | pd.Series) -> str | None
```

Generates a data-quality inspection report. For a `DataFrame`, it reports NA (null) stats per
column; for a `Series`, it delegates to `series_na_report`. Returns a formatted multi-line
string (or `None`).

```python
import pandas as pd
from parsimony_agents.quality import inspect_object

df = pd.DataFrame({
    "A": [1, 2, None, None, None, 6, 7],
    "B": [None] * 7,  # 100% missing
})
print(inspect_object(df))
```

### `series_na_report(series, ...)` — null-run detection on a single Series

```python
series_na_report(series: pd.Series, top_k: int = 5, min_run: int = 2) -> str
```

Produces a compact NA report for one Series: the global NA ratio plus the top-`k` largest
*consecutive* NA runs. Runs that are both highly concentrated and large are flagged as
`WARNING`. `min_run` is the smallest consecutive-NA cluster size worth reporting — raise it to
ignore isolated or paired nulls and focus on real gaps.

```python
from parsimony_agents.quality import series_na_report

s = pd.Series([1, None, None, 3, 4, None] * 5)
print(series_na_report(s, top_k=3, min_run=3))
```

### `check_code(code, ...)` — advisory linting of analysis code

```python
check_code(code: str, type_map: dict[str, type] | None = None) -> list[str]
```

Runs the four AST-based linters over a string of Python and returns a list of **advisory**
issues (it never raises, and it never blocks execution):

- **`IndexPolicyLinter`** — flags reads of `.index` and index-producing operations (`groupby`,
  `pivot`, `merge`, `join`, `set_index`, `sort_index`, `reindex`, `stack`, `unstack`,
  `resample`) that aren't cleared with a following `.reset_index()`.
- **`RollingLinter`** — flags `.rolling(...)` calls with no explicit `min_periods`, which can
  silently introduce NAs.
- **`RawParquetIOLinter`** — discourages raw `df.to_parquet(...)` / `pd.read_parquet(...)` in
  favor of typed dataset I/O.
- **`FrameworkImportLinter`** — rejects `import parsimony_agents...` / `from parsimony_agents...`,
  since framework helpers are pre-injected.

```python
from parsimony_agents.quality import check_code

code = """
df = df.groupby('category').sum()   # missing reset_index!
df.to_parquet('output.parquet')     # should use return_dataset
"""

for issue in check_code(code):
    print(issue)
```

Pass a `type_map` (variable name → type) to scope the pandas-specific checks precisely and
avoid false positives on non-pandas objects:

```python
type_map = {"df": pd.DataFrame}
issues = check_code(code, type_map=type_map)
```

These are **soft** lints by design: the agent sees the issues alongside its notebook output and
can self-correct on the next turn, rather than being hard-stopped. See
[Failure handling & recovery](../concepts/failure-and-recovery.md) for how the loop responds to
harder problems.

## Putting files in front of the agent (`file_store` / `files_dir`)

For the agent to read a PDF or workbook, the file has to be reachable from the kernel's working
directory. That mapping is the job of the **`FileStore`** protocol, passed to the `Agent` as
`file_store=`:

```python
@runtime_checkable
class FileStore(Protocol):
    async def list_files(self) -> list[str]: ...
    def get_files_dir(self) -> Path: ...
```

It is a session-scoped, two-method contract, importable from `parsimony_agents.agent.config`:

- **`list_files()`** — an async method returning the file names visible to this session. The
  agent uses this to discover what's available before reaching for a document helper.
- **`get_files_dir()`** — returns the `Path` of the directory those files live in: the
  workspace the kernel's relative paths (`"reports/q1.pdf"`) resolve against.

Any object satisfying that protocol works (it's `@runtime_checkable`). A minimal local
implementation that hands the agent a directory of files:

```python
import asyncio
from pathlib import Path

from parsimony_agents import Agent, stream_to_display


class LocalFileStore:
    def __init__(self, root: str) -> None:
        self._root = Path(root)

    async def list_files(self) -> list[str]:
        return [p.name for p in self._root.iterdir() if p.is_file()]

    def get_files_dir(self) -> Path:
        return self._root


async def main() -> None:
    agent = Agent(
        model="claude-sonnet-4-6",
        file_store=LocalFileStore("/path/to/my/files"),
    )

    await stream_to_display(
        agent,
        "Read q1-summary.pdf and pull the headline revenue figures into a table.",
    )


if __name__ == "__main__":
    asyncio.run(main())
```

With `file_store` wired up, the agent can call `list_files` to see what's there, then run
`read_pdf_text("q1-summary.pdf")` (or `read_excel(...)` / `read_pptx_text(...)`) against the
directory `get_files_dir()` points at — all inside the kernel, with no imports.

`stream_to_display` lives in `parsimony_agents` and renders the run live. For the lower-level
event-by-event API, see [Streaming and displaying results](streaming-and-displaying-results.md);
for the full constructor surface, see the [Agent reference](../reference/agent.md).

## Related pages

- [Code execution](../concepts/code-execution.md) — the kernel, typed outputs, and pre-injected globals
- [Installation](../getting-started/installation.md) — the `documents`, `sql`, and `all` extras
- [Artifacts, identity & lineage](../concepts/artifacts.md) — typed dataset I/O vs. raw Parquet
- [Saving and loading artifacts](saving-loading-artifacts.md)
- [Execution reference](../reference/execution.md) and [Agent tools](../reference/agent-tools.md)
