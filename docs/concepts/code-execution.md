# Code execution

When an agent answers a data question, it does not hand you a paragraph — it
writes Python, runs it, and shows you the result. This page explains the engine
that runs that code: a persistent in-process kernel that injects analysis
libraries and connectors, captures every result as a *typed* output, sandboxes
the code against secret-leaking system calls, enforces per-cell timeouts, and
records who produced each variable so artifacts stay traceable.

If you only embed the [`Agent`](../reference/agent.md), you rarely touch the
executor directly — it is created and driven for you. But understanding it tells
you exactly what the agent's code can and cannot do, what comes back from each
cell, and how to extend the kernel (custom output types, remote sandboxes,
storage backends) when you host Parsimony Agents in your own application.

The executor lives in `parsimony_agents.execution`. The in-process default is
`CodeExecutor`; the abstract contract every executor satisfies is
`BaseCodeExecutor`.

```python
from pathlib import Path

from parsimony_agents.execution import CodeExecutor, OutputFactory

workspace = Path("/tmp/my_workspace")
executor = CodeExecutor(
    cwd=str(workspace),
    output_factory=OutputFactory(local_dir=workspace),
)
```

`CodeExecutor.execute` and `eval` are coroutines, so the host always drives them
inside an event loop:

```python
import asyncio


async def main() -> None:
    executor = CodeExecutor(
        cwd="/tmp/my_workspace",
        output_factory=OutputFactory(local_dir="/tmp/my_workspace"),
    )
    result = await executor.execute("df = pd.DataFrame({'a': [1, 2, 3]})\ndf")
    print(result.outputs)


if __name__ == "__main__":
    asyncio.run(main())
```

For the full executor surface, see the [execution reference](../reference/execution.md).
For how outputs travel through the agent loop, see
[How it works](how-it-works.md) and [Events](events.md).

## The kernel namespace (`pd`, `np`, `alt`, `datetime`, connectors, `load_dataset`)

A single `CodeExecutor` owns one **kernel** — a persistent Python namespace
(`locals` dict) that survives across `execute()` calls. Variables you define in
one cell are visible in the next, exactly like cells in a Jupyter notebook
sharing one runtime.

```python
await executor.execute("import math\nradius = 3")
result = await executor.execute("area = math.pi * radius**2\narea")
# `radius` and `math` are still bound; the kernel is stateful.
```

Every kernel boots with a fixed base namespace so agent code never has to import
the common analysis stack:

| Name | Bound to |
|---|---|
| `pd` | `pandas` |
| `np` | `numpy` |
| `alt` | `altair` |
| `datetime`, `timedelta`, `timezone` | the corresponding classes from the `datetime` module |
| `load_dataset` | the dataset-loading primitive (see below) |
| `read_pdf_text`, `read_excel`, `read_pptx_text` | document readers (see below) |
| each bound connector | injected via `set_connectors` |

Connectors are injected separately with `set_connectors`, which accepts a single
`Connectors` bundle or a `Mapping[str, Connectors]` for named bindings:

```python
from parsimony_fred import CONNECTORS as FRED

await executor.set_connectors(FRED.bind(api_key="..."))
# or, for an explicit binding name:
await executor.set_connectors({"fred": FRED.bind(api_key="...")})

result = await executor.execute('gdp = fred["GDPC1"](series_id="GDPC1")')
```

Under the hood each bundle is wrapped in a `MemoizingConnectorBundle`: identical
calls (same connector, same canonical args) are served from a per-kernel
`ConnectorCache` instead of re-fetching. The cache is a kernel-lifetime concern —
`set_cwd()` and `clear_namespace()` clear it. See [Connectors](connectors.md)
for the connector model itself.

`clear_namespace()` resets the kernel to that base state (dropping any
agent-defined names), re-applies connectors, and re-runs any registered setup
snippets:

```python
await executor.execute("def my_helper(x): return x * 2")
assert "my_helper" in executor.locals

await executor.clear_namespace()
assert "my_helper" not in executor.locals
assert "pd" in executor.locals  # base namespace restored
```

## Typed kernel outputs (`DataFrameObject`, `FigureObject`, `PrimitiveObject`, `ExceptionObject`)

The executor never returns raw Python objects to the agent loop. Every value a
cell *displays* — the trailing expression, a `print()`, an explicit
`display()` — is converted into a **typed kernel output** by the
`OutputFactory`. There are four built-in types:

| Type | Wraps | Notes |
|---|---|---|
| `DataFrameObject` | a pandas DataFrame | parquet-backed via a `DataframeRef`, with head/tail previews and column dtypes |
| `FigureObject` | an Altair chart or Vega-Lite spec dict | serializes to a spec, computes a base64 PNG for the LLM |
| `PrimitiveObject` | a scalar (`str`, `int`, `float`, `bool`, `None`) | also the fallthrough for anything else, via `str(value)` |
| `ExceptionObject` | an exception | renders a redacted traceback; typed `parsimony.errors` surface their message without a traceback |

`OutputFactory.from_value` performs the dispatch — custom handlers first, then a
built-in `isinstance` chain (DataFrame → `DataFrameObject`, Altair →
`FigureObject`, scalar → `PrimitiveObject`, `Exception` → `ExceptionObject`, else
`str(value)` → `PrimitiveObject`):

```python
from parsimony_agents.execution import (
    DataFrameObject,
    FigureObject,
    PrimitiveObject,
    ExceptionObject,
)

output = await executor.get("df")  # wraps the kernel variable `df`
if isinstance(output, DataFrameObject):
    llm_blocks = output.to_llm(mode="default")  # paginated CSV text blocks
```

Each typed output knows how to render itself for the model via `to_llm()`:
DataFrames and long strings paginate (see `TablePaginator` / `StringPaginator`),
and figures emit a `data:image/png;base64` URI. These same objects are the
*payloads* of published artifacts — a `Dataset` carries a `DataFrameObject`, a
`Chart` carries a `FigureObject`. See [Artifacts](artifacts.md) for the
curation layer on top.

You can teach the factory new types. `OutputFactory.register` is a class method;
handlers are checked before the built-in chain and receive `(value, *,
local_dir, backend)`:

```python
import polars as pl

from parsimony_agents.execution import DataFrameObject, DataframeRef, OutputFactory


def handle_polars(val, *, local_dir, backend):
    return DataFrameObject(
        ref=DataframeRef.from_pandas(
            val.to_pandas(),
            ref="polars_result",
            local_dir=local_dir,
            backend=backend,
        )
    )


OutputFactory.register(pl.DataFrame, handle_polars)
```

## `KernelOutput` and the fetch log

A single `execute()` or `eval()` call returns one `KernelOutput`, not a bare
list. It bundles two things:

- `outputs: list[KernelOutputType]` — the typed outputs produced by the cell.
- `fetch_log: list[FetchLogEntry]` — a record of every connector fetch observed
  during the run.

```python
result = await executor.execute('gdp = fred["GDPC1"](series_id="GDPC1")\ngdp')

for output in result.outputs:
    ...  # DataFrameObject / FigureObject / PrimitiveObject / ExceptionObject

for entry in result.fetch_log:
    print(entry.provenance, entry.row_count, entry.column_names)
    print(entry.data_object_ref)  # ArtifactRef to the persisted parquet snapshot
```

Each `FetchLogEntry` carries the fetch `provenance`, `row_count`,
`column_names`, head/tail samples, and an optional `data_object_ref`
(`ArtifactRef`) pointing at the immutable parquet snapshot persisted into the
content-addressed data-object pool. The fetch log is how the agent — and you —
know what raw data a cell pulled, independent of what it assigned. Post-fetch
hooks (the data-object persister and the fetch logger) run on *every* connector
call, cached or not, so the log stays truthful even when `MemoizingConnectorBundle`
serves a result from cache.

`KernelOutput.to_llm()` renders the whole bundle into the list of
`{type, text|image_url}` blocks the model consumes, paginating large tables and
strings per the view configuration.

## Safe builtins + AST sanitizer (blocked env/subprocess access)

Agent-written code runs in-process, so the kernel restricts what it can reach.
Two layers do this.

**Restricted `__builtins__`.** The exec namespace is given a curated
`_SAFE_BUILTINS` dict rather than the full builtin set. It keeps the
data-analysis primitives (types, introspection, `itertools`, I/O, the exception
classes) and re-binds `__import__` so ordinary `import` statements still work,
but it omits the obvious foot-guns.

**An AST sanitizer.** Before every `compile()`, `assert_safe_code` walks the
parse tree and refuses code that tries to read secrets or spawn processes. It
blocks, at compile time:

- `os.environ` in any access shape — `os.environ["X"]`, `os.environ.get(...)`,
  `os.environ.copy()`, etc.
- `os.getenv(...)`.
- `subprocess.*` — any attribute access under the `subprocess` module.
- String literals that reference `/proc/<pid>/environ` (the literal-pattern
  case).

```python
result = await executor.execute('key = os.getenv("FRED_API_KEY")')
assert isinstance(result.outputs[0], ExceptionObject)
# message: "os.getenv is blocked in agent code (secrets are not in scope)"
```

The point is that connector credentials and other server secrets live in the
host process's real environment; agent code has no business reading them, so the
sanitizer makes the attempt fail loudly instead of silently exfiltrating a key
through a returned value. Note that `import subprocess` itself is permitted (it
is just a name) — what fails is every attribute access on it.

There is one escape hatch for local debugging: set the environment variable
`OCKHAM_DISABLE_SANITIZE` to `1`, `true`, or `yes` and `assert_safe_code`
becomes a no-op. Leave it unset in any deployment that runs untrusted
agent-authored code.

## Timeouts and the daemon-thread execution model

Every cell runs under a wall-clock timeout. The default is
`DEFAULT_CELL_TIMEOUT_S`, read once at import from the `EXECUTOR_CELL_TIMEOUT_S`
environment variable and defaulting to **300 seconds**. You can override it
per-call with `timeout_seconds`:

```python
result = await executor.execute("while True: pass", timeout_seconds=1)

assert isinstance(result.outputs[0], ExceptionObject)
assert "timeout" in result.outputs[0].value.lower()

# The executor is not wedged — it remains usable afterwards.
followup = await executor.execute("x = 42")
```

Mechanically, the synchronous cell body runs in a dedicated **daemon thread**
(`_run_sync_in_thread_with_timeout`) while the event loop waits. On timeout the
executor injects `SystemExit` into that thread via
`PyThreadState_SetAsyncExc`. This is **best-effort**: the exception is only
delivered at the next Python bytecode boundary, so a cell blocked inside a C
extension (a long NumPy/pandas call, a blocking native I/O call) **cannot be
interrupted**. In that case the timed-out daemon thread is abandoned and reaped
when the process exits; the event loop and the executor itself stay responsive
and ready for the next cell. A process-global lock serializes overlapping
executions because the executor changes the process working directory.

## Producer attribution: `OriginLedger`, `RunScope`, `VariableOrigin`

The kernel records *who produced each variable* and *what data it depended on*,
so that when the agent publishes an artifact, the lineage is already known. This
is opt-in per call: pass `producer_notebook_path` to `execute()` and the
executor opens a **`RunScope`** around the run.

```python
result = await executor.execute(
    'df = pd.DataFrame({"a": [1, 2, 3]})',
    producer_notebook_path="notebooks/analysis.py",
)

origin = await executor.get_origin("df")
print(origin.notebook_path)  # "notebooks/analysis.py"
print(origin.load_refs, origin.fetch_refs)
```

The pieces:

- **`OriginLedger`** — an in-memory map of variable name → `VariableOrigin`, one
  per kernel lifetime. Cleared on `set_cwd()` / `clear_namespace()`.
- **`RunScope`** — a *mutable* per-run accumulator opened by `scope()`. As the
  cell runs it collects the load/fetch refs the code touched. Scopes do not
  nest: opening one inside another raises `RuntimeError`.
- **`VariableOrigin`** — the *frozen* result. At scope exit the executor diffs
  the kernel namespace before and after, then stamps every newly assigned name
  with a `VariableOrigin` recording the producing notebook plus the `load_refs`
  and `fetch_refs` it depended on. `VariableOrigin` is a `@dataclass(frozen=True)`
  — once stamped, lineage is immutable.

```python
from dataclasses import dataclass

# Conceptual shape (frozen):
# @dataclass(frozen=True)
# class VariableOrigin:
#     notebook_path: str
#     load_refs: tuple[ArtifactRef, ...] = ()
#     fetch_refs: tuple[ArtifactRef, ...] = ()
```

Omit `producer_notebook_path` (and `dry_run=True` runs) and **no scope opens** —
the run leaves no lineage, which is the right behaviour for scratch evaluation.
For how these origins feed the dual-identity artifact model, see
[Artifacts](artifacts.md).

## `load_dataset` and cross-terminal gating

`load_dataset` is the injected primitive agent code uses to pull a previously
published dataset back into the kernel. It takes a single positional **live
name** (the dataset's `live_name` slug), resolves it from the workspace's
`.ockham/datasets/*/curation.json` plus `log.jsonl`, records the load on the open
`RunScope` (if any), and returns the materialized DataFrame:

```python
result = await executor.execute(
    'inflation = load_dataset("inflation_monthly")',
    producer_notebook_path="notebooks/fed_analysis.py",
)
```

If the slug does not match a published dataset, or is ambiguous, the call raises
`LoadDatasetError` (a `KeyError` subclass) with guidance.

`load_dataset` also enforces a **cross-terminal gate**. A workspace can host
several terminals, each with its own set of live names it has actually seen. When
the executor is given `seen_live_names` (a `set[tuple[str, str]]` of `(kind,
live_name)` pairs) and `("dataset", slug)` is *not* in it, the underlying
`resolve_dataset_slug` raises `LiveNameCollisionError` — the dataset belongs to a
*sibling* terminal, and the agent must `read_artifact` it before loading. (The
`load_dataset` wrapper surfaces that collision to agent code as a
`LoadDatasetError` carrying the collision message.)

```python
result = await executor.execute(
    'inflation = load_dataset("inflation_monthly")',
    producer_notebook_path="notebooks/fed_analysis.py",
    seen_live_names={
        ("dataset", "inflation_monthly"),
        ("notebook", "fed_analysis.py"),
    },
)
# Drop ("dataset", "inflation_monthly") from the set and the load is rejected:
# the slug belongs to a sibling terminal.
```

This keeps one terminal from silently reaching into another terminal's
namespace by slug collision. `LiveNameCollisionError` is importable from
`parsimony_agents.identity`.

## Document helpers (`read_pdf_text`, `read_excel`, `read_pptx_text`)

Three readers are injected into the kernel so agent code can pull text and tables
out of office documents without importing third-party packages itself:

```python
# Inside agent code (all three are pre-bound in the kernel):
text = read_pdf_text("reports/q4.pdf", max_pages=5)
df = read_excel("data/sales.xlsx", sheet_name=0)
slides = read_pptx_text("deck.pptx")  # list of {"index", "text"} dicts
```

| Helper | Returns | Backed by |
|---|---|---|
| `read_pdf_text(path, *, max_pages=None)` | extracted text (`str`) | `pypdf` |
| `read_excel(path, sheet_name=0, **kwargs)` | a pandas `DataFrame` (xlsx, `openpyxl` engine) | `openpyxl` |
| `read_pptx_text(path)` | `list[dict]` of per-slide text extracts | `python-pptx` |

These backends are part of the optional **`[documents]`** extra (also pulled in
by `[all]`), and each helper **imports its package lazily, at call time**. That
means `parsimony-agents` installs and the executor boots fine *without* the
extra — calling a document helper without the backend installed raises an
`ImportError` telling you to install the `documents` extra, rather than failing
at import time. See [Installation](../getting-started/installation.md) for the
extras and [SQL and document inputs](../guides/sql-and-documents.md) for the
broader document-ingestion workflow.

## `DataframeRef` content addressing and `StorageBackend` healing

A `DataFrameObject` does not embed its rows — it carries a `DataframeRef`, an
immutable, content-addressed handle to a parquet file:

```python
from parsimony_agents.execution import DataframeRef

ref = DataframeRef.from_pandas(df, ref="q4_results", local_dir="/tmp/dfo")
# ref.content_hash  — MD5 of the canonicalized data (one hash, one file)
# ref.local_path    — parquet path under the session dir
# ref.remote_key    — optional key for cross-environment healing
```

The model is intentionally version-free: data is immutable, so one content hash
maps to exactly one parquet file. `materialize_sync()` reads the frame back,
trying the current session-dir layout first, then the stored absolute path, then
a remote download — this is the **healing** path that lets a `DataframeRef`
resolve even after it has moved between environments (a saved artifact reopened
on a different host, say).

Remote persistence is optional and goes through the `StorageBackend` protocol,
which any object with `upload(key, local_path)` and `download(key, local_path) ->
bool` satisfies:

```python
from parsimony_agents.execution import set_default_backend, set_default_local_root

set_default_backend(my_backend)      # used by materialize_sync() when none passed
set_default_local_root("/var/sessions/abc")  # default local root for path resolution
```

`set_default_backend` and `set_default_local_root` install process-level
defaults so `DataframeRef.materialize_sync()` can heal without every call site
threading a backend through. When no backend is configured, refs resolve purely
from the local session directory. For the persisted, curated form of these
frames — `Dataset` snapshots, the `.ockham` layout, and the data-object pool —
see [Artifacts](artifacts.md) and the [I/O reference](../reference/io.md).
