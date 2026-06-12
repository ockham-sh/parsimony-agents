# Execution reference

Field-level reference for the execution subsystem: the executors that run
agent-written Python, the output factory that turns raw values into typed
kernel outputs, the typed outputs themselves, the kernel result container,
lineage types, connector memoization, the LLM-rendering paginators, and the
storage backend helpers.

Everything documented here imports from a single module:

```python
from parsimony_agents.execution import (
    BaseCodeExecutor,
    CodeExecutor,
    OutputFactory,
    DataFrameObject,
    FigureObject,
    PrimitiveObject,
    ExceptionObject,
    KernelOutput,
    KernelOutputType,
    FetchLogEntry,
    DataframeRef,
    StorageBackend,
    StringPaginator,
    TablePaginator,
    StructuredStreamCapturer,
    finalize_spec,
    generate_cell_id,
    set_default_backend,
    set_default_local_root,
    get_default_local_root,
)
from parsimony_agents.execution.sandbox import (
    SandboxedCodeExecutor,
    create_executor,
    selected_capability_tier,
    detect_bwrap_support,
)
```

This is the low-level substrate that the agent loop drives. Most host
integrators never construct a `CodeExecutor` directly — they configure an
[`Agent`](agent.md) and consume [events](events.md) and
[artifacts](artifacts.md). Reach for this page when you are embedding the
executor in a host, swapping in a remote/sandboxed executor, teaching the
output factory about a new type, or inspecting lineage. For the conceptual
picture, see [Code execution](../concepts/code-execution.md) and
[Artifacts, identity & lineage](../concepts/artifacts.md).

## BaseCodeExecutor and CodeExecutor (methods)

`BaseCodeExecutor` is the abstract protocol for code execution. The default
in-process implementation is `CodeExecutor`; a remote or sandboxed executor
subclasses `BaseCodeExecutor` and implements the same surface.

```python
class BaseCodeExecutor(ABC): ...
```

### Abstract methods (`BaseCodeExecutor`)

| Method | Signature | Purpose |
| --- | --- | --- |
| `execute` | `async execute(code: str, dry_run: bool = False, timeout_seconds: float \| None = None, producer_notebook_path: str \| None = None, seen_live_names: set[tuple[str, str]] \| None = None) -> KernelOutput` | Run `code` in the kernel namespace. |
| `eval` | `async eval(expr: str, dry_run: bool = False, timeout_seconds: float \| None = None) -> KernelOutput` | Evaluate a single expression. |
| `get` | `async get(key: str) -> KernelOutputType \| None` | Fetch a kernel variable, wrapped as a typed output. |
| `set_cwd` | `async set_cwd(cwd: str, session_id: str \| None = None) -> None` | Change the working directory (workspace switch). |
| `clear_namespace` | `async clear_namespace() -> None` | Reset the kernel to its base locals. |

The base class also provides four abstract workspace-file methods
(`read_workspace_file`, `write_workspace_file`, `delete_workspace_file`,
`list_workspace_files`) and an abstract `execute_workspace` (see below). These
four methods are also the storage seam the framework uses to persist `return_*`
deliverables: `parsimony_agents.execution.artifact_store` writes each
dataset/chart/report (and its notebook recipe) as the
`.ockham/<kind>s/<logical_id>/{curation.json, log.jsonl, <content_sha>.<ext>}`
triplet through `write_workspace_file`, so the same persistence works whether the
executor is in-process (local fs) or a remote sandbox. A custom executor must
therefore implement these methods to accept `.ockham/` dotpaths and write
atomically (tmp-write + replace), since `artifact_store` reads each snapshot
straight back to verify it (`SnapshotIntegrityError`). See
[Artifacts, identity & lineage](../concepts/artifacts.md) for the persistence
layout.

`set_connectors` and `get_origin` are **not** abstract — `BaseCodeExecutor`
ships concrete defaults (`set_connectors` is a no-op; `get_origin` reads the
in-process `origin_ledger`, returning `None` when there is none). A remote
executor overrides them to answer over the wire. Both are `async`, so a remote
kernel can answer connector injection and lineage queries asynchronously.

### CodeExecutor

```python
class CodeExecutor(BaseCodeExecutor):
    def __init__(
        self,
        *,
        cwd: str,
        output_factory: OutputFactory,
        file_session_materializer: Callable[[str], Awaitable[None]] | None = None,
    ): ...
```

`CodeExecutor` is in-process and stateful: it maintains a persistent `locals`
namespace across `execute()` calls, initialized with `pd`, `np`, `alt`,
`datetime`, the document helpers (`read_pdf_text`, `read_excel`,
`read_pptx_text`), `load_dataset`, and any injected connector bundles. Code
runs via `exec`/`eval` against a restricted `__builtins__`, captures
stdout/`display`/`print` into structured outputs, enforces a per-cell timeout,
and attributes assigned variables to producer notebooks through the
`OriginLedger`.

#### Concrete methods

```python
from pathlib import Path
from parsimony_agents.execution import CodeExecutor, OutputFactory

workspace = Path("/tmp/my_workspace")
output_factory = OutputFactory(local_dir=workspace)
executor = CodeExecutor(cwd=str(workspace), output_factory=output_factory)
```

**`execute`** — run code in the persistent namespace.

```python
result = await executor.execute(
    'df = pd.DataFrame({"a": [1, 2, 3]})',
    producer_notebook_path="notebooks/analysis.py",
)
```

When `producer_notebook_path` is set, `execute` opens a `RunScope`, diffs the
locals before and after the run, and stamps each newly-assigned name with a
lineage origin (notebook path + the load/fetch refs the run touched). It also
handles top-level `await`, timeout enforcement, and fetch logging. With
`producer_notebook_path=None` no `RunScope` is opened — the run produces no
lineage (use this for scratch/dry execution). `dry_run=True` copies the locals
dict before evaluating so mutations are isolated and discarded, and skips
origin attribution. `seen_live_names` is the cross-terminal access gate consumed
by `load_dataset` (see below).

**`execute_workspace`** — run code in a *fresh* namespace.

```python
result = await executor.execute_workspace("print('hello')")
```

Same signature as `execute`. It clears the locals, origin ledger, and connector
cache first, so each call starts clean. Used for workspace/IDE mode where no
lineage is tracked.

**`eval`** — evaluate an expression in the current persistent namespace.

```python
result = await executor.eval("df.shape")
```

**`get`** — fetch a kernel variable, wrapped as a typed output.

```python
output = await executor.get("df")          # KernelOutputType | None
if output is not None:
    llm_blocks = output.to_llm(mode="default")
```

`get` runs the value through `OutputFactory.from_value()` and returns the typed
output, or `None` if the name is not bound.

**`set_cwd`** — change the working directory.

```python
await executor.set_cwd("/tmp/other_workspace", session_id="run-42")
```

Clears the connector cache and origin ledger (a workspace switch is a kernel
lifetime boundary) and rebinds the fetch logger to the new cwd for data-object
persistence.

**`clear_namespace`** — reset to base.

```python
await executor.execute("def my_func(x): return x * 2")
assert "my_func" in executor.locals

await executor.clear_namespace()
assert "my_func" not in executor.locals
assert "pd" in executor.locals          # base is restored
```

Resets locals to the base set (`pd`, `np`, `alt`, `datetime`, the document
helpers, `load_dataset`), clears the origin ledger and connector cache,
re-applies connectors, and re-runs any registered setup snippets.

**`add_setup_snippet`** — register code that re-runs on every
`clear_namespace()`.

```python
executor.add_setup_snippet("import matplotlib; matplotlib.use('Agg')")
```

**`get_locals`** — return a fresh dict of user-bound kernel names.

```python
names = executor.get_locals().keys()
```

It is a new (mutable) dict copied from the live `locals`, with the injected
prelude filtered out (`pd`, `np`, `alt`, `display`, `print`, `__builtins__`,
and the document helpers), so it shows only what the agent's code bound. To
reach the full live namespace (including the prelude), read `executor.locals`
directly.

**`get_origin`** — look up the lineage origin of a variable.

```python
origin = await executor.get_origin("df")
if origin is not None:
    print(f"df came from {origin.notebook_path}")
```

Returns the `VariableOrigin` for the name, or `None` if the variable was never
attributed (no producer notebook on the run that created it).

**`set_connectors`** — inject connector bundles.

```python
from parsimony_fred import CONNECTORS as FRED

await executor.set_connectors(FRED.bind(api_key="..."))
# or, for a named binding:
await executor.set_connectors({"fred": FRED.bind(api_key="...")})
```

Accepts a `Connectors` bundle or a `Mapping[str, Connectors]`. Internally,
each connector is converted to a secret-free `ConnectorManifest` and wrapped in a
`ConnectorProxy`, backed by a `ConnectorTransport` (in-process or socket-based).
The kernel-side namespace receives only the `ConnectorProxy` objects—metadata
and authority to call, but no credential (`bound_arguments` or secrets). A
`MemoizingConnectorTransport` wraps the inner transport with the per-kernel cache and
post-fetch hooks (data-object persister, fetch logger) so lineage stays truthful
across cached and uncached calls. See [Connectors](../concepts/connectors.md)
for the connector model and [Memoization](#memoization-connectorcache-memoizingconnectortransport)
below.

#### Timeout behaviour

A timed-out cell returns gracefully — it does not wedge the executor.

```python
from parsimony_agents.execution import ExceptionObject

result = await executor.execute("while True: pass", timeout_seconds=1)

assert isinstance(result.outputs[0], ExceptionObject)
assert "timeout" in result.outputs[0].value.lower()

# Executor is still usable afterwards
await executor.execute("x = 42")
```

The timeout default is process-configurable (300s) and code runs in a dedicated
daemon thread. Cancellation is best-effort and cannot interrupt blocking
C-extension calls.

## OutputFactory (from_value, register)

`OutputFactory` converts Python values to typed `KernelOutputType` objects. It
owns the parquet `local_dir` and an optional `StorageBackend` used when a value
needs to be persisted (DataFrames).

```python
class OutputFactory:
    def __init__(self, *, local_dir: str | Path, backend: StorageBackend | None = None): ...
```

### `from_value`

```python
def from_value(self, value: Any, ref: str = "anonymous") -> KernelOutputType: ...
```

Dispatch order: registered custom handlers are checked first, then the built-in
`isinstance` chain:

| Input value | Output |
| --- | --- |
| pandas `DataFrame` / `Series` | `DataFrameObject` (with a parquet `DataframeRef`) |
| Altair chart | `FigureObject` (or `ExceptionObject` if the spec fails to compile) |
| scalar (`str`, `int`, `float`, `bool`, `None`) | `PrimitiveObject` |
| numpy scalar (`np.generic`) | `PrimitiveObject` (unwrapped via `.item()`) |
| `Exception` | `ExceptionObject` |
| anything else | `PrimitiveObject` wrapping `str(value)` |

`ref` becomes the slug used when writing the parquet snapshot for DataFrames.

### `register`

```python
@classmethod
def register(cls, type_: type, handler: OutputHandler) -> None: ...
```

`register` is a **class method** — it installs a handler for a type globally,
and registered handlers are consulted before the built-in chain. A handler has
the shape `(value, *, local_dir, backend) -> KernelOutputType`.

```python
import polars as pl
from parsimony_agents.execution import OutputFactory, DataFrameObject, DataframeRef

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

## Typed outputs (DataFrameObject, FigureObject, PrimitiveObject, ExceptionObject)

`KernelOutputType` is the union of the four typed outputs. Each is a structured
output object that knows how to render itself to LLM blocks via `to_llm(...)`.

### DataFrameObject

```python
class DataFrameObject(BaseOutputObject):
    ref: DataframeRef
```

Wraps a parquet-backed DataFrame. It carries head/tail previews and column
dtypes, and can heal across environments via the content-addressed
`DataframeRef`. Its `to_llm` paginates the preview as CSV:

```python
def to_llm(
    self,
    mode: Literal["default", "minimal"] = "default",
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]: ...
```

`overrides` accepts the same view knobs the table paginator uses
(`page_rows`, `show_dtypes`, `display_pages`, `max_cell_length`).

### FigureObject

```python
class FigureObject(BaseOutputObject):
    value: alt.TopLevelMixin | dict[str, Any]
```

Wraps an Altair/Vega-Lite visualization (either a live Altair object or a spec
dict). It serializes to a spec dict and can compute a base64 PNG for LLM image
rendering:

```python
def calc_base64_image(self, force_recalc: bool = False) -> str: ...
```

### PrimitiveObject

```python
class PrimitiveObject(BaseOutputObject):
    value: str | int | float | bool | None
```

Holds a scalar. Non-scalar fall-through values are stored as their string
representation.

### ExceptionObject

```python
class ExceptionObject(BaseOutputObject):
    value: str
```

Captures an exception. The stored `value` is the traceback string, redacted for
secrets. For typed `parsimony` errors it surfaces the message directly without a
full traceback (see [Failure handling & recovery](../concepts/failure-and-recovery.md)).
Timeouts surface here too — the `value` contains a "timeout" message.

## KernelOutput and FetchLogEntry

### KernelOutput

`KernelOutput` is the top-level result of `execute`/`eval`/`execute_workspace`.

```python
class KernelOutput(MessageContent):
    outputs: list[KernelOutputType]
    fetch_log: list[FetchLogEntry]
```

`outputs` is the ordered list of typed outputs the cell produced (captured
stdout/`display` results and the final expression value). `fetch_log` records
every connector fetch observed during the run. `KernelOutput` renders the whole
result for the LLM:

```python
def to_llm(self, mode: str = "default") -> list[dict[str, Any]]: ...
```

Returns a list of `{type, text | image_url}` blocks, paginating large
DataFrames and strings per the view config.

### FetchLogEntry

```python
class FetchLogEntry(BaseModel):
    provenance: Provenance
    row_count: int
    column_names: list[str]
    columns: list[dict[str, Any]]
    head: dict[str, Any] | None = None
    tail: dict[str, Any] | None = None
    data_object_ref: ArtifactRef | None = None
    version: int | None = None
```

One record per connector fetch: the `Provenance` of the call (exposed via the
`source`, `source_description`, and `params` convenience properties), the
result's `row_count`, `column_names` and per-column `columns` metadata, optional
`head`/`tail` previews, and — when the fetch result was mirrored into the
content-addressed data-object pool — a `data_object_ref` (`ArtifactRef` with
`kind="data_object"`) pointing at the persisted parquet snapshot. `version` is
always `None` for immutable object-pool entries. See
[Artifacts reference](artifacts.md) for `Provenance` and `ArtifactRef`.

## DataframeRef (from_pandas, materialize)

`DataframeRef` is an immutable, content-addressed reference to a parquet-backed
DataFrame.

```python
class DataframeRef(BaseModel):
    ref: str
    local_path: str
    content_hash: str
    remote_key: str | None = None
```

`ref` is the slug, `content_hash` is the hash of the canonicalized data,
`local_path` is the session-relative or stored absolute path, and `remote_key`
(optional) lets the ref heal across environments through a `StorageBackend`.
Data is immutable — one hash, one file, no versioning.

### `from_pandas`

```python
@classmethod
def from_pandas(
    cls,
    dataframe: pd.DataFrame | pd.Series,
    ref: str = "anonymous",
    *,
    local_dir: str | Path,
    backend: StorageBackend | None = None,
) -> DataframeRef: ...
```

Computes the content hash, writes parquet to `local_dir/ref/{hash}.parquet`,
and (if a `backend` is given) uploads the file.

### `materialize` / `materialize_sync`

```python
def materialize_sync(self, backend: StorageBackend | None = None) -> pd.DataFrame: ...

async def materialize(self, backend: StorageBackend | None = None) -> pd.DataFrame: ...
```

`materialize_sync` reads the DataFrame back, trying the current session-dir
layout first, then the stored absolute path, then a remote download via the
backend. `materialize` is the async wrapper (`materialize_sync` run in a
thread).

```python
ref = DataframeRef.from_pandas(df, ref="gdp", local_dir=workspace)
restored = await ref.materialize()
```

When no `backend` is passed explicitly, both methods fall back to the
process-level default set by
[`set_default_backend`](#backends-and-helpers-storagebackend-set_default_backend-set_default_local_root-get_default_local_root)
and resolve paths against the default local root.

## Lineage (OriginLedger, RunScope, VariableOrigin)

Lineage answers "which notebook produced this variable, and what data did it
depend on?" It is recorded automatically when you call `execute(...)` with a
`producer_notebook_path`. The three types below are the machinery; `OriginLedger`
is not re-exported from `parsimony_agents.execution` directly (query it through
`CodeExecutor.get_origin`), while `VariableOrigin` is what `get_origin` returns.

### VariableOrigin

```python
@dataclass(frozen=True)
class VariableOrigin:
    notebook_path: str
    load_refs: tuple[ArtifactRef, ...] = ()
    fetch_refs: tuple[ArtifactRef, ...] = ()
```

Immutable record of provenance: the notebook that produced the variable, the
dataset loads it depended on (`load_refs`), and the connector fetches it
depended on (`fetch_refs`).

### RunScope

```python
@dataclass
class RunScope:
    notebook_path: str
    load_refs: list[ArtifactRef] = field(default_factory=list)
    fetch_refs: list[ArtifactRef] = field(default_factory=list)
```

The mutable per-run accumulator opened around a notebook execution. It collects
load/fetch events as the run progresses; at scope exit it is frozen into a
`VariableOrigin`.

### OriginLedger

The in-memory map of variable name → `VariableOrigin`. One per executor kernel
lifetime; cleared on `set_cwd`/`clear_namespace`. Opened and stamped through a
scope context manager.

| Method | Signature | Purpose |
| --- | --- | --- |
| `scope` | `scope(notebook_path: str) -> Iterator[RunScope]` | Context manager opening a per-run scope. Nested scopes raise `RuntimeError`. |
| `stamp` | `stamp(names: list[str], scope: RunScope) -> None` | Attribute the given names to the scope's frozen origin. |
| `get` | `get(name: str) -> VariableOrigin \| None` | Retrieve the origin for a name, or `None`. |
| `clear` | `clear() -> None` | Wipe the ledger and close any open scope. |

The executor drives this for you: `execute(producer_notebook_path=...)` opens a
`scope`, runs the code, diffs pre/post locals to find newly-assigned names, and
calls `stamp(names, scope)`. You read results back with
`await executor.get_origin(name)`.

```python
result = await executor.execute(
    'df = pd.DataFrame({"a": [1, 2, 3]})',
    producer_notebook_path="notebooks/analysis.py",
)
origin = await executor.get_origin("df")
print(origin.notebook_path)   # "notebooks/analysis.py"
```

The `load_dataset` primitive injected into the kernel records a load on the open
`RunScope` when it resolves a dataset, so loaded data flows into
`VariableOrigin.load_refs`. It is cross-terminal gated: when `seen_live_names` is
passed to `execute` and the requested dataset slug is not in that set,
`load_dataset` raises a collision error rather than reading another terminal's
data.

```python
result = await executor.execute(
    'inflation = load_dataset("inflation_monthly")',
    producer_notebook_path="notebooks/fed_analysis.py",
    seen_live_names={("dataset", "inflation_monthly"), ("notebook", "fed_analysis.py")},
)
```

See [Artifacts, identity & lineage](../concepts/artifacts.md) for the full
lineage model.

## Memoization (ConnectorCache, MemoizingConnectorTransport)

Connector calls are memoized within one kernel lifetime so an agent re-running
the same fetch doesn't hit the network twice. `set_connectors` wires this up
automatically. The kernel receives a `Mapping[str, ConnectorProxy]` — each proxy
points to a `ConnectorManifest` and delegates to a `ConnectorTransport`. A
`MemoizingConnectorTransport` wraps the inner transport with per-kernel memoization and
post-fetch hooks; the two types below are the moving parts.

### ConnectorCache

```python
class ConnectorCache:
    def get(self, name: str, args_key: str) -> Result | None: ...
    def put(self, name: str, args_key: str, result: Result) -> None: ...
    def clear(self) -> None: ...
```

A store keyed by `(connector_name, canonical_args_key)` → `Result`. Cleared on
`clear_namespace`/`set_cwd`.

### MemoizingConnectorTransport

```python
class MemoizingConnectorTransport:
    def __init__(
        self,
        inner: ConnectorTransport,
        cache: ConnectorCache,
        post_hooks: tuple[Callable[[Result], Any], ...],
    ): ...
```

Wraps any inner `ConnectorTransport` (in-process or socket-based) with
memoization and post-fetch hooks. Identical-argument calls return the cached
`Result` without invoking the inner transport. Crucially, the `post_hooks`
(the data-object persister and the fetch logger) run on **every** call — cached
or not — so the `fetch_log` and lineage stay truthful even when a fetch is
served from cache.

```python
result = await executor.execute('data = fred["gdpc1"](series_id="GDPC1")')
# result.fetch_log has one FetchLogEntry with a persisted data_object_ref
```

### ConnectorProxy and ConnectorManifest

The kernel never receives a bound `Connector` (which would carry the credential
in its `bound_arguments`). It receives a `ConnectorProxy` minted from the
connector's secret-free `ConnectorManifest`. The proxy exposes connector
metadata (parameters, return types, etc.) and the authority to call via the
transport, but carries no credential. See the `parsimony` library for
`ConnectorProxy`, `ConnectorManifest`, `ConnectorTransport`, and
`Connector.to_manifest()`.

## Pagination (StringPaginator, TablePaginator) and StructuredStreamCapturer

The typed outputs use these paginators to fit large results into LLM context.
They are independently usable.

### StringPaginator

```python
class StringPaginator:
    def __init__(self, text: str, chars_per_page: int): ...
    def iter_pages(self, display_pages: list[int] | None = None) -> Iterator[str]: ...
```

Splits a long string on word boundaries at the `chars_per_page` threshold,
yielding page blocks with char offsets and continuation markers. `display_pages`
selects a subset of pages (e.g. `[0, -1]` for first and last).

```python
from parsimony_agents.execution import StringPaginator

pages = list(StringPaginator(long_text, chars_per_page=2000).iter_pages([0, -1]))
```

### TablePaginator

```python
class TablePaginator:
    def __init__(self, df: pd.DataFrame, rows_per_page: int, show_dtypes: bool = True): ...
    def iter_pages(
        self,
        display_pages: list[int] | None = None,
        *,
        na_rep: str = "<NULL>",
        max_cell_length: int = 100,
    ) -> Iterator[str]: ...
```

Splits a DataFrame on `rows_per_page`, yielding CSV page blocks with row ranges
and (when `show_dtypes=True`) dtype hints. Cells longer than `max_cell_length`
are truncated; nulls render as `na_rep`.

```python
from parsimony_agents.execution import TablePaginator

for block in TablePaginator(df, rows_per_page=50).iter_pages(max_cell_length=80):
    print(block)
```

### StructuredStreamCapturer

```python
class StructuredStreamCapturer:
    def __init__(self, output_factory: OutputFactory): ...
```

Captures `stdout.write`, `display()`, and `print()` calls made during code
execution and structures them into `KernelOutputType` objects via the supplied
`OutputFactory`. The executor uses it internally to assemble
`KernelOutput.outputs`.

## Backends and helpers (StorageBackend, set_default_backend, set_default_local_root, get_default_local_root, generate_cell_id, finalize_spec)

### StorageBackend

```python
@runtime_checkable
class StorageBackend(Protocol):
    def upload(self, key: str, local_path: Path) -> None: ...
    def download(self, key: str, local_path: Path) -> bool: ...
```

The optional remote-parquet persistence protocol. `upload` pushes a local file
under `key`; `download` fetches it back, returning `True` on success. Used by
`DataframeRef` and the data-object persister to heal references across
environments. Any object with these two methods satisfies the protocol
(`runtime_checkable`).

### Process-level defaults

```python
def set_default_backend(backend: StorageBackend | None) -> None: ...
def set_default_local_root(path: Path | str | None) -> None: ...
def get_default_local_root() -> Path | None: ...
```

- `set_default_backend` sets the process-level `StorageBackend` used by
  `DataframeRef.materialize_sync()` when no explicit backend is passed.
- `set_default_local_root` sets the default local session directory used to
  resolve parquet paths across environments.
- `get_default_local_root` returns whatever `set_default_local_root` last set
  (or `None`).

```python
from pathlib import Path
from parsimony_agents.execution import set_default_local_root, get_default_local_root

set_default_local_root("/tmp/session-root")
assert get_default_local_root() == Path("/tmp/session-root")
```

### generate_cell_id

```python
def generate_cell_id(length: int = 6) -> str: ...
```

Generates a unique alphanumeric cell ID (6 chars by default) for tracing a cell
through the run.

```python
from parsimony_agents.execution import generate_cell_id

cell_id = generate_cell_id()        # e.g. "a3Kf9z"
```

### finalize_spec

```python
def finalize_spec(spec: dict) -> dict: ...
```

Applies the default sizing/autosize rules (width, height, autosize type) to a
Vega-Lite spec dict. `FigureObject` uses it when normalizing a chart for
rendering.

```python
from parsimony_agents.execution import finalize_spec

normalized = finalize_spec({"mark": "bar", "encoding": {...}})
```

## Where this fits

The executor is the engine the agent loop turns. To put it in context:

- [Code execution](../concepts/code-execution.md) — the conceptual model.
- [Artifacts, identity & lineage](../concepts/artifacts.md) — how `FetchLogEntry`,
  `DataframeRef`, and `VariableOrigin` become durable artifacts.
- [Agent reference](agent.md) — the high-level `Agent` that owns an executor.
- [Events reference](events.md) — what the loop emits as the executor runs.
- [Artifacts reference](artifacts.md) — `Provenance` and `ArtifactRef`.
- [Streaming and displaying results](../guides/streaming-and-displaying-results.md)
  — consuming `to_llm` blocks in a host.

A full end-to-end agent program (the executor is created and driven for you):

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
