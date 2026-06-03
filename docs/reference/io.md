# I/O functions reference

The top-level `parsimony_agents` package exports a small family of
serialize / deserialize / read / save functions for persisting the three
curated artifact kinds — **datasets**, **charts**, and **reports** — plus
**notebooks** and their regenerable output cache. This page is the reference
for those functions, together with the lower-level helpers in
`parsimony_agents.report_format`, `parsimony_agents.virtual_path`,
`parsimony_agents.closure`, and `parsimony_agents.storage`.

These functions are the storage seam a host integrates against when it wants
to read, write, or walk the content-addressed `.ockham` artifact store
directly — for example to render a saved chart, mirror artifacts to cloud
storage, or compute a report's dependency closure. For the artifact models
themselves (`Dataset`, `Chart`, `Report`, `Script`, `ArtifactRef`) and the
dual-identity model behind them, see the
[Artifacts reference](artifacts.md) and the
[Artifacts, identity & lineage](../concepts/artifacts.md) concept page.

## Naming conventions

Every codec family follows the same shape, so once you know one you know all
three:

| Verb | Direction | Returns / takes |
| --- | --- | --- |
| `read_*` | file path → in-memory | reads a file from disk |
| `serialize_*` / `write_*_bytes` | in-memory → `bytes` | encodes to the on-disk byte format |
| `deserialize_*` | `bytes` → in-memory | decodes from the on-disk byte format |
| `save_*` | in-memory → file path | writes a file to disk |

`serialize_dataset` / `serialize_chart` are dispatcher-friendly aliases for
`write_dataset_bytes` / `write_chart_bytes` respectively — same function,
two names.

All four artifact formats are **self-contained**: a plain Parquet, Vega-Lite,
or Markdown renderer can open them with no `parsimony_agents` dependency.
Curation metadata (title, lineage, pins) is embedded in a side channel each
format already supports (Arrow schema metadata, Vega-Lite `usermeta`, YAML
frontmatter), and a vanilla file with no such metadata round-trips into an
empty curation envelope rather than failing.

---

## Dataset I/O

A dataset persists as a `.parquet` file whose Arrow schema metadata carries the
`Dataset` curation envelope under a `parsimony_agents` key. The payload is a
`parsimony` `Result` (a DataFrame plus provenance), wrapped at write time in a
`DataFrameObject`.

```python
from parsimony_agents import (
    Dataset,
    read_dataset,
    serialize_dataset,    # alias for write_dataset_bytes
    deserialize_dataset,
)
from parsimony_agents.dataset_io import write_dataset_bytes
```

### `read_dataset(path) -> tuple[Result, Dataset]`

```python
def read_dataset(path: str | Path) -> tuple[Result, Dataset]: ...
```

Read a curated `.parquet` dataset from disk. Returns a 2-tuple of the live
`parsimony.Result` (frame + provenance) and the `Dataset` curation envelope.

### `serialize_dataset(dataset, payload) -> bytes`

```python
serialize_dataset = write_dataset_bytes

def write_dataset_bytes(dataset: Dataset, payload: DataFrameObject) -> bytes: ...
```

Render a `Dataset` plus its `DataFrameObject` payload to Parquet bytes,
embedding the curation envelope in the Arrow schema metadata under the
`parsimony_agents` key. `serialize_dataset` is a back-compat alias for the same
function, intended for code that dispatches by a `serialize_*` name.

### `deserialize_dataset(data) -> tuple[Result, Dataset]`

```python
def deserialize_dataset(data: bytes) -> tuple[Result, Dataset]: ...
```

Inverse of `write_dataset_bytes`. Decode Parquet bytes back into a
`(Result, Dataset)` pair. A vanilla Parquet file with no embedded
`parsimony_agents` metadata decodes cleanly — the `Result` is fully populated
and the `Dataset` is an empty curation envelope.

```python
result, dataset = deserialize_dataset(parquet_bytes)

print(result.df.shape)          # the live DataFrame
print(dataset.title)            # curation metadata (empty if vanilla parquet)
print(dataset.notebook_refs)    # lineage back to the producing notebook
print(dataset.variable_name)    # kernel variable name (recipe field)
```

---

## Chart I/O

A chart persists as a `.vl.json` file — a Vega-Lite spec whose
`usermeta.parsimony_agents` slot carries the `Chart` curation envelope. The
payload is a `FigureObject`, which accepts either a raw Vega-Lite `dict` or an
Altair chart object.

```python
from parsimony_agents import (
    Chart,
    read_chart,
    serialize_chart,      # alias for write_chart_bytes
    deserialize_chart,
)
from parsimony_agents.chart_io import (
    write_chart_bytes,
    split_chart_data,
    inline_chart_data,
    chart_data_refs,
)
```

### `read_chart(path) -> tuple[Chart, dict]`

```python
def read_chart(path: str | Path) -> tuple[Chart, dict[str, Any]]: ...
```

Read a `.vl.json` chart file from disk. Returns a 2-tuple of the `Chart`
curation envelope and the Vega-Lite spec `dict`.

### `serialize_chart(chart, payload) -> bytes`

```python
serialize_chart = write_chart_bytes

def write_chart_bytes(chart: Chart, payload: FigureObject) -> bytes: ...
```

Render a `Chart` plus its `FigureObject` payload to `.vl.json` bytes with the
curation embedded in `usermeta.parsimony_agents`. Altair charts and plain
`dict` specs are both normalized into a Vega-Lite spec. `serialize_chart` is a
back-compat alias for dispatcher-friendly use.

### `deserialize_chart(data) -> tuple[Chart, dict]`

```python
def deserialize_chart(data: bytes) -> tuple[Chart, dict[str, Any]]: ...
```

Inverse of `write_chart_bytes`. Decode `.vl.json` bytes into a `(Chart, spec)`
pair. A vanilla Vega-Lite spec with no `usermeta.parsimony_agents` block
returns an empty `Chart()`.

```python
import altair as alt
import pandas as pd

from parsimony_agents import Chart, read_chart, serialize_chart
from parsimony_agents.execution.outputs import FigureObject
from parsimony_agents.identity import ArtifactRef, chart_logical_id

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
chart_obj = alt.Chart(df).mark_line().encode(x="x", y="y")

nb_ref = ArtifactRef(kind="notebook", logical_id="analysis", content_sha="abc123")
ds_refs = [ArtifactRef(kind="dataset", logical_id="ds-lid", content_sha="ds-csha")]

chart = Chart(
    logical_id=chart_logical_id(
        notebook_ref=nb_ref,
        chart_variable_name="trend_chart",
        source_dataset_refs=ds_refs,
        source_refs=[],
    ),
    title="Trend Chart",
    notebook_ref=nb_ref,
    source_dataset_refs=ds_refs,
    variable_name="trend_chart",
).with_payload(FigureObject(value=chart_obj))

chart.save("/tmp/trend.vl.json")

recovered, spec = read_chart("/tmp/trend.vl.json")
assert recovered.title == "Trend Chart"
assert spec["mark"] == "line"
```

### Chart data pool: `split_chart_data` / `inline_chart_data` / `chart_data_refs`

Vega-Lite specs inline their plotted data as JSON arrays. For large charts this
makes every re-style produce a full new snapshot. The chart-data-pool helpers
de-inline that data into a content-addressed pool so a spec carries only marker
references, not the data itself.

```python
def split_chart_data(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]: ...
def inline_chart_data(spec: dict[str, Any], data_map: dict[str, bytes]) -> dict[str, Any]: ...
def chart_data_refs(spec: dict[str, Any]) -> set[str]: ...
```

- **`split_chart_data(spec)`** extracts inline data arrays into a pool and
  returns `(deinlined_spec, pool)`. The de-inlined spec replaces each data array
  with a `__parsimony_chart_data_ref__` marker; `pool` maps each
  content-addressed key to its bytes.
- **`inline_chart_data(spec, data_map)`** is the inverse — it re-inlines pooled
  data back into the spec for rendering.
- **`chart_data_refs(spec)`** returns the set of every chart-data-pool
  `content_sha` referenced by a de-inlined spec.

```python
from parsimony_agents.chart_io import (
    split_chart_data,
    inline_chart_data,
    chart_data_refs,
)

spec = {
    "data": {"values": [{"x": 1, "y": 2}, {"x": 2, "y": 4}]},
    "mark": "bar",
    "encoding": {"x": {"field": "x"}, "y": {"field": "y"}},
}

deinlined, pool = split_chart_data(spec)
refs = chart_data_refs(deinlined)        # set of pool content_shas
full = inline_chart_data(deinlined, pool)
assert full["data"]["values"] == spec["data"]["values"]
```

De-inlining is a **storage optimization** internal to the store. `read_chart`
and `write_chart_bytes` still produce and consume fully self-contained
`.vl.json` files — losing a pool entry degrades gracefully (a missing data
series) rather than hard-failing.

---

## Notebook I/O

A notebook persists as a plain `.py` file: `serialize_notebook` and
`deserialize_notebook` round-trip the raw Python source with **no** embedded
metadata block. (A notebook's identity is its working-copy path, not a hash of
its inputs — see [Artifacts, identity & lineage](../concepts/artifacts.md).)
Separately, a notebook's *output* — `KernelOutput` from running it — has a
regenerable, content-addressed cache keyed on the code's SHA.

```python
from parsimony_agents import (
    Script,
    read_notebook,
    serialize_notebook,
    deserialize_notebook,
    save_notebook,
    save_notebook_state,
    load_notebook_state,
    notebook_state_cache_key,
    decode_notebook_state,
)
```

### Source round-trip

```python
def read_notebook(path: str | Path) -> Script: ...
def deserialize_notebook(data: bytes, *, path: str | None = None) -> Script: ...
def serialize_notebook(script: Script) -> bytes: ...
def save_notebook(script: Script, path: str | Path) -> None: ...
```

- **`read_notebook(path)`** reads a `.py` file into a `Script` (plain Python,
  no metadata block).
- **`deserialize_notebook(data, *, path=None)`** decodes `.py` bytes into a
  `Script`. Newlines are normalized; pass `path` to set the script's workspace
  path.
- **`serialize_notebook(script)`** renders a `Script` back to `.py` bytes
  (plain Python source).
- **`save_notebook(script, path)`** writes a `Script` to a `.py` file on disk.
  The `path` must end in `.py`.

### Output state cache

The notebook-state cache stores a `Script`'s `KernelOutput` at
`notebook-state/<code_sha>.json` under a root. Because the key is the SHA of the
notebook's code, editing the notebook automatically invalidates the cache.

```python
def notebook_state_cache_key(script: Script) -> str: ...
def save_notebook_state(script: Script, root: str | Path) -> None: ...
def load_notebook_state(script: Script, root: str | Path) -> KernelOutput | None: ...
def decode_notebook_state(blob: bytes, *, script: Script) -> KernelOutput | None: ...
```

- **`notebook_state_cache_key(script)`** returns the canonical relative cache
  path, `notebook-state/<code_sha>.json`.
- **`save_notebook_state(script, root)`** persists `script.output` to the cache
  under `root`. It is a **no-op** when there is nothing worth caching (no cell
  outputs and no connector fetch log).
- **`load_notebook_state(script, root)`** restores the cached `KernelOutput`,
  or returns `None` on a miss (no file, or the cached `code_sha` no longer
  matches the script's code).
- **`decode_notebook_state(blob, *, script)`** decodes raw cache bytes into a
  `KernelOutput`, returning `None` when the blob is invalid, the schema version
  is unsupported, or the cached `code_sha` is stale relative to `script.code`.
  The on-disk envelope is a `NotebookStateDocument` with
  `schema_version` (`Literal[1]`), `code_sha`, and `output` fields.

```python
from pathlib import Path

from parsimony_agents import (
    Script,
    save_notebook_state,
    load_notebook_state,
)
from parsimony_agents.execution.outputs import KernelOutput, PrimitiveObject

script = Script(
    path="notebooks/analysis.py",
    code="x = 42\nprint(x)",
    output=KernelOutput(
        outputs=[PrimitiveObject(value=42)],
        fetch_log=[],
    ),
)

root = Path("/workspace")
save_notebook_state(script, root)        # writes notebook-state/<code_sha>.json

recovered = load_notebook_state(script, root)
if recovered:
    print(recovered.outputs)             # cache hit — code unchanged
else:
    print("cache miss or code changed")
```

---

## Report format

A report persists as a Markdown document with a deterministic YAML frontmatter
block carrying its `formats`, pins, title, and subtitle. The `Report` model's
`snapshot_bytes()` method is the single source of truth for the on-disk shape;
the two functions below are the underlying compose / parse pair.

```python
from parsimony_agents.report_format import (
    compose_snapshot,
    parse_snapshot,
    ParsedSnapshot,
)
```

### `parse_snapshot(text) -> ParsedSnapshot`

```python
def parse_snapshot(text: str) -> ParsedSnapshot: ...
```

Split a report snapshot's text into a `ParsedSnapshot`, parsing the YAML
frontmatter into its `formats`, `pins`, `body`, `title`, and `subtitle`
components.

### `compose_snapshot(...) -> str`

```python
def compose_snapshot(
    formats: list[str],
    pins: dict[str, ArtifactRef],
    body: str,
    *,
    title: str,
    subtitle: str = "",
) -> str: ...
```

Compose a snapshot string: a deterministic YAML frontmatter block, a blank
line, then the body. Pins are emitted in sorted order; `formats` preserve the
input order. Determinism here is what makes a report's bytes — and therefore its
`content_sha` — stable across renders.

### `ParsedSnapshot`

```python
class ParsedSnapshot(NamedTuple):
    formats: list[str]
    pins: dict[str, ArtifactRef]
    body: str
    title: str
    subtitle: str
```

A `NamedTuple` of the parsed snapshot fields. It supports both positional
unpacking and attribute access.

```python
from parsimony_agents import Report
from parsimony_agents.identity import ArtifactRef
from parsimony_agents.report_format import compose_snapshot, parse_snapshot

trend_ref = ArtifactRef(kind="chart", logical_id="trend-lid", content_sha="a" * 64)
pins = {"trend": trend_ref}

report = Report(
    logical_id="earnings-report-lid",
    title="Q4 2025 Earnings",
    subtitle="Revenue beat by 8%",
    markdown="The trend chart shows strong growth.\n",
    formats=["html", "pdf"],
    live_name_pins=pins,
)

text = report.snapshot_bytes().decode("utf-8")   # frontmatter + body
parsed = parse_snapshot(text)
assert parsed.title == "Q4 2025 Earnings"
assert parsed.formats == ["html", "pdf"]
assert "trend" in parsed.pins
```

---

## Virtual path

The workspace presents agents with a **virtual live-tree** — friendly paths like
`notebooks/analysis.py`, `data/sales.parquet`, `charts/trend.vl.json`,
`reports/q4.qmd` — synthesized from the canonical `.ockham` storage layout. The
`virtual_path` helpers map those paths back to canonical snapshots and validate
that splicing a name into an `.ockham` path is safe.

```python
from parsimony_agents.virtual_path import (
    resolve_virtual_entry,
    latest_content_sha,
    is_safe_name,
)
```

### `resolve_virtual_entry(local_dir, path, *, workspace_id) -> str | None`

```python
def resolve_virtual_entry(local_dir: Path, path: str, *, workspace_id: str) -> str | None: ...
```

Map a virtual live-tree `path` to its canonical `.ockham` snapshot path. It
scans `.ockham/<kind>s/*/curation.json` to find the artifact whose `live_name`
matches the requested name, then reads the latest `content_sha` from that
artifact's `log.jsonl`. Returns the canonical relative path (e.g.
`.ockham/notebooks/analysis/<content_sha>.py`), or `None` when there is no
matching curation entry or no log history.

The supported virtual directories and their canonical kinds/extensions are
defined by the `VIRTUAL_LIVE_KINDS` constant: `notebooks` → `(notebook, .py)`,
`data` → `(dataset, .parquet)`, `charts` → `(chart, .vl.json)`, `reports` →
`(report, .qmd)`.

```python
from pathlib import Path

from parsimony_agents.virtual_path import resolve_virtual_entry

canonical = resolve_virtual_entry(
    Path("/workspace"),
    "notebooks/analysis.py",
    workspace_id="ws-123",
)
if canonical:
    print(f"resolved to {canonical}")   # .ockham/notebooks/analysis/<sha>.py
else:
    print("artifact not found in workspace")
```

> Resolution is an `O(artifacts-of-this-kind)` scan over `curation.json` files —
> fast for the tens-of-artifacts case in practice, but worth knowing if you ever
> have thousands.

### `latest_content_sha(log_path) -> str | None`

```python
def latest_content_sha(log_path: Path) -> str | None: ...
```

Extract the most recent `content_sha` from a `log.jsonl` file, or `None` if the
log is missing or empty. This is the version-history primitive
`resolve_virtual_entry` uses once it has located the right artifact directory.

### `is_safe_name(name) -> bool`

```python
def is_safe_name(name: str) -> bool: ...
```

Validate that a name is safe to splice into an `.ockham` path. Rejects path
traversal (`..`, slashes), NUL bytes, and hidden-file (`.`-prefixed) names — a
guard against escaping the workspace directory.

---

## Closure

Artifacts form a DAG: reports depend on datasets and charts, which depend on
notebooks and other datasets, which bottom out at notebooks and data objects
(the leaves). The `closure` helpers walk that DAG. Both are **async** because
resolving an artifact's children requires reading and deserializing its snapshot
through an executor.

```python
from parsimony_agents.closure import enumerate_closure, child_refs
```

### `child_refs(ref, *, executor) -> list[ArtifactRef]`

```python
async def child_refs(ref: ArtifactRef, *, executor: _Executor) -> list[ArtifactRef]: ...
```

Return an artifact's direct, typed source refs — the single source of truth for
the edges of the artifact DAG. Leaf kinds (`notebook`, `data_object`) return an
empty list.

### `enumerate_closure(root, *, executor) -> list[ArtifactRef]`

```python
async def enumerate_closure(root: ArtifactRef, *, executor: _Executor) -> list[ArtifactRef]: ...
```

Post-order depth-first traversal of the artifact DAG rooted at `root`. Returns
every reachable ref **including** `root`, with dependencies emitted before their
dependents, deduplicated by `(kind, logical_id, content_sha)`. Cycles are safe
(dedup terminates them).

```python
import asyncio

from parsimony_agents.identity import ArtifactRef
from parsimony_agents.closure import enumerate_closure


async def walk() -> None:
    executor = ...  # an executor exposing read_workspace_file
    root = ArtifactRef(
        kind="report",
        logical_id="earnings-report-lid",
        content_sha="report-csha",
    )
    closure = await enumerate_closure(root, executor=executor)
    for ref in closure:                # leaves first, root last
        print(ref.kind, ref.logical_id[:8], ref.content_sha[:8])


asyncio.run(walk())
```

Use the closure to copy an artifact and everything it transitively depends on —
for example, exporting a report and its full lineage to another store.

---

## Storage

The codecs above produce and consume `bytes` and file paths; the
`FileStorage` protocol is the backend-agnostic key-value seam those bytes flow
through. A host supplies any implementation of the protocol; `LocalFileStorage`
is the filesystem-backed implementation shipped in the package.

```python
from parsimony_agents.storage import FileStorage, LocalFileStorage
```

### `FileStorage` protocol

```python
@runtime_checkable
class FileStorage(Protocol):
    async def read(self, key: str) -> bytes: ...
    async def write(self, key: str, data: bytes) -> None: ...
    async def append(self, key: str, data: bytes) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def list_keys(self, prefix: str = "") -> list[str]: ...
    async def delete_prefix(self, prefix: str) -> None: ...
    async def materialize_prefix(self, prefix: str) -> Path: ...
    async def sync_back(self, local_dir: Path, prefix: str) -> None: ...
```

A backend-agnostic, fully-async key-value file store. Keys are slash-separated
paths (e.g. `.ockham/datasets/<logical_id>/<content_sha>.parquet`). The
protocol splits into **per-key** operations (`read`, `write`, `append`,
`delete`, `exists`) and **per-prefix** operations:

- **`list_keys(prefix="")`** — every key under a prefix.
- **`delete_prefix(prefix)`** — delete every key under a prefix.
- **`materialize_prefix(prefix)`** — copy a prefix's keys to a local directory
  and return its `Path` (so non-async tooling, e.g. a notebook kernel, can read
  real files).
- **`sync_back(local_dir, prefix)`** — write a local directory's files back
  under a prefix (the inverse of `materialize_prefix`).

It is `@runtime_checkable`, so `isinstance(obj, FileStorage)` works for a
structural check.

### `LocalFileStorage`

```python
class LocalFileStorage:
    def __init__(self, root: Path) -> None: ...
    async def read(self, key: str) -> bytes: ...
    async def write(self, key: str, data: bytes) -> None: ...
    async def append(self, key: str, data: bytes) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
    async def list_keys(self, prefix: str = "") -> list[str]: ...
    async def delete_prefix(self, prefix: str) -> None: ...
    async def materialize_prefix(self, prefix: str) -> Path: ...
    async def sync_back(self, local_dir: Path, prefix: str) -> None: ...
```

The filesystem-backed implementation of `FileStorage`. Construct it with a
`root` directory; every `key` is resolved as a path under that root.

```python
import asyncio
from pathlib import Path

from parsimony_agents.storage import LocalFileStorage


async def main() -> None:
    storage = LocalFileStorage(root=Path("/workspace"))

    await storage.write("notes/hello.txt", b"hi")
    assert await storage.exists("notes/hello.txt")
    assert await storage.read("notes/hello.txt") == b"hi"

    keys = await storage.list_keys(prefix="notes/")
    print(keys)                              # ['notes/hello.txt']


asyncio.run(main())
```

---

## See also

- [Artifacts reference](artifacts.md) — the `Dataset`, `Chart`, `Report`,
  `Script`, and `ArtifactRef` models these functions operate on.
- [Artifacts, identity & lineage](../concepts/artifacts.md) — the dual-identity
  (`logical_id` / `content_sha`) and `.ockham` storage model.
- [Saving and loading artifacts](../guides/saving-loading-artifacts.md) —
  task-oriented walkthrough of persisting and reloading artifacts.
- [Execution reference](execution.md) — `KernelOutput`, `DataFrameObject`, and
  `FigureObject`, the payload types these codecs carry.
