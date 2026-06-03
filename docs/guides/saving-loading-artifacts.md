# Saving and loading artifacts

Parsimony Agents produces **artifacts** — `Dataset`, `Chart`, and `Report` — plus
the notebook (`Script`) that generates them. As a host or integrator, you sometimes
need to write these to disk yourself, read them back outside the agent loop, or walk
their lineage. This guide covers the I/O surface for doing that, and how the
agent-facing *virtual paths* resolve to the canonical `.ockham` storage layout.

For the conceptual model behind dual identity (`logical_id` vs `content_sha`),
snapshots, and the artifact DAG, see [Artifacts, identity &
lineage](../concepts/artifacts.md). This page is the practical, code-led companion;
the full signatures live in the [I/O functions reference](../reference/io.md) and
the [Artifacts reference](../reference/artifacts.md).

Every symbol below is importable from the top-level `parsimony_agents` package unless
the example says otherwise.

## Saving a Dataset/Chart/Report (`.save` and `with_payload`)

Each artifact model is a **curation envelope** — title, description, tags, and
lineage refs — that is separate from its **payload** (the actual `DataFrameObject`,
`FigureObject`, or markdown body). For `Dataset` and `Chart`, the payload is an
in-process-only private attribute. You must attach it with `.with_payload(...)`
**before** calling `.save(...)`, or `.save()` raises `ValueError`:

```python
import pandas as pd

from parsimony_agents import Dataset
from parsimony_agents.execution.outputs import DataFrameObject
from parsimony_agents.identity import ArtifactRef, dataset_logical_id

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

nb_ref = ArtifactRef(kind="notebook", logical_id="analysis", content_sha="abc123")
source_refs: list[ArtifactRef] = []  # upstream data_objects, if any

logical_id = dataset_logical_id(
    notebook_refs=[nb_ref],
    variable_name="results",
    source_refs=source_refs,
)

dataset = Dataset(
    logical_id=logical_id,
    content_sha="",  # computed from the serialized bytes at persist time
    title="Q4 Results",
    description="Quarterly analysis",
    tags=["important"],
    notebook_refs=[nb_ref],
    source_refs=source_refs,
    variable_name="results",
    live_name="q4_results",
).with_payload(DataFrameObject.from_pandas(df, local_dir="/tmp/dfo"))

dataset.save("/tmp/q4_results.parquet")
```

`with_payload()` attaches the payload to the model **in place** and returns the same
`Dataset` (so you can chain `.with_payload(...).save(...)`). The payload is read-only
via the `.payload` property and is never written into the snapshot bytes beyond the
dataframe itself; the curation metadata is embedded alongside.

`Chart.save` works the same way. Its payload is a `FigureObject`, which accepts
either an Altair chart or a raw Vega-Lite dict:

```python
import altair as alt
import pandas as pd

from parsimony_agents import Chart
from parsimony_agents.execution.outputs import FigureObject
from parsimony_agents.identity import ArtifactRef, chart_logical_id

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
fig = alt.Chart(df).mark_line().encode(x="x", y="y")

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
).with_payload(FigureObject(value=fig))

chart.save("/tmp/trend.vl.json")
```

`Report` is different: its body is the plain `markdown` field (a string), so there is
no `with_payload()` step. Instead, `Report.save` calls `snapshot_bytes()`, which
composes YAML frontmatter (title, subtitle, formats, pins) plus the body. An empty
`markdown` raises `ValueError`:

```python
from parsimony_agents import Report
from parsimony_agents.identity import ArtifactRef, report_logical_id

trend_ref = ArtifactRef(kind="chart", logical_id="trend-lid", content_sha="a" * 64)
sales_ref = ArtifactRef(kind="dataset", logical_id="sales-lid", content_sha="b" * 64)
pins = {"trend": trend_ref, "sales": sales_ref}

report = Report(
    logical_id=report_logical_id(
        embedded_refs=[trend_ref, sales_ref],
        title="Q4 2025 Earnings",
    ),
    title="Q4 2025 Earnings",
    subtitle="Revenue beat by 8%",
    markdown="The trend chart shows strong growth.\n",
    formats=["html", "pdf"],
    live_name_pins=pins,
)

report.save("/tmp/earnings.qmd")          # writes YAML frontmatter + body
raw = report.snapshot_bytes()             # same bytes, without touching disk
```

### `.save` rules at a glance

| Method | Payload requirement | Required path suffix | Raises `ValueError` when |
|---|---|---|---|
| `Dataset.save(path)` | `.with_payload(DataFrameObject)` first | `.parquet` | no payload attached, or wrong suffix |
| `Chart.save(path)` | `.with_payload(FigureObject)` first | `.vl.json` | no payload attached, or wrong suffix |
| `Report.save(path)` | none — uses `markdown` | `.qmd` | `markdown` is empty, or wrong suffix |

All three create parent directories as needed (`parents=True, exist_ok=True`).

## Reading back: `read_dataset`, `read_chart`, `read_notebook`

The read functions are the inverse of `.save` and return the curation envelope **plus**
the live payload, so you get both the metadata and something renderable.

`read_dataset` returns a `(Result, Dataset)` tuple — `Result` is the
`parsimony` (parsimony-core) result carrying the live dataframe and provenance, and
`Dataset` is the recovered curation envelope:

```python
from parsimony_agents import read_dataset

result, dataset = read_dataset("/tmp/q4_results.parquet")

print(result.df.shape)          # the live pandas DataFrame
print(dataset.title)            # "Q4 Results"
print(dataset.notebook_refs)    # recovered lineage
print(dataset.variable_name)    # "results"
```

`read_chart` returns a `(Chart, dict)` tuple — the `Chart` curation envelope and the
raw Vega-Lite spec as a plain dict you can hand straight to a renderer:

```python
from parsimony_agents import read_chart

chart, vega_spec = read_chart("/tmp/trend.vl.json")

print(chart.title)              # "Trend Chart"
print(vega_spec["mark"]["type"])  # "line"  (Altair serializes mark as {"type": "line"})
```

`read_notebook` returns a `Script` (plain Python source — there is no metadata block
embedded in a `.py` file):

```python
from parsimony_agents import read_notebook

script = read_notebook("/tmp/analysis.py")
print(script.path)              # e.g. "/tmp/analysis.py"
print(script.code)              # the Python source
```

To write a notebook back out, use `save_notebook(script, path)`. It validates that
the path ends in `.py` and raises `ValueError` otherwise.

> **Vanilla files round-trip too.** A plain Parquet file with no embedded
> `parsimony_agents` metadata deserializes to a populated `Result` with an *empty*
> `Dataset()`. A plain Vega-Lite spec with no `usermeta.parsimony_agents` block
> deserializes to its `dict` plus an empty `Chart()`. The artifacts are
> self-contained: a generic renderer can open them without this library installed.

## Bytes codecs: `serialize_` / `deserialize_` functions

When you are moving artifacts through cloud storage, a queue, or any
bytes-in/bytes-out boundary rather than the filesystem, use the codec pair directly.
These are the same encoders `.save` and the read functions call under the hood:

| Kind | Encode (object → bytes) | Decode (bytes → object) | Wire format |
|---|---|---|---|
| Dataset | `serialize_dataset(dataset, payload)` | `deserialize_dataset(data) -> (Result, Dataset)` | Parquet (curation in Arrow metadata) |
| Chart | `serialize_chart(chart, payload)` | `deserialize_chart(data) -> (Chart, dict)` | `.vl.json` (curation in `usermeta`) |
| Notebook | `serialize_notebook(script)` | `deserialize_notebook(data, *, path=None) -> Script` | plain `.py` source |

`serialize_dataset` and `serialize_chart` are dispatcher-friendly aliases for
`write_dataset_bytes` and `write_chart_bytes`; they take the artifact and its raw
payload (not a `with_payload`-wrapped model):

```python
from parsimony_agents import serialize_dataset, deserialize_dataset

raw: bytes = serialize_dataset(dataset, DataFrameObject.from_pandas(df, local_dir="/tmp/dfo"))

# Later, somewhere else (different process, machine, or object store):
result, recovered = deserialize_dataset(raw)
print(result.df.shape, recovered.title)
```

`deserialize_chart` raises `ValueError("Chart bytes must decode to a Vega-Lite JSON
object.")` if the bytes are not a JSON object. `deserialize_notebook` normalizes
newlines and accepts an optional `path` keyword to stamp the resulting `Script`.

## Notebook output cache (`save_notebook_state` / `load_notebook_state`)

Kernel output (the result of running a notebook) is **not** part of the `.py` snapshot
— it is regenerable, so it lives in a separate content-addressed cache keyed by the
**code SHA**. Editing the notebook changes the SHA and therefore misses the cache,
which is exactly what you want: stale output is never served for changed code.

`save_notebook_state(script, root)` writes `script.output` to
`notebook-state/<code_sha>.json` under `root`. It is a no-op when there is no output
worth caching. `load_notebook_state(script, root)` returns the cached `KernelOutput`
or `None` on a miss:

```python
from pathlib import Path

from parsimony_agents import (
    Script,
    save_notebook_state,
    load_notebook_state,
    notebook_state_cache_key,
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
save_notebook_state(script, root)         # -> /workspace/notebook-state/<sha>.json

recovered = load_notebook_state(script, root)
if recovered is not None:
    print("cache hit:", recovered.outputs)
else:
    print("cache miss or code changed")

print(notebook_state_cache_key(script))   # "notebook-state/<code_sha>.json"
```

`notebook_state_cache_key(script)` returns the canonical relative path,
`notebook-state/<code_sha>.json`, if you need to manage the cache file yourself. The
on-disk envelope carries a `schema_version` of `1`; decoding an unknown version is
treated as a cache miss (returns `None`).

## Virtual paths → canonical `.ockham` (`resolve_virtual_entry`)

The agent reads and writes deliverables by *virtual live-tree paths* — friendly
names like `notebooks/<name>.py`, `data/<name>.parquet`, `charts/<name>.vl.json`, and
`reports/<name>.qmd`. Those bytes do not physically live at those paths. Canonical
storage is content-addressed under `.ockham`:

```
.ockham/notebooks/<logical_id>/<content_sha>.py
.ockham/datasets/<logical_id>/<content_sha>.parquet
.ockham/charts/<logical_id>/<content_sha>.vl.json
.ockham/reports/<logical_id>/<content_sha>.qmd
```

The mapping from a virtual directory to its `(kind, extension)` is the authoritative
`VIRTUAL_LIVE_KINDS` bijection (`notebooks → (notebook, .py)`, `data → (dataset,
.parquet)`, `charts → (chart, .vl.json)`, `reports → (report, .qmd)`).

`resolve_virtual_entry` performs the lookup. Given a materialized workspace directory,
a virtual path, and a `workspace_id`, it returns the canonical `.ockham` path for the
*latest* snapshot of that live name, or `None` if nothing matches:

```python
from pathlib import Path

from parsimony_agents.virtual_path import resolve_virtual_entry

local_dir = Path("/workspace")

canonical = resolve_virtual_entry(
    local_dir,
    "notebooks/analysis.py",
    workspace_id="ws-123",
)

if canonical is not None:
    # e.g. ".ockham/notebooks/analysis/abc123xyz.py"
    print(f"Resolved to: {canonical}")
else:
    # No curation.json with this live_name, or its log.jsonl is empty.
    print("Artifact not found in workspace")
```

Internally it scans `.ockham/<kind>s/*/curation.json` for a `live_name` match, then
reads the most recent `content_sha` from the sibling `log.jsonl` (via
`latest_content_sha`). It is a synchronous, single-pass scan over artifacts of one
kind — fast for the tens-of-artifacts case typical in practice, so call it via
`asyncio.to_thread` if you are inside an async path. The `<name>` segment is treated
as untrusted: `is_safe_name` rejects path traversal, NUL bytes, and hidden files
before any path splicing happens.

## Walking lineage with `enumerate_closure`

Artifacts form a DAG: reports depend on datasets and charts, which depend on
notebooks, datasets, and data objects; notebooks and data objects are leaves. To
collect everything a given artifact transitively depends on (inclusive of itself), use
`enumerate_closure`. It is an `async` function because it must read and deserialize
each snapshot to discover its child refs, so it takes an executor that can read
workspace files:

```python
import asyncio

from parsimony_agents.identity import ArtifactRef
from parsimony_agents.closure import enumerate_closure, child_refs


class WorkspaceExecutor:
    """Read-only adapter over your workspace storage."""

    cwd = None

    async def read_workspace_file(self, path: str) -> bytes:
        # Return the bytes at a `.ockham/...` path from your storage backend.
        raise NotImplementedError


async def main() -> None:
    executor = WorkspaceExecutor()

    report_ref = ArtifactRef(
        kind="report",
        logical_id="earnings-report-lid",
        content_sha="report-csha",
    )

    # Post-order DFS: dependencies appear before their dependents, deduped.
    closure = await enumerate_closure(report_ref, executor=executor)
    for ref in closure:
        print(f"{ref.kind} {ref.logical_id[:8]}… ({ref.content_sha[:8]}…)")

    # One level only — the direct source refs of a single artifact:
    direct = await child_refs(report_ref, executor=executor)
    print("direct children:", direct)


if __name__ == "__main__":
    asyncio.run(main())
```

`child_refs` is the single source of truth for DAG edges (leaves such as notebooks and
data objects return `[]`). `enumerate_closure` returns refs in post-order — leaves
first — deduplicated by `(kind, logical_id, content_sha)`, so identical refs are
emitted exactly once and cycles are safe. This is the building block for tasks such as
packaging a report with all of its inputs, or pruning storage to a reachable set.

`ArtifactRef` itself is a frozen dataclass (`kind`, `logical_id`, `content_sha`) and
knows its own canonical layout — `ref.workspace_file_path` gives the `.ockham` path
for that exact snapshot, which pairs naturally with `executor.read_workspace_file`
when you walk a closure.

## See also

- [Artifacts, identity & lineage](../concepts/artifacts.md) — the dual-identity and snapshot model
- [Code execution](../concepts/code-execution.md) — how notebooks run and produce `KernelOutput`
- [I/O functions reference](../reference/io.md) — full signatures for every function here
- [Artifacts reference](../reference/artifacts.md) — full field listings for `Dataset`, `Chart`, `Report`, `Script`
- [Embedding in a host application](embedding-in-a-host.md) — wiring storage and executors into your own app
