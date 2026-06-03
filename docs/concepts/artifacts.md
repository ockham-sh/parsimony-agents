# Artifacts, identity & lineage

When an agent does analytical work, it produces things worth keeping: a cleaned
dataset, a chart, a written report. Parsimony Agents models each of these as an
**artifact** — a small Pydantic curation envelope (title, description, tags,
lineage) plus a content-addressed snapshot on disk. This page explains the three
artifact kinds (and the notebook that produces them), the dual-identity scheme
that distinguishes *which artifact* from *which version*, the `.ockham` storage
layout, and how lineage is enumerated.

Everything here is built on two ideas that recur throughout:

- **`logical_id` answers "which artifact"; `content_sha` answers "which
  version."** A logical artifact accumulates immutable snapshots over its
  lifetime, each addressed by its own `content_sha`.
- **Curation is separate from payload.** The artifact model carries metadata and
  lineage. The actual data (a DataFrame, a Vega-Lite figure, markdown) is the
  *payload*, and it lives in memory only — never serialized into the snapshot by
  the model itself. Codecs marry the two when writing bytes.

See also [Code execution](code-execution.md) for how notebooks run, and
[Connectors](connectors.md) for where the upstream `data_object` fetches come
from. The full symbol reference lives in
[reference/artifacts.md](../reference/artifacts.md) and
[reference/io.md](../reference/io.md).

## The three artifact kinds (Dataset, Chart, Report) and Script

There are five **snapshot kinds** in the system —
`Literal["notebook", "data_object", "dataset", "chart", "report"]` — but only
three are first-class curated artifacts you'll construct directly:

| Kind | Class | Importable from | Payload | Snapshot extension |
|---|---|---|---|---|
| `dataset` | `Dataset` | `parsimony_agents` | `DataFrameObject` | `.parquet` |
| `chart` | `Chart` | `parsimony_agents` | `FigureObject` (Vega-Lite / Altair) | `.vl.json` |
| `report` | `Report` | `parsimony_agents` | markdown string | `.qmd` |

All three subclass `_ArtifactBase`, which provides the common **curation**
fields:

```python
# _ArtifactBase (shared by Dataset, Chart, Report)
schema_version: int = 2
logical_id: str = ""                  # filled in at compute/persist time
content_sha: str = ""                 # filled in at persist time
title: str = ""
description: str = ""
tags: list[str] = Field(default_factory=list)
notes: list[str] = Field(default_factory=list)
live_name: str | None = None          # None → hidden from the live workspace tree
```

The remaining two kinds are leaves in the lineage graph:

- **`notebook`** — the Python working copy that *produces* artifacts. It's
  modeled by `Script` (the in-memory file: `path`, `code`, `output`,
  `data_objects`), not by `_ArtifactBase`. Notebooks have their own identity
  rules (see [Recipe fields and refresh](#recipe-fields-variable_name-and-refresh-semantics)).
- **`data_object`** — an upstream connector fetch. It has no curation envelope
  class; it's referenced only by `ArtifactRef` and stored in a flat
  content-addressed object pool.

```python
from parsimony_agents import Dataset, Chart, Report, Script
```

A `Script` is the live notebook the kernel runs. Its UI projection is
`ScriptPreview`, which parses the code into displayable `steps`:

```python
from parsimony_agents import Script, ScriptPreview

script = Script(path="notebooks/analysis.py", code="x = 42\nprint(x)")
preview: ScriptPreview = script.to_preview()
```

## Dual identity: `logical_id` vs `content_sha`

Every snapshot is pinned by two strings, and keeping them straight is the key to
the whole model:

- **`logical_id`** is *stable identity*. It answers **"which artifact is this?"**
  For datasets, charts, and reports it's a hash of the artifact's *recipe* — the
  inputs that define it (source refs, the variable name, the title) — so the same
  logical artifact keeps the same `logical_id` even when its underlying data is
  refreshed.
- **`content_sha`** is *version identity*. It answers **"which version is this?"**
  It's the SHA-256 of the serialized snapshot bytes, so any change to the bytes
  produces a new `content_sha`.

```python
from parsimony_agents.identity import content_sha

content_sha(b"some snapshot bytes")  # -> lowercase hex SHA-256 string
```

The `logical_id` of each kind is computed by a dedicated function in
`parsimony_agents.identity`. Each one **sorts its ref inputs** so that call-site
ordering can't change the identity:

```python
from parsimony_agents.identity import (
    dataset_logical_id,    # hash of notebook_refs + variable_name + source_refs
    chart_logical_id,      # hash of notebook_ref + variable_name + source_dataset_refs + source_refs
    report_logical_id,     # hash of embedded_refs + title
    data_object_logical_id,# hash of provenance (excludes fetched_at / properties)
    notebook_logical_id,   # the working-copy basename — NOT a hash (see below)
)
```

Because `report_logical_id` folds in the `title`, two reports that pin the exact
same artifacts but carry different titles get distinct `logical_id`s — they're
genuinely different reports. And because `data_object_logical_id` deliberately
*excludes* `fetched_at` and `properties`, the same upstream series keeps a stable
identity across data refreshes.

## Snapshot vs logical artifact: the `.ockham` layout

A **logical artifact** is the long-lived thing you name and curate. A
**snapshot** is one immutable, content-addressed version of it. As an artifact
evolves, new snapshots accumulate under the same `logical_id` — nothing is ever
mutated in place; a new version simply forks a new `content_sha`.

This maps directly onto the on-disk layout. The canonical store lives under
`.ockham/`:

```
.ockham/
├── notebooks/<logical_id>/<content_sha>.py
├── datasets/<logical_id>/<content_sha>.parquet
├── charts/<logical_id>/<content_sha>.vl.json
├── reports/<logical_id>/<content_sha>.qmd
└── objects/<sha[:2]>/<sha[2:]>.parquet      # immutable data_object pool
```

The versioned path is computed by `ArtifactRef.workspace_file_path`:

```python
# ArtifactRef.workspace_file_path
return f".ockham/{self.kind}s/{self.logical_id}/{self.content_sha}{ext}"
```

Note the `s` — `notebook` → `notebooks/`, `dataset` → `datasets/`, and so on.
Sitting next to the snapshots in each `.ockham/<kind>s/<logical_id>/` directory
are two sibling files:

- **`curation.json`** — *mutable* metadata that points at the latest version
  (this is where `live_name` lives, so the workspace can find the artifact by
  its friendly name).
- **`log.jsonl`** — the *append-only version history*. Each line records a
  snapshot; the last line's `content_sha` is the current version.

You rarely read these by hand. To resolve a friendly live-tree path back to a
canonical snapshot, use `resolve_virtual_entry`:

```python
from pathlib import Path
from parsimony_agents.virtual_path import resolve_virtual_entry

# Agent asks for "notebooks/analysis.py"; resolve to the latest .ockham snapshot.
canonical = resolve_virtual_entry(
    Path("/workspace"),
    "notebooks/analysis.py",
    workspace_id="ws-123",
)
# -> ".ockham/notebooks/analysis/<content_sha>.py", or None if not found
```

To pull just the most recent version out of a `log.jsonl`:

```python
from pathlib import Path
from parsimony_agents.virtual_path import latest_content_sha

latest_content_sha(Path(".ockham/datasets/sales/log.jsonl"))  # -> sha or None
```

The mapping from the agent-facing live tree to canonical kinds and extensions is
fixed by `VIRTUAL_LIVE_KINDS`:

```python
# parsimony_agents.virtual_path.VIRTUAL_LIVE_KINDS
{
    "notebooks": ("notebook", ".py"),
    "data":      ("dataset", ".parquet"),
    "charts":    ("chart", ".vl.json"),
    "reports":   ("report", ".qmd"),
}
```

## Curation envelope vs payload (`.with_payload`, in-process only)

The artifact model is a *curation envelope*. The data it describes — the
DataFrame, the figure — is the **payload**, and the model never serializes it.
On `Dataset` and `Chart` the payload is a Pydantic `PrivateAttr`, readable
through a `@property` but settable only via `with_payload()`, which returns a new
copy:

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
    content_sha="",                 # filled in at persist time
    title="Q4 Results",
    description="Quarterly analysis",
    tags=["important"],
    notebook_refs=[nb_ref],
    source_refs=source_refs,
    variable_name="results",
    live_name="q4_results",
).with_payload(DataFrameObject.from_pandas(df, local_dir="/tmp/dfo"))

assert dataset.payload is not None        # read-only @property
```

Why bother separating them? Because the payload is heavy and the envelope is
cheap. The envelope round-trips through XML for the agent and JSON for the
frontend without ever dragging a DataFrame along. The codec is the only place the
two meet: `Dataset.save()` / `Chart.save()` pull the payload, serialize bytes,
and write the `.parquet` / `.vl.json` file.

A few sharp edges worth knowing:

- `Dataset.save()` and `Chart.save()` **raise `ValueError` if no payload is
  attached** — `with_payload()` is mandatory before saving.
- Accessing `._payload` directly bypasses the typed `@property`. Always use
  `with_payload()` to set and `.payload` to read.
- `Report` is different: its content *is* the `markdown` string on the model, so
  there's no separate payload. `Report.save()` raises `ValueError` on empty
  markdown.

## Lineage: `ArtifactRef`, source refs, `child_refs` and `enumerate_closure`

Artifacts form a **DAG**. A report pins charts and datasets; a chart points at
its source datasets; a dataset points at the notebooks and data_objects that
produced it. Notebooks and data_objects are **leaves**.

The currency of that graph is `ArtifactRef` — a frozen, immutable reference to
exactly one snapshot:

```python
from parsimony_agents.identity import ArtifactRef

ref = ArtifactRef(kind="dataset", logical_id="sales", content_sha="b" * 64)
ref.workspace_file_path      # -> ".ockham/datasets/sales/bbbb...b.parquet"
ref.to_self_closing_tag()    # -> '<ref kind="dataset" logical_id="sales" content_sha="bbbb..."/>'
```

Each artifact declares its edges through typed fields:

- **`Dataset`** — `notebook_refs` (the producing notebooks; multi-notebook
  pipelines are fine) and `source_refs` (upstream data_objects and/or composing
  datasets).
- **`Chart`** — `notebook_ref` (singular), `source_dataset_refs` (plural), and
  `source_refs` (the uncommon path: a chart drawn straight from data_objects,
  bypassing a published dataset).
- **`Report`** — `live_name_pins`, a frozen `live_name → ArtifactRef` map whose
  values become `embedded_refs`.

Rather than have every caller re-discover those field names, `child_refs` is the
**single source of truth for the DAG's edges**. It maps any ref to its immediate
children:

```python
from parsimony_agents.closure import child_refs

# report -> pins; chart -> notebook_ref + source_dataset_refs + source_refs;
# dataset -> notebook_refs + source_refs; notebook / data_object -> []
children = await child_refs(some_ref, executor=executor)
```

Leaves return `[]`. A notebook returns `[]` even though it *ran* connector
fetches — the `fetch_log` recording those `data_object`s is a kernel artifact and
is **not** serialized into the `.py` snapshot. data_objects are still fully
reachable in any closure, just through the `source_refs` of the downstream
dataset or chart that captured them at publish time. Every data_object in a
published lineage shows up in *some* descendant's `source_refs`.

To walk the whole reachable graph, `enumerate_closure` does a **post-order DFS**
— dependencies are emitted before the things that depend on them, every ref
appears exactly once (deduped by the `(kind, logical_id, content_sha)` triple),
and cycles are safe:

```python
import asyncio
from parsimony_agents.identity import ArtifactRef
from parsimony_agents.closure import enumerate_closure


async def main() -> None:
    report_ref = ArtifactRef(
        kind="report",
        logical_id="earnings-report-lid",
        content_sha="report-csha",
    )
    # `executor` is the same workspace executor the agent uses; it must expose
    # read_workspace_file(path) so snapshots can be deserialized.
    closure = await enumerate_closure(report_ref, executor=executor)
    for ref in closure:
        print(ref.kind, ref.logical_id, ref.content_sha)  # leaves first, root last


asyncio.run(main())
```

The closure (inclusive of the root) is exactly the set of bytes you'd need to
reproduce or export an artifact in full — datasets and charts before the report
that embeds them, notebooks and data_objects before the datasets they produced.

## Recipe fields (`variable_name`) and refresh semantics

`variable_name` on `Dataset` and `Chart` is a **recipe field**: it participates
in the `logical_id` hash *and* it's the kernel variable the refresh machinery
re-extracts. It records *which variable in the producing notebook* held the
published value.

Because it's part of the recipe, it cannot be edited after the fact without
changing the artifact's identity — replay semantics depend on it staying fixed.

Refreshing an artifact re-runs its lineage and **appends a new snapshot** under
the same `logical_id`:

```python
from parsimony_agents.refresh import refresh_artifact

new_ref = await refresh_artifact(dataset_ref, executor=executor)
# new_ref shares dataset_ref.logical_id but carries a fresh content_sha
```

The mechanics by kind:

- **dataset** → recurse into source datasets, re-run the producing notebooks
  (which auto-refresh their connector data_objects), re-extract `variable_name`
  from the kernel, persist a new snapshot.
- **chart** → recurse into source datasets, re-run the chart's notebook,
  re-extract, persist.
- **notebook** → *not* refreshable this way; notebooks are working copies
  (re-publish via `return_notebook(execute=True)`).
- **data_object** → refreshes implicitly through the connector layer when its
  producing notebook re-runs.

Refresh is idempotent: if nothing upstream changed, re-extraction produces
identical bytes and therefore the same `content_sha`. Artifacts published without
a `variable_name` (older snapshots) can't be refreshed — `refresh_artifact`
raises.

## Embedded self-describing metadata

A defining property of every snapshot format is that it stays **self-contained**:
a plain renderer with no knowledge of Parsimony Agents can still open the file.
The curation envelope rides along in a sidecar slot that the host format ignores.

| Kind | Format | Where curation lives |
|---|---|---|
| Chart | Vega-Lite JSON | `usermeta.parsimony_agents` |
| Dataset | Parquet / Arrow | Arrow schema `metadata.parsimony_agents` (JSON-encoded) |
| Report | Markdown | YAML frontmatter `parsimony` block |

Charts round-trip the envelope through Vega-Lite's `usermeta` slot:

```python
import json
from parsimony_agents import deserialize_chart

spec_with_meta = {
    "data": {"values": [{"x": 1}, {"x": 2}]},
    "mark": "point",
    "encoding": {"x": {"field": "x", "type": "quantitative"}},
    "usermeta": {
        "parsimony_agents": {
            "type": "chart",
            "schema_version": 2,
            "logical_id": "my-chart-lid",
            "content_sha": "my-chart-csha",
            "title": "Simple Chart",
            "notebook_ref": {
                "kind": "notebook",
                "logical_id": "nb-lid",
                "content_sha": "nb-csha",
            },
            "source_dataset_refs": [],
        }
    },
}

chart, spec = deserialize_chart(json.dumps(spec_with_meta).encode("utf-8"))
assert chart.title == "Simple Chart"
assert spec["mark"] == "point"
```

A *vanilla* Vega-Lite spec with no `usermeta` deserializes cleanly too — you just
get an empty `Chart()`. The same forgiving rule holds for datasets:
`deserialize_dataset(plain_parquet_bytes)` returns a populated `Result` and an
empty `Dataset`.

```python
from parsimony_agents import read_chart, read_dataset

chart, spec = read_chart("/path/to/trend.vl.json")   # -> (Chart, Vega-Lite spec dict)
result, dataset = read_dataset("/path/to/data.parquet")  # -> (parsimony Result, Dataset)
```

Reports compose deterministic YAML frontmatter (formats and pins) followed by the
markdown body. `snapshot_bytes()` is the single source of truth for the on-disk
shape; `parse_snapshot` is its inverse:

```python
from parsimony_agents import Report
from parsimony_agents.identity import ArtifactRef, report_logical_id
from parsimony_agents.report_format import parse_snapshot

trend_ref = ArtifactRef(kind="chart", logical_id="trend-lid", content_sha="a" * 64)
pins = {"trend": trend_ref}

report = Report(
    logical_id=report_logical_id(embedded_refs=[trend_ref], title="Q4 2025 Earnings"),
    title="Q4 2025 Earnings",
    subtitle="Revenue beat by 8%",
    markdown="The trend chart shows strong growth.\n",
    formats=["html", "pdf"],
    live_name_pins=pins,
)

text = report.snapshot_bytes().decode("utf-8")  # YAML frontmatter + blank line + body
parsed = parse_snapshot(text)
assert parsed.title == "Q4 2025 Earnings"
assert "trend" in parsed.pins
```

`Report.embedded_refs` is derived on the fly from `markdown` + `live_name_pins`
(not stored); empty markdown or empty pins yields `[]`.

## Notebook identity vs content

Notebooks are the one kind whose identity is *not* a hash of inputs. A
notebook's **`logical_id` IS its `live_name`** — the basename of the working-copy
path:

```python
from parsimony_agents.identity import notebook_logical_id, notebook_content_sha

notebook_logical_id("notebooks/analysis.py")  # -> "analysis"
notebook_content_sha("x = 1\n")               # -> SHA-256 of the source bytes
```

This is the git model. Editing a notebook produces a new `content_sha` (the hash
of the `.py` source, with trailing whitespace stripped for round-trip
invariance) appended under the same `logical_id`. **Renaming** a notebook creates
a brand-new `logical_id` with a fresh log — the old snapshots stay reachable
under the old name in `.ockham/notebooks/<old_name>/`. The other kinds work the
opposite way: their `logical_id` is a hash of the *recipe*, and their
`content_sha` is the hash of the produced *content*.

Notebook execution output is cached separately in a *regenerable* (not
authoritative) content-addressed store, keyed by the code SHA so an edit
invalidates it:

```python
from pathlib import Path
from parsimony_agents import save_notebook_state, load_notebook_state

save_notebook_state(script, Path("/workspace"))     # -> notebook-state/<code_sha>.json
cached = load_notebook_state(script, Path("/workspace"))  # KernelOutput or None on miss
```

## The chart data pool (`split_chart_data` / `inline_chart_data`)

Vega-Lite specs inline their plotted data as a JSON array. For a large series
that bloats every snapshot, and re-styling a chart (same data, new color) would
needlessly rewrite all those rows. The chart data pool fixes this by
**de-inlining** plotted data into a separate content-addressed pool, leaving the
spec carrying only a small marker (`__parsimony_chart_data_ref__`):

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

deinlined_spec, pool = split_chart_data(spec)   # pool: {content_sha -> bytes}
refs = chart_data_refs(deinlined_spec)          # every pool sha the spec points at

full_spec = inline_chart_data(deinlined_spec, pool)   # inverse
assert full_spec["data"]["values"] == spec["data"]["values"]
```

Now restyling a chart costs a single small spec snapshot and **zero new pool
bytes** — the unchanged data already lives in the pool under its existing
`content_sha`. De-inlining is purely a storage optimization:
`write_chart_bytes` (aliased as `serialize_chart`) still produces a fully
self-contained `.vl.json`, and a missing pool entry degrades gracefully (a
dropped data series) rather than failing hard.

---

**Related reading**

- [Code execution](code-execution.md) — how the kernel runs notebooks and
  captures output.
- [Connectors](connectors.md) — where upstream `data_object` fetches originate.
- [Saving and loading artifacts](../guides/saving-loading-artifacts.md) —
  task-oriented walkthrough.
- [reference/artifacts.md](../reference/artifacts.md) and
  [reference/io.md](../reference/io.md) — exhaustive symbol reference.
