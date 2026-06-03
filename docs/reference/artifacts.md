# Artifacts reference

Field-level reference for the artifact models — `Dataset`, `Chart`, `Report`, `Script`,
`ScriptPreview` — plus `ArtifactRef`, the `SnapshotKind` union, and the identity functions that
compute logical IDs and content hashes.

For the conceptual model behind these types (dual identity, snapshots, lineage, the `.ockham`
layout), read [Artifacts, identity & lineage](../concepts/artifacts.md) first. For the
read/write codecs (`read_dataset`, `read_chart`, `deserialize_*`, `save_notebook`, …) see the
[I/O functions reference](io.md).

All user-facing models import from the top-level package:

```python
from parsimony_agents import Dataset, Chart, Report, Script, ScriptPreview
```

`ArtifactRef`, `SnapshotKind`, `LiveNameCollisionError`, and every identity function live in the
`parsimony_agents.identity` module; `VIRTUAL_LIVE_KINDS` lives in `parsimony_agents.virtual_path`.

---

## Shared curation envelope

`Dataset`, `Chart`, and `Report` all extend an internal base that carries the same identity and
curation fields. Every one of these models has:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | `int` | `2` | Version of the embedded-metadata schema. |
| `logical_id` | `str` | `""` | Stable identity — "which artifact." Computed from the relevant identity function; empty until assigned. |
| `content_sha` | `str` | `""` | Hash of the serialized snapshot bytes — "which version." Computed at persist time. |
| `title` | `str` | `""` | Human title (curation). |
| `description` | `str` | `""` | Longer prose description (curation). |
| `tags` | `list[str]` | `[]` | Free-form tags (curation). |
| `notes` | `list[str]` | `[]` | Bullet notes (curation). |
| `live_name` | `str \| None` | `None` | Workspace-tree slug. `None` hides the artifact from the live tree; an empty string is mapped to `slug_from_title(title)` at persist time by the registry. |

These are Pydantic models that act as **curation envelopes** — the lineage and metadata around a
payload, not the payload itself. The actual data (a `DataFrameObject` for a dataset, a
`FigureObject` for a chart, the markdown body for a report) is attached separately and is never
re-serialized blindly: codecs pull it out at persist time.

---

## Dataset

```python
class Dataset(_ArtifactBase):
    type: Literal["dataset"] = "dataset"
    notebook_refs: list[ArtifactRef]   # producing notebooks
    source_refs: list[ArtifactRef]     # upstream data_objects / composing datasets
    variable_name: str                 # kernel variable name (recipe field)
    # _payload: DataFrameObject | None  (PrivateAttr, in-process only)
```

A curated, published dataframe. On disk it lives at
`.ockham/datasets/<logical_id>/<content_sha>.parquet` with sibling `curation.json` and
`log.jsonl`.

### Dataset fields

| Field | Type | Notes |
| --- | --- | --- |
| `notebook_refs` | `list[ArtifactRef]` | Refs to the notebook(s) that produced the dataset. Multi-notebook pipelines are allowed. |
| `source_refs` | `list[ArtifactRef]` | Refs to upstream `data_object`s and/or composing datasets. |
| `variable_name` | `str` | The kernel variable the agent extracted. A **recipe field**: it participates in `dataset_logical_id` and is required for refresh to re-extract from the kernel. The PATCH endpoint rejects edits to preserve replay semantics. |

Plus all the [shared curation fields](#shared-curation-envelope).

### Dataset methods

- **`with_payload(payload: DataFrameObject) -> Dataset`** — attaches the in-process dataframe
  payload and returns the same instance (for chaining). Raises `TypeError` if `payload` is not a
  `DataFrameObject`. The payload is stored on a private attribute and is **never serialized into
  the model** — it is only read by the codec at `.save()` time.
- **`payload -> DataFrameObject | None`** — read-only property exposing the attached payload
  (`None` until `with_payload` is called).
- **`to_llm(mode="default") -> list[dict[str, Any]]`** — renders the artifact as compact XML text
  blocks for the agent: a `<dataset title="…">` wrapper with optional `<description>`/`<notes>`
  and a `<sources>` block of `source_refs`.
- **`to_frontend_dict() -> dict[str, Any]`** — flat JSON-serializable dict for a host UI:
  `type`, `schema_version`, `logical_id`, `content_sha`, `title`, `description`, `notes`, `tags`,
  `live_name`, `notebook_refs` (as dicts), `source_refs` (as dicts), and `variable_name`.
- **`save(path: str | Path) -> None`** — writes the Parquet snapshot via
  `write_dataset_bytes`. Raises `ValueError` if no payload is attached, or if `path` does not end
  in `.parquet`. Parent directories are created.

```python
import pandas as pd
from parsimony_agents import Dataset
from parsimony_agents.execution.outputs import DataFrameObject
from parsimony_agents.identity import ArtifactRef, dataset_logical_id

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

nb_ref = ArtifactRef(kind="notebook", logical_id="analysis", content_sha="a" * 64)
source_refs: list[ArtifactRef] = []

logical_id = dataset_logical_id(
    notebook_refs=[nb_ref],
    variable_name="results",
    source_refs=source_refs,
)

dataset = (
    Dataset(
        logical_id=logical_id,
        title="Q4 Results",
        description="Quarterly analysis",
        tags=["important"],
        notebook_refs=[nb_ref],
        source_refs=source_refs,
        variable_name="results",
        live_name="q4_results",
    )
    .with_payload(DataFrameObject.from_pandas(df, local_dir="/tmp/dfo"))
)

dataset.save("/tmp/q4.parquet")
```

> `content_sha` is computed from the serialized bytes at persist time — you do not need to set it
> when constructing the model.

---

## Chart

```python
class Chart(_ArtifactBase):
    type: Literal["chart"] = "chart"
    notebook_ref: ArtifactRef | None           # singular — one rendering notebook
    source_dataset_refs: list[ArtifactRef]      # plural — multi-dataset charts
    source_refs: list[ArtifactRef]              # rare: drawn straight from data_objects
    variable_name: str                          # kernel variable (recipe field)
    # _payload: FigureObject | None  (PrivateAttr, in-process only)
```

A curated, published chart. On disk: `.ockham/charts/<logical_id>/<content_sha>.vl.json` with
sibling curation/log files. The payload is a `FigureObject`, which accepts either an Altair chart
or a raw Vega-Lite spec dict.

### Chart fields

| Field | Type | Notes |
| --- | --- | --- |
| `notebook_ref` | `ArtifactRef \| None` | **Singular** — a chart is rendered in exactly one notebook. Required at persist time; optional on the model so vanilla `.vl.json` round-trips without curation don't fail on construction. |
| `source_dataset_refs` | `list[ArtifactRef]` | Refs to source datasets. Plural — multi-dataset comparison charts welcome. |
| `source_refs` | `list[ArtifactRef]` | Refs to upstream `data_object`s for the uncommon case of a chart drawn straight from data, bypassing an intermediate dataset. |
| `variable_name` | `str` | The kernel variable. A **recipe field** (same semantics as `Dataset.variable_name`): participates in `chart_logical_id`, required for refresh. |

Plus all the [shared curation fields](#shared-curation-envelope).

### Chart methods

- **`with_payload(payload: FigureObject) -> Chart`** — attaches the in-process figure and returns
  `self`.
- **`payload -> FigureObject | None`** — read-only property.
- **`to_llm(mode="default") -> list[dict[str, Any]]`** — XML text blocks: a `<chart title="…">`
  wrapper with optional `<description>`/`<notes>`, a `<source_datasets>` block of
  `source_dataset_refs`, and a `<sources>` block of `source_refs`.
- **`to_frontend_dict() -> dict[str, Any]`** — flat UI dict including `notebook_ref`,
  `source_dataset_refs`, `source_refs`, `variable_name`, and the shared identity/curation fields.
- **`save(path: str | Path) -> None`** — writes the `.vl.json` snapshot. Raises `ValueError` if
  no payload is attached.

```python
import altair as alt
import pandas as pd
from parsimony_agents import Chart
from parsimony_agents.execution.outputs import FigureObject
from parsimony_agents.identity import ArtifactRef, chart_logical_id

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
fig = alt.Chart(df).mark_line().encode(x="x", y="y")

nb_ref = ArtifactRef(kind="notebook", logical_id="analysis", content_sha="a" * 64)
ds_refs = [ArtifactRef(kind="dataset", logical_id="ds-lid", content_sha="b" * 64)]

logical_id = chart_logical_id(
    notebook_ref=nb_ref,
    chart_variable_name="trend_chart",
    source_dataset_refs=ds_refs,
    source_refs=[],
)

chart = Chart(
    logical_id=logical_id,
    title="Trend Chart",
    notebook_ref=nb_ref,
    source_dataset_refs=ds_refs,
    variable_name="trend_chart",
).with_payload(FigureObject(value=fig))

chart.save("/tmp/trend.vl.json")
```

---

## Report

```python
class Report(_ArtifactBase):
    type: Literal["report"] = "report"
    markdown: str = ""
    subtitle: str = ""
    formats: list[str] = ["html"]          # default_factory
    live_name_pins: dict[str, ArtifactRef] = {}
    # embedded_refs is a derived property, not a stored field
```

A user-readable markdown deliverable. On disk: `.ockham/reports/<logical_id>/<content_sha>.qmd` —
a valid Quarto document with a YAML frontmatter block carrying `formats` and `pins`. Because the
frontmatter participates in `content_sha`, changing the format list or pin map forks a new
snapshot under the same `logical_id`.

### Report fields

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `markdown` | `str` | `""` | The report body. Must be non-empty to serialize (see `snapshot_bytes`). |
| `subtitle` | `str` | `""` | Optional secondary line under the title. Empty means no subtitle is rendered. |
| `formats` | `list[str]` | `["html"]` | Output formats requested at publish time (e.g. `html`, `pdf`, `pptx`, `dashboard`, `revealjs`). Order is preserved on serialization. |
| `live_name_pins` | `dict[str, ArtifactRef]` | `{}` | Frozen `live_name → ArtifactRef` map. The body's `file://./<dir>/<live_name>.<ext>` URIs resolve against this snapshot-local map, so old reports stay byte-stable even when artifacts are later renamed. It is the single source of truth for embedded refs. |

Plus all the [shared curation fields](#shared-curation-envelope). Reports have no separate
payload attribute — the markdown body *is* the payload.

### Report members

- **`embedded_refs -> list[ArtifactRef]`** — a derived (computed) property, **not stored**. It is
  derived from `markdown` plus `live_name_pins`, in body order and deduped. If either `markdown`
  or `live_name_pins` is empty, it returns `[]`.
- **`to_llm(mode="default") -> list[dict[str, Any]]`** — XML text blocks: a `<report title="…">`
  wrapper with optional `<description>`/`<notes>` and an `<embedded>` block of `embedded_refs`.
- **`to_frontend_dict() -> dict[str, Any]`** — flat UI dict including `subtitle`, `formats`,
  `embedded_refs` (as dicts), `live_name_pins` (as a `{live_name: ref_dict}` map), and the shared
  fields.
- **`snapshot_bytes() -> bytes`** — the canonical on-disk shape: deterministic YAML frontmatter
  (title + subtitle + formats + pins) followed by the body, UTF-8 encoded. This is the **single
  source of truth** for what hits disk and what gets hashed into `content_sha`. Raises
  `ValueError` if the markdown is empty/whitespace-only.
- **`save(path: str | Path) -> None`** — writes `snapshot_bytes()` to disk. Raises `ValueError`
  unless `path` ends in `.qmd`. Parent directories are created.

```python
from parsimony_agents import Report
from parsimony_agents.identity import ArtifactRef, report_logical_id

trend_ref = ArtifactRef(kind="chart", logical_id="trend-lid", content_sha="a" * 64)
sales_ref = ArtifactRef(kind="dataset", logical_id="sales-lid", content_sha="b" * 64)
pins = {"trend": trend_ref, "sales": sales_ref}

logical_id = report_logical_id(
    embedded_refs=[trend_ref, sales_ref],
    title="Q4 2025 Earnings",
)

report = Report(
    logical_id=logical_id,
    title="Q4 2025 Earnings",
    subtitle="Revenue beat by 8%",
    markdown="The trend chart shows strong growth.\n",
    formats=["html", "pdf"],
    live_name_pins=pins,
)

snapshot = report.snapshot_bytes()   # YAML frontmatter + body
report.save("/tmp/earnings.qmd")
```

Round-trip the snapshot with `parse_snapshot` / `compose_snapshot` — see the
[I/O functions reference](io.md).

---

## Script and ScriptPreview

A `Script` is a workspace notebook file — a path, its Python source, and any kernel output from
the last run. Its identity is the workspace path (not a hash of inputs like the curated
artifacts). Both import from the top level:

```python
from parsimony_agents import Script, ScriptPreview
```

The default notebook path is `notebooks/main.py`.

### Script

```python
class Script(BaseModel):
    type: Literal["script"] = "script"
    path: str = "notebooks/main.py"
    code: str = ""
    output: KernelOutput               # default: empty KernelOutput
    data_objects: list[FetchLogEntry]  # default: []
```

| Field | Type | Notes |
| --- | --- | --- |
| `path` | `str` | Workspace path, e.g. `notebooks/<name>.py`. Defaults to `notebooks/main.py`. |
| `code` | `str` | Full Python source. |
| `output` | `KernelOutput` | Cell results from a kernel run. Set after execution; empty otherwise. |
| `data_objects` | `list[FetchLogEntry]` | Connector fetches recorded during the run. Set after a kernel run for UI previews. |

Execution is not implicit — `output` and `data_objects` are populated only after a kernel run
(e.g. via the `return_notebook` / `edit_notebook` agent tools with `execute=True`).

**Methods:**

- **`to_preview() -> ScriptPreview`** — projects to a UI-oriented `ScriptPreview`. It scans
  `output.outputs` for the first exception and surfaces its first line as `error_message`, copies
  `data_objects`, and includes `output` only when there is actually something to show (outputs or
  a fetch log).
- **`to_frontend_dict() -> dict[str, Any]`** — JSON-serializable form, equivalent to
  `to_preview().model_dump(mode="json")`.

### ScriptPreview

```python
class ScriptPreview(BaseModel):
    type: Literal["script_preview"] = "script_preview"
    path: str = "notebooks/main.py"
    code: str
    error_message: str | None = None
    data_objects: list[FetchLogEntry]   # default: []
    output: KernelOutput | None = None
    ui_message: str | None = None
    # steps: computed property
```

| Field | Type | Notes |
| --- | --- | --- |
| `path` | `str` | Workspace path. |
| `code` | `str` | Python source (required). |
| `error_message` | `str \| None` | First line of the first exception in the run, if any. |
| `data_objects` | `list[FetchLogEntry]` | Connector fetches. |
| `output` | `KernelOutput \| None` | Present only when there is output worth showing. |
| `ui_message` | `str \| None` | Optional non-technical detail after `>` in Created/… labels (used by `return_notebook`, not `edit_notebook`). |

**Computed property:**

- **`steps -> list[ScriptStepPreview]`** — a `@computed_field` derived from `code` by parsing the
  comment/code structure into ordered steps. It serializes alongside the other fields.

```python
from parsimony_agents import Script
from parsimony_agents.execution.outputs import KernelOutput, PrimitiveObject

script = Script(
    path="notebooks/analysis.py",
    code="x = 42\nprint(x)",
    output=KernelOutput(
        outputs=[PrimitiveObject(value=42, type="int")],
        fetch_log=[],
    ),
)

preview = script.to_preview()
print(preview.path)           # notebooks/analysis.py
print(preview.error_message)  # None (no exception in outputs)
print(preview.steps)          # parsed ScriptStepPreview list
```

See the [I/O functions reference](io.md) for `read_notebook`, `serialize_notebook`,
`save_notebook`, and the notebook-state cache helpers.

---

## ArtifactRef and SnapshotKind

```python
from parsimony_agents.identity import ArtifactRef, SnapshotKind
```

`SnapshotKind` is the union of the five artifact kinds:

```python
SnapshotKind = Literal["notebook", "data_object", "dataset", "chart", "report"]
```

`ArtifactRef` is a **frozen dataclass** — immutable after construction — that pins exactly one
`content_sha` of one logical artifact:

```python
@dataclass(frozen=True)
class ArtifactRef:
    kind: SnapshotKind
    logical_id: str
    content_sha: str
```

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | `SnapshotKind` | Which kind of artifact this references. |
| `logical_id` | `str` | "Which artifact" — the stable identity. |
| `content_sha` | `str` | "Which snapshot" — the version. |

Construction validates eagerly: an unsupported `kind`, or an empty `logical_id`/`content_sha`,
raises `ValueError`.

### ArtifactRef members

- **`workspace_file_path -> str`** — the workspace-relative on-disk path for this snapshot.
  Versioned kinds resolve to `.ockham/<kind>s/<logical_id>/<content_sha>.<ext>`, where the
  extension is `.py` (notebook), `.parquet` (dataset/data_object), `.vl.json` (chart), or `.qmd`
  (report). A `data_object` resolves to its immutable object-pool path (see
  [`object_pool_path`](#object_pool_path)) — addressed only by `content_sha`.
- **`to_dict() -> dict[str, str]`** — `{"kind", "logical_id", "content_sha"}`, e.g. for
  `log.jsonl` rows or JSON wire.
- **`from_dict(data) -> ArtifactRef`** (classmethod) — inverse of `to_dict`.
- **`from_workspace_file_path(path) -> ArtifactRef | None`** (classmethod) — parses a canonical
  `.ockham/...` path back into a ref. Returns `None` for any path outside the canonical layout,
  giving callers a clean miss signal.
- **`to_xml_attrs() -> str`** — the inline attribute fragment
  `kind="…" logical_id="…" content_sha="…"`, for composing tags that carry extra attributes.
- **`to_self_closing_tag(tag="ref") -> str`** — a self-closing tag
  `<{tag} kind="…" logical_id="…" content_sha="…"/>`. The default `tag="ref"` matches the generic
  `<ref/>` lineage form; pass a more specific tag (`"notebook_ref"`, `"data_object_ref"`) where it
  helps the agent.

```python
from parsimony_agents.identity import ArtifactRef

ref = ArtifactRef(kind="dataset", logical_id="sales-lid", content_sha="b" * 64)

ref.workspace_file_path          # .ockham/datasets/sales-lid/bbbb…bbbb.parquet
ref.to_dict()                    # {"kind": "dataset", "logical_id": "sales-lid", ...}
ref.to_self_closing_tag()        # <ref kind="dataset" logical_id="sales-lid" .../>
ref.to_xml_attrs()               # kind="dataset" logical_id="sales-lid" content_sha="…"

roundtrip = ArtifactRef.from_workspace_file_path(ref.workspace_file_path)
assert roundtrip == ref          # frozen dataclasses compare by value
```

---

## Identity functions

All identity helpers live in `parsimony_agents.identity`:

```python
from parsimony_agents.identity import (
    content_sha,
    notebook_content_sha,
    notebook_logical_id,
    dataset_logical_id,
    chart_logical_id,
    report_logical_id,
    data_object_logical_id,
    object_pool_path,
    slug_from_title,
)
```

The dual-identity split is the load-bearing idea: a **`logical_id`** answers "which artifact"
(stable across data refreshes, derived from a recipe) and a **`content_sha`** answers "which
version" (the hash of the serialized bytes). See
[Artifacts, identity & lineage](../concepts/artifacts.md) for the full model.

### content_sha

```python
def content_sha(blob: bytes) -> str: ...
```

SHA-256 of `blob`, returned as a lowercase hex string. This is the generic snapshot-hashing
primitive used for dataset/chart/report bytes.

### notebook_content_sha

```python
def notebook_content_sha(code: str) -> str: ...
```

SHA-256 of a notebook's UTF-8 source bytes, **with trailing whitespace stripped** so the hash is
invariant under the serialize → deserialize round-trip (on-disk files gain a trailing newline;
parsed source has it stripped). This is the canonical `content_sha` for a notebook snapshot — it
is *not* the notebook's `logical_id`.

### notebook_logical_id

```python
def notebook_logical_id(path: str) -> str: ...
```

Derives a notebook's `logical_id` from its working-copy path: `notebooks/foo.py → "foo"`.
Notebooks are special — **the live name IS the logical_id**, not a hash of inputs. Renaming a
notebook produces a new `logical_id` and a fresh log (git-style); old snapshots stay reachable
under the old name. The path must start with `notebooks/`, be flat (no subdirectories), and end in
`.py`; otherwise `ValueError`.

### dataset_logical_id

```python
def dataset_logical_id(
    *,
    notebook_refs: list[ArtifactRef],
    variable_name: str,
    source_refs: list[ArtifactRef],
) -> str: ...
```

Hashes a dataset's recipe: producing notebooks, `variable_name`, and source refs.
`notebook_refs` and `source_refs` are sorted by `logical_id` so call-site ordering does not
perturb identity, and the notebook **`logical_id`** (not its `content_sha`) is hashed — so notebook
edits append a new snapshot under the unchanged dataset `logical_id` rather than forking a new
artifact. Raises `ValueError` if `variable_name` is empty.

### chart_logical_id

```python
def chart_logical_id(
    *,
    notebook_ref: ArtifactRef,
    chart_variable_name: str,
    source_dataset_refs: list[ArtifactRef],
    source_refs: list[ArtifactRef],
) -> str: ...
```

Hashes a chart's recipe: rendering notebook, `chart_variable_name`, source datasets, and source
refs. `source_dataset_refs` and `source_refs` are sorted by `logical_id`. Raises `ValueError` if
`notebook_ref.kind != "notebook"` or if `chart_variable_name` is empty.

### report_logical_id

```python
def report_logical_id(
    *,
    embedded_refs: list[ArtifactRef],
    title: str,
) -> str: ...
```

Hashes a report's identity: its embedded refs (sorted by `logical_id`) plus the `title`. The
title participates so that two distinct reports referencing the same artifact set get distinct
`logical_id`s. Raises `ValueError` if `title` is empty.

### data_object_logical_id

```python
def data_object_logical_id(provenance: Any) -> str: ...
```

Hashes a data object's provenance, **excluding `fetched_at` and `properties`**. The result is
stable across data refreshes: the same source plus the same parameters yields the same
`logical_id`, regardless of when the fetch happened or what bytes came back.

### object_pool_path

```python
def object_pool_path(content_sha: str) -> str: ...
```

Workspace-relative path for an immutable object-pool Parquet entry:
`.ockham/objects/<sha[:2]>/<sha[2:]>.parquet`. Data objects are content-addressed only — there is
no `logical_id` segment in the path. Raises `ValueError` if `content_sha` is shorter than 3 hex
chars.

### slug_from_title

```python
def slug_from_title(text: str, max_len: int = 40) -> str: ...
```

ASCII-folds a title into a lowercase snake_case slug: Unicode is NFKD-normalized then ASCII-folded,
non-alphanumeric runs collapse to `_`, leading/trailing `_` is stripped, and the result is capped
at `max_len` (default 40). Empty or all-non-ASCII input yields `"untitled"`. Used to derive
`live_name` defaults at persist time.

```python
from parsimony_agents.identity import slug_from_title

slug_from_title("Q4 2025 Earnings")      # "q4_2025_earnings"
slug_from_title("US-GDP, 2020-2024")     # "us_gdp_2020_2024"
slug_from_title("ñ café")                # "n_cafe"   (ASCII folding)
slug_from_title("   ")                    # "untitled"
slug_from_title("a" * 50)                # "a" * 40   (capped)
```

---

## LiveNameCollisionError and VIRTUAL_LIVE_KINDS

### LiveNameCollisionError

```python
from parsimony_agents.identity import LiveNameCollisionError

class LiveNameCollisionError(Exception):
    def __init__(
        self,
        *,
        live_name: str,
        existing_logical_id: str,
        kind: SnapshotKind = "notebook",
    ) -> None: ...
```

Raised when a `live_name` already belongs to another terminal's artifact in the same workspace —
i.e. a write or refresh would silently coalesce with an artifact this terminal has never
interacted with. Three keyword-only parameters are load-bearing and exposed as attributes:

| Attribute | Meaning |
| --- | --- |
| `live_name` | The slug both halves of the agent surface share — what the agent typed, and what it must use to read. |
| `existing_logical_id` | The colliding artifact's `logical_id`. Retrying with the same `live_name` after reading returns this value (continuation), not a fresh slug. |
| `kind` | Which artifact kind collided. The seen-set is keyed on `(kind, live_name)`. Defaults to `"notebook"`. |

The recovery loop is encoded in the exception message: read the existing artifact first (to bring
it into this terminal's seen-set), then re-issue the write — or pick a different `live_name`.

### VIRTUAL_LIVE_KINDS

```python
from parsimony_agents.virtual_path import VIRTUAL_LIVE_KINDS
```

Maps each virtual live-tree directory to its canonical `(kind, extension)` pair:

```python
VIRTUAL_LIVE_KINDS: Final[dict[str, tuple[str, str]]] = {
    "notebooks": ("notebook", ".py"),
    "data":      ("dataset", ".parquet"),
    "charts":    ("chart", ".vl.json"),
    "reports":   ("report", ".qmd"),
}
```

This is the source of truth that connects the agent-facing virtual paths (`notebooks/<name>.py`,
`data/<name>.parquet`, `charts/<name>.vl.json`, `reports/<name>.qmd`) to the canonical `.ockham`
storage layout. Note the agent-facing directory for datasets is `data/`, while the canonical
storage directory is `.ockham/datasets/`. It is used by `resolve_virtual_entry` to map a live-tree
path back to its `.ockham/<kind>s/<logical_id>/<content_sha>` snapshot — see the
[I/O functions reference](io.md).

---

## See also

- [Artifacts, identity & lineage](../concepts/artifacts.md) — the conceptual model.
- [I/O functions reference](io.md) — codecs, snapshot serialization, virtual-path resolution, and
  closure traversal.
- [Code execution](../concepts/code-execution.md) — how `KernelOutput` and `FetchLogEntry` are
  produced.
- [Saving and loading artifacts](../guides/saving-loading-artifacts.md) — task-oriented guide.
