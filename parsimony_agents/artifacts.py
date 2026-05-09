"""Published deliverables (datasets, charts, reports) — content-addressed identity.

Datasets, Charts, and Reports follow the dual-identity model defined in
``CONTENT_ADDRESSED_ARTIFACTS_PLAN.md`` §2.1:

- ``logical_id`` — "Which artifact is this?" derived from inputs minus
  their content. Stable across data refreshes. Computed from the
  per-kind formulas in :mod:`parsimony_agents.identity` (§2.2).
- ``content_sha`` — "What does it currently look like?" hash of the
  rendered bytes. Computed at persist time by the framework, not the
  agent.

Each artifact carries lineage as :class:`~parsimony_agents.identity.ArtifactRef`
values pointing at frozen snapshots of upstream artifacts:

- Datasets reference notebooks (multi-notebook pipelines OK) and
  upstream data_objects.
- Charts reference one notebook + N source datasets. ``source_refs``
  remains for the uncommon "chart from raw data_objects without an
  intermediate dataset" case.
- Reports reference any embedded artifact (dataset, chart, or other
  report) — frozen by default so the report stays reproducible.

Curation (title, description, tags, notes, live_name) is embedded in
the artifact for in-process use and ALSO mirrored to a sidecar
``.ockham/<kind>/<logical_id>/curation.json`` for editable, identity-stable
renames. The embedded form is **frozen at persist time** (§5.9): a
later rename of the title bumps the sidecar but never rewrites bytes.

Payload contract (single, typed):

- ``Dataset._payload: DataFrameObject | None`` — executor wrapper for a
  pandas DataFrame.
- ``Chart._payload: FigureObject | None`` — executor wrapper for an
  Altair chart or Vega-Lite spec.
- ``Report.markdown: str`` — the markdown source itself (no executor
  payload — reports are markdown bytes, not in-kernel objects).
"""

from __future__ import annotations

__all__ = [
    "Chart",
    "Dataset",
    "Report",
]

from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from parsimony_agents._naming import slug_from_title
from parsimony_agents.agent.xml_render import escape_attr, escape_text
from parsimony_agents.execution.outputs import DataFrameObject, FigureObject
from parsimony_agents.identity import ArtifactRef, ExportFormat
from parsimony_agents.messages import MessageContent


# ============================================================================
# Shared identity + payload mechanics
# ============================================================================


class _ArtifactBase(MessageContent):
    """Identity, curation, and lineage shared by Dataset / Chart / Report."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 2
    logical_id: str = ""
    content_sha: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    #: ``None`` → hidden from the live workspace tree (§4.6).
    #: Empty string sentinel mapped to a slugged default at persist time
    #: by the artifact registry.
    live_name: str | None = None


# ============================================================================
# Dataset
# ============================================================================


class Dataset(_ArtifactBase):
    """Curation + lineage for a published dataset.

    Lives on disk at ``.ockham/datasets/<logical_id>/<content_sha>.parquet``
    with a sibling ``curation.json`` and ``log.jsonl``. The Pydantic
    model is the in-process projection; the codec
    :func:`~parsimony_agents.dataset_io.write_dataset_bytes` embeds it
    in arrow metadata for portability outside the workspace.
    """

    type: Literal["dataset"] = "dataset"

    notebook_refs: list[ArtifactRef] = Field(
        default_factory=list,
        description="Frozen refs to notebooks used to produce this dataset (multi-notebook pipelines OK).",
    )
    source_refs: list[ArtifactRef] = Field(
        default_factory=list,
        description="Frozen refs to upstream data_objects (and/or composing datasets).",
    )
    variable_name: str = Field(
        default="",
        description=(
            "Kernel variable name the agent extracted to produce this dataset. "
            "Recipe field — participates in ``logical_id`` and is required for "
            "``refresh`` to re-extract from the kernel after re-running the "
            "producing notebook. Persisted in arrow ``usermeta`` and curation; "
            "the PATCH endpoint rejects edits."
        ),
    )

    _payload: DataFrameObject | None = PrivateAttr(default=None)

    def with_payload(self, payload: DataFrameObject) -> Dataset:
        if not isinstance(payload, DataFrameObject):
            raise TypeError(
                f"Dataset.with_payload expects a DataFrameObject; got "
                f"{type(payload).__name__}. Wrap raw frames with "
                f"DataFrameObject.from_pandas(df, local_dir=...)."
            )
        self._payload = payload
        return self

    @property
    def payload(self) -> DataFrameObject | None:
        return self._payload

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        title = self.title or "(untitled)"
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f'<dataset title="{escape_attr(title)}">\n'},
        ]
        if self.description:
            blocks.append(
                {"type": "text", "text": f"<description>{escape_text(self.description)}</description>\n"}
            )
        if self.notes:
            blocks.append({"type": "text", "text": "<notes>\n"})
            blocks.extend({"type": "text", "text": f"- {escape_text(note)}\n"} for note in self.notes)
            blocks.append({"type": "text", "text": "</notes>\n"})
        blocks.extend(_refs_blocks("sources", self.source_refs))
        blocks.append({"type": "text", "text": "</dataset>\n"})
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "content_sha": self.content_sha,
            "title": self.title,
            "description": self.description,
            "notes": list(self.notes),
            "tags": list(self.tags),
            "live_name": self.live_name,
            "notebook_refs": [r.to_dict() for r in self.notebook_refs],
            "source_refs": [r.to_dict() for r in self.source_refs],
            "variable_name": self.variable_name,
        }

    def save(self, path: str | Path) -> None:
        from parsimony_agents.dataset_io import write_dataset_bytes

        if self._payload is None:
            raise ValueError(
                "Dataset.save: no payload attached. Call .with_payload(...) first."
            )
        target = Path(path)
        if target.suffix != ".parquet":
            raise ValueError(f"Dataset.save: path must end in .parquet, got {path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(write_dataset_bytes(self, self._payload))


# ============================================================================
# Chart
# ============================================================================


class Chart(_ArtifactBase):
    """Curation + lineage for a published chart.

    Lives on disk at ``.ockham/charts/<logical_id>/<content_sha>.vl.json``
    with sibling curation/log files. ``notebook_ref`` is singular — a
    chart is rendered in exactly one notebook. ``source_dataset_refs``
    is plural for multi-dataset comparison charts. ``source_refs``
    covers the rare case of a chart drawn straight from data_objects
    without an intermediate ``return_dataset``.
    """

    type: Literal["chart"] = "chart"

    notebook_ref: ArtifactRef | None = Field(
        default=None,
        description=(
            "Frozen ref to the notebook that renders this chart (kind='notebook'). "
            "Required at persist time; left optional on the model so codec round-trips "
            "of vanilla vl.json without curation don't fail on construction."
        ),
    )
    source_dataset_refs: list[ArtifactRef] = Field(
        default_factory=list,
        description="Frozen refs to source datasets (kind='dataset'). Plural — multi-dataset charts welcome.",
    )
    source_refs: list[ArtifactRef] = Field(
        default_factory=list,
        description="Frozen refs to upstream data_objects when bypassing return_dataset (uncommon).",
    )
    variable_name: str = Field(
        default="",
        description=(
            "Kernel variable name the agent extracted to produce this chart. "
            "Recipe field — participates in ``logical_id`` and is required for "
            "``refresh`` to re-extract from the kernel after re-running the "
            "producing notebook. Persisted in vl.json ``usermeta.parsimony_agents`` "
            "and curation; the PATCH endpoint rejects edits."
        ),
    )

    _payload: FigureObject | None = PrivateAttr(default=None)

    def with_payload(self, payload: FigureObject) -> Chart:
        if not isinstance(payload, FigureObject):
            raise TypeError(
                f"Chart.with_payload expects a FigureObject; got "
                f"{type(payload).__name__}. Wrap raw specs/Altair charts "
                f"with FigureObject(value=spec_or_chart)."
            )
        self._payload = payload
        return self

    @property
    def payload(self) -> FigureObject | None:
        return self._payload

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        title = self.title or "(untitled)"
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f'<chart title="{escape_attr(title)}">\n'},
        ]
        if self.description:
            blocks.append(
                {"type": "text", "text": f"<description>{escape_text(self.description)}</description>\n"}
            )
        if self.notes:
            blocks.append({"type": "text", "text": "<notes>\n"})
            blocks.extend({"type": "text", "text": f"- {escape_text(note)}\n"} for note in self.notes)
            blocks.append({"type": "text", "text": "</notes>\n"})
        blocks.extend(_refs_blocks("source_datasets", self.source_dataset_refs))
        blocks.extend(_refs_blocks("sources", self.source_refs))
        blocks.append({"type": "text", "text": "</chart>\n"})
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "content_sha": self.content_sha,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "notes": list(self.notes),
            "live_name": self.live_name,
            "notebook_ref": self.notebook_ref.to_dict() if self.notebook_ref else None,
            "source_dataset_refs": [r.to_dict() for r in self.source_dataset_refs],
            "source_refs": [r.to_dict() for r in self.source_refs],
            "variable_name": self.variable_name,
        }

    def save(self, path: str | Path) -> None:
        from parsimony_agents.chart_io import write_chart_bytes

        if self._payload is None:
            raise ValueError(
                "Chart.save: no payload attached. Call .with_payload(...) first."
            )
        target = Path(path)
        if "".join(target.suffixes[-2:]) != ".vl.json":
            raise ValueError(f"Chart.save: path must end in .vl.json, got {path!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(write_chart_bytes(self, self._payload))


# ============================================================================
# Report
# ============================================================================


class Report(_ArtifactBase):
    """Curation + lineage for a published Quarto report.

    Reports are user-readable deliverables. The agent authors a markdown
    body; the framework persists it as a single ``.qmd`` file with a
    minimal YAML preamble (``title`` + ``ockham.formats``) — the server
    builds the full Quarto YAML at render time. Snapshots live at
    ``.ockham/reports/<logical_id>/<content_sha>.qmd`` with sibling
    curation/log files. ``embedded_refs`` are frozen by default (§2.7) —
    a re-author against newer data produces a new report snapshot whose
    embedded refs may be newer; old snapshots stay byte-stable and
    reproducible.
    """

    type: Literal["report"] = "report"

    markdown: str = ""
    embedded_refs: list[ArtifactRef] = Field(
        default_factory=list,
        description="Frozen refs to artifacts embedded in the markdown source.",
    )
    formats: list[ExportFormat] = Field(
        default_factory=lambda: ["html", "pdf"],
        description=(
            "Quarto output formats this report should render to. "
            "Persisted in the .qmd YAML preamble; the server reads it to "
            "build per-format render config."
        ),
    )

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        title = self.title or "(untitled)"
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f'<report title="{escape_attr(title)}">\n'},
        ]
        if self.description:
            blocks.append(
                {"type": "text", "text": f"<description>{escape_text(self.description)}</description>\n"}
            )
        if self.notes:
            blocks.append({"type": "text", "text": "<notes>\n"})
            blocks.extend({"type": "text", "text": f"- {escape_text(note)}\n"} for note in self.notes)
            blocks.append({"type": "text", "text": "</notes>\n"})
        blocks.extend(_refs_blocks("embedded", self.embedded_refs))
        blocks.append({"type": "text", "text": "</report>\n"})
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "content_sha": self.content_sha,
            "title": self.title,
            "description": self.description,
            "tags": list(self.tags),
            "notes": list(self.notes),
            "live_name": self.live_name,
            "embedded_refs": [r.to_dict() for r in self.embedded_refs],
            "formats": list(self.formats),
        }


# ============================================================================
# Helpers
# ============================================================================


def _refs_blocks(name: str, refs: list[ArtifactRef]) -> list[dict[str, Any]]:
    """Render a list of ArtifactRefs as ``<name>``/``<ref />`` text blocks for ``to_llm``."""
    if not refs:
        return []
    out: list[dict[str, Any]] = [
        {"type": "text", "text": f'<{name} count="{len(refs)}">\n'},
    ]
    for r in refs:
        out.append({"type": "text", "text": f"  {r.to_self_closing_tag()}\n"})
    out.append({"type": "text", "text": f"</{name}>\n"})
    return out


def derive_live_name(title: str) -> str:
    """Derive a file-tree-friendly slug from a curation title (§4.6 default)."""
    return slug_from_title(title)
