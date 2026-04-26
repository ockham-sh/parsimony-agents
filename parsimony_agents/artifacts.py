"""Published deliverables (datasets, charts) — typed curation primitives.

Datasets and Charts are *durable curation metadata*: artifact_id, title,
description, lineage. They are deliberately decoupled from the live payload
and from any storage location.

Payload contract (single, typed):

- ``Dataset._payload: DataFrameObject | None`` — the executor wrapper for a
  pandas DataFrame.
- ``Chart._payload: FigureObject | None`` — the executor wrapper for an
  Altair chart or Vega-Lite spec.

There is exactly one producer in production for each (the ``return_dataset``
/ ``return_chart`` agent tools), and that producer always has the executor
wrapper. Tests and ad-hoc scripts construct the same wrapper via
``DataFrameObject.from_pandas(...)`` / ``FigureObject(value=...)``. There is
no overload, no coercion table, and no ``Any`` payload anywhere in the
pipeline — wrong payload type raises :class:`TypeError` at the call site,
not five hops deep in the streaming dispatcher.

Where the payload is written on disk is not the artifact's concern. The
agent calls :meth:`Dataset.save` / :meth:`Chart.save` when it wants to
publish to a user-visible workspace path (``data/foo.parquet``); the
terminal writes a framework-managed snapshot under
``.ockham/cards/<artifact_id>/v<n>/<title_slug>.<ext>`` to back the chat
card. Both paths embed the same curation metadata via the open-format
codecs.

Why no ``Artifact[T]`` generic
------------------------------
The two artifact kinds share roughly fifteen lines of identity machinery
(``artifact_id``/``version`` defaults + ``populate_identity`` validator,
``with_payload`` + ``payload`` property), captured here in
:class:`_ArtifactBase`. Their domain fields, on-disk codecs (Parquet vs
``.vl.json``), and rendering metadata (``to_llm`` / ``to_frontend_dict``)
are entirely disjoint, so a Pydantic ``Generic[T]`` payload abstraction
would buy nothing beyond a writer registry of two single-entry maps.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from parsimony.result import Provenance

from parsimony_agents._naming import slug_from_title
from parsimony_agents.execution.outputs import DataFrameObject, FigureObject
from parsimony_agents.messages import MessageContent

# ============================================================================
# Snapshot path layout — the framework-managed namespace for ``return_*``
# artifacts. Lives here (not in the terminal) because path layout *is* part
# of the artifact contract: charts reference their source dataset by path.
# ============================================================================

CARDS_NAMESPACE: Final[str] = ".ockham/cards"


def snapshot_path(*, artifact_id: str, version: int, kind: str, title: str) -> str:
    """Workspace-relative snapshot path for a published artifact version.

    Layout: ``.ockham/cards/<artifact_id>/v<n>/<title_slug>.<ext>`` so the
    basename is a human-readable snake_case slug derived from curation title.

    Lives under ``.ockham/cards/`` so it stays out of the user's editable
    workspace tree but still travels with the workspace blob (and so is
    accessible to viewer endpoints under the existing storage backend).
    """

    if not artifact_id:
        raise ValueError("snapshot_path requires a non-empty artifact_id")
    if version < 1:
        raise ValueError(f"snapshot_path requires version >= 1, got {version}")
    match kind:
        case "dataset":
            ext = "parquet"
        case "chart":
            ext = "vl.json"
        case _:
            raise ValueError(f"snapshot_path: unsupported artifact kind {kind!r}")
    slug = slug_from_title(title)
    return f"{CARDS_NAMESPACE}/{artifact_id}/v{version}/{slug}.{ext}"


_DATASET_SNAPSHOT_PATH: re.Pattern[str] = re.compile(
    r"^\.ockham/cards/([^/]+)/v\d+/[^/]+\.parquet$",
)


def artifact_id_from_dataset_snapshot_path(path: str) -> str | None:
    """Return the ``artifact_id`` segment from a dataset snapshot path, or ``None``."""

    p = (path or "").replace("\\", "/").strip()
    if not p:
        return None
    m = _DATASET_SNAPSHOT_PATH.match(p)
    return m.group(1) if m else None


# ============================================================================
# Shared identity + payload mechanics
# ============================================================================


class _ArtifactBase(MessageContent):
    """Identity machinery shared by :class:`Dataset` / :class:`Chart`.

    Subclass-specific payload typing is enforced by each subclass's
    :meth:`with_payload` (``isinstance`` + ``TypeError``) — never
    ``ValueError``. ``provenance`` is permissive by design so reading
    vanilla files (no embedded curation) yields an empty placeholder
    envelope instead of failing validation. Non-emptiness of title/
    description is enforced at the agent tool boundary.
    """

    schema_version: int = 1
    artifact_id: str = ""
    version: int = 1
    provenance: Provenance = Field(default_factory=Provenance)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_curation(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        migrated = dict(values)
        provenance_raw = migrated.get("provenance")
        if provenance_raw is None:
            provenance_payload: dict[str, Any] = {}
        elif isinstance(provenance_raw, dict):
            provenance_payload = dict(provenance_raw)
        else:
            return migrated

        legacy_title = migrated.pop("title", None)
        legacy_description = migrated.pop("description", None)
        legacy_tags = migrated.pop("tags", None)
        if legacy_title is not None and not provenance_payload.get("title"):
            provenance_payload["title"] = legacy_title
        if legacy_description is not None and not provenance_payload.get("description"):
            provenance_payload["description"] = legacy_description
        if legacy_tags is not None and not provenance_payload.get("tags"):
            provenance_payload["tags"] = legacy_tags
        if provenance_payload:
            migrated["provenance"] = provenance_payload
        return migrated

    @model_validator(mode="after")
    def populate_identity(self) -> _ArtifactBase:
        if not self.artifact_id:
            self.artifact_id = str(uuid.uuid4())
        if self.version < 1:
            self.version = 1
        return self


# ============================================================================
# Curation primitives
# ============================================================================


class Dataset(_ArtifactBase):
    """Curation metadata for a published dataset.

    Pure metadata: no DataFrame, no path, no executor handle. The live
    payload (DataFrame or :class:`parsimony.Result`) is supplied at publish
    time via :meth:`save` (user-space) or by the streaming dispatcher
    (snapshot under ``.ockham/cards/``).
    """

    # Old snapshot files (pre "path is identity") may carry fields we have
    # since dropped (e.g. ``derived_from``). ``extra="ignore"`` lets us
    # deserialize them without a migration script — the on-disk schema is
    # framework-managed under ``.ockham/`` and never quoted by users.
    model_config = ConfigDict(extra="ignore")

    type: Literal["dataset"] = "dataset"
    notebook_refs: list[str] = Field(
        default_factory=list,
        description="Notebook paths used to produce this dataset.",
    )

    # In-process transport: optionally carries the executor's DataFrameObject
    # for this dataset, so a single ``Dataset`` instance can be handed off
    # to the streaming dispatcher (or :meth:`save`) without a side channel.
    # Never serialized.
    _payload: DataFrameObject | None = PrivateAttr(default=None)

    def with_payload(self, payload: DataFrameObject) -> Dataset:
        """Attach the executor's DataFrameObject for this dataset and return self."""

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
        title = self.provenance.title or "(untitled)"
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f'<dataset title="{title}">\n'},
        ]
        if self.provenance.description:
            blocks.append({"type": "text", "text": f"<description>{self.provenance.description}</description>\n"})
        if self.notes:
            blocks.append({"type": "text", "text": "<notes>\n"})
            blocks.extend({"type": "text", "text": f"- {note}\n"} for note in self.notes)
            blocks.append({"type": "text", "text": "</notes>\n"})
        blocks.append({"type": "text", "text": "</dataset>\n"})
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "title": self.provenance.title or "",
            "description": self.provenance.description or "",
            "notes": list(self.notes),
            "tags": list(self.provenance.tags),
            "notebook_refs": list(self.notebook_refs),
        }

    def save(self, path: str | Path) -> None:
        """Persist this dataset to ``path`` as Parquet with embedded curation.

        ``path`` is the explicit, user-visible workspace location (e.g.
        ``data/sp500_daily.parquet``). The on-disk file is plain Parquet
        readable by any client; the curation metadata is recoverable via
        :func:`parsimony_agents.deserialize_dataset`.

        Requires an in-process ``_payload`` attached via :meth:`with_payload`
        — there is no per-call payload override. ``return_dataset`` is the
        only producer that mints datasets in production; if you're calling
        ``.save`` from a script or test, attach a payload first via
        ``Dataset(...).with_payload(DataFrameObject.from_pandas(df, local_dir=...))``.
        """

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


class Chart(_ArtifactBase):
    """Curation metadata for a published chart.

    Pure metadata: no Vega-Lite spec, no path, no executor handle. The live
    figure is supplied at publish time via :meth:`save` (user-space) or by
    the streaming dispatcher (snapshot under ``.ockham/cards/``).

    Charts reference their source dataset by its workspace-relative
    snapshot path (``source_dataset_path``); the path *is* the identity
    (see ``snapshot_path``). Staleness is derived from path resolution,
    not stored on the chart.
    """

    # See ``Dataset.model_config`` — same rationale (back-compat with old
    # snapshots that carried ``source_dataset_artifact_id`` /
    # ``source_dataset_version`` before the swap to ``source_dataset_path``).
    model_config = ConfigDict(extra="ignore")

    type: Literal["chart"] = "chart"
    source_dataset_path: str = ""
    chart_notebook_ref: str = ""

    _payload: FigureObject | None = PrivateAttr(default=None)

    def with_payload(self, payload: FigureObject) -> Chart:
        """Attach the executor's FigureObject for this chart and return self."""

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
        title = self.provenance.title or "(untitled)"
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f'<chart title="{title}" notebook="{self.chart_notebook_ref}">\n'},
        ]
        if self.provenance.description:
            blocks.append({"type": "text", "text": f"<description>{self.provenance.description}</description>\n"})
        if self.notes:
            blocks.append({"type": "text", "text": "<notes>\n"})
            blocks.extend({"type": "text", "text": f"- {note}\n"} for note in self.notes)
            blocks.append({"type": "text", "text": "</notes>\n"})
        blocks.append({"type": "text", "text": "</chart>\n"})
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "title": self.provenance.title or "",
            "description": self.provenance.description or "",
            "notes": list(self.notes),
            "source_dataset_path": self.source_dataset_path,
            "chart_notebook_ref": self.chart_notebook_ref,
        }

    def save(self, path: str | Path) -> None:
        """Persist this chart to ``path`` as Vega-Lite JSON with embedded curation.

        ``path`` must end in ``.vl.json``; the on-disk file is plain
        Vega-Lite that any vega-embed-compatible viewer can render.
        Curation lives under ``spec.usermeta.parsimony_agents``.

        Requires an in-process ``_payload`` attached via :meth:`with_payload`
        — there is no per-call payload override (see :meth:`Dataset.save`
        for the rationale).
        """

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
