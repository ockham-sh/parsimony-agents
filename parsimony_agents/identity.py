"""Content-addressed identity primitives for workspace artifacts.

Every artifact (notebook, data_object, dataset, chart, report) carries
two hashes:

- ``logical_id`` — "Which artifact is this?" Stable across data
  refreshes and edits to the same logical thing.
- ``content_sha`` — "What does it currently look like?" Hash of the
  bytes of this specific snapshot.

The two are always independent. Each kind derives ``logical_id``
differently, but the storage layout is uniform:
``.ockham/<kind>s/<logical_id>/<content_sha>.<ext>``. A logical artifact
accumulates immutable snapshots over time; ``log.jsonl`` next to the
snapshots is the version history.

| Kind        | logical_id derivation                                           |
| ----------- | --------------------------------------------------------------- |
| notebook    | live_name from path (``notebooks/foo.py`` → ``"foo"``)          |
| dataset     | hash of inputs (notebook_refs, var_name, sources)               |
| chart       | hash of inputs (notebook_ref, var_name, source_datasets)        |
| report      | hash of inputs (embedded_refs, title)                           |
| data_object | hash of provenance minus ``fetched_at`` and ``properties`` |

Notebook identity follows git's model: ``logical_id`` IS the
working-copy basename, so renaming a notebook starts a fresh
``logical_id`` and a fresh log. Pre-rename snapshots stay reachable
(content-addressed; the old path under
``.ockham/notebooks/<old_name>/`` doesn't move), but the version
sequence resets — pills emitted before the rename keep working, while
the new name accumulates v1, v2, … from scratch. This matches how
git treats renames (delete + add) and lets all five kinds share a
"logical_id derived from inputs" mental model.
"""

from __future__ import annotations

__all__ = [
    "ArtifactRef",
    "LiveNameCollisionError",
    "OBJECTS_NAMESPACE",
    "SNAPSHOT_KINDS",
    "SnapshotKind",
    "chart_logical_id",
    "content_sha",
    "data_object_logical_id",
    "dataset_logical_id",
    "notebook_content_sha",
    "notebook_logical_id",
    "object_pool_path",
    "report_logical_id",
    "slug_from_title",
]

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

from parsimony_agents._naming import slug_from_title

SnapshotKind = Literal["notebook", "data_object", "dataset", "chart", "report"]

SNAPSHOT_KINDS: Final[tuple[SnapshotKind, ...]] = get_args(SnapshotKind)

_EXT_BY_KIND: Final[dict[SnapshotKind, str]] = {
    "notebook": ".py",
    "data_object": ".parquet",
    "dataset": ".parquet",
    "chart": ".vl.json",
    "report": ".qmd",
}

OBJECTS_NAMESPACE = ".ockham/objects"


class LiveNameCollisionError(Exception):
    """A ``live_name`` already belongs to a sibling terminal's artifact.

    Raised by resolvers when a write or refresh would silently coalesce with
    an artifact this terminal has never interacted with. The recovery loop is
    encoded in the message itself: read the existing artifact first to bring
    it into this terminal's seen-set, then re-issue the write — or pick a
    different ``live_name``.

    Three parameters are loadbearing for callers:
    - ``live_name``: the slug both halves of the agent surface share — the
      argument the agent typed and the argument it must use to read.
    - ``existing_logical_id``: the colliding artifact's ``logical_id``. If
      the agent retries with the same ``live_name`` after reading the
      artifact, the resolver returns this value (continuation), not a
      fresh slug.
    - ``kind``: which artifact kind collided. The seen-set is keyed on
      ``(kind, live_name)``, so the error names both halves for the caller.
    """

    def __init__(
        self,
        *,
        live_name: str,
        existing_logical_id: str,
        kind: SnapshotKind = "notebook",
    ) -> None:
        self.live_name = live_name
        self.existing_logical_id = existing_logical_id
        self.kind = kind
        super().__init__(
            f"{kind} live_name {live_name!r} is already in use by another "
            f"terminal in this workspace (existing logical_id={existing_logical_id!r}). "
            f"To continue working on the existing artifact, call "
            f"read_artifact(live_name={live_name!r}, kind={kind!r}) first — "
            f"this adds it to your context and your next return_* call will "
            f"publish a revision. To start a separate artifact, pick a "
            f"different live_name."
        )


@dataclass(frozen=True)
class ArtifactRef:
    """Frozen reference: pins one ``content_sha`` of a logical artifact.

    All five kinds are uniform: ``logical_id`` answers "which artifact"
    and ``content_sha`` answers "which snapshot." Per-kind logical_id
    derivation differs (see module docstring) but the storage layout
    and the wire shape are the same everywhere.
    """

    kind: SnapshotKind
    logical_id: str
    content_sha: str

    def __post_init__(self) -> None:
        if self.kind not in _EXT_BY_KIND:
            raise ValueError(f"ArtifactRef: unsupported kind {self.kind!r}")
        if not self.logical_id:
            raise ValueError("ArtifactRef.logical_id must be non-empty")
        if not self.content_sha:
            raise ValueError("ArtifactRef.content_sha must be non-empty")

    @property
    def workspace_file_path(self) -> str:
        """Workspace-relative on-disk path for this snapshot.

        Versioned kinds live under ``.ockham/<kind>s/<logical_id>/``.
        ``data_object`` bytes are immutable pool entries addressed only by
        ``content_sha`` under :data:`OBJECTS_NAMESPACE`.
        """
        if self.kind == "data_object":
            return object_pool_path(self.content_sha)
        ext = _EXT_BY_KIND[self.kind]
        return f".ockham/{self.kind}s/{self.logical_id}/{self.content_sha}{ext}"

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict (e.g. for log.jsonl, JSON wire)."""
        return {
            "kind": self.kind,
            "logical_id": self.logical_id,
            "content_sha": self.content_sha,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        return cls(
            kind=data["kind"],
            logical_id=data["logical_id"],
            content_sha=data["content_sha"],
        )

    @classmethod
    def from_workspace_file_path(cls, path: str) -> ArtifactRef | None:
        """Inverse of :attr:`workspace_file_path` — parse a canonical path.

        Returns ``None`` for paths outside the canonical layout
        ``.ockham/<kind>s/<logical_id>/<content_sha>.<ext>``, so callers
        get a clean miss-signal instead of guessing kinds.
        """
        if not path.startswith(".ockham/"):
            return None
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "objects" and parts[3].endswith(".parquet"):
            content_sha = f"{parts[2]}{parts[3][: -len('.parquet')]}"
            if not content_sha:
                return None
            return cls(kind="data_object", logical_id=content_sha, content_sha=content_sha)
        if len(parts) != 4:
            return None
        kind_plural, logical_id, last = parts[1], parts[2], parts[3]
        for kind, ext in _EXT_BY_KIND.items():
            if kind_plural == f"{kind}s" and last.endswith(ext):
                content_sha = last[: -len(ext)]
                if not logical_id or not content_sha:
                    return None
                return cls(kind=kind, logical_id=logical_id, content_sha=content_sha)
        return None

    # ---- XML rendering ---------------------------------------------------
    #
    # The framework speaks to the LLM in compact XML — refs surface as
    # attributes on tags like ``<notebook_ref/>``, ``<data_object_ref/>``,
    # and ``<artifact .../>``. Centralising the attr format here is the
    # single source of truth for that wire shape; callers compose the
    # tag/body/extra-attrs they need on top.

    def to_xml_attrs(self) -> str:
        """Inline ``kind="…" logical_id="…" content_sha="…"`` attribute fragment.

        Use when composing a tag that needs additional attributes
        (e.g. ``<artifact path="…" {ref.to_xml_attrs()}>summary</artifact>``).
        """
        return (
            f'kind="{self.kind}" '
            f'logical_id="{self.logical_id}" '
            f'content_sha="{self.content_sha}"'
        )

    def to_self_closing_tag(self, tag: str = "ref") -> str:
        """Self-closing ``<{tag} kind="…" logical_id="…" content_sha="…"/>``.

        The default ``tag="ref"`` matches the generic ``<ref/>`` form
        used in artifact lineage outlines; callers pick a more specific
        tag (``"notebook_ref"``, ``"data_object_ref"``) where it adds
        clarity for the agent.
        """
        return f"<{tag} {self.to_xml_attrs()}/>"


# ---------------------------------------------------------------------------
# content_sha: hash of bytes
# ---------------------------------------------------------------------------


def content_sha(blob: bytes) -> str:
    """SHA-256 of ``blob`` as lowercase hex."""
    return hashlib.sha256(blob).hexdigest()


def object_pool_path(content_sha: str) -> str:
    """Workspace-relative path for an immutable object-pool parquet entry."""
    if len(content_sha) < 3:
        raise ValueError("object_pool_path: content_sha must be at least 3 hex chars")
    return f"{OBJECTS_NAMESPACE}/{content_sha[:2]}/{content_sha[2:]}.parquet"


# ---------------------------------------------------------------------------
# logical_id: hash of inputs (kind-specific)
# ---------------------------------------------------------------------------


def _hash_canonical(payload: Any) -> str:
    """SHA-256 of a JSON-canonicalized payload.

    Sort keys, sort lists where order is irrelevant *before* calling.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def notebook_content_sha(code: str) -> str:
    """Hash of a notebook's source bytes (UTF-8); strips trailing whitespace.

    Trailing whitespace is stripped so the hash is invariant under the
    serialize → deserialize round-trip (on-disk files get a trailing
    newline; parsed source has it stripped). This is the canonical
    ``content_sha`` for a notebook snapshot — NOT its ``logical_id``.

    A notebook's ``logical_id`` is its working-copy basename (the live
    name), so the same logical notebook accumulates snapshots as its
    code evolves. Renames produce a new ``logical_id`` and start a
    fresh log — git-style, content-addressed snapshots stay reachable.
    """
    return hashlib.sha256(code.rstrip().encode("utf-8")).hexdigest()


def notebook_logical_id(path: str) -> str:
    """Derive a notebook's logical_id from its working-copy path.

    ``notebooks/foo.py`` → ``"foo"``. The live_name IS the logical_id —
    rename = new logical_id (history splits, mirroring git's model).
    Single-segment under ``notebooks/`` only — subdirectories are not
    supported (matches dataset/chart/report flat layout).
    """
    cleaned = path.strip().strip("/")
    if not cleaned.startswith("notebooks/"):
        raise ValueError(
            f"notebook path must start with 'notebooks/', got {path!r}"
        )
    rest = cleaned[len("notebooks/"):]
    if "/" in rest:
        raise ValueError(
            f"notebook path must be flat (no subdirectories), got {path!r}"
        )
    if not rest.endswith(".py"):
        raise ValueError(
            f"notebook path must end with '.py', got {path!r}"
        )
    name = rest[: -len(".py")]
    if not name:
        raise ValueError("notebook live_name must be non-empty")
    return name


def data_object_logical_id(provenance: Any) -> str:
    """Hash the canonical provenance of a data_object, excluding ``fetched_at``.

    Same source + same params → same logical_id, regardless of when the
    fetch occurred or what bytes came back. ``properties`` is excluded —
    provider facts belong in result data columns, not provenance identity.
    """
    canonical = provenance.model_dump(mode="json", exclude={"fetched_at", "properties"})
    return _hash_canonical({"data_object": canonical})


def dataset_logical_id(
    *,
    notebook_refs: list[ArtifactRef],
    variable_name: str,
    source_refs: list[ArtifactRef],
) -> str:
    """Hash a dataset's identity inputs.

    Sorts both ``notebook_refs`` and ``source_refs`` by logical_id so
    ordering at the call site doesn't perturb identity. Hashing
    notebook ``logical_id`` (not ``content_sha``) means notebook edits
    are byte-level snapshots of the same logical dataset — refresh
    appends a new ``content_sha`` under the unchanged ``logical_id``,
    rather than forking a new artifact.
    """
    if not variable_name:
        raise ValueError("dataset_logical_id: variable_name must be non-empty")
    notebooks = sorted(r.logical_id for r in notebook_refs if r.kind == "notebook")
    sources = sorted(r.logical_id for r in source_refs)
    return _hash_canonical(
        {
            "kind": "dataset",
            "notebooks": notebooks,
            "variable_name": variable_name,
            "sources": sources,
        }
    )


def chart_logical_id(
    *,
    notebook_ref: ArtifactRef,
    chart_variable_name: str,
    source_dataset_refs: list[ArtifactRef],
    source_refs: list[ArtifactRef],
) -> str:
    """Hash a chart's identity inputs.

    Sorts ``source_dataset_refs`` and ``source_refs`` by logical_id so
    ordering at the call site doesn't perturb identity.
    """
    if notebook_ref.kind != "notebook":
        raise ValueError(
            f"chart_logical_id: notebook_ref must be kind='notebook', got {notebook_ref.kind!r}"
        )
    if not chart_variable_name:
        raise ValueError("chart_logical_id: chart_variable_name must be non-empty")
    datasets = sorted(r.logical_id for r in source_dataset_refs)
    sources = sorted(r.logical_id for r in source_refs)
    return _hash_canonical(
        {
            "kind": "chart",
            "notebook": notebook_ref.logical_id,
            "variable_name": chart_variable_name,
            "source_datasets": datasets,
            "sources": sources,
        }
    )


def report_logical_id(
    *,
    embedded_refs: list[ArtifactRef],
    title: str,
) -> str:
    """Hash a report's identity inputs.

    Sorts ``embedded_refs`` by logical_id. Title participates so two
    distinct reports referencing the same artifact set get distinct
    logical_ids.
    """
    if not title:
        raise ValueError("report_logical_id: title must be non-empty")
    embedded = sorted(r.logical_id for r in embedded_refs)
    return _hash_canonical(
        {
            "kind": "report",
            "title": title,
            "embedded": embedded,
        }
    )
