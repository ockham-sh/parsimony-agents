"""Closure — transitive enumeration of an artifact's typed dependencies.

The artifact DAG is the same graph ``refresh`` walks: report.pin_map →
chart/dataset; chart.notebook_ref + chart.source_dataset_refs +
chart.source_refs (uncommon) → datasets / data_objects;
dataset.notebook_refs + dataset.source_refs → notebooks / datasets /
data_objects. ``data_object`` and ``notebook`` are leaves — they carry
no typed source refs that the system can statically follow.

This module exposes two primitives:

- :func:`child_refs` — the single source of truth for "what edges does
  this artifact have?" Every walker that needs to traverse the DAG
  consults this one function. Adding a new artifact kind with new
  typed source fields adds one branch here.

- :func:`enumerate_closure` — post-order DFS from a root. Returns every
  reachable ref deps-before-dependents (topologically sorted). Uses a
  ``(kind, logical_id, content_sha)`` visited set so identical refs are
  emitted exactly once and any structural cycle terminates safely.

Notebooks have a wrinkle the closure walker is honest about: a
notebook's persisted bytes are ``.py`` source only — the ``fetch_log``
that records which ``data_object`` refs the kernel produced is not
serialised into the snapshot. So :func:`child_refs` returns ``[]`` for
a notebook leaf. data_objects ARE still reachable in the closure,
just through ``dataset.source_refs`` / ``chart.source_refs`` (where
``return_dataset`` / ``return_chart`` captured the fetch_log at publish
time), never through the notebook ref directly. This is correct: every
data_object in a published lineage appears in some downstream
dataset's or chart's source_refs.

Relationship to refresh
-----------------------
:mod:`parsimony_agents.refresh` walks the same DAG with kind-specific
re-derivation logic (notebooks via ``_rerun_notebooks``, datasets via
recursive ``_refresh``, data_objects implicitly via the connector
callback during a kernel run). Each ``_refresh_<kind>`` accesses
``dataset.notebook_refs`` / ``dataset.source_refs`` / ``chart.notebook_ref``
/ ``chart.source_dataset_refs`` / ``chart.source_refs`` / ``snap.pins``
directly because the **edge handling** is kind-specific, not just the
edge enumeration — routing all edges through a flat ``child_refs``
list would force refresh to immediately re-dispatch on ``child.kind``,
adding lines rather than removing them. ``child_refs`` is the canonical
edge primitive for read-only consumers (closure publishing, bundle
size estimation, lineage validation); refresh's typed-field access
stays in place because its per-edge work is structurally different.
"""

from __future__ import annotations

__all__ = ["child_refs", "enumerate_closure"]

from typing import Protocol

from parsimony_agents.identity import ArtifactRef


class _Executor(Protocol):
    """The minimal executor surface ``child_refs`` needs.

    Same shape as :class:`parsimony_agents.refresh._Executor` (the two
    walkers share a stub in tests), but only ``read_workspace_file`` is
    actually called from here — closure enumeration is read-only.
    """

    cwd: str | None

    async def read_workspace_file(self, path: str) -> bytes: ...


async def child_refs(ref: ArtifactRef, *, executor: _Executor) -> list[ArtifactRef]:
    """Return ``ref``'s typed source refs.

    Single source of truth for edges in the artifact DAG. Refresh and
    closure-publish both consult this so the "what counts as a
    dependency?" knowledge lives in exactly one place. Ordering inside
    the returned list matches the artifact's declared field order
    (notebook(s) first, then datasets, then misc source_refs) — callers
    that need a different ordering should sort themselves; the DFS in
    :func:`enumerate_closure` is order-insensitive.

    Leaves return ``[]``:

    - ``data_object`` — no source refs by design (provenance is the
      identity, not a sub-DAG).
    - ``notebook`` — fetch_log is a kernel-run artefact, not persisted
      with the snapshot bytes.

    Raises :class:`ValueError` when the snapshot's bytes are missing or
    unparseable, with a message that names the offending ref.
    """
    if ref.kind == "report":
        return await _report_children(ref, executor=executor)
    if ref.kind == "chart":
        return await _chart_children(ref, executor=executor)
    if ref.kind == "dataset":
        return await _dataset_children(ref, executor=executor)
    if ref.kind in ("notebook", "data_object"):
        return []
    raise ValueError(f"closure: unsupported kind {ref.kind!r}")


async def enumerate_closure(
    root: ArtifactRef, *, executor: _Executor
) -> list[ArtifactRef]:
    """Topological closure of ``root`` under typed source refs.

    Returns every ref reachable from ``root`` (inclusive), in post-order:
    dependencies appear before the artifacts that reference them, with
    ``root`` always last. Deduplicated by ``(kind, logical_id,
    content_sha)`` so a diamond DAG emits each shared dep exactly once.

    Side-effect-free apart from snapshot reads. Idempotent. Cycle-safe
    (the visited set terminates any pathological back-edge — none should
    exist in a healthy graph, but the walker doesn't assume it).
    """
    visited: set[tuple[str, str, str]] = set()
    order: list[ArtifactRef] = []

    async def visit(node: ArtifactRef) -> None:
        key = (node.kind, node.logical_id, node.content_sha)
        if key in visited:
            return
        visited.add(key)
        for child in await child_refs(node, executor=executor):
            await visit(child)
        order.append(node)

    await visit(root)
    return order


# ---------------------------------------------------------------------------
# Per-kind edge extractors — one place each typed ref field is named.
# ---------------------------------------------------------------------------


async def _report_children(
    ref: ArtifactRef, *, executor: _Executor
) -> list[ArtifactRef]:
    """Pin-map values, in YAML insertion order.

    Body URIs (``file://./charts/<n>.vl.json`` etc.) speak in
    ``live_name``s; the pin map resolves them to typed ``ArtifactRef``s.
    The pin map is the authoritative edge list — body parsing is only
    needed by the renderer.
    """
    from parsimony_agents.report_format import parse_snapshot

    blob = await _read_snapshot(executor, ref)
    snap = parse_snapshot(blob.decode("utf-8"))
    return list(snap.pins.values())


async def _chart_children(
    ref: ArtifactRef, *, executor: _Executor
) -> list[ArtifactRef]:
    """notebook_ref (when set) + source_dataset_refs + source_refs.

    ``source_refs`` is the "uncommon" path (chart drawn straight from
    data_objects bypassing return_dataset). Including it unconditionally
    is correct because closure must follow every typed edge, not just
    the common ones.
    """
    from parsimony_agents.chart_io import deserialize_chart

    blob = await _read_snapshot(executor, ref)
    chart, _spec = deserialize_chart(blob)
    out: list[ArtifactRef] = []
    if chart.notebook_ref is not None:
        out.append(chart.notebook_ref)
    out.extend(chart.source_dataset_refs)
    out.extend(chart.source_refs)
    return out


async def _dataset_children(
    ref: ArtifactRef, *, executor: _Executor
) -> list[ArtifactRef]:
    """notebook_refs + source_refs (mixed dataset / data_object kinds)."""
    from parsimony_agents.dataset_io import deserialize_dataset

    blob = await _read_snapshot(executor, ref)
    _result, dataset = deserialize_dataset(blob)
    return [*dataset.notebook_refs, *dataset.source_refs]


async def _read_snapshot(executor: _Executor, ref: ArtifactRef) -> bytes:
    try:
        return await executor.read_workspace_file(ref.workspace_file_path)
    except FileNotFoundError as e:
        raise ValueError(
            f"closure: snapshot bytes missing for {ref.kind} "
            f"{ref.logical_id}@{ref.content_sha[:8]}. The artifact may have "
            "been deleted from disk."
        ) from e


