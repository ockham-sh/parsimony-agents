"""Refresh — re-derive an artifact from the latest upstream data.

Walks the lineage of a dataset / chart / report bottom-up:

- **dataset** → recurse into source datasets, re-run producing
  notebooks (which auto-refresh connector data_objects via the
  persister callback wired into ``KernelOutput.fetch_log``), re-extract
  the published variable from the kernel, persist a new snapshot
  under the unchanged ``logical_id``.
- **chart** → recurse into source datasets, re-run the chart's
  notebook, re-extract, persist.
- **report** → recurse into embedded refs, rewrite the markdown's
  embedded ``content_sha`` references to point at the latest
  snapshots, persist.

The orchestrator lives in ``parsimony-agents`` so it sits next to the
``refresh`` agent tool. Storage I/O goes through
:meth:`BaseCodeExecutor.read_workspace_file` /
:meth:`~BaseCodeExecutor.write_workspace_file` — the same abstraction
the rest of the agent uses, so refresh works in any workspace mode
(local or sync-back).

Versioning semantic (uniform across kinds): refresh appends a new
``content_sha`` snapshot to the same ``logical_id`` whenever any byte
in the lineage changed. If nothing upstream changed, every layer's
``content_sha`` is unchanged and the persister returns the existing
version — refresh is fully idempotent.

Notebooks themselves are working copies — refresh does not "refresh" a
notebook (re-publish via ``return_notebook(execute=True)``). data_objects refresh
implicitly through the producing notebook's connector calls — exposed
as a separate operation would create a second entry point for the same
intent.
"""

from __future__ import annotations

__all__ = ["embedded_refs_from_markdown", "refresh_artifact"]

import json
import re
from datetime import UTC, datetime
from typing import Any, Protocol

from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.chart_io import deserialize_chart, write_chart_bytes
from parsimony_agents.dataset_io import deserialize_dataset, write_dataset_bytes
from parsimony_agents.identity import (
    ArtifactRef,
    SnapshotKind,
    content_sha,
    notebook_content_sha,
)
from parsimony_agents.notebook_io import deserialize_notebook, read_latest_notebook


_REFRESHABLE_KINDS: frozenset[SnapshotKind] = frozenset({"dataset", "chart", "report"})


class _Executor(Protocol):
    """Minimal executor surface refresh needs.

    Aligned with :class:`BaseCodeExecutor`, but typed loosely so tests
    can stub it without inheriting the whole abstract base.
    """

    cwd: str | None

    async def read_workspace_file(self, path: str) -> bytes: ...
    async def write_workspace_file(self, path: str, data: bytes) -> None: ...
    async def execute(self, code: str) -> Any: ...
    async def get(self, key: str) -> Any: ...


async def refresh_artifact(
    ref: ArtifactRef,
    *,
    executor: _Executor,
) -> ArtifactRef:
    """Refresh ``ref`` and every upstream artifact it depends on.

    Returns the latest :class:`ArtifactRef` for ``ref``'s
    ``logical_id`` — same ``logical_id`` always, fresh ``content_sha``
    when any byte in the lineage changed.

    Raises :class:`ValueError` for unsupported kinds, missing upstream
    snapshots, or artifacts published before R2 (no ``variable_name``
    persisted on dataset/chart).
    """
    if ref.kind not in _REFRESHABLE_KINDS:
        raise ValueError(
            f"refresh: unsupported kind {ref.kind!r}; refresh applies to "
            "dataset / chart / report only. Notebooks are working copies "
            "(re-publish via return_notebook(execute=True)); data_objects refresh implicitly "
            "via the notebook that produced them."
        )
    return await _refresh(ref, executor=executor)


# ---------------------------------------------------------------------------
# Per-kind walkers
# ---------------------------------------------------------------------------


async def _refresh(ref: ArtifactRef, *, executor: _Executor) -> ArtifactRef:
    if ref.kind == "dataset":
        return await _refresh_dataset(ref, executor=executor)
    if ref.kind == "chart":
        return await _refresh_chart(ref, executor=executor)
    if ref.kind == "report":
        return await _refresh_report(ref, executor=executor)
    raise AssertionError(f"refresh: unreachable kind {ref.kind!r}")


async def _refresh_dataset(ref: ArtifactRef, *, executor: _Executor) -> ArtifactRef:
    blob = await _read_snapshot(executor, ref)
    _result, dataset = deserialize_dataset(blob)

    if not dataset.variable_name:
        raise ValueError(
            f"refresh: dataset {ref.logical_id!r} has no variable_name "
            "persisted. This artifact predates R2 — re-publish via "
            "return_dataset to enable refresh."
        )

    refreshed_datasets = await _refresh_dataset_sources(
        dataset.source_refs, executor=executor
    )
    new_notebook_refs, new_data_object_refs = await _rerun_notebooks(
        dataset.notebook_refs, executor=executor
    )

    out_obj = await executor.get(dataset.variable_name)
    if out_obj is None:
        raise ValueError(
            f"refresh: dataset {ref.logical_id!r} variable "
            f"{dataset.variable_name!r} was not produced by re-running "
            "its notebooks. The notebook may no longer assign that "
            "variable, or the kernel may have been restarted mid-walk."
        )

    new_source_refs = _compose_source_refs(
        original=dataset.source_refs,
        refreshed_datasets=refreshed_datasets,
        new_data_objects=new_data_object_refs,
    )

    new_dataset = Dataset(
        logical_id=ref.logical_id,
        title=dataset.title,
        description=dataset.description,
        tags=list(dataset.tags),
        notes=list(dataset.notes),
        live_name=dataset.live_name,
        notebook_refs=new_notebook_refs,
        source_refs=new_source_refs,
        variable_name=dataset.variable_name,
    )
    new_blob = write_dataset_bytes(new_dataset, out_obj)
    return await _persist_layer(
        executor=executor,
        kind="dataset",
        artifact=new_dataset,
        blob=new_blob,
        log_inputs={
            "notebooks": [r.content_sha for r in new_notebook_refs],
            "sources": [r.content_sha for r in new_source_refs],
        },
    )


async def _refresh_chart(ref: ArtifactRef, *, executor: _Executor) -> ArtifactRef:
    blob = await _read_snapshot(executor, ref)
    chart, _spec = deserialize_chart(blob)

    if not chart.variable_name:
        raise ValueError(
            f"refresh: chart {ref.logical_id!r} has no variable_name "
            "persisted. This artifact predates R2 — re-publish via "
            "return_chart to enable refresh."
        )
    if chart.notebook_ref is None:
        raise ValueError(
            f"refresh: chart {ref.logical_id!r} has no notebook_ref. "
            "This snapshot is malformed — re-publish via return_chart."
        )

    refreshed_datasets = await _refresh_dataset_sources(
        chart.source_dataset_refs, executor=executor
    )
    new_notebook_refs, new_data_object_refs = await _rerun_notebooks(
        [chart.notebook_ref], executor=executor
    )

    fig_obj = await executor.get(chart.variable_name)
    if fig_obj is None:
        raise ValueError(
            f"refresh: chart {ref.logical_id!r} variable "
            f"{chart.variable_name!r} was not produced by re-running its "
            "notebook. The notebook may no longer assign that variable."
        )

    new_source_dataset_refs = [
        refreshed_datasets.get(r.logical_id, r) for r in chart.source_dataset_refs
    ]
    new_source_refs = _compose_source_refs(
        original=chart.source_refs,
        refreshed_datasets={},
        new_data_objects=new_data_object_refs,
    )

    new_chart = Chart(
        logical_id=ref.logical_id,
        title=chart.title,
        description=chart.description,
        tags=list(chart.tags),
        notes=list(chart.notes),
        live_name=chart.live_name,
        notebook_ref=new_notebook_refs[0],
        source_dataset_refs=new_source_dataset_refs,
        source_refs=new_source_refs,
        variable_name=chart.variable_name,
    )
    new_blob = write_chart_bytes(new_chart, fig_obj)
    return await _persist_layer(
        executor=executor,
        kind="chart",
        artifact=new_chart,
        blob=new_blob,
        log_inputs={
            "notebook": new_notebook_refs[0].content_sha,
            "source_datasets": [r.content_sha for r in new_source_dataset_refs],
            "sources": [r.content_sha for r in new_source_refs],
        },
    )


async def _refresh_report(ref: ArtifactRef, *, executor: _Executor) -> ArtifactRef:
    from parsimony_agents.report_format import parse_snapshot

    blob = await _read_snapshot(executor, ref)
    # Snapshot bytes carry the leading ``formats:`` line — separate it
    # from the body so ref-substitution only touches what the agent
    # authored, and the formats list survives the refresh.
    formats, markdown = parse_snapshot(blob.decode("utf-8"))
    embedded = embedded_refs_from_markdown(markdown)
    curation = await _load_report_curation(executor, ref)

    new_embedded: list[ArtifactRef] = []
    new_markdown = markdown
    for emb in embedded:
        if emb.kind not in _REFRESHABLE_KINDS:
            # data_objects can't be refreshed standalone; notebooks
            # aren't valid embed targets in reports. Pass through.
            new_embedded.append(emb)
            continue
        refreshed = await _refresh(emb, executor=executor)
        new_embedded.append(refreshed)
        if refreshed.content_sha != emb.content_sha:
            new_markdown = new_markdown.replace(
                emb.workspace_file_path, refreshed.workspace_file_path
            )

    new_report = Report(
        logical_id=ref.logical_id,
        title=curation.get("title", "") or "",
        description=curation.get("description", "") or "",
        tags=list(curation.get("tags") or []),
        notes=list(curation.get("notes") or []),
        live_name=curation.get("live_name"),
        markdown=new_markdown,
        embedded_refs=new_embedded,
        formats=formats,
    )
    new_blob = new_report.snapshot_bytes()
    return await _persist_layer(
        executor=executor,
        kind="report",
        artifact=new_report,
        blob=new_blob,
        log_inputs={"embedded": [r.content_sha for r in new_embedded]},
    )


# ---------------------------------------------------------------------------
# Lineage helpers
# ---------------------------------------------------------------------------


async def _refresh_dataset_sources(
    refs: list[ArtifactRef], *, executor: _Executor
) -> dict[str, ArtifactRef]:
    """Recursively refresh every ``kind="dataset"`` ref; key by ``logical_id``."""
    out: dict[str, ArtifactRef] = {}
    for src in refs:
        if src.kind == "dataset":
            out[src.logical_id] = await _refresh(src, executor=executor)
    return out


async def _rerun_notebooks(
    notebook_refs: list[ArtifactRef], *, executor: _Executor
) -> tuple[list[ArtifactRef], list[ArtifactRef]]:
    """Run each notebook in declared order; collect fresh refs.

    Notebook source bytes come from the latest content-addressed
    snapshot (``.ockham/notebooks/<lid>/<csha>.py``), NOT the
    transient working copy ``notebooks/<live_name>.py`` — that file
    is deleted after the agent's persist step (§4.1) and only exists
    mid-edit. The canonical bytes live in the snapshot tree.

    Returns ``(new_notebook_refs, new_data_object_refs)``. Notebook
    refs keep their ``logical_id`` (slug-based identity) and pin the
    snapshot's ``content_sha``. Data-object refs are gathered from
    each run's ``KernelOutput.fetch_log`` and deduped by
    ``workspace_file_path`` (same logical_id + content_sha = same
    snapshot).
    """
    new_notebook_refs: list[ArtifactRef] = []
    new_data_object_refs: list[ArtifactRef] = []
    seen_paths: set[str] = set()

    for nb_ref in notebook_refs:
        if nb_ref.kind != "notebook":
            raise ValueError(
                f"refresh: expected kind='notebook' in notebook_refs, got {nb_ref.kind!r}"
            )
        try:
            raw, latest_csha = await read_latest_notebook(
                executor, logical_id=nb_ref.logical_id
            )
        except FileNotFoundError as e:
            raise ValueError(
                f"refresh: notebook {nb_ref.logical_id!r} has no persisted "
                "snapshot (log missing or empty)."
            ) from e
        snapshot_path = f".ockham/notebooks/{nb_ref.logical_id}/{latest_csha}.py"
        script = deserialize_notebook(raw, path=snapshot_path)
        kernel_output = await executor.execute(script.code)
        # Pin the current snapshot's content_sha so the refreshed
        # downstream artifact's log entry records the exact upstream
        # bytes that flowed in.
        new_notebook_refs.append(
            ArtifactRef(
                kind="notebook",
                logical_id=nb_ref.logical_id,
                content_sha=latest_csha or notebook_content_sha(script.code),
            )
        )
        for entry in getattr(kernel_output, "fetch_log", None) or []:
            ref = getattr(entry, "data_object_ref", None)
            if ref is None:
                continue
            key = ref.workspace_file_path
            if key in seen_paths:
                continue
            seen_paths.add(key)
            new_data_object_refs.append(ref)

    return new_notebook_refs, new_data_object_refs


def _compose_source_refs(
    *,
    original: list[ArtifactRef],
    refreshed_datasets: dict[str, ArtifactRef],
    new_data_objects: list[ArtifactRef],
) -> list[ArtifactRef]:
    """Rebuild a ``source_refs`` list after a refresh.

    Keep the original declaration order. For ``data_object`` entries,
    prefer a freshly-fetched ref (matched by ``logical_id``) and fall
    back to the original if absent (e.g. a stale ref no longer fetched
    by any re-run). For ``dataset`` entries, swap in the refreshed
    descendant. Non-matching kinds pass through.
    """
    by_logical_id = {ref.logical_id: ref for ref in new_data_objects}
    out: list[ArtifactRef] = []
    for ref in original:
        if ref.kind == "data_object":
            out.append(by_logical_id.get(ref.logical_id, ref))
        elif ref.kind == "dataset":
            out.append(refreshed_datasets.get(ref.logical_id, ref))
        else:
            out.append(ref)
    # Append any newly-fetched data_objects not already present.
    seen_ids = {r.logical_id for r in out if r.kind == "data_object"}
    for ref in new_data_objects:
        if ref.logical_id not in seen_ids:
            out.append(ref)
            seen_ids.add(ref.logical_id)
    return out


# ---------------------------------------------------------------------------
# Snapshot / curation / log primitives (executor-mediated FS)
# ---------------------------------------------------------------------------


async def _read_snapshot(executor: _Executor, ref: ArtifactRef) -> bytes:
    try:
        return await executor.read_workspace_file(ref.workspace_file_path)
    except FileNotFoundError as e:
        raise ValueError(
            f"refresh: snapshot bytes missing for {ref.kind} "
            f"{ref.logical_id}@{ref.content_sha[:8]}. The artifact may "
            "have been deleted from disk."
        ) from e


async def _load_report_curation(
    executor: _Executor, ref: ArtifactRef
) -> dict[str, Any]:
    """Read a report's editable curation sidecar as a plain dict."""
    cur_path = f".ockham/reports/{ref.logical_id}/curation.json"
    try:
        raw = await executor.read_workspace_file(cur_path)
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


_EMBEDDED_PATH_RE = re.compile(
    r"\.ockham/(?:notebooks|data_objects|datasets|charts|reports)/[^\s)\"']+"
)


def embedded_refs_from_markdown(markdown: str) -> list[ArtifactRef]:
    """Recover embedded ArtifactRefs by parsing ``.ockham/...`` paths.

    Reports embed snapshots via
    ``![](file://./.ockham/<kind>s/<lid>/<csha>.<ext>)`` — the path IS
    the ref, so we re-derive it via
    :meth:`ArtifactRef.from_workspace_file_path` rather than threading a
    duplicate field through the log. Order-preserving and dedup'd by
    ``workspace_file_path`` (same path = same ref).
    """
    seen: set[str] = set()
    out: list[ArtifactRef] = []
    for match in _EMBEDDED_PATH_RE.finditer(markdown):
        path = match.group(0)
        if path in seen:
            continue
        seen.add(path)
        ref = ArtifactRef.from_workspace_file_path(path)
        if ref is not None:
            out.append(ref)
    return out


async def _persist_layer(
    *,
    executor: _Executor,
    kind: SnapshotKind,
    artifact: Dataset | Chart | Report,
    blob: bytes,
    log_inputs: dict[str, Any],
) -> ArtifactRef:
    """Write snapshot + curation + log entry; return the new ArtifactRef.

    Mirrors :func:`server.api.workspace.artifact_registry.persist_return_artifact`
    but in-process so refresh doesn't need a terminal-side import. All
    writes are idempotent: same bytes → no-op snapshot write, dedup'd
    log line. Same identity model — ``logical_id`` from the artifact,
    ``content_sha`` from the rendered bytes.
    """
    csha = content_sha(blob)
    artifact.content_sha = csha
    ref = ArtifactRef(kind=kind, logical_id=artifact.logical_id, content_sha=csha)

    # 1. Snapshot — idempotent (same path = same bytes under
    #    content-addressing).
    snapshot_path = ref.workspace_file_path
    if not await _file_exists(executor, snapshot_path):
        await executor.write_workspace_file(snapshot_path, blob)

    # 2. Curation sidecar — overwrite with the latest fields.
    curation_path = f".ockham/{kind}s/{artifact.logical_id}/curation.json"
    await _write_curation(
        executor=executor,
        curation_path=curation_path,
        artifact=artifact,
        kind=kind,
    )

    # 3. Log entry — append-deduplicate on content_sha.
    log_path = f".ockham/{kind}s/{artifact.logical_id}/log.jsonl"
    await _append_log_dedup(
        executor=executor,
        log_path=log_path,
        content_sha=csha,
        inputs=log_inputs,
    )
    return ref


async def _file_exists(executor: _Executor, path: str) -> bool:
    try:
        await executor.read_workspace_file(path)
        return True
    except FileNotFoundError:
        return False


async def _write_curation(
    *,
    executor: _Executor,
    curation_path: str,
    artifact: Dataset | Chart | Report,
    kind: SnapshotKind,
) -> None:
    """Write the curation sidecar.

    Schema mirrors :class:`server.api.workspace.snapshot_store.Curation`
    (the canonical wire format on disk). Preserves ``created_at`` from
    any pre-existing record so the user's first-publish timestamp
    sticks across refreshes.
    """
    now = _now_iso_z()
    created_at = now
    try:
        prior = await executor.read_workspace_file(curation_path)
        prior_data = json.loads(prior.decode("utf-8"))
        if isinstance(prior_data, dict):
            existing_created = prior_data.get("created_at")
            if isinstance(existing_created, str) and existing_created:
                created_at = existing_created
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        pass

    live_name = artifact.live_name
    if live_name is None:
        from parsimony_agents.artifacts import derive_live_name

        live_name = derive_live_name(artifact.title or kind)

    payload: dict[str, Any] = {
        "kind": kind,
        "logical_id": artifact.logical_id,
        "title": artifact.title,
        "description": artifact.description,
        "tags": list(artifact.tags),
        "notes": list(artifact.notes),
        "live_name": live_name,
        "variable_name": getattr(artifact, "variable_name", None) or None,
        "created_at": created_at,
        "updated_at": now,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    await executor.write_workspace_file(curation_path, blob)


async def _append_log_dedup(
    *,
    executor: _Executor,
    log_path: str,
    content_sha: str,
    inputs: dict[str, Any],
) -> int:
    """Append a log line, dedup'd on ``content_sha``.

    Read-modify-write under the executor's serialized FS — refreshes
    are turn-serial in the agent. Returns the entry's 1-based version.
    """
    existing_text = ""
    try:
        existing = await executor.read_workspace_file(log_path)
        existing_text = existing.decode("utf-8")
    except FileNotFoundError:
        pass

    shas = _content_shas_from_jsonl(existing_text)
    if content_sha in shas:
        return shas.index(content_sha) + 1

    entry = {
        "ts": _now_iso_z(),
        "content_sha": content_sha,
        "inputs": inputs,
    }
    line = json.dumps(entry, sort_keys=True)
    if existing_text and not existing_text.endswith("\n"):
        new_text = existing_text + "\n" + line + "\n"
    else:
        new_text = existing_text + line + "\n"
    await executor.write_workspace_file(log_path, new_text.encode("utf-8"))
    return len(shas) + 1


async def _read_log(
    executor: _Executor, kind: SnapshotKind, logical_id: str
) -> list[dict[str, Any]]:
    log_path = f".ockham/{kind}s/{logical_id}/log.jsonl"
    try:
        raw = await executor.read_workspace_file(log_path)
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _content_shas_from_jsonl(jsonl_text: str) -> list[str]:
    out: list[str] = []
    for line in jsonl_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        sha = data.get("content_sha")
        if isinstance(sha, str):
            out.append(sha)
    return out


def _now_iso_z() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
