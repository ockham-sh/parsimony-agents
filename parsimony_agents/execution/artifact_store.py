"""Persist ``return_*`` deliverables into the content-addressed ``.ockham/`` tree.

This is the framework's single, opinionated artifact registry. It turns a typed
``Dataset`` / ``Chart`` / ``Report`` (and the notebook recipe behind them) into the
on-disk layout the rest of the agent already reads from::

    .ockham/<kind>s/<logical_id>/curation.json        # editable sidecar
    .ockham/<kind>s/<logical_id>/log.jsonl            # append-only history
    .ockham/<kind>s/<logical_id>/<content_sha>.<ext>  # immutable snapshot

All I/O goes through the executor's ``read_workspace_file`` /
``write_workspace_file`` — the same abstraction code execution uses — so the same
registry works whether the executor is in-process (local fs) or a remote sandbox.
That executor boundary is the only storage seam; the registry layout, content
hashing, log dedup, and lineage ``inputs`` are uniform across every host.

Connector fetches (``data_object``) persist separately, content-addressed in the
flat object pool, via :func:`parsimony_agents.execution.data_objects.make_data_object_persister`.
This module is the deliverable-side counterpart.

The report trust boundary (executable fences / active HTML / out-of-allowlist
refs) is host-injected via ``report_validator`` and enforced here at write time —
see :func:`persist_artifact` — so it is the single chokepoint every report write
(return_report and refresh) routes through. Other host concerns stay in the
terminal layer: blob durability / sync-back, multi-tenant routing,
etag-on-the-wire, and archival compaction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Protocol

from parsimony_agents.artifacts import Chart, Dataset, Report, derive_live_name
from parsimony_agents.chart_io import write_chart_bytes
from parsimony_agents.dataset_io import write_dataset_bytes
from parsimony_agents.identity import (
    ArtifactRef,
    SnapshotKind,
    content_sha,
)
from parsimony_agents.notebook import Script
from parsimony_agents.notebook_io import serialize_notebook

__all__ = [
    "PersistExecutor",
    "ReportValidationError",
    "ReportValidator",
    "SnapshotIntegrityError",
    "log_inputs_for",
    "persist_artifact",
    "persist_notebook",
    "render_artifact_bytes",
]


class ReportValidationError(ValueError):
    """A report body failed the host-injected ``report_validator`` and was not persisted.

    A distinct type so the agent loop can tell a *trust-boundary rejection* (the
    agent must rewrite the body — drop active HTML / executable fences / disallowed
    refs) apart from a generic persist/storage failure. Raised by
    :func:`persist_artifact` before any bytes are written, so an unsafe report never
    reaches the workspace tree.
    """


class SnapshotIntegrityError(RuntimeError):
    """A snapshot's stored bytes do not match the bytes that were written.

    Raised by the verify-after-write step in :func:`persist_artifact` /
    :func:`persist_notebook`: a freshly written snapshot is read straight back
    and its bytes must equal those passed in, or the write is a partial write /
    storage fault and must not be treated as durable. Snapshots are immutable
    and content-addressed, so this byte round-trip is the integrity floor every
    downstream reader (and the host's archival compaction) builds on.
    """


class PersistExecutor(Protocol):
    """Minimal executor surface the registry needs (a subset of BaseCodeExecutor)."""

    async def read_workspace_file(self, path: str) -> bytes: ...
    async def write_workspace_file(self, path: str, data: bytes) -> None: ...


class ReportValidator(Protocol):
    """Host-injected trust-boundary check for agent-authored report bodies.

    Called by :func:`persist_artifact` for ``kind="report"`` BEFORE the snapshot is
    written, so an unsafe body (active HTML, executable code fences, out-of-allowlist
    file refs) never reaches the workspace tree and every downstream read/render path
    can trust the on-disk bytes. Returns ``None`` when the body is safe; raises on
    rejection. ``pin_map_keys`` is the set of ``live_name`` strings the snapshot's
    frozen pin map provides — when given, every body ref must resolve within it.

    The standalone agent injects nothing (the author reads their own output); a
    workspace host (terminal) injects ``validate_report_body``.
    """

    def __call__(self, body: str, *, pin_map_keys: frozenset[str] | None = None) -> None: ...


# ---------------------------------------------------------------------------
# Byte rendering + lineage inputs (per kind)
# ---------------------------------------------------------------------------


def render_artifact_bytes(artifact: Dataset | Chart | Report, kind: SnapshotKind) -> bytes:
    """Render a typed deliverable to its snapshot bytes.

    Raises ``ValueError`` when the artifact is missing the payload / lineage the
    codec requires — these are framework invariants the ``return_*`` tools must
    satisfy before persistence.
    """
    match artifact:
        case Dataset() as dataset:
            if dataset.payload is None:
                raise ValueError(
                    f"Dataset(logical_id={dataset.logical_id!r}) is missing its live "
                    "payload; return_dataset must attach it via .with_payload(...)."
                )
            return write_dataset_bytes(dataset, dataset.payload)
        case Chart() as chart:
            if chart.payload is None:
                raise ValueError(
                    f"Chart(logical_id={chart.logical_id!r}) is missing its live "
                    "payload; return_chart must attach it via .with_payload(...)."
                )
            if chart.notebook_ref is None:
                raise ValueError(
                    f"Chart(logical_id={chart.logical_id!r}) has no notebook_ref; "
                    "the framework requires a notebook ref at persist time."
                )
            return write_chart_bytes(chart, chart.payload)
        case Report() as report:
            if not report.markdown.strip():
                raise ValueError(
                    f"Report(logical_id={report.logical_id!r}) has empty markdown; "
                    "return_report must populate it at construction time."
                )
            # The agent-authored report bytes, rendered as-is. The trust-boundary
            # check (executable fences / active HTML / out-of-allowlist refs) runs
            # in persist_artifact BEFORE these bytes are written, whenever the host
            # injects a report_validator — so the on-disk snapshot is safe for every
            # read/render path. A standalone CLI user injects none and reads their
            # own output, so nothing is rejected for them.
            return report.snapshot_bytes()
        case _:
            raise ValueError(f"Unsupported artifact type for persistence: {type(artifact).__name__}")


def log_inputs_for(artifact: Dataset | Chart | Report, kind: SnapshotKind) -> dict[str, Any]:
    """Per-kind ``inputs`` payload recorded on the log line (lineage by content_sha)."""
    match artifact:
        case Dataset() as dataset:
            return {
                "notebooks": [r.content_sha for r in dataset.notebook_refs],
                "sources": [r.content_sha for r in dataset.source_refs],
            }
        case Chart() as chart:
            return {
                "notebook": chart.notebook_ref.content_sha if chart.notebook_ref else None,
                "source_datasets": [r.content_sha for r in chart.source_dataset_refs],
                "sources": [r.content_sha for r in chart.source_refs],
            }
        case Report() as report:
            return {
                "embedded": [r.content_sha for r in report.embedded_refs],
                "formats": list(report.formats),
            }
        case _:
            return {}


# ---------------------------------------------------------------------------
# Persist entry points
# ---------------------------------------------------------------------------


async def persist_artifact(
    executor: PersistExecutor,
    *,
    kind: SnapshotKind,
    artifact: Dataset | Chart | Report,
    blob: bytes,
    log_inputs: dict[str, Any],
    report_validator: ReportValidator | None = None,
) -> ArtifactRef:
    """Write snapshot + curation + log for a dataset/chart/report; return its ref.

    Stamps ``artifact.content_sha`` from the rendered bytes (so the caller's mint
    check sees it). Idempotent: identical bytes → no-op snapshot write and a
    deduped log line; the curation sidecar is always overwritten with the latest
    title/description/tags. ``artifact.logical_id`` must already be set by the
    framework.

    For ``kind="report"`` with a ``report_validator`` present, the body is validated
    BEFORE any bytes are written; an unsafe body raises :class:`ReportValidationError`
    and nothing is persisted. This is the single write-time chokepoint — every report
    writer (return_report and refresh) routes through here — so an unsafe body can
    never reach disk when a host validator is in play.
    """
    if not artifact.logical_id:
        raise ValueError(
            f"{type(artifact).__name__}(logical_id='') — persist requires the "
            "framework to set logical_id before persistence."
        )

    # Trust boundary (reports only), enforced before any write so the on-disk
    # snapshot is safe for every reader. Host errors are normalized to
    # ReportValidationError so the caller can distinguish a body-rejection (rewrite
    # the report) from a storage failure.
    if kind == "report" and report_validator is not None and isinstance(artifact, Report):
        try:
            report_validator(artifact.markdown, pin_map_keys=frozenset(artifact.live_name_pins or {}))
        except ReportValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize any host validator error
            raise ReportValidationError(str(exc)) from exc

    csha = content_sha(blob)
    artifact.content_sha = csha
    ref = ArtifactRef(kind=kind, logical_id=artifact.logical_id, content_sha=csha)

    await _write_snapshot_verified(executor, path=ref.workspace_file_path, blob=blob)

    live_name = artifact.live_name or derive_live_name(artifact.title or kind)
    await _write_curation(
        executor,
        curation_path=f".ockham/{kind}s/{artifact.logical_id}/curation.json",
        base={
            "kind": kind,
            "logical_id": artifact.logical_id,
            "title": artifact.title,
            "description": artifact.description,
            "tags": list(artifact.tags),
            "notes": list(artifact.notes),
            "live_name": live_name,
            "variable_name": getattr(artifact, "variable_name", None) or None,
        },
    )
    await _append_log_dedup(
        executor,
        log_path=f".ockham/{kind}s/{artifact.logical_id}/log.jsonl",
        content_sha=csha,
        inputs=log_inputs,
    )
    return ref


async def persist_notebook(
    executor: PersistExecutor,
    *,
    ref: ArtifactRef,
    code: str,
    notebook_path: str,
) -> ArtifactRef:
    """Write a notebook recipe's snapshot + curation + log.

    The producing-recipe counterpart to :func:`persist_artifact`. ``ref`` is the
    already-computed notebook ref (``content_sha = notebook_content_sha(code)``);
    the snapshot bytes are ``serialize_notebook(...)`` written to that ref's path
    (the notebook convention: the filename hashes the *code*, the file holds the
    serialized form). Without this, ``return_dataset``'s "recipe published?"
    lineage check fails and deliverables can't be minted.

    The notebook curation carries only a ``live_name`` (the path slug); notebooks
    have no agent-authored title/description. The UI-only kernel-output state
    cache the terminal host writes is intentionally skipped here.
    """
    from parsimony_agents.identity import notebook_logical_id

    if ref.kind != "notebook":
        raise ValueError(f"persist_notebook: expected kind='notebook', got {ref.kind!r}")

    raw = serialize_notebook(Script(path=notebook_path, code=code))
    await _write_snapshot_verified(executor, path=ref.workspace_file_path, blob=raw)

    try:
        live_name = notebook_logical_id(notebook_path)
    except ValueError:
        live_name = ref.logical_id

    await _write_curation(
        executor,
        curation_path=f".ockham/notebooks/{ref.logical_id}/curation.json",
        base={
            "kind": "notebook",
            "logical_id": ref.logical_id,
            "title": "",
            "description": "",
            "tags": [],
            "notes": [],
            "live_name": live_name,
            "variable_name": None,
        },
    )
    await _append_log_dedup(
        executor,
        log_path=f".ockham/notebooks/{ref.logical_id}/log.jsonl",
        content_sha=ref.content_sha,
        inputs={"path": notebook_path},
    )
    return ref


# ---------------------------------------------------------------------------
# Snapshot / curation / log primitives (executor-mediated FS)
# ---------------------------------------------------------------------------


async def _file_exists(executor: PersistExecutor, path: str) -> bool:
    try:
        await executor.read_workspace_file(path)
        return True
    except FileNotFoundError:
        return False


async def _write_snapshot_verified(executor: PersistExecutor, *, path: str, blob: bytes) -> None:
    """Write an immutable content-addressed snapshot with verify-after-write.

    Idempotent: an existing path is left untouched (content-addressed, so the
    same path implies the same bytes — verified when first written). A fresh
    write is read straight back and its bytes must equal those passed in, or
    :class:`SnapshotIntegrityError` is raised so a partial write / storage fault
    surfaces before the snapshot is treated as durable. This is the same
    integrity floor the terminal host applies at its own write boundary; here it
    rides the executor seam so it holds for every backend.

    Skip-on-exists assumes the executor's ``write_workspace_file`` is atomic
    (tmp-write + rename), so a failed write never leaves a partial file at the
    final path that a later call would then accept unverified. Both supported
    backends satisfy this (the local-fs ``CodeExecutor`` and the terminal sandbox
    both write via tmp + atomic replace); a backend that does not must re-verify.
    """
    if await _file_exists(executor, path):
        return
    await executor.write_workspace_file(path, blob)
    written = await executor.read_workspace_file(path)
    if written != blob:
        raise SnapshotIntegrityError(
            f"snapshot at {path} read back {len(written)} bytes that differ from "
            f"the {len(blob)} bytes written — stored bytes are corrupt"
        )


async def _write_curation(
    executor: PersistExecutor,
    *,
    curation_path: str,
    base: dict[str, Any],
) -> None:
    """Write a curation sidecar, preserving first-publish ``created_at``.

    Schema mirrors the canonical on-disk curation (``snapshot_store.Curation`` in
    the host). ``base`` carries the kind-specific fields; this adds the
    timestamps and writes deterministic JSON.
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

    payload = {**base, "created_at": created_at, "updated_at": now}
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    await executor.write_workspace_file(curation_path, blob)


async def _append_log_dedup(
    executor: PersistExecutor,
    *,
    log_path: str,
    content_sha: str,
    inputs: dict[str, Any],
) -> int:
    """Append a log line, deduplicated on ``content_sha``; return its 1-based version.

    Read-modify-write under the executor's serialized FS — agent turns are
    serial, so no lock is needed for the single-terminal case. Multi-terminal
    concurrency (the host) is a separate concern.
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

    entry = {"ts": _now_iso_z(), "content_sha": content_sha, "inputs": inputs}
    line = json.dumps(entry, sort_keys=True)
    if existing_text and not existing_text.endswith("\n"):
        new_text = existing_text + "\n" + line + "\n"
    else:
        new_text = existing_text + line + "\n"
    await executor.write_workspace_file(log_path, new_text.encode("utf-8"))
    return len(shas) + 1


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
