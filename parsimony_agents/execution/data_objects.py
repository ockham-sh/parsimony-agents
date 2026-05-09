"""Content-addressed persistence for connector ``Result`` fetches.

Every connector fetch the executor observes is mirrored to a parquet
file under the dual-identity layout described in
``CONTENT_ADDRESSED_ARTIFACTS_PLAN.md`` §2.3:

::

    .ockham/data_objects/<logical_id>/log.jsonl
    .ockham/data_objects/<logical_id>/<content_sha>.parquet

- ``logical_id`` = hash(canonical_provenance excluding ``fetched_at``).
  Same source + same params → same logical_id, regardless of when or
  what bytes came back.
- ``content_sha`` = hash(canonical Arrow IPC bytes). Refreshes
  propagate on this axis only.

The persister returns an :class:`ArtifactRef` (``kind="data_object"``)
which the fetch logger stamps onto each :class:`FetchLogEntry` as
``data_object_ref``. That typed ref is the single handle the rest of
the system uses to reference the snapshot — it carries both the
logical and content axes.

Why local-FS, not the host's storage service
--------------------------------------------
The executor already runs against a materialized workspace directory
(``CodeExecutor.cwd``). For the local backend, that *is* the canonical
store; for remote backends, the host's ``sync_back`` step pushes the
materialized directory back to blob storage at end-of-turn. Writing
straight to the local FS therefore reaches blob storage by the same
path every other workspace file takes — no new transport, no sync
shim, no per-fetch round trip.

GC: none. The cache grows monotonically per workspace until the
workspace is deleted.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from parsimony_agents.identity import (
    ArtifactRef,
    content_sha,
    data_object_logical_id,
)

DATA_OBJECTS_NAMESPACE = ".ockham/data_objects"


def make_data_object_persister(
    workspace_root: Path,
) -> Callable[[Any], Awaitable[tuple[ArtifactRef, int] | None]]:
    """Build an async connector callback that snapshots each ``Result``.

    Returns ``(ref, version)`` on success or ``None`` on failure.
    ``version`` is the 1-based position of this ``content_sha`` in the
    data_object's ``log.jsonl`` — same semantic as datasets/charts/reports
    under the unified versioning model. Identical-content republishes
    return the existing version (full dedup).
    """

    root = workspace_root.resolve()

    def _do_persist_sync(result: Any) -> tuple[ArtifactRef, int] | None:
        try:
            table = _canonicalize_arrow_table(result.to_arrow())
            logical_id = data_object_logical_id(result.provenance)
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            blob = buffer.getvalue()
            csha = content_sha(blob)
            ref = ArtifactRef(
                kind="data_object", logical_id=logical_id, content_sha=csha
            )
            target = root / ref.workspace_file_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                tmp = target.with_suffix(target.suffix + ".tmp")
                try:
                    tmp.write_bytes(blob)
                    tmp.replace(target)
                except Exception:
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)
                    raise
            version = _append_log_entry(root, logical_id, csha)
            _write_curation_sidecar(root, logical_id, result.provenance)
            return ref, version
        except Exception:
            return None

    async def _persist(result: Any) -> tuple[ArtifactRef, int] | None:
        return await asyncio.to_thread(_do_persist_sync, result)

    return _persist


def _append_log_entry(root: Path, logical_id: str, csha: str) -> int:
    """Append a JSONL line and return the 1-based position of *csha* in the log.

    Full content_sha dedup: any prior line with matching ``content_sha``
    short-circuits the write and the function returns its existing
    index. Same semantic as
    :func:`server.api.workspace.snapshot_store.append_dedup_jsonl` —
    duplicated here because parsimony-agents must not import from the
    terminal layer. Lock on a sentinel sibling file to serialize
    concurrent writes.
    """
    log_path = root / DATA_OBJECTS_NAMESPACE / logical_id / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = log_path.parent / (log_path.name + ".lock")
    import fcntl

    with open(lock_path, "ab") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            existing = log_path.read_bytes() if log_path.exists() else b""
            shas = _content_shas_from_jsonl(existing.decode("utf-8"))
            if csha in shas:
                return shas.index(csha) + 1
            entry = {
                "ts": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "content_sha": csha,
                "inputs": {},
            }
            line = json.dumps(entry, sort_keys=True)
            with open(log_path, "ab") as f:
                if existing and not existing.endswith(b"\n"):
                    f.write(b"\n")
                f.write(line.encode("utf-8"))
                f.write(b"\n")
                f.flush()
            return len(shas) + 1
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


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


def _write_curation_sidecar(root: Path, logical_id: str, provenance: Any) -> None:
    """Write ``curation.json`` next to the parquet with the provenance record.

    Mirrors :class:`server.api.workspace.snapshot_store.DataObjectCuration` —
    we cannot import from the terminal layer (the dependency points the
    other way), so the schema is duplicated as a plain dict. The terminal
    reader (``read_curation``) validates the JSON against
    ``DataObjectCuration`` on load; field drift is caught there.

    Idempotent: ``created_at`` is preserved across rewrites by reading the
    existing file first. Last-writer-wins on identical content is fine
    (same logical_id + same canonical params → same payload).
    """
    sidecar = root / DATA_OBJECTS_NAMESPACE / logical_id / "curation.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    safe = provenance.safe_dump()
    fetched_at = safe.get("fetched_at")
    if fetched_at is not None and not isinstance(fetched_at, str):
        fetched_at = str(fetched_at)

    created_at = now
    if sidecar.exists():
        try:
            prior = json.loads(sidecar.read_text(encoding="utf-8"))
            prior_created = prior.get("created_at")
            if isinstance(prior_created, str) and prior_created:
                created_at = prior_created
        except (OSError, json.JSONDecodeError):
            pass

    payload = {
        "kind": "data_object",
        "logical_id": logical_id,
        "source": str(safe.get("source") or ""),
        "source_description": str(safe.get("source_description") or ""),
        "params": safe.get("params") or {},
        "fetched_at": fetched_at,
        "properties": safe.get("properties") or {},
        "created_at": created_at,
        "updated_at": now,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    try:
        tmp.write_bytes(blob)
        tmp.replace(sidecar)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


_RESULT_SCHEMA_META_KEY = b"parsimony.result"


def _canonicalize_arrow_table(table: pa.Table) -> pa.Table:
    """Strip ``fetched_at`` from the embedded ``parsimony.result`` metadata.

    The arrow table coming out of :meth:`Result.to_arrow` carries the
    full provenance JSON in schema metadata under
    ``parsimony.result``. That JSON includes ``fetched_at`` — but
    ``fetched_at`` does NOT participate in the artifact's logical
    identity (§2.2), so two fetches with identical data must produce
    identical parquet bytes for content-addressing to dedup. We strip
    ``fetched_at`` from the JSON envelope before write so the parquet
    bytes are stable across refreshes.

    Field-level metadata is left untouched.
    """
    raw_meta = table.schema.metadata or {}
    if _RESULT_SCHEMA_META_KEY not in raw_meta:
        return table
    payload_bytes = raw_meta[_RESULT_SCHEMA_META_KEY]
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return table
    prov = payload.get("provenance")
    if isinstance(prov, dict) and "fetched_at" in prov:
        prov.pop("fetched_at", None)
    new_meta = dict(raw_meta)
    new_meta[_RESULT_SCHEMA_META_KEY] = json.dumps(payload, default=str).encode("utf-8")
    return table.replace_schema_metadata(new_meta)


def _content_hash_legacy(provenance: Any, table: pa.Table) -> str:
    """SHA-256 of canonical provenance + canonical Arrow IPC bytes.

    Retained for tests that pin the legacy hash. Not used by the
    persister anymore — the new layout content-addresses by parquet
    bytes (``content_sha(blob)``), which is invariant under read paths
    that consume parquet directly.
    """
    canonical_prov = provenance.model_dump(mode="json", exclude={"fetched_at"})
    canonical_prov_bytes = json.dumps(
        canonical_prov, sort_keys=True, default=str
    ).encode("utf-8")
    canonical_table = _strip_table_metadata(table)
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, canonical_table.schema) as writer:
        writer.write_table(canonical_table)
    h = hashlib.sha256()
    h.update(canonical_prov_bytes)
    h.update(b"\x00")
    h.update(sink.getvalue())
    return h.hexdigest()


def _strip_table_metadata(table: pa.Table) -> pa.Table:
    """Return *table* with schema- and field-level metadata cleared.

    Used solely for the legacy hash; the persisted parquet retains
    original metadata so :func:`Result.from_arrow` can recover
    provenance later.
    """
    bare_fields = [
        pa.field(field.name, field.type, nullable=field.nullable, metadata=None)
        for field in table.schema
    ]
    bare_schema = pa.schema(bare_fields, metadata=None)
    return table.replace_schema_metadata(None).cast(bare_schema)
