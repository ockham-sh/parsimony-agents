"""Content-addressed persistence for connector ``Result`` fetches.

Every connector fetch the executor observes is mirrored to a parquet file
in the immutable object pool:

::

    .ockham/objects/<content_sha[:2]>/<content_sha[2:]>.parquet

- ``content_sha`` = hash(canonical parquet bytes). Identical data → one file.
- ``logical_id`` = hash(canonical provenance excluding ``fetched_at``). Same
  source + same params → same logical identity on the wire, regardless of
  when or what bytes came back.

The persister returns an :class:`ArtifactRef` (``kind="data_object"``)
which the fetch logger stamps onto each :class:`FetchLogEntry` as
``data_object_ref``. Provenance lives in the parquet's embedded
``parsimony.result`` metadata — there is no per-object log or curation
sidecar.

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

import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from parsimony_agents.identity import (
    ArtifactRef,
    content_sha,
    data_object_logical_id,
    object_pool_path,
)

__all__ = ["make_data_object_persister"]


def make_data_object_persister(
    workspace_root: Path,
) -> Callable[[Any], tuple[ArtifactRef, None] | None]:
    """Build a connector callback that snapshots each ``Result``.

    The callback is synchronous — connectors are sync, so the post-fetch
    hook chain runs inline on the calling thread. Returns ``(ref, None)``
    on success or ``None`` on failure. Data objects are not versioned —
    each fetch is an immutable pool entry keyed by ``content_sha``.
    """

    root = workspace_root.resolve()

    def _persist(result: Any) -> tuple[ArtifactRef, None] | None:
        try:
            table = _canonicalize_arrow_table(result.to_arrow())
            logical_id = data_object_logical_id(result.provenance)
            buffer = io.BytesIO()
            pq.write_table(table, buffer)
            blob = buffer.getvalue()
            csha = content_sha(blob)
            ref = ArtifactRef(kind="data_object", logical_id=logical_id, content_sha=csha)
            target = root / object_pool_path(csha)
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
            return ref, None
        except Exception:
            return None

    return _persist


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
    canonical_prov_bytes = json.dumps(canonical_prov, sort_keys=True, default=str).encode("utf-8")
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
    original metadata so :func:`TabularResult.from_arrow` can recover
    provenance later.
    """
    bare_fields = [pa.field(field.name, field.type, nullable=field.nullable, metadata=None) for field in table.schema]
    bare_schema = pa.schema(bare_fields, metadata=None)
    return table.replace_schema_metadata(None).cast(bare_schema)
