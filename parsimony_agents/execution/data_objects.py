"""Content-addressed persistence for connector ``Result`` fetches.

Each connector fetch the executor observes is mirrored to a parquet file
at ``<workspace_root>/.ockham/data_objects/<title_slug>_<short_sha>.parquet``.
The ``<short_sha>`` prefix (12 hex chars) of the full SHA-256 is the
dedup key; ``<title_slug>`` comes from provenance title or source. The SHA is
content-addressed (canonical provenance excluding ``fetched_at``, plus
the Arrow IPC bytes), so two fetches that produce the same data with
the same parameters dedup to one file.

Path is identity (see ``terminal/AGENTS.md``): the workspace-relative
path is the only handle the rest of the system needs to render a data
object as a clickable artifact (notebook viewer pill, MetadataRenderer
``file://`` link). The persister returns that path, which the fetch
logger stamps on each :class:`FetchLogEntry` as ``workspace_path``.

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
workspace is deleted. A sweep policy is a follow-up if disk pressure
shows.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from parsimony_agents._naming import short_sha, slug_from_title

DATA_OBJECTS_NAMESPACE = ".ockham/data_objects"


def make_data_object_persister(
    workspace_root: Path,
) -> Callable[[Any], Awaitable[str | None]]:
    """Build an async connector callback that snapshots each ``Result`` to a content-addressed parquet.

    The Arrow conversion, hashing, and parquet write run in a worker
    thread to keep the agent event loop responsive. Returns the
    workspace-relative path on success or ``None`` on failure.
    """

    root = workspace_root.resolve()

    def _do_persist_sync(result: Any) -> str | None:
        try:
            table = result.to_arrow()
            sha = _content_hash(result.provenance, table)
            title_src = result.provenance.source or "fetch"
            rel = (
                f"{DATA_OBJECTS_NAMESPACE}/{slug_from_title(title_src)}_{short_sha(sha)}.parquet"
            )
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                buffer = io.BytesIO()
                pq.write_table(table, buffer)
                blob = buffer.getvalue()
                tmp = target.with_suffix(target.suffix + ".tmp")
                try:
                    tmp.write_bytes(blob)
                    tmp.replace(target)
                except Exception:
                    if tmp.exists():
                        tmp.unlink(missing_ok=True)
                    raise
            return rel
        except Exception:
            return None

    async def _persist(result: Any) -> str | None:
        return await asyncio.to_thread(_do_persist_sync, result)

    return _persist


def _content_hash(provenance: Any, table: pa.Table) -> str:
    """SHA-256 of canonical provenance + canonical Arrow IPC bytes.

    ``fetched_at`` is excluded from provenance so identical fetches at
    different timestamps dedup. The Arrow table is serialized through
    IPC after stripping all schema/field metadata — that metadata
    embeds the same ``fetched_at`` and other instance-specific
    bookkeeping that would otherwise leak into the hash. The remaining
    bytes are the column names, dtypes, and row values — exactly what
    "same content" should mean.
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

    Used solely for hashing; the persisted parquet retains the original
    metadata so :func:`Result.from_arrow` can recover provenance later.
    """

    bare_fields = [
        pa.field(field.name, field.type, nullable=field.nullable, metadata=None)
        for field in table.schema
    ]
    bare_schema = pa.schema(bare_fields, metadata=None)
    return table.replace_schema_metadata(None).cast(bare_schema)
