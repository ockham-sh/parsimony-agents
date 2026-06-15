"""Fetch logging callback for connector result tracking.

Captures per-fetch metadata (source, params, columns, provenance,
head/tail samples) so the agent can reason about what data it pulled
and when. The log is consumed by the executor to produce
:class:`~parsimony_agents.execution.outputs.FetchLogEntry` records on
each :class:`~parsimony_agents.execution.outputs.KernelOutput`.

When ``persist_fn`` is supplied, each fetch result is also written to a
content-addressed file under
``.ockham/objects/<content_sha[:2]>/<content_sha[2:]>.parquet`` and the
returned :class:`ArtifactRef` is recorded on the active
:class:`~parsimony_agents.execution.run_scope.RunScope` (if any), so
the producing notebook's lineage automatically accumulates fetch edges
— the agent never types them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pandas as pd

from parsimony_agents.execution.run_scope import OriginLedger
from parsimony_agents.identity import ArtifactRef

PersistFn = Callable[[Any], tuple[ArtifactRef, int | None] | None]


def make_fetch_logger(
    persist_fn: PersistFn | None = None,
    *,
    ledger: OriginLedger | None = None,
) -> tuple[list[dict[str, Any]], Callable[[Any], None]]:
    """Create a fetch-log list and a callback that appends to it.

    The callback is synchronous: connectors are sync, so the post-fetch
    hook chain runs inline on the calling thread.

    When *persist_fn* is supplied, it is invoked with each ``Result`` and
    its ``(ref, version)`` return is split onto the entry as
    ``data_object_ref`` and ``version`` respectively. ``version`` is
    always ``None`` for immutable object-pool entries.

    When *ledger* is supplied, the data_object ref is also recorded on
    the ledger's current :class:`RunScope` so the producing notebook's
    fetch lineage accumulates automatically.
    """

    fetch_log: list[dict[str, Any]] = []

    def _log_fetch(result: Any) -> None:
        entry: dict[str, Any] = {
            "provenance": result.provenance,
            "columns": [c.model_dump(mode="json") for c in result.columns],
        }
        if isinstance(result.data, pd.DataFrame):
            df = result.data
            entry["row_count"] = len(df)
            entry["column_names"] = list(df.columns)
            entry["head"] = json.loads(df.head(5).to_json(orient="table"))
            entry["tail"] = json.loads(df.tail(5).to_json(orient="table")) if len(df) > 10 else None
        else:
            entry["row_count"] = 1
            entry["column_names"] = []
            entry["head"] = {"data": str(result.data)[:500]}
            entry["tail"] = None
        if persist_fn is not None:
            value = persist_fn(result)
            if value is not None:
                ref, version = value
                entry["data_object_ref"] = ref
                entry["version"] = version
                if ledger is not None and ledger.current is not None:
                    ledger.current.record_fetch(ref)
        fetch_log.append(entry)

    return fetch_log, _log_fetch
