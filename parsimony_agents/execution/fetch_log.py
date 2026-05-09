"""Fetch logging callback for connector result tracking.

Captures per-fetch metadata (source, params, columns, provenance, head/tail
samples) so the agent can reason about what data it pulled and when. The
log is consumed by the executor to produce
:class:`~parsimony_agents.execution.outputs.FetchLogEntry` records on each
:class:`~parsimony_agents.execution.outputs.KernelOutput`.

Optional persister
------------------
When *persist_fn* is supplied, each fetch result is also written to a
content-addressed file under
``.ockham/data_objects/<logical_id>/<content_sha>.parquet`` (see
:mod:`parsimony_agents.execution.data_objects`). The persister returns
``(ref, version)`` — the typed :class:`ArtifactRef` and the 1-based
position of this ``content_sha`` in the data_object's ``log.jsonl``.
Both ride the entry: ``data_object_ref`` for lineage, ``version`` for
the user-facing v{N} indicator that the connector produced
fresh/refreshed data.

When *persist_fn* is ``None`` (default), fetch logs remain observational
metadata only — no ref, no version.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pandas as pd

from parsimony_agents.identity import ArtifactRef

PersistFn = Callable[
    [Any],
    Awaitable[tuple[ArtifactRef, int] | None] | tuple[ArtifactRef, int] | None,
]


def make_fetch_logger(
    persist_fn: PersistFn | None = None,
) -> tuple[list[dict[str, Any]], Callable[[Any], Awaitable[None]]]:
    """Create a fetch-log list and an async callback that appends to it.

    When *persist_fn* is supplied, it is invoked with each ``Result`` and
    its ``(ref, version)`` return is split onto the entry as
    ``data_object_ref`` and ``version`` respectively.
    """

    fetch_log: list[dict[str, Any]] = []

    async def _log_fetch(result: Any) -> None:
        entry: dict[str, Any] = {
            "provenance": result.provenance,
            "columns": [c.model_dump(mode="json") for c in result.columns],
        }
        if isinstance(result.data, pd.DataFrame):
            df = result.data
            entry["row_count"] = len(df)
            entry["column_names"] = list(df.columns)
            entry["head"] = json.loads(df.head(5).to_json(orient="table"))
            entry["tail"] = (
                json.loads(df.tail(5).to_json(orient="table")) if len(df) > 10 else None
            )
        else:
            entry["row_count"] = 1
            entry["column_names"] = []
            entry["head"] = {"data": str(result.data)[:500]}
            entry["tail"] = None
        if persist_fn is not None:
            ret = persist_fn(result)
            value = await ret if inspect.isawaitable(ret) else ret
            if value is not None:
                ref, version = value
                entry["data_object_ref"] = ref
                entry["version"] = version
        fetch_log.append(entry)

    return fetch_log, _log_fetch
