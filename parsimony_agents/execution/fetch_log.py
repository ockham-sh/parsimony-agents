"""Fetch logging callback for connector result tracking.

Captures per-fetch metadata (source, params, columns, provenance, head/tail
samples) so the agent can reason about what data it pulled and when. The
log is consumed by the executor to produce
:class:`~parsimony_agents.execution.outputs.FetchLogEntry` records on each
:class:`~parsimony_agents.execution.outputs.KernelOutput`.

Optional persister
------------------
When *persist_fn* is supplied, each fetch result is also written to a
content-addressed file under ``<workspace>/.ockham/data_objects/<sha>.parquet``
(see :mod:`parsimony_agents.execution.data_objects`). The persister
returns the workspace-relative path, which is stamped on the entry as
``workspace_path``. Path is identity: that one string is the only handle
the rest of the system needs to render the data object as a clickable
artifact (notebook viewer pill, MetadataRenderer link, etc).

When *persist_fn* is ``None`` (default), fetch logs remain observational
metadata only.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pandas as pd

PersistFn = Callable[[Any], Awaitable[str | None] | str | None]


def make_fetch_logger(
    persist_fn: PersistFn | None = None,
) -> tuple[list[dict[str, Any]], Callable[[Any], Awaitable[None]]]:
    """Create a fetch-log list and an async callback that appends to it.

    When *persist_fn* is supplied, it is invoked with each ``Result`` and
    its return value (sync or async) is recorded on the entry as
    ``workspace_path`` and stamped onto ``provenance.data_object_path``.
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
            workspace_path = await ret if inspect.isawaitable(ret) else ret
            entry["workspace_path"] = workspace_path
            if workspace_path is not None:
                try:
                    result.provenance.data_object_path = workspace_path
                except Exception:
                    pass
        fetch_log.append(entry)

    return fetch_log, _log_fetch
