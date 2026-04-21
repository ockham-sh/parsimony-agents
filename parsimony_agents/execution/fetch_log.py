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

import json
from collections.abc import Callable
from typing import Any

import pandas as pd


def make_fetch_logger(
    persist_fn: Callable[[Any], str | None] | None = None,
) -> tuple[list[dict[str, Any]], Callable[[Any], None]]:
    """Create a fetch-log list and a callback that appends to it.

    Returns ``(fetch_log, log_fetch_callback)``. Attach the callback via
    :meth:`~parsimony.connector.Connectors.with_callback`; the executor
    drains *fetch_log* after each code execution to produce
    :class:`~parsimony_agents.execution.outputs.FetchLogEntry` records.

    When *persist_fn* is supplied, it is invoked with the live ``Result``
    and is expected to return the workspace-relative path of a
    content-addressed parquet snapshot (or ``None`` to skip stamping).
    The path is recorded on the entry as ``workspace_path``.
    """

    fetch_log: list[dict[str, Any]] = []

    def _log_fetch(result: Any) -> None:
        entry: dict[str, Any] = {
            "source": result.provenance.source,
            "source_description": result.provenance.source_description,
            "params": result.provenance.params,
            "columns": [c.model_dump(mode="json") for c in result.columns],
            "provenance": result.provenance,
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
            entry["workspace_path"] = persist_fn(result)
        fetch_log.append(entry)

    return fetch_log, _log_fetch
