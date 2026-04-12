"""Fetch logging callback for connector result tracking."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pandas as pd


def make_fetch_logger() -> tuple[list[dict[str, Any]], Callable[[Any], None]]:
    """Create a fetch-log list and a callback that appends to it.

    Returns ``(fetch_log, log_fetch_callback)``.  Attach the callback via
    :meth:`~parsimony.connector.Connectors.with_callback`; the executor
    drains *fetch_log* after each code execution to produce
    :class:`~parsimony_agents.execution.outputs.FetchLogEntry` records.
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
        fetch_log.append(entry)

    return fetch_log, _log_fetch
