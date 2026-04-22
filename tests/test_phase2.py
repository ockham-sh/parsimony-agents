"""Lightweight roundtrip checks for FetchLogEntry and Result.from_dataframe.

Executor-level coverage (fetch-log draining, await-cell handling, connector
binding) lives in the host product's test suite, where ``CodeExecutor`` is
actually wired up. The previous tests here imported ``server.execution``
from the terminal application — a layering violation that also broke
collection in a clean ``parsimony-agents`` checkout.
"""

from __future__ import annotations

import pandas as pd
from parsimony.result import Provenance, Result

from parsimony_agents.execution.outputs import FetchLogEntry


def test_fetch_log_entry_roundtrip() -> None:
    raw = {
        "source": "fred",
        "params": {"series_id": "GDPC1"},
        "row_count": 2,
        "column_names": ["date", "value"],
        "columns": [
            {"name": "date", "dtype": "datetime", "role": "data"},
            {"name": "value", "dtype": "numeric", "role": "data"},
        ],
        "provenance": {"source": "fred", "params": {"series_id": "GDPC1"}},
        "head": {"schema": {}, "data": []},
        "tail": None,
    }
    e = FetchLogEntry.model_validate(raw)
    assert e.source == "fred"
    assert e.row_count == 2
    dumped = e.model_dump(mode="json")
    e2 = FetchLogEntry.model_validate(dumped)
    assert e2.source == e.source


def test_result_from_dataframe_roundtrip() -> None:
    df = pd.DataFrame({"a": [1]})
    prov = Provenance(source="t", params={})
    r = Result.from_dataframe(df, prov)
    assert isinstance(r, Result)
