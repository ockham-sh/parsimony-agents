"""Tests for: inject_connectors, KernelOutput.fetch_log."""

from __future__ import annotations

import tempfile
from typing import Any

import pandas as pd
import pytest
from ockham.result import Provenance, Result

from ockham_agents.execution.helpers import inject_connectors
from ockham_agents.execution.outputs import FetchLogEntry


class _Exec:
    locals: dict[str, Any]


# ── inject_connectors ─────────────────────────────────────────────────────


def test_inject_connectors_sets_locals() -> None:
    ex = _Exec()
    ex.locals = {}
    sentinel = object()
    inject_connectors(ex, sentinel)
    assert ex.locals["client"] is sentinel


# ── FetchLogEntry / CodeExecutor.fetch_log ────────────────────────────────


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


@pytest.mark.asyncio
async def test_code_executor_drains_fetch_log() -> None:
    from server.execution import CodeExecutor

    cwd = tempfile.mkdtemp()
    ex = CodeExecutor(cwd=cwd)
    code = """
_fetch_log = []
_fetch_log.append({
    "source": "stub",
    "params": {"x": 1},
    "row_count": 1,
    "column_names": ["a"],
    "columns": [{"name": "a", "dtype": "auto", "role": "data"}],
    "provenance": {"source": "stub", "params": {}},
    "head": None,
    "tail": None,
})
"""
    out = await ex.execute(code)
    assert len(out.fetch_log) == 1
    assert out.fetch_log[0].source == "stub"


@pytest.mark.asyncio
async def test_code_executor_await_cell_runs() -> None:
    """Cells containing ``await`` use the async wrapper and must execute without error."""
    from server.execution import CodeExecutor

    cwd = tempfile.mkdtemp()
    ex = CodeExecutor(cwd=cwd)
    out = await ex.execute("import asyncio\nawait asyncio.sleep(0)")
    assert out.fetch_log == []


def test_result_from_dataframe_roundtrip() -> None:
    df = pd.DataFrame({"a": [1]})
    prov = Provenance(source="t", params={})
    r = Result.from_dataframe(df, prov)
    assert isinstance(r, Result)
