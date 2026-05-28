"""Tests for: normalize_connector_bundles, KernelOutput.fetch_log, executor wiring.

The two ``CodeExecutor`` tests use ``pytest.importorskip`` on
``server.execution`` (terminal repo's executor package). They run when
the workspace contains the terminal checkout and skip cleanly in a
standalone parsimony-agents CI environment.
"""

from __future__ import annotations

import tempfile

import pandas as pd
import pytest
from parsimony.connector import Connectors
from parsimony.result import Result, TabularResult

from parsimony_agents.execution.helpers import normalize_connector_bundles
from parsimony_agents.execution.outputs import FetchLogEntry


def test_normalize_connector_bundles_default_binding() -> None:
    bundle = Connectors([])
    out = normalize_connector_bundles(bundle)
    assert out == {"client": bundle}


def test_normalize_connector_bundles_mapping_passthrough() -> None:
    a, b = Connectors([]), Connectors([])
    out = normalize_connector_bundles({"fred": a, "fmp": b})
    assert out == {"fred": a, "fmp": b}


def test_normalize_connector_bundles_none_is_empty() -> None:
    assert normalize_connector_bundles(None) == {}


def test_normalize_connector_bundles_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        normalize_connector_bundles([])  # type: ignore[arg-type]


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
        "provenance": {
            "source": "fred_fetch",
            "source_description": "St. Louis Fed FRED",
            "params": {"series_id": "GDPC1"},
        },
        "head": {"schema": {}, "data": []},
        "tail": None,
    }
    e = FetchLogEntry.model_validate(raw)
    assert e.source == "fred_fetch"
    assert e.row_count == 2
    dumped = e.model_dump(mode="json")
    e2 = FetchLogEntry.model_validate(dumped)
    assert e2.source == e.source
    assert e2.provenance.source_description == "St. Louis Fed FRED"
    assert e2.provenance.params == {"series_id": "GDPC1"}


@pytest.mark.asyncio
async def test_code_executor_drains_fetch_log() -> None:
    server_execution = pytest.importorskip(
        "server.execution",
        reason="requires terminal repo's server.execution package",
    )
    CodeExecutor = server_execution.CodeExecutor

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
    server_execution = pytest.importorskip(
        "server.execution",
        reason="requires terminal repo's server.execution package",
    )
    CodeExecutor = server_execution.CodeExecutor

    cwd = tempfile.mkdtemp()
    ex = CodeExecutor(cwd=cwd)
    out = await ex.execute("import asyncio\nawait asyncio.sleep(0)")
    assert out.fetch_log == []


def test_tabular_result_from_dataframe_roundtrip() -> None:
    df = pd.DataFrame({"a": [1]})
    r = TabularResult.from_dataframe(df)
    assert isinstance(r, TabularResult)
    assert isinstance(r, Result)
