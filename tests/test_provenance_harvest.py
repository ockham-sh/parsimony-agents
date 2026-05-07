"""Tests for ``_resolve_sources_from_variables``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from parsimony.connector import Connectors, connector
from parsimony.result import Result
from pydantic import BaseModel

from parsimony_agents.agent.agent import _resolve_sources_from_variables
from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory


class _FetchParams(BaseModel):
    series_id: str


@connector()
async def fred_fetch(params: _FetchParams) -> Result:
    """Fetch FRED time series."""
    return Result(data=pd.DataFrame({"date": ["2024-01-01"], "value": [1.0]}))


@connector()
async def fred_search(params: _FetchParams) -> Result:
    """Search FRED time series."""
    return Result(data=pd.DataFrame({"id": ["X"], "title": ["Y"]}))


async def _make_executor(td: str) -> CodeExecutor:
    of = OutputFactory(local_dir=Path(td))
    ex = CodeExecutor(cwd=td, output_factory=of)
    await ex.set_connectors({"connectors": Connectors([fred_fetch, fred_search])})
    return ex


async def test_resolve_empty_list_returns_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        ex = await _make_executor(td)
        sources = await _resolve_sources_from_variables(ex, [])
        assert sources == []


async def test_resolve_single_fetched_variable() -> None:
    with tempfile.TemporaryDirectory() as td:
        ex = await _make_executor(td)
        await ex.execute(
            "gdp_raw = await connectors['fred_fetch'](series_id='GDPC1')\n"
        )
        sources = await _resolve_sources_from_variables(ex, ["gdp_raw"])
        assert len(sources) == 1
        assert sources[0].source == "fred_fetch"
        assert sources[0].params == {"series_id": "GDPC1"}


async def test_resolve_only_declared_variables_excluding_discovery() -> None:
    with tempfile.TemporaryDirectory() as td:
        ex = await _make_executor(td)
        await ex.execute(
            "search1 = await connectors['fred_search'](series_id='ignored')\n"
            "search2 = await connectors['fred_search'](series_id='ignored')\n"
            "gdp_raw = await connectors['fred_fetch'](series_id='GDPC1')\n"
            "unrate_raw = await connectors['fred_fetch'](series_id='UNRATE')\n"
        )
        sources = await _resolve_sources_from_variables(
            ex, ["gdp_raw", "unrate_raw"]
        )
        assert {s.source for s in sources} == {"fred_fetch"}
        assert len(sources) == 2
        assert {s.params["series_id"] for s in sources} == {"GDPC1", "UNRATE"}


async def test_resolve_unknown_variable_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        ex = await _make_executor(td)
        with pytest.raises(ValueError, match="not in the kernel"):
            await _resolve_sources_from_variables(ex, ["does_not_exist"])


async def test_resolve_non_result_variable_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        ex = await _make_executor(td)
        await ex.execute("df = pd.DataFrame({'x': [1, 2, 3]})\n")
        with pytest.raises(ValueError, match="not in the kernel or"):
            await _resolve_sources_from_variables(ex, ["df"])


async def test_resolve_carries_data_object_path_when_persister_stamps() -> None:
    with tempfile.TemporaryDirectory() as td:
        ex = await _make_executor(td)
        await ex.execute(
            "raw = await connectors['fred_fetch'](series_id='GDPC1')\n"
        )
        sources = await _resolve_sources_from_variables(ex, ["raw"])
        assert len(sources) == 1
        assert sources[0].data_object_path is not None
        assert sources[0].data_object_path.startswith(".ockham/data_objects/")


async def test_resolve_preserves_passed_order() -> None:
    with tempfile.TemporaryDirectory() as td:
        ex = await _make_executor(td)
        await ex.execute(
            "first = await connectors['fred_fetch'](series_id='A')\n"
            "second = await connectors['fred_fetch'](series_id='B')\n"
        )
        sources = await _resolve_sources_from_variables(
            ex, ["second", "first"]
        )
        assert [s.params["series_id"] for s in sources] == ["B", "A"]
