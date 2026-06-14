"""Tests for the new no-refs return surface.

Brief §5: the agent never types a ref. ``return_dataset`` /
``return_chart`` derive lineage from the executor's origin ledger;
``return_report`` derives embedded refs from the markdown body itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import altair as alt
import pandas as pd
import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.models import AgentContext
from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.execution.run_scope import OriginLedger, VariableOrigin
from parsimony_agents.identity import ArtifactRef, notebook_content_sha
from parsimony_agents.notebook import Script
from parsimony_agents.notebook_io import serialize_notebook


def _bare_executor(tmp_path: Path) -> MagicMock:
    """Build a MagicMock executor seeded with a producer notebook snapshot."""
    files: dict[str, bytes] = {}

    code = "result_df = 1\n"
    csha = notebook_content_sha(code)
    files[f".ockham/notebooks/producer/{csha}.py"] = serialize_notebook(Script(path="notebooks/producer.py", code=code))
    files[".ockham/notebooks/producer/log.jsonl"] = (
        json.dumps({"ts": "t1", "content_sha": csha, "inputs": {}}) + "\n"
    ).encode("utf-8")

    ex = MagicMock()
    ex.origin_ledger = OriginLedger()
    ex.get_origin = AsyncMock(side_effect=lambda name: ex.origin_ledger.get(name))
    ex.cwd = str(tmp_path)
    ex.write_workspace_file = AsyncMock(side_effect=lambda p, d: files.update({p: d}))

    async def _read(p):
        if p not in files:
            raise FileNotFoundError(p)
        return files[p]

    ex.read_workspace_file = AsyncMock(side_effect=_read)
    ex.execute = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.clear_namespace = AsyncMock()
    ex.set_cwd = AsyncMock()
    ex.set_connectors = AsyncMock()
    ex.eval = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.delete_workspace_file = AsyncMock()

    async def _list(prefix: str = "") -> list[tuple[str, int]]:
        return [(p, len(d)) for p, d in files.items() if p.startswith(prefix)]

    ex.list_workspace_files = AsyncMock(side_effect=_list)
    ex.execute_workspace = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.get_locals = MagicMock(return_value={})
    return ex, csha


@pytest.mark.asyncio
async def test_return_dataset_derives_lineage_from_origin(tmp_path: Path) -> None:
    ex, csha = _bare_executor(tmp_path)
    factory = OutputFactory(local_dir=str(tmp_path))
    df_out = factory.from_value(pd.DataFrame({"x": [1, 2]}), ref="result_df")
    ex.get = AsyncMock(return_value=df_out)
    ex.origin_ledger._origins["result_df"] = VariableOrigin(
        notebook_path="notebooks/producer.py",
        load_refs=(),
        fetch_refs=(),
    )

    agent = Agent(model="m", code_executor=ex)
    ctx = AgentContext(session_id="s")
    r = await agent.return_dataset(
        context=ctx,
        dataset_variable_name="result_df",
        title="Result",
        description="x",
        notes=[],
        live_name="my_result",
    )
    assert r.success, getattr(r, "exception_message", "")
    ds = r.data
    assert isinstance(ds, Dataset)
    assert ds.live_name == "my_result"
    assert len(ds.notebook_refs) == 1
    assert ds.notebook_refs[0].kind == "notebook"
    assert ds.notebook_refs[0].logical_id == "producer"


@pytest.mark.asyncio
async def test_return_dataset_rejects_variable_without_origin(tmp_path: Path) -> None:
    ex, _ = _bare_executor(tmp_path)
    factory = OutputFactory(local_dir=str(tmp_path))
    df_out = factory.from_value(pd.DataFrame({"x": [1]}), ref="result_df")
    ex.get = AsyncMock(return_value=df_out)
    # No origin seeded → "publish a notebook first" rule fires.

    agent = Agent(model="m", code_executor=ex)
    ctx = AgentContext(session_id="s")
    r = await agent.return_dataset(
        context=ctx,
        dataset_variable_name="result_df",
        title="Result",
        description="x",
        notes=[],
        live_name="my_result",
    )
    assert not r.success
    assert "producing notebook" in r.exception_message.lower() or "scratch" in r.exception_message.lower()


@pytest.mark.asyncio
async def test_return_chart_partitions_load_and_fetch(tmp_path: Path) -> None:
    ex, _ = _bare_executor(tmp_path)
    factory = OutputFactory(local_dir=str(tmp_path))
    fig_out = factory.from_value(
        alt.Chart(pd.DataFrame({"x": [1]})).mark_line().encode(x="x:Q"),
        ref="result_chart",
    )
    ex.get = AsyncMock(return_value=fig_out)
    load_ref = ArtifactRef(kind="dataset", logical_id="ds-a", content_sha="cs-a")
    fetch_ref = ArtifactRef(kind="data_object", logical_id="do-x", content_sha="cs-x")
    ex.origin_ledger._origins["result_chart"] = VariableOrigin(
        notebook_path="notebooks/producer.py",
        load_refs=(load_ref,),
        fetch_refs=(fetch_ref,),
    )

    agent = Agent(model="m", code_executor=ex)
    ctx = AgentContext(session_id="s")
    r = await agent.return_chart(
        context=ctx,
        title="A chart",
        chart_variable_name="result_chart",
        description="x",
        notes=[],
        live_name="chart_one",
    )
    assert r.success, getattr(r, "exception_message", "")
    ch = r.data
    assert isinstance(ch, Chart)
    assert ch.source_dataset_refs == [load_ref]
    assert ch.source_refs == [fetch_ref]


@pytest.mark.asyncio
async def test_return_report_pins_embedded_live_names(tmp_path: Path) -> None:
    """Body URIs name embeds by live_name; return_report resolves each against
    curation and freezes the pin map into the snapshot bytes."""
    import json as _json

    from parsimony_agents.identity import content_sha

    ex, _ = _bare_executor(tmp_path)
    agent = Agent(model="m", code_executor=ex)
    ctx = AgentContext(session_id="s")

    # Seed a published dataset with live_name "sales".
    blob = b"FAKE_PARQUET_BYTES"
    csha = content_sha(blob)
    await ex.write_workspace_file(f".ockham/datasets/lid_x/{csha}.parquet", blob)
    await ex.write_workspace_file(
        ".ockham/datasets/lid_x/curation.json",
        _json.dumps({"live_name": "sales"}).encode("utf-8"),
    )
    await ex.write_workspace_file(
        ".ockham/datasets/lid_x/log.jsonl",
        (_json.dumps({"ts": "t1", "content_sha": csha, "inputs": {}}) + "\n").encode("utf-8"),
    )

    md = "# Title\n\nSee the dataset:\n\n![](file://./data/sales.parquet)\n"
    r = await agent.return_report(
        context=ctx,
        title="My Title",
        markdown=md,
        description="x",
        notes=[],
        live_name="my_report",
    )
    assert r.success, getattr(r, "exception_message", "")
    rep = r.data
    assert isinstance(rep, Report)
    assert len(rep.embedded_refs) == 1
    assert rep.embedded_refs[0].kind == "dataset"
    assert rep.embedded_refs[0].logical_id == "lid_x"
    # The pin map travels with the snapshot bytes — body URIs resolve
    # against it, not against current curation.
    assert "sales" in rep.live_name_pins
    assert rep.live_name_pins["sales"].kind == "dataset"


@pytest.mark.asyncio
async def test_return_report_rejects_unresolved_embed(tmp_path: Path) -> None:
    """Body references a live_name with no published snapshot in this workspace."""
    ex, _ = _bare_executor(tmp_path)
    agent = Agent(model="m", code_executor=ex)
    ctx = AgentContext(session_id="s")
    md = "# Hi\n\n![](file://./data/ghost.parquet)\n"
    r = await agent.return_report(
        context=ctx,
        title="X",
        markdown=md,
        description="",
        notes=[],
        live_name="ghost_report",
    )
    assert not r.success
    msg = r.exception_message.lower()
    assert "ghost" in msg
    assert "no published snapshot" in msg or "not found" in msg
