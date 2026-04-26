"""Contract: file-backed notebooks, no VariableStore in snapshot, code_set is no-exec."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.models import AgentContext, AgentContextSnapshot
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.notebook_io import deserialize_notebook


def test_context_snapshot_excludes_data_context_and_notebook_tags() -> None:
    async def _load() -> str:
        snap = await AgentContext(session_id="s").to_snapshot(connectors=None)
        return "".join(c["text"] for c in snap.to_llm())

    joined = asyncio.run(_load())
    assert "<data_context>" not in joined
    assert "<notebooks" not in joined
    assert "<notebook" not in joined


def test_code_set_does_not_call_execute() -> None:
    written: list[bytes] = []
    ex = MagicMock()
    ex.write_workspace_file = AsyncMock(side_effect=lambda _p, d: written.append(d))
    ex.read_workspace_file = AsyncMock(
        side_effect=lambda p: written[-1] if written else b"",
    )
    ex.execute = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.clear_namespace = AsyncMock()
    ex.set_cwd = AsyncMock()
    ex.set_connectors = AsyncMock()
    ex.get = AsyncMock(return_value=None)
    ex.eval = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.delete_workspace_file = AsyncMock()
    ex.list_workspace_files = AsyncMock(return_value=[])
    ex.execute_workspace = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.get_locals = MagicMock(return_value={})

    agent = Agent(model="m", code_executor=ex)  # type: ignore[arg-type]
    ctx = AgentContext(session_id="sid")

    async def _run() -> None:
        r = await agent.code_set(context=ctx, path="n.py", code="x=1\n")
        assert r.success
        ex.execute.assert_not_awaited()
        ex.write_workspace_file.assert_awaited()
        raw = written[-1]
        assert deserialize_notebook(raw, path="n.py").code == "x=1"

    asyncio.run(_run())


def test_code_set_with_execute_calls_execute_once() -> None:
    written: list[bytes] = []
    ex = MagicMock()
    ex.write_workspace_file = AsyncMock(side_effect=lambda _p, d: written.append(d))
    ex.read_workspace_file = AsyncMock(
        side_effect=lambda p: written[-1] if written else b"",
    )
    ex.execute = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.clear_namespace = AsyncMock()
    ex.set_cwd = AsyncMock()
    ex.set_connectors = AsyncMock()
    ex.get = AsyncMock(return_value=None)
    ex.eval = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.delete_workspace_file = AsyncMock()
    ex.list_workspace_files = AsyncMock(return_value=[])
    ex.execute_workspace = AsyncMock(return_value=KernelOutput(outputs=[]))
    ex.get_locals = MagicMock(return_value={})

    agent = Agent(model="m", code_executor=ex)  # type: ignore[arg-type]
    ctx = AgentContext(session_id="sid")

    async def _run() -> None:
        r = await agent.code_set(context=ctx, path="n.py", code="x=1\n", execute=True)
        assert r.success
        ex.execute.assert_awaited_once()
        ex.write_workspace_file.assert_awaited()
        assert ex.read_workspace_file.await_count >= 1

    asyncio.run(_run())


def test_to_snapshot_default_has_empty_connectors_catalog() -> None:
    snap = asyncio.run(AgentContext(session_id="x").to_snapshot())
    assert snap.connectors_catalog == ""
    assert isinstance(snap, AgentContextSnapshot)
