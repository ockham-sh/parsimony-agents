"""Contract: return_notebook is no-exec by default, agent does not touch a transient working copy."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.models import AgentContext, AgentContextSnapshot
from parsimony_agents.execution.outputs import KernelOutput


def test_context_snapshot_excludes_data_context_and_notebook_tags() -> None:
    async def _load() -> str:
        snap = await AgentContext(session_id="s").to_snapshot()
        return "".join(c["text"] for c in snap.to_llm())

    joined = asyncio.run(_load())
    assert "<data_context>" not in joined
    assert "<notebooks" not in joined
    assert "<notebook" not in joined


def test_return_notebook_does_not_call_execute() -> None:
    """no-exec ``return_notebook`` runs zero kernel calls and zero executor file writes.

    Notebook bytes are persisted by the streaming layer (out of this test's
    scope). The agent itself never writes a transient working copy.
    """
    ex = MagicMock()
    ex.write_workspace_file = AsyncMock()
    ex.read_workspace_file = AsyncMock(side_effect=FileNotFoundError)
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
        r = await agent.return_notebook(context=ctx, path="n.py", code="x=1\n")
        assert r.ok
        ex.execute.assert_not_awaited()
        ex.write_workspace_file.assert_not_awaited()
        # The returned message confirms publication without exposing a ref.
        assert "Published n.py" in r.data  # type: ignore[arg-type]
        assert "<notebook_ref" not in r.data  # type: ignore[arg-type]

    asyncio.run(_run())


def test_return_notebook_with_execute_calls_execute_once() -> None:
    """``return_notebook(..., execute=True)`` runs the kernel once and never writes via the executor."""
    ex = MagicMock()
    ex.write_workspace_file = AsyncMock()
    ex.read_workspace_file = AsyncMock(side_effect=FileNotFoundError)
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
        r = await agent.return_notebook(context=ctx, path="n.py", code="x=1\n", execute=True)
        assert r.ok
        ex.execute.assert_awaited_once()
        ex.write_workspace_file.assert_not_awaited()

    asyncio.run(_run())


def test_to_snapshot_default_returns_snapshot() -> None:
    snap = asyncio.run(AgentContext(session_id="x").to_snapshot())
    assert isinstance(snap, AgentContextSnapshot)


def test_return_notebook_no_exec_returns_publication_confirmation() -> None:
    """The no-exec ``return_notebook`` returns a publication confirmation.

    Refs are framework-internal under the new model; the agent only
    needs to know the notebook landed.
    """
    written: list[bytes] = []
    ex = MagicMock()
    ex.write_workspace_file = AsyncMock(side_effect=lambda _p, d: written.append(d))
    ex.read_workspace_file = AsyncMock(side_effect=lambda p: written[-1] if written else b"")
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

    async def _run() -> str:
        r = await agent.return_notebook(context=ctx, path="n.py", code="x = 1\n")
        return r.data  # type: ignore[no-any-return]

    body = asyncio.run(_run())
    assert "Published n.py" in body
    assert "<notebook_ref" not in body
    assert "logical_id" not in body
    assert "content_sha" not in body


def test_return_notebook_with_execute_does_not_surface_ref_to_llm() -> None:
    """When ``execute=True`` the KernelOutput must NOT surface a notebook_ref.

    Under the new model, refs are framework-internal — the agent never
    consumes a hash. The metadata may still carry the ref (used by the
    streaming layer's ref ledger) but ``to_llm`` must hide it.
    """
    written: list[bytes] = []
    ex = MagicMock()
    ex.write_workspace_file = AsyncMock(side_effect=lambda _p, d: written.append(d))
    ex.read_workspace_file = AsyncMock(side_effect=lambda p: written[-1] if written else b"")
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
    code = "y = 2\n"

    async def _run() -> KernelOutput:
        r = await agent.return_notebook(context=ctx, path="n.py", code=code, execute=True)
        return r.data  # type: ignore[no-any-return]

    ko = asyncio.run(_run())
    flat = "".join(b.get("text", "") for b in ko.to_llm())
    assert "logical_id" not in flat
    assert "content_sha" not in flat
    assert "<notebook_ref" not in flat
