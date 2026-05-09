"""Contract: return_notebook is no-exec by default, agent does not touch a transient working copy."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.models import AgentContext, AgentContextSnapshot
from parsimony_agents.execution.outputs import KernelOutput


def test_context_snapshot_excludes_data_context_and_notebook_tags() -> None:
    async def _load() -> str:
        snap = await AgentContext(session_id="s").to_snapshot(connectors=None)
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
        assert r.success
        ex.execute.assert_not_awaited()
        ex.write_workspace_file.assert_not_awaited()
        # The returned message round-trips the canonical notebook ref tag.
        assert "<notebook_ref " in r.data  # type: ignore[arg-type]

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
        assert r.success
        ex.execute.assert_awaited_once()
        ex.write_workspace_file.assert_not_awaited()

    asyncio.run(_run())


def test_to_snapshot_default_has_empty_connectors_catalog() -> None:
    snap = asyncio.run(AgentContext(session_id="x").to_snapshot())
    assert snap.connectors_catalog == ""
    assert isinstance(snap, AgentContextSnapshot)


def test_return_notebook_no_exec_returns_notebook_ref_in_string() -> None:
    """The return_notebook tool result must include the canonical notebook ref.

    Without this the agent has no way to learn the notebook's
    ``logical_id``/``content_sha`` short of recomputing the hash by hand
    (which diverges from ``notebook_content_sha``'s whitespace-stripped
    canonical form). Surfaced inline so the agent sees it the moment it
    writes the notebook, not after a session_state rebuild that never
    happens mid-turn.
    """

    from parsimony_agents.identity import notebook_content_sha

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
    # ``path="n.py"`` is outside the canonical ``notebooks/`` prefix, so
    # ``notebook_logical_id`` raises and the agent falls back to using
    # ``content_sha`` as logical_id (standalone parsimony-agents usage).
    ctx = AgentContext(session_id="sid")
    code = "x = 1\n"
    expected_sha = notebook_content_sha(code)

    async def _run() -> str:
        r = await agent.return_notebook(context=ctx, path="n.py", code=code)
        return r.data  # type: ignore[no-any-return]

    body = asyncio.run(_run())
    assert "<notebook_ref " in body
    assert f'logical_id="{expected_sha}"' in body
    assert f'content_sha="{expected_sha}"' in body


def test_return_notebook_with_execute_stamps_notebook_ref_metadata() -> None:
    """When ``execute=True`` the KernelOutput carries the notebook ref in metadata.

    ``KernelOutput.to_llm`` reads it from there to emit the same
    ``<notebook_ref/>`` block the no-exec path appends to its string.
    """

    from parsimony_agents.identity import notebook_content_sha

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
    expected_sha = notebook_content_sha(code)

    async def _run() -> KernelOutput:
        r = await agent.return_notebook(context=ctx, path="n.py", code=code, execute=True)
        return r.data  # type: ignore[no-any-return]

    ko = asyncio.run(_run())
    assert ko.metadata is not None
    nb_ref = ko.metadata.get("notebook_ref")
    assert nb_ref == {
        "kind": "notebook",
        "logical_id": expected_sha,
        "content_sha": expected_sha,
    }
    flat = "".join(b.get("text", "") for b in ko.to_llm())
    assert f'logical_id="{expected_sha}"' in flat
