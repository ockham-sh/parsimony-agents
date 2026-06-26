"""CodeExecutor: async gate serializes overlapping kernel work (no event-loop deadlock)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory


@pytest.mark.asyncio
async def test_concurrent_awaits_to_execute_complete_without_deadlock(
    tmp_path: Path,
) -> None:
    of = OutputFactory(local_dir=tmp_path)
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=of)

    async def run() -> int:
        out = await ex.execute("x = 1\nprint(x)\n")
        return len(out.outputs)

    results = await asyncio.gather(*(run() for _ in range(3)))
    assert all(n >= 1 for n in results)
