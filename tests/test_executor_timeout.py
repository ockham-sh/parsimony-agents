"""Tests for CodeExecutor cell-timeout enforcement (B1 fix).

Covers:
- Infinite-loop cell times out and returns a graceful TimeoutError KernelOutput.
- After a timeout the SAME executor is still usable (no wedge).
- Normal multi-cell namespace persistence is unaffected.
- Top-level-await cells still execute correctly.
- execute_workspace with an infinite loop also times out gracefully.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.outputs import ExceptionObject


def _make_executor(tmp_path: Path) -> CodeExecutor:
    of = OutputFactory(local_dir=tmp_path)
    return CodeExecutor(cwd=str(tmp_path), output_factory=of)


# ---------------------------------------------------------------------------
# Test: infinite loop times out, result is a graceful TimeoutError output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_infinite_loop_times_out(tmp_path: Path) -> None:
    """execute("while True: pass", timeout_seconds=1) must return within ~3 s."""
    ex = _make_executor(tmp_path)

    out = await asyncio.wait_for(
        ex.execute("while True: pass", timeout_seconds=1),
        timeout=5,  # hard outer guard so the test itself never hangs
    )

    assert len(out.outputs) == 1
    obj = out.outputs[0]
    assert isinstance(obj, ExceptionObject), f"expected ExceptionObject, got {type(obj)}: {obj}"
    # obj.value contains the stringified exception; must mention timeout.
    assert "timeout" in obj.value.lower() or "TimeoutError" in obj.value, (
        f"expected timeout-related message in value, got: {obj.value!r}"
    )


# ---------------------------------------------------------------------------
# Test: executor is still usable after a timeout (no wedge)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_executor_still_usable_after_timeout(tmp_path: Path) -> None:
    """After a timeout the executor must not be permanently wedged."""
    ex = _make_executor(tmp_path)

    # Step 1: trigger a timeout.
    timeout_out = await asyncio.wait_for(
        ex.execute("while True: pass", timeout_seconds=1),
        timeout=5,
    )
    assert len(timeout_out.outputs) == 1
    assert isinstance(timeout_out.outputs[0], ExceptionObject)

    # Step 2: the executor must still accept new cells.
    await asyncio.wait_for(ex.execute("x = 1", timeout_seconds=5), timeout=10)

    # Step 3: eval should see the assignment from step 2.
    eval_out = await asyncio.wait_for(ex.eval("x", timeout_seconds=5), timeout=10)
    assert len(eval_out.outputs) == 1
    assert not isinstance(eval_out.outputs[0], ExceptionObject), (
        f"eval after timeout raised: {eval_out.outputs[0]}"
    )
    # The output value for the integer 1 should contain '1'.
    assert "1" in str(eval_out.outputs[0])


# ---------------------------------------------------------------------------
# Test: normal multi-cell namespace persistence is unaffected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_namespace_persistence_unaffected(tmp_path: Path) -> None:
    """Persistent namespace still works normally after the timeout fix."""
    ex = _make_executor(tmp_path)

    await ex.execute("a = 5")
    await ex.execute("b = a + 1")
    eval_out = await ex.eval("b")

    assert len(eval_out.outputs) == 1
    assert not isinstance(eval_out.outputs[0], ExceptionObject)
    assert "6" in str(eval_out.outputs[0])


# ---------------------------------------------------------------------------
# Test: top-level await cell still executes correctly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_top_level_await_cell_works(tmp_path: Path) -> None:
    """A cell with top-level ``await`` must still run to completion."""
    ex = _make_executor(tmp_path)

    # asyncio.sleep(0) is a simple top-level await that does nothing harmful
    # but forces the coroutine path in the executor.
    out = await asyncio.wait_for(
        ex.execute(
            "import asyncio\nawait asyncio.sleep(0)\nresult = 42\n",
            timeout_seconds=5,
        ),
        timeout=10,
    )
    assert not any(isinstance(o, ExceptionObject) for o in out.outputs), (
        f"unexpected exception in top-level await test: {out.outputs}"
    )

    eval_out = await ex.eval("result")
    assert not isinstance(eval_out.outputs[0], ExceptionObject)
    assert "42" in str(eval_out.outputs[0])


# ---------------------------------------------------------------------------
# Test: execute_workspace with an infinite loop also times out gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_execute_workspace_infinite_loop_times_out(tmp_path: Path) -> None:
    """execute_workspace must also time out and return a graceful error."""
    ex = _make_executor(tmp_path)

    out = await asyncio.wait_for(
        ex.execute_workspace("while True: pass", timeout_seconds=1),
        timeout=5,
    )

    assert len(out.outputs) == 1
    obj = out.outputs[0]
    assert isinstance(obj, ExceptionObject), f"expected ExceptionObject, got {type(obj)}: {obj}"
    assert "timeout" in obj.value.lower() or "TimeoutError" in obj.value, (
        f"expected timeout message in value, got: {obj.value!r}"
    )
