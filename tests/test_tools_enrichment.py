"""Phase 6 tests for ``parsimony_agents.tools`` enrichment.

Verifies (PLAN Phase 6 done criteria, tests 1-3):
- ``Tool.idempotent`` etc. default to False / None.
- ``ToolResult.failure`` populated → ``ok`` is False.
- ``ToolResult.partial_data`` is independent of ``failure``.
"""

from __future__ import annotations

from parsimony_agents.agent.failure import Failure, FailureKind
from parsimony_agents.tools import Tool, ToolResult


def _noop_tool() -> Tool:
    async def fn(**_):
        return None

    return Tool(
        function=fn,
        name="noop",
        description="d",
        parameters_schema={"type": "object", "properties": {}},
        tool_type="utility",
    )


def test_tool_phase6_fields_default_safely() -> None:
    """All new structural declarations default to safe values."""
    t = _noop_tool()
    assert t.idempotent is False
    assert t.retryable_on_error is False
    assert t.parallelizable is False
    assert t.timeout_s is None


def test_tool_phase6_fields_can_be_set() -> None:
    """The new fields are forwarded to the Tool instance."""

    async def fn(**_):
        return None

    t = Tool(
        function=fn,
        name="x",
        description="d",
        parameters_schema={"type": "object", "properties": {}},
        tool_type="utility",
        idempotent=True,
        retryable_on_error=True,
        parallelizable=True,
        timeout_s=30.0,
    )
    assert t.idempotent is True
    assert t.retryable_on_error is True
    assert t.parallelizable is True
    assert t.timeout_s == 30.0


def test_toolresult_ok_false_when_failure_populated() -> None:
    """A populated ``failure`` field makes ``ok`` False even with no legacy exception_message."""
    failure = Failure(kind=FailureKind.tool_error, explanation="kernel died")
    result = ToolResult(exception_message=None, data=None, failure=failure)
    assert result.ok is False
    assert result.success is False  # alias


def test_toolresult_ok_true_when_clean() -> None:
    result = ToolResult(exception_message=None, data={"x": 1})
    assert result.ok is True


def test_toolresult_ok_false_for_legacy_exception_message() -> None:
    """Legacy ``exception_message`` path still flags failure."""
    result = ToolResult(exception_message="boom", data=None)
    assert result.ok is False
    assert result.failure is None  # legacy path doesn't populate failure


def test_toolresult_partial_data_independent_of_failure() -> None:
    """Partial-success: failure populated AND partial_data populated."""
    failure = Failure(
        kind=FailureKind.tool_error,
        explanation="fetched 90% of rows before timeout",
    )
    result = ToolResult(
        exception_message=None,
        data=None,
        failure=failure,
        partial_data={"rows": 900, "expected": 1000},
    )
    assert result.ok is False
    assert result.partial_data == {"rows": 900, "expected": 1000}


def test_toolresult_from_failure_classmethod_populates_both_fields() -> None:
    failure = Failure(kind=FailureKind.tool_error, explanation="boom")
    result = ToolResult.from_failure(failure, partial_data={"x": 1})
    assert result.failure is failure
    assert result.partial_data == {"x": 1}
    assert result.exception_message == "boom"
    assert result.ok is False
