"""Tests for ``parsimony_agents.tools`` ToolResult behaviour.

Verifies:
- ``ToolResult.failure`` populated → ``ok`` is False.
- ``ToolResult.partial_data`` is independent of ``failure``.
"""

from __future__ import annotations

from parsimony_agents.agent.failure import Failure, FailureKind
from parsimony_agents.tools import ToolResult


def test_toolresult_ok_false_when_failure_populated() -> None:
    """A populated ``failure`` field makes ``ok`` False even with no legacy exception_message."""
    failure = Failure(kind=FailureKind.tool_error, explanation="kernel died")
    result = ToolResult(exception_message=None, data=None, failure=failure)
    assert result.ok is False


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
