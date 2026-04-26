from __future__ import annotations

from parsimony_agents.execution.outputs import ExceptionObject
from parsimony_agents.tools import ToolResult


def test_tool_result_from_exception_redacts_sensitive_query_values() -> None:
    error = Exception("failed on https://api.example.com/path?api_key=super-secret&series=UNRATE")
    result = ToolResult.from_exception(error)
    assert result.exception_message is not None
    assert "super-secret" not in result.exception_message
    assert "series=UNRATE" in result.exception_message


def test_exception_object_redacts_sensitive_query_values_in_traceback() -> None:
    try:
        raise RuntimeError("bad url https://api.example.com/path?token=t0psecret&series=GDP")
    except RuntimeError as exc:
        output = ExceptionObject(value=exc)
    assert "t0psecret" not in output.value
    assert "series=GDP" in output.value
