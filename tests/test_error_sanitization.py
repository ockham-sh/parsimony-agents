from __future__ import annotations

from parsimony.errors import (
    ConnectorError,
    EmptyDataError,
    ProviderError,
    RateLimitError,
    UnauthorizedError,
)

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


# ---------------------------------------------------------------------------
# Typed connector errors short-circuit to str(exc) — no traceback, kernel
# default messages already carry agent-loop directives.
# ---------------------------------------------------------------------------


def test_unauthorized_error_renders_without_traceback() -> None:
    exc = UnauthorizedError(provider="fred", env_var="FRED_API_KEY")
    output = ExceptionObject(value=exc)
    assert output.value == str(exc)
    assert "Traceback" not in output.value
    # Kernel default carries the directive verbatim.
    assert "FRED_API_KEY" in output.value
    assert "DO NOT retry with different arguments" in output.value


def test_rate_limit_error_renders_without_traceback() -> None:
    exc = RateLimitError(provider="fred", retry_after=60.0, quota_exhausted=True)
    output = ExceptionObject(value=exc)
    assert output.value == str(exc)
    assert "Traceback" not in output.value
    assert "DO NOT retry" in output.value


def test_provider_error_renders_status_bucketed_directive() -> None:
    exc = ProviderError(provider="fmp", status_code=503)
    output = ExceptionObject(value=exc)
    assert output.value == str(exc)
    assert "HTTP 503" in output.value
    assert "transient" in output.value


def test_empty_data_error_does_not_carry_do_not_retry() -> None:
    exc = EmptyDataError(provider="fred")
    output = ExceptionObject(value=exc)
    assert output.value == str(exc)
    assert "DO NOT" not in output.value


def test_bare_connector_error_renders_author_message_verbatim() -> None:
    """The bare-ConnectorError contract: author message reaches the agent unmodified."""
    exc = ConnectorError("Set PARSIMONY_X to enable this tool", provider="x")
    output = ExceptionObject(value=exc)
    assert output.value == "Set PARSIMONY_X to enable this tool"
