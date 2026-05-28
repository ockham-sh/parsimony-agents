"""Phase 2 tests for ``parsimony_agents.agent.failure.detectors``.

Verifies the 10 detector criteria from PLAN Phase 2. Uses simple namespace stubs
for LLM responses / tool results so the tests don't need a real LLM client.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.agent.failure import (
    Failure,
    FailureKind,
    accumulate_usage,
    loop_signature,
    post_llm,
    post_tool,
    pre_step,
    record_tool_call,
)
from parsimony_agents.agent.state import RunState

# ---------------------------------------------------------------------------
# Stubs for LLM-response and tool-result shapes
# ---------------------------------------------------------------------------


def _llm_response(
    *,
    finish_reason: str | None = "stop",
    tool_calls: list[tuple[str, str]] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> SimpleNamespace:
    """Build a minimal LLM response stub.

    ``tool_calls`` is a list of (name, arguments_json) tuples.
    """
    calls = []
    for name, args_json in tool_calls or []:
        calls.append(SimpleNamespace(function=SimpleNamespace(name=name, arguments=args_json)))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(tool_calls=calls or None),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


# ---------------------------------------------------------------------------
# Tests #1–#3: pre_step budget detectors
# ---------------------------------------------------------------------------


def test_iteration_limit_fires_at_threshold() -> None:
    """``iteration_limit`` fires when ``state.iteration >= guardrails.max_iterations``."""
    guard = AgentGuardrails(max_iterations=5)
    state = RunState(run_id="r1", session_id="s1", iteration=5)
    failure = pre_step(state, guard)
    assert failure is not None
    assert failure.kind == FailureKind.iteration_limit
    assert failure.metadata["iteration"] == 5


def test_iteration_limit_does_not_fire_below_threshold() -> None:
    state = RunState(run_id="r1", session_id="s1", iteration=4)
    assert pre_step(state, AgentGuardrails(max_iterations=5)) is None


def test_time_limit_fires_when_elapsed_exceeds_threshold() -> None:
    """``time_limit`` fires when ``elapsed_seconds() >= max_execution_time_s``."""
    guard = AgentGuardrails(max_execution_time_s=10.0)
    state = RunState(
        run_id="r1",
        session_id="s1",
        # started_at backed up so elapsed > threshold immediately.
        started_at=datetime.now(UTC) - timedelta(seconds=20),
        # last_event_time_s set to "now" so stall detector doesn't preempt.
        last_event_time_s=time.monotonic(),
    )
    failure = pre_step(state, guard)
    assert failure is not None
    assert failure.kind == FailureKind.time_limit


def test_no_progress_fires_after_stall_threshold() -> None:
    """``no_progress`` fires when ``time.monotonic() - state.last_event_time_s > stall_threshold_s``."""
    guard = AgentGuardrails(stall_threshold_s=5.0)
    state = RunState(
        run_id="r1",
        session_id="s1",
        last_event_time_s=time.monotonic() - 10.0,  # 10s ago
    )
    failure = pre_step(state, guard)
    assert failure is not None
    assert failure.kind == FailureKind.no_progress
    assert failure.metadata["silence_s"] > 5.0


# ---------------------------------------------------------------------------
# Test #9: pre_step precedence
# ---------------------------------------------------------------------------


def test_pre_step_precedence_iteration_beats_stall() -> None:
    """When both ``iteration_limit`` and ``no_progress`` would fire, iteration wins."""
    state = RunState(
        run_id="r1",
        session_id="s1",
        iteration=10,
        last_event_time_s=time.monotonic() - 9999.0,  # ancient → would also fire no_progress
    )
    guard = AgentGuardrails(max_iterations=10, stall_threshold_s=5.0)
    failure = pre_step(state, guard)
    assert failure is not None
    assert failure.kind == FailureKind.iteration_limit


# ---------------------------------------------------------------------------
# Test #6: post_llm finish_reason mapping
# ---------------------------------------------------------------------------


def test_output_truncated_fires_on_length() -> None:
    response = _llm_response(finish_reason="length")
    failure = post_llm(response, RunState(run_id="r1", session_id="s1"), AgentGuardrails())
    assert failure is not None
    assert failure.kind == FailureKind.output_truncated


@pytest.mark.parametrize("finish_reason", ["content_filter", "refusal"])
def test_output_refused_fires_on_content_filter_or_refusal(finish_reason: str) -> None:
    response = _llm_response(finish_reason=finish_reason)
    failure = post_llm(response, RunState(run_id="r1", session_id="s1"), AgentGuardrails())
    assert failure is not None
    assert failure.kind == FailureKind.output_refused
    assert failure.metadata["finish_reason"] == finish_reason


def test_normal_finish_reasons_produce_no_failure() -> None:
    for reason in ("stop", "tool_calls", None):
        response = _llm_response(finish_reason=reason)
        assert post_llm(response, RunState(run_id="r1", session_id="s1"), AgentGuardrails()) is None


# ---------------------------------------------------------------------------
# Tests #4–#5: loop detection
# ---------------------------------------------------------------------------


def test_loop_signature_normalizes_ui_message() -> None:
    """Stripping ``_ui_message`` means two calls with different prefixes collapse to one signature."""
    sig_a = loop_signature("read_data", {"path": "data.csv", "_ui_message": "first try"})
    sig_b = loop_signature("read_data", {"path": "data.csv", "_ui_message": "trying again"})
    assert sig_a == sig_b


def test_loop_signature_differs_for_different_args() -> None:
    """Two calls with different real args produce different signatures (no false-positive)."""
    sig_a = loop_signature("read_data", {"path": "A.csv"})
    sig_b = loop_signature("read_data", {"path": "B.csv"})
    assert sig_a != sig_b


def test_loop_detected_fires_at_hard_threshold() -> None:
    """``loop_detected`` Failure fires when the would-be repeat count hits the hard threshold."""
    state = RunState(run_id="r1", session_id="s1")
    sig = loop_signature("read_data", {"path": "A.csv"})
    # Prime history with hard_threshold - 1 repeats; the next call is the trigger.
    state.last_repeat_counts[sig] = 5  # one more makes 6 → hard threshold

    response = _llm_response(
        tool_calls=[("read_data", '{"path": "A.csv"}')],
        finish_reason="tool_calls",
    )
    failure = post_llm(response, state, AgentGuardrails(loop_hard_threshold=6))
    assert failure is not None
    assert failure.kind == FailureKind.loop_detected
    assert failure.metadata["repeat_count"] == 6
    assert failure.metadata["tool_name"] == "read_data"


def test_loop_soft_threshold_returns_none(caplog) -> None:
    """Soft-threshold breaches log but do not produce a Failure."""
    state = RunState(run_id="r1", session_id="s1")
    sig = loop_signature("read_data", {"path": "A.csv"})
    state.last_repeat_counts[sig] = 2  # one more makes 3 → above soft (2), below hard (6)

    response = _llm_response(
        tool_calls=[("read_data", '{"path": "A.csv"}')],
        finish_reason="tool_calls",
    )
    with caplog.at_level("INFO", logger="parsimony_agents.agent.failure.detectors"):
        failure = post_llm(state=state, response=response, guardrails=AgentGuardrails())
    assert failure is None
    assert any("loop_soft_threshold_hit" in rec.message for rec in caplog.records)


def test_loop_detection_skips_malformed_arguments() -> None:
    """Non-JSON ``arguments`` falls back to ``{"_raw_args": ...}`` and still signs cleanly."""
    response = _llm_response(
        tool_calls=[("read_data", "not-json")],
        finish_reason="tool_calls",
    )
    state = RunState(run_id="r1", session_id="s1")
    assert post_llm(response, state, AgentGuardrails()) is None  # one call, well below threshold


# ---------------------------------------------------------------------------
# Test #7: post_tool
# ---------------------------------------------------------------------------


def test_post_tool_fires_on_new_shape_failure() -> None:
    """When the new ``ToolResult.failure`` is populated, ``post_tool`` returns it verbatim."""
    fail = Failure(kind=FailureKind.tool_error, explanation="kernel died")
    result = SimpleNamespace(failure=fail, exception_message=None)
    out = post_tool(result, SimpleNamespace(name="execute_code"), RunState(run_id="r1", session_id="s1"))
    assert out is fail


def test_post_tool_fires_on_legacy_exception_message() -> None:
    """When only the legacy ``exception_message`` is present, wrap it into a ``tool_error`` Failure."""
    result = SimpleNamespace(failure=None, exception_message="connector timeout")
    out = post_tool(result, SimpleNamespace(name="load_dataset"), RunState(run_id="r1", session_id="s1"))
    assert out is not None
    assert out.kind == FailureKind.tool_error
    assert out.explanation == "connector timeout"
    assert out.metadata["tool_name"] == "load_dataset"


def test_post_tool_no_failure_returns_none() -> None:
    """A clean tool result produces no failure."""
    result = SimpleNamespace(failure=None, exception_message=None)
    assert post_tool(result, SimpleNamespace(name="x"), RunState(run_id="r1", session_id="s1")) is None


# ---------------------------------------------------------------------------
# Test #10: accumulate_usage
# ---------------------------------------------------------------------------


def test_accumulate_usage_sums_across_calls() -> None:
    """``accumulate_usage`` updates state cumulative counters in-place."""
    state = RunState(run_id="r1", session_id="s1")

    with patch("litellm.completion_cost", return_value=0.05):
        prompt, completion, cost = accumulate_usage(
            state,
            _llm_response(prompt_tokens=100, completion_tokens=50),
            model="claude-opus-4-7",
        )
    assert (prompt, completion) == (100, 50)
    assert cost == 0.05
    assert state.cumulative_prompt_tokens == 100
    assert state.cumulative_completion_tokens == 50
    assert state.cumulative_cost_usd == 0.05

    with patch("litellm.completion_cost", return_value=0.07):
        prompt, completion, cost = accumulate_usage(
            state,
            _llm_response(prompt_tokens=200, completion_tokens=80),
            model="claude-opus-4-7",
        )
    assert state.cumulative_prompt_tokens == 300
    assert state.cumulative_completion_tokens == 130
    assert abs(state.cumulative_cost_usd - 0.12) < 1e-9


def test_accumulate_usage_swallows_litellm_failure() -> None:
    """If ``litellm.completion_cost`` raises, the call still completes; cost recorded as 0."""
    state = RunState(run_id="r1", session_id="s1")
    with patch("litellm.completion_cost", side_effect=RuntimeError("boom")):
        _, _, cost = accumulate_usage(
            state,
            _llm_response(prompt_tokens=10, completion_tokens=5),
            model="claude-opus-4-7",
        )
    assert cost == 0.0
    assert state.cumulative_cost_usd == 0.0
    assert state.cumulative_prompt_tokens == 10


def test_accumulate_usage_without_model_skips_cost() -> None:
    """No ``model`` arg → don't even import litellm; cost is 0 but tokens still accumulate."""
    state = RunState(run_id="r1", session_id="s1")
    _, _, cost = accumulate_usage(state, _llm_response(prompt_tokens=10, completion_tokens=5))
    assert cost == 0.0
    assert state.cumulative_prompt_tokens == 10
    assert state.cumulative_completion_tokens == 5


# ---------------------------------------------------------------------------
# record_tool_call helper
# ---------------------------------------------------------------------------


def test_record_tool_call_appends_history_and_counter() -> None:
    """The helper grows both ``tool_call_history`` and ``last_repeat_counts``."""
    state = RunState(run_id="r1", session_id="s1")
    sig1 = record_tool_call(state, "read_data", {"path": "A.csv"})
    sig2 = record_tool_call(state, "read_data", {"path": "A.csv"})
    sig3 = record_tool_call(state, "read_data", {"path": "B.csv"})
    assert sig1 == sig2 != sig3
    assert state.tool_call_history == [sig1, sig2, sig3]
    assert state.last_repeat_counts[sig1] == 2
    assert state.last_repeat_counts[sig3] == 1
