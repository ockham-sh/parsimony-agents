"""Phase 8 tests for suspend/resume.

Verifies (PLAN Phase 8 done criteria):
- ``ask_user`` tool raises :class:`SuspensionRequest`; loop yields ``UserInputRequested``.
- Resume with valid token continues the loop with the user's reply.
- Resume preserves cumulative cost / tokens / tool_call_history / elapsed time.
- Tampered token → :class:`SuspensionTokenMismatch`.
- Resume on cancelled suspension → ``RunCancelled``.
- Stale suspension → :class:`SuspensionExpired`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.agent.events import (
    RunCancelled,
    ToolEvent,
    UserInputRequested,
)
from parsimony_agents.agent.failure import (
    DefaultPolicy,
    SuspensionExpired,
    SuspensionTokenMismatch,
)
from parsimony_agents.agent.loop import resume_run, run_loop
from parsimony_agents.agent.state import RunState, SuspensionRecord
from parsimony_agents.agent.termination_tools import TERMINATION_TOOLS
from parsimony_agents.tools import Tools


# ---------------------------------------------------------------------------
# Reuse helpers from test_loop
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, chunks: list[Any]):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _stream_chunk(*, content: str | None = None, tool_call: tuple[str, str, str] | None = None) -> SimpleNamespace:
    tcs = None
    if tool_call is not None:
        tc_id, name, args = tool_call
        tcs = [SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args))]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content, reasoning_content=None, tool_calls=tcs))
        ]
    )


def _assembled(*, content: str = "", tool_calls=None, finish_reason: str = "tool_calls") -> SimpleNamespace:
    calls = []
    for tc_id, name, args in tool_calls or []:
        calls.append(SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args)))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, reasoning_content="", tool_calls=calls),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _LLMScript:
    def __init__(self, turns):
        self._turns = list(turns)
        self._idx = 0

    async def acompletion(self, *_, **__):
        chunks, _ = self._turns[self._idx]
        self._idx += 1
        return _FakeStream(chunks)

    def stream_chunk_builder(self, *_, **__):
        _, assembled = self._turns[self._idx - 1]
        return assembled


def _agent(tools=None, guardrails=None):
    return SimpleNamespace(
        guardrails=guardrails or AgentGuardrails(max_iterations=10),
        policy=DefaultPolicy(),
        suspension_secret="topsecret",
        model_config={"model": "claude-opus-4-7"},
        instructions="you are an agent",
        tools=Tools(list(tools or []) + TERMINATION_TOOLS),
    )


async def _drain(agen):
    return [event async for event in agen]


# ---------------------------------------------------------------------------
# Suspend → resume happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspend_then_resume_continues_with_user_reply() -> None:
    """ask_user → suspended; resume with user reply → run continues and returns done."""
    # Turn 1: agent calls ask_user → suspends.
    # Turn 2 (after resume): agent calls return_done.
    script = _LLMScript(
        turns=[
            (
                [_stream_chunk(tool_call=("c1", "ask_user", ""))],
                _assembled(tool_calls=[("c1", "ask_user", json.dumps({"question": "which one?"}))]),
            ),
            (
                [_stream_chunk(tool_call=("c2", "return_done", ""))],
                _assembled(tool_calls=[("c2", "return_done", json.dumps({"summary": "done"}))]),
            ),
        ]
    )

    with (
        patch("litellm.acompletion", side_effect=script.acompletion),
        patch("litellm.stream_chunk_builder", side_effect=script.stream_chunk_builder),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        agent = _agent()
        state = RunState(run_id="r1", session_id="s1")
        events1 = await _drain(run_loop(agent, state))

        suspensions = [e for e in events1 if isinstance(e, UserInputRequested)]
        assert len(suspensions) == 1
        record = suspensions[0].suspension_record

        # Pre-resume snapshot of accumulators.
        prior_cost = state.cumulative_cost_usd
        prior_tool_count = len(state.tool_call_history)

        # Resume with a user reply.
        events2 = await _drain(
            resume_run(agent, record, user_reply="use dataset A")
        )

    # Resume produced events for the second turn.
    assert any(isinstance(e, ToolEvent) and e.tool_name == "return_done" for e in events2)

    # Cost and tool history carried forward (accumulators preserved).
    assert prior_cost > 0  # something was billed in the first call
    # The resume created a NEW state but it inherited the accumulators from the record.
    # Verify the record itself carries them:
    assert record.cumulative_cost_usd == prior_cost
    assert len(record.tool_call_history) == prior_tool_count


# ---------------------------------------------------------------------------
# Token tamper detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_with_tampered_token_raises_mismatch() -> None:
    """A record whose run_id was edited after token issuance fails verification."""
    from parsimony_agents.agent.failure import compute_suspension_token

    agent = _agent()
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        started_at=datetime.now(timezone.utc),
        pending_question="?",
    )
    tampered = record.model_copy(update={"run_id": "r2"})

    with pytest.raises(SuspensionTokenMismatch):
        async for _ in resume_run(agent, tampered, user_reply="hi"):
            pass


# ---------------------------------------------------------------------------
# Stale suspension
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_on_stale_suspension_raises_expired() -> None:
    """A suspension older than ``max_suspension_age_s`` raises :class:`SuspensionExpired`."""
    from parsimony_agents.agent.failure import compute_suspension_token

    agent = _agent()
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        suspended_at=datetime.now(timezone.utc) - timedelta(seconds=3600),  # 1h old
        started_at=datetime.now(timezone.utc) - timedelta(seconds=3700),
        pending_question="?",
    )

    with pytest.raises(SuspensionExpired):
        async for _ in resume_run(agent, record, user_reply="hi", max_suspension_age_s=60.0):
            pass


# ---------------------------------------------------------------------------
# Cancellation on resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_with_preset_cancellation_yields_run_cancelled() -> None:
    """If cancellation is set before resume, the loop's first cancel check fires."""
    from parsimony_agents.agent.failure import compute_suspension_token

    agent = _agent()
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        started_at=datetime.now(timezone.utc),
        pending_question="?",
    )
    cancellation = CancellationRequest()
    cancellation.set()

    events = await _drain(
        resume_run(agent, record, user_reply="hi", cancellation=cancellation)
    )
    assert any(isinstance(e, RunCancelled) for e in events)


# ---------------------------------------------------------------------------
# Empty user_reply rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_with_empty_reply_raises_value_error() -> None:
    """Empty or whitespace-only ``user_reply`` is rejected before any LLM call."""
    from parsimony_agents.agent.failure import compute_suspension_token

    agent = _agent()
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        started_at=datetime.now(timezone.utc),
        pending_question="?",
    )

    with pytest.raises(ValueError, match="non-empty user_reply"):
        async for _ in resume_run(agent, record, user_reply="   "):
            pass


# ---------------------------------------------------------------------------
# RunState.from_suspension preserves accumulators
# ---------------------------------------------------------------------------


def test_runstate_from_suspension_preserves_accumulators() -> None:
    """Cumulative cost / tokens / tool_call_history / lessons_learned all carry over."""
    from parsimony_agents.agent.failure import (
        Failure,
        FailureKind,
        compute_suspension_token,
    )

    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="x"),
        started_at=datetime.now(timezone.utc) - timedelta(seconds=120),
        elapsed_seconds=120.0,
        iteration_count=7,
        tool_call_history=["read_data:abc", "read_data:def"],
        pending_question="?",
        cumulative_cost_usd=0.42,
        cumulative_prompt_tokens=1000,
        cumulative_completion_tokens=500,
        lessons_learned=[Failure(kind=FailureKind.tool_error, explanation="x")],
        last_repeat_counts={"read_data:abc": 2},
    )

    state = RunState.from_suspension(record)
    assert state.iteration == 7
    assert state.cumulative_cost_usd == 0.42
    assert state.cumulative_prompt_tokens == 1000
    assert state.tool_call_history == ["read_data:abc", "read_data:def"]
    assert state.last_repeat_counts == {"read_data:abc": 2}
    assert len(state.lessons_learned) == 1
    assert state.accumulated_elapsed_s == 120.0
    assert state.done is False
    assert state.pending_instruction is None  # cleared on resume


def _budget_suspension_record(kind: "FailureKind") -> "SuspensionRecord":
    from parsimony_agents.agent.failure import compute_suspension_token

    return SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="x"),
        started_at=datetime.now(timezone.utc) - timedelta(seconds=900),
        elapsed_seconds=34.8,
        iteration_count=20,
        pending_question="continue?",
        originating_failure_kind=kind,
    )


def test_from_suspension_resets_time_budget_when_user_continued_past_time_limit() -> None:
    """A run that suspended on ``time_limit`` gets a fresh time budget on resume.

    Regression: without this, ``pre_step`` re-trips ``time_limit`` on the first
    post-resume iteration because ``accumulated_elapsed_s`` still holds the
    pre-suspension elapsed (already over budget).
    """
    from parsimony_agents.agent.failure import FailureKind

    state = RunState.from_suspension(_budget_suspension_record(FailureKind.time_limit))
    assert state.accumulated_elapsed_s == 0.0
    # started_at reset to resume moment → elapsed_seconds() reflects only this turn.
    assert state.elapsed_seconds() < 1.0
    # The iteration counter is a different budget; it is left untouched.
    assert state.iteration == 20


def test_from_suspension_resets_iteration_budget_when_user_continued_past_iteration_limit() -> None:
    """A run that suspended on ``iteration_limit`` gets a fresh iteration budget on resume."""
    from parsimony_agents.agent.failure import FailureKind

    state = RunState.from_suspension(_budget_suspension_record(FailureKind.iteration_limit))
    assert state.iteration == 0
    # The time budget is a different guardrail; its accumulator is left untouched.
    assert state.accumulated_elapsed_s == 34.8


def test_from_suspension_keeps_budgets_for_non_budget_suspension() -> None:
    """A non-budget suspension (``ambiguous_input``) must NOT reset budget counters.

    Otherwise a run could dodge a budget by suspending on an unrelated question.
    """
    from parsimony_agents.agent.failure import FailureKind

    state = RunState.from_suspension(_budget_suspension_record(FailureKind.ambiguous_input))
    assert state.iteration == 20
    assert state.accumulated_elapsed_s == 34.8
