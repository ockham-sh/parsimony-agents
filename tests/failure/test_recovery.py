"""Tests for ``parsimony_agents.agent.failure.recovery``.

Verifies:
- ``handle_failure(narrow_scope)`` sets pending_instruction, increments
  ``failure_attempts``, yields ``AgentError``, does *not* set ``state.done``.
- ``handle_failure(ask_user)`` yields ``UserInputRequested`` with a valid
  ``SuspensionRecord`` token, sets ``state.done = True``.
- ``handle_failure(handoff)`` yields ``Handoff`` with blockers,
  sets ``state.done = True``.
- ``handle_failure(retry)`` with budget exhausted promotes to handoff.
- Two consecutive ``no_progress`` failures: first → narrow_scope; second → handoff.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from parsimony_agents.agent.events import (
    AgentError,
    Handoff,
    PartialRunSummary,
    UserInputRequested,
)
from parsimony_agents.agent.failure import (
    Action,
    DefaultPolicy,
    Failure,
    FailureKind,
    handle_failure,
    verify_suspension_token,
)
from parsimony_agents.agent.state import RunState, SuspensionRecord


def _agent(secret: str = "topsecret") -> SimpleNamespace:
    """Minimal agent-like for the recovery funnel."""
    return SimpleNamespace(suspension_secret=secret, policy=DefaultPolicy())


async def _collect(agen) -> list:
    return [event async for event in agen]


@pytest.mark.asyncio
async def test_narrow_scope_sets_pending_instruction() -> None:
    """``narrow_scope`` action: pending_instruction set, AgentError yielded, run continues."""
    state = RunState(run_id="r1", session_id="s1")
    failure = Failure(kind=FailureKind.no_progress, explanation="text-only response")

    events = await _collect(handle_failure(failure, agent=_agent(), state=state))

    assert len(events) == 1
    assert isinstance(events[0], AgentError)
    assert events[0].failure is failure
    assert state.pending_instruction is not None
    # The no_progress corrective prompt must surface ask_user as a first-class
    # option (not steer the agent past it toward "make progress"), and the
    # other explicit-termination tools.
    assert "ask_user" in state.pending_instruction
    assert "return_done" in state.pending_instruction
    assert "return_unable" in state.pending_instruction
    assert state.failure_attempts[FailureKind.no_progress] == 1
    assert state.done is False


@pytest.mark.asyncio
async def test_narrow_scope_instruction_is_kind_specific() -> None:
    """no_progress gets the act/ask_user/end prompt; scope_too_large keeps shrink-the-step."""
    from parsimony_agents.agent.failure.recovery import _narrow_scope_instruction

    no_prog = _narrow_scope_instruction(Failure(kind=FailureKind.no_progress, explanation="text only"))
    assert "ask_user" in no_prog
    # The no_progress prompt must NOT push toward "make progress" — that bias is
    # what previously buried ask_user when the agent meant to ask the user.
    assert "smallest piece of work" not in no_prog

    scope = _narrow_scope_instruction(Failure(kind=FailureKind.scope_too_large, explanation="too broad"))
    assert "smallest piece of work" in scope


@pytest.mark.asyncio
async def test_ask_user_yields_user_input_requested() -> None:
    """``ask_user`` action: UserInputRequested with valid suspension token, run ends."""
    state = RunState(run_id="r1", session_id="s1")
    failure = Failure(kind=FailureKind.ambiguous_input, explanation="which dataset?")

    events = await _collect(handle_failure(failure, agent=_agent(), state=state))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, UserInputRequested)
    assert event.originating_failure_kind == "ambiguous_input"
    assert isinstance(event.suspension_record, SuspensionRecord)
    # Token must verify under the same secret.
    assert verify_suspension_token(record=event.suspension_record, secret="topsecret") is True
    assert state.done is True


@pytest.mark.asyncio
async def test_handoff_yields_handoff_event() -> None:
    """``handoff`` action: Handoff event with blockers, run ends."""
    state = RunState(run_id="r1", session_id="s1", iteration=4)
    failure = Failure(
        kind=FailureKind.capability_gap,
        explanation="no connector for SAP",
        blockers=("missing SAP connector",),
    )

    events = await _collect(handle_failure(failure, agent=_agent(), state=state))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, Handoff)
    assert "missing SAP connector" in event.blockers
    assert state.done is True


@pytest.mark.asyncio
async def test_retry_with_budget_exhausted_promotes_to_handoff() -> None:
    """``retry`` after retry_budget(transient_provider)=3 prior attempts → handoff."""
    state = RunState(
        run_id="r1",
        session_id="s1",
        failure_attempts={FailureKind.transient_provider: 3},
    )
    failure = Failure(kind=FailureKind.transient_provider, explanation="429")

    events = await _collect(handle_failure(failure, agent=_agent(), state=state))

    assert len(events) == 1
    assert isinstance(events[0], Handoff)
    assert state.done is True


@pytest.mark.asyncio
async def test_two_consecutive_no_progress_escalate_to_handoff() -> None:
    """First ``no_progress`` → narrow_scope; second → handoff."""
    state = RunState(run_id="r1", session_id="s1")
    failure = Failure(kind=FailureKind.no_progress, explanation="text only")

    # First strike: narrow_scope.
    events1 = await _collect(handle_failure(failure, agent=_agent(), state=state))
    assert isinstance(events1[0], AgentError)
    assert state.pending_instruction is not None
    assert state.failure_attempts[FailureKind.no_progress] == 1
    assert state.done is False

    # Reset pending_instruction to simulate the renderer clearing it next turn.
    state.pending_instruction = None

    # Second strike: handoff.
    events2 = await _collect(handle_failure(failure, agent=_agent(), state=state))
    assert isinstance(events2[0], Handoff)
    assert state.failure_attempts[FailureKind.no_progress] == 2
    assert state.done is True


@pytest.mark.asyncio
async def test_stop_action_yields_partial_run_summary() -> None:
    """``stop`` action: PartialRunSummary, run ends."""
    state = RunState(run_id="r1", session_id="s1")
    failure = Failure(
        kind=FailureKind.tool_error,
        explanation="unrecoverable",
        blockers=("budget exhausted",),
        suggested_action=Action.stop,
    )

    events = await _collect(handle_failure(failure, agent=_agent(), state=state))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PartialRunSummary)
    assert "budget exhausted" in event.missing
    assert state.done is True


@pytest.mark.asyncio
async def test_handle_failure_tracks_lessons_learned_with_dedup() -> None:
    """Lessons_learned: same-kind failures collapse to most-recent; cap at 5 distinct kinds."""
    state = RunState(run_id="r1", session_id="s1")
    for i in range(3):
        await _collect(
            handle_failure(
                Failure(kind=FailureKind.tool_error, explanation=f"attempt {i}"),
                agent=_agent(),
                state=state,
            )
        )
    # Only the most recent tool_error lesson remains.
    assert len(state.lessons_learned) == 1
    assert state.lessons_learned[0].explanation == "attempt 2"


@pytest.mark.asyncio
async def test_lessons_learned_caps_at_five_distinct_kinds() -> None:
    """A 6th distinct kind evicts the oldest."""
    state = RunState(run_id="r1", session_id="s1")
    distinct_kinds = [
        FailureKind.tool_error,
        FailureKind.transient_provider,
        FailureKind.output_refused,
        FailureKind.capability_gap,
        FailureKind.policy_violation,
        FailureKind.kernel_invalidated,  # the 6th — should evict tool_error
    ]
    for kind in distinct_kinds:
        await _collect(
            handle_failure(
                Failure(kind=kind, explanation=f"{kind.value} happened"),
                agent=_agent(),
                state=state,
            )
        )
    assert len(state.lessons_learned) == 5
    remaining_kinds = {f.kind for f in state.lessons_learned}
    assert FailureKind.tool_error not in remaining_kinds
    assert FailureKind.kernel_invalidated in remaining_kinds


@pytest.mark.asyncio
async def test_time_limit_grants_one_finalize_turn_then_hands_off() -> None:
    """First time_limit → narrow_scope 'publish what you have' turn; second → handoff."""
    state = RunState(run_id="r1", session_id="s1")
    failure = Failure(kind=FailureKind.time_limit, explanation="hit the 300s budget")

    # First strike: finalize turn, run continues (not suspended).
    events = await _collect(handle_failure(failure, agent=_agent(), state=state))
    assert len(events) == 1
    assert isinstance(events[0], AgentError)
    assert state.done is False
    instr = state.pending_instruction
    assert instr is not None
    # Must steer to publish-in-hand-then-terminate, not keep exploring; the
    # resume-friendly ask_user escape hatch stays available for the needs-more case.
    assert "return_dataset" in instr
    assert "return_done" in instr
    assert "return_unable" in instr
    assert "ask_user" in instr
    assert state.failure_attempts[FailureKind.time_limit] == 1

    # Second strike (grace exhausted): bounded — hands off, run ends.
    events2 = await _collect(handle_failure(failure, agent=_agent(), state=state))
    assert any(isinstance(e, Handoff) for e in events2)
    assert state.done is True
