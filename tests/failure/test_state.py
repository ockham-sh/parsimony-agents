"""Phase 1 tests for ``parsimony_agents.agent.state``.

Verifies (BRIEF gaps 44–48, plan Phase 1 done criteria):
- :class:`RunState` JSON round-trips via ``model_dump_json`` / ``model_validate_json``.
- :class:`SuspensionRecord` round-trips; suspension tokens are unique per record.
- Runtime services (``files``, ``code_executor``, ``cancellation``) are excluded from output.
- HMAC suspension tokens compute / verify / tamper-detect correctly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from parsimony_agents.agent.failure import Action, Failure, FailureKind
from parsimony_agents.agent.state import (
    RunState,
    SuspensionRecord,
    TurnSubstate,
    compute_suspension_token,
    verify_suspension_token,
)


def test_runstate_round_trips_through_json() -> None:
    """``RunState`` survives a ``model_dump_json`` → ``model_validate_json`` cycle."""
    state = RunState(
        run_id="r1",
        session_id="s1",
        iteration=3,
        failure_attempts={FailureKind.no_progress: 1},
        pending_instruction="narrow it",
        lessons_learned=[
            Failure(kind=FailureKind.tool_error, explanation="exec failed", blockers=("kernel",)),
        ],
        cumulative_cost_usd=0.42,
        cumulative_prompt_tokens=1234,
        cumulative_completion_tokens=567,
        tool_call_history=["read_data:abc123de", "read_data:abc123de"],
    )

    payload = state.model_dump_json()
    rehydrated = RunState.model_validate_json(payload)

    assert rehydrated.run_id == "r1"
    assert rehydrated.iteration == 3
    assert rehydrated.failure_attempts[FailureKind.no_progress] == 1
    assert rehydrated.pending_instruction == "narrow it"
    assert len(rehydrated.lessons_learned) == 1
    assert rehydrated.lessons_learned[0].kind == FailureKind.tool_error
    assert rehydrated.lessons_learned[0].blockers == ("kernel",)
    assert rehydrated.lessons_learned[0].suggested_action == Action.retry
    assert rehydrated.cumulative_cost_usd == 0.42


def test_runstate_excludes_runtime_services_from_serialization() -> None:
    """``files``, ``code_executor``, ``cancellation`` are ``Field(exclude=True)``."""
    state = RunState(
        run_id="r1",
        session_id="s1",
        files=object(),  # Any sentinel
        code_executor=object(),
        cancellation=object(),
    )
    payload = json.loads(state.model_dump_json())
    assert "files" not in payload
    assert "code_executor" not in payload
    assert "cancellation" not in payload


def test_turnsubstate_default_factory_isolation() -> None:
    """Each :class:`RunState` gets its own :class:`TurnSubstate` (no shared mutable default)."""
    s1 = RunState(run_id="r1", session_id="s1")
    s2 = RunState(run_id="r2", session_id="s2")
    s1.turn.tool_calls_this_turn = 5
    assert s2.turn.tool_calls_this_turn == 0


def test_record_failure_attempt_increments() -> None:
    """``record_failure_attempt`` returns the new count and persists it."""
    state = RunState(run_id="r1", session_id="s1")
    assert state.record_failure_attempt(FailureKind.no_progress) == 1
    assert state.record_failure_attempt(FailureKind.no_progress) == 2
    assert state.failure_attempts[FailureKind.no_progress] == 2


def test_suspension_record_round_trips() -> None:
    """:class:`SuspensionRecord` survives JSON round-trip with all gap-44–48 fields."""
    started = datetime.now(timezone.utc)
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        started_at=started,
        elapsed_seconds=12.5,
        iteration_count=5,
        tool_call_history=["read_data:abc", "read_data:abc"],
        pending_question="which dataset?",
        pending_question_context="user wrote 'the one'",
        originating_failure_kind=FailureKind.ambiguous_input,
        accumulated_reasoning="thinking step 1\nthinking step 2",
        accumulated_reasoning_duration_s=3.14,
        last_repeat_counts={"read_data:abc": 2},
        cumulative_cost_usd=0.10,
        cumulative_prompt_tokens=100,
        cumulative_completion_tokens=50,
    )

    payload = record.model_dump_json()
    rehydrated = SuspensionRecord.model_validate_json(payload)

    assert rehydrated.run_id == "r1"
    assert rehydrated.iteration_count == 5
    assert rehydrated.elapsed_seconds == 12.5
    assert rehydrated.originating_failure_kind == FailureKind.ambiguous_input
    assert rehydrated.accumulated_reasoning_duration_s == 3.14
    assert rehydrated.last_repeat_counts == {"read_data:abc": 2}
    assert rehydrated.suspension_token == record.suspension_token


def test_suspension_tokens_are_unique_per_record() -> None:
    """Two suspensions of the same (run_id, session_id) produce distinct tokens (nonce)."""
    t1 = compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret")
    t2 = compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret")
    assert t1 != t2


def test_suspension_token_verifies() -> None:
    """A freshly-issued token verifies under the same secret."""
    started = datetime.now(timezone.utc)
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        started_at=started,
        pending_question="?",
    )
    assert verify_suspension_token(record=record, secret="topsecret") is True


def test_suspension_token_rejects_tamper() -> None:
    """A tampered token (wrong run_id) fails verification."""
    started = datetime.now(timezone.utc)
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        started_at=started,
        pending_question="?",
    )
    tampered = record.model_copy(update={"run_id": "r2"})
    assert verify_suspension_token(record=tampered, secret="topsecret") is False


def test_suspension_token_rejects_wrong_secret() -> None:
    """A token issued under one secret fails verification under another."""
    started = datetime.now(timezone.utc)
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token=compute_suspension_token(run_id="r1", session_id="s1", secret="topsecret"),
        started_at=started,
        pending_question="?",
    )
    assert verify_suspension_token(record=record, secret="othersecret") is False


def test_suspension_token_rejects_malformed() -> None:
    """A token missing the ``nonce.digest`` separator fails verification cleanly."""
    started = datetime.now(timezone.utc)
    record = SuspensionRecord(
        run_id="r1",
        session_id="s1",
        suspension_token="garbage",
        started_at=started,
        pending_question="?",
    )
    assert verify_suspension_token(record=record, secret="topsecret") is False
