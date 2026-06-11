"""Suspension primitives: exception class and HMAC token helpers.

Three pieces:

- :class:`SuspensionRequest` — raised by the ``ask_user`` tool (and synthesized by the
  recovery funnel for ``Action.ask_user`` decisions) to bubble a suspension request up
  through the loop. The loop catches it, builds a :class:`SuspensionRecord` from the
  current :class:`RunState`, yields :class:`UserInputRequested`, and exits cleanly.

- :func:`compute_suspension_token` — issues an HMAC-SHA256 token bound to
  ``(run_id, session_id, nonce)`` under a shared secret. The wire format is
  ``"{nonce}.{hexdigest}"``; the nonce ensures distinct suspensions of the same
  ``(run_id, session_id)`` produce distinct tokens (one token per suspension).

- :func:`verify_suspension_token` — constant-time verify a record's token matches
  the secret.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import UTC, datetime

from parsimony_agents.agent.failure.kinds import FailureKind
from parsimony_agents.agent.state import RunState, SuspensionRecord


class SuspensionRequest(Exception):
    """Raised when the agent suspends pending a user reply.

    Carries the question + optional context + the originating
    :class:`~parsimony_agents.agent.failure.kinds.FailureKind` when the suspension was
    synthesized by the recovery funnel (None when the agent directly called ``ask_user``).
    The loop's tool-execution phase catches this and yields a
    :class:`~parsimony_agents.agent.events.UserInputRequested` event.

    Cancellation precedence: if a cancellation fires before the loop catches the
    :class:`SuspensionRequest`, the exception is suppressed and the run emits
    :class:`~parsimony_agents.agent.events.RunCancelled` instead.
    """

    def __init__(
        self,
        question: str,
        *,
        context: str | None = None,
        choices: list[str] | None = None,
        originating_failure_kind: str | None = None,
    ):
        super().__init__(question)
        self.question = question
        self.context = context
        self.choices = choices
        self.originating_failure_kind = originating_failure_kind


def compute_suspension_token(
    *,
    run_id: str,
    session_id: str,
    secret: str,
    nonce: str | None = None,
) -> str:
    """Issue the HMAC-SHA256 suspension token.

    Wire format: ``"{nonce}.{hexdigest}"``. ``nonce`` defaults to a fresh 16-byte hex
    string so two suspensions of the same ``(run_id, session_id)`` produce distinct
    tokens (replay protection).
    """
    if not secret:
        raise ValueError("suspension_token secret is empty; refuse to issue a token")
    if nonce is None:
        nonce = secrets.token_hex(16)
    elif "." in nonce:
        # The wire format is ``"{nonce}.{hexdigest}"`` and verification splits on
        # the first ".". A nonce containing "." would corrupt that boundary.
        raise ValueError("suspension_token nonce must not contain '.'")
    payload = f"{run_id}:{session_id}:{nonce}".encode()
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"{nonce}.{digest}"


def verify_suspension_token(
    *,
    record: SuspensionRecord,
    secret: str,
) -> bool:
    """Constant-time verify that ``record.suspension_token`` matches ``secret``."""
    if not secret:
        return False
    token = record.suspension_token
    if "." not in token:
        return False
    nonce, provided_digest = token.split(".", 1)
    payload = f"{record.run_id}:{record.session_id}:{nonce}".encode()
    expected_digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(provided_digest, expected_digest)


class SuspensionTokenMismatch(Exception):
    """Raised by :meth:`Agent.resume` when the presented token fails verification."""


class SuspensionExpired(Exception):
    """Raised by :meth:`Agent.resume` when the suspension is older than the configured max age."""


def build_suspension_record(
    state: RunState,
    *,
    question: str,
    context: str | None,
    secret: str,
    originating_kind: FailureKind | str | None = None,
) -> SuspensionRecord:
    """Snapshot ``state`` into a JSON-serializable :class:`SuspensionRecord`.

    The single builder for both suspension exits — the ``ask_user`` tool path and
    the recovery funnel's ``Action.ask_user`` decision — so adding a carried-over
    field means touching one place. The token binds (run_id, session_id, nonce)
    under ``secret``.
    """
    return SuspensionRecord(
        run_id=state.run_id,
        session_id=state.session_id,
        model_id=state.model_id,
        suspension_token=compute_suspension_token(
            run_id=state.run_id,
            session_id=state.session_id,
            secret=secret,
        ),
        messages=list(state.messages),
        iteration_count=state.iteration,
        tool_call_history=list(state.tool_call_history),
        minted_refs=list(state.minted_refs),
        minted_live_names=dict(state.minted_live_names),
        started_at=state.started_at,
        elapsed_seconds=state.elapsed_seconds(),
        pending_question=question,
        pending_question_context=context,
        originating_failure_kind=originating_kind,
        accumulated_reasoning=state.accumulated_reasoning,
        accumulated_reasoning_duration_s=state.accumulated_reasoning_duration_s,
        last_repeat_counts=dict(state.last_repeat_counts),
        cumulative_cost_usd=state.cumulative_cost_usd,
        cumulative_prompt_tokens=state.cumulative_prompt_tokens,
        cumulative_completion_tokens=state.cumulative_completion_tokens,
        lessons_learned=list(state.lessons_learned),
        failure_attempts=dict(state.failure_attempts),
    )


def validate_suspension(
    record: SuspensionRecord,
    user_reply: str,
    *,
    secret: str,
    max_age_s: float | None,
) -> None:
    """Validate a resume request: non-empty reply, token authenticity, staleness.

    The single validator shared by ``Agent.resume`` and the bare-spine
    ``resume_run``. Raises :class:`ValueError` for an empty reply,
    :class:`SuspensionTokenMismatch` for a bad token, and
    :class:`SuspensionExpired` when older than ``max_age_s`` (skip the staleness
    check by passing ``None``).
    """
    if not user_reply or not user_reply.strip():
        raise ValueError("resume requires a non-empty user_reply")
    if not verify_suspension_token(record=record, secret=secret):
        raise SuspensionTokenMismatch(f"suspension token failed verification for run_id={record.run_id!r}")
    if max_age_s is not None:
        age = (datetime.now(UTC) - record.suspended_at).total_seconds()
        if age > max_age_s:
            raise SuspensionExpired(f"suspension is {age:.0f}s old (max {max_age_s:.0f}s)")


__all__ = [
    "SuspensionExpired",
    "SuspensionRequest",
    "SuspensionTokenMismatch",
    "build_suspension_record",
    "compute_suspension_token",
    "validate_suspension",
    "verify_suspension_token",
]
