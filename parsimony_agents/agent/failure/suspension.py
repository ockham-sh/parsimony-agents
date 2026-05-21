"""Suspension primitives: exception class and HMAC token helpers.

Three pieces:

- :class:`SuspensionRequest` — raised by the ``ask_user`` tool (and synthesized by the
  recovery funnel for ``Action.ask_user`` decisions) to bubble a suspension request up
  through the loop. The loop catches it, builds a :class:`SuspensionRecord` from the
  current :class:`RunState`, yields :class:`UserInputRequested`, and exits cleanly.

- :func:`compute_suspension_token` — issues an HMAC-SHA256 token bound to
  ``(run_id, session_id, nonce)`` under a shared secret. The wire format is
  ``"{nonce}.{hexdigest}"``; the nonce ensures distinct suspensions of the same
  ``(run_id, session_id)`` produce distinct tokens (replay protection per BRIEF gap 45).

- :func:`verify_suspension_token` — constant-time verify a record's token matches
  the secret.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from parsimony_agents.agent.state import SuspensionRecord


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
    payload = f"{run_id}:{session_id}:{nonce}".encode("utf-8")
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
    payload = f"{record.run_id}:{record.session_id}:{nonce}".encode("utf-8")
    expected_digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(provided_digest, expected_digest)


class SuspensionTokenMismatch(Exception):
    """Raised by :meth:`Agent.resume` when the presented token fails verification."""


class SuspensionExpired(Exception):
    """Raised by :meth:`Agent.resume` when the suspension is older than the configured max age."""


__all__ = [
    "SuspensionExpired",
    "SuspensionRequest",
    "SuspensionTokenMismatch",
    "compute_suspension_token",
    "verify_suspension_token",
]
