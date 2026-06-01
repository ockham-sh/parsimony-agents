"""Recovery funnel: one async generator that takes a :class:`Failure` and a
:class:`RecoveryPolicy`, dispatches the chosen :class:`Action`, and yields the
appropriate agent events.

The funnel is the single integration point between failure detection and the
event stream the host sees. Detector → ``handle_failure`` → 0..N
:class:`AgentEvent`. The loop drives :func:`handle_failure`, consumes the events,
and inspects ``state.done`` to decide whether to continue.

Action dispatch (each branch documented inline):

- ``retry``: sleep per policy.backoff, yield ``AgentError(failure=...)``, leave
  ``state.done`` unchanged. Caller's next iteration will re-attempt.
- ``narrow_scope``: set ``state.pending_instruction`` to a corrective prompt and
  yield ``AgentError``. Leave ``state.done`` unchanged.
- ``ask_user``: synthesize a :class:`SuspensionRecord`, yield
  :class:`UserInputRequested`, set ``state.done = True``.
- ``handoff``: yield :class:`Handoff` with structured blockers, set ``state.done = True``.
- ``stop``: yield :class:`PartialRunSummary` carrying the structured gap, set
  ``state.done = True``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Protocol

from parsimony_agents.agent.events import (
    AgentError,
    AgentEvent,
    Handoff,
    PartialRunSummary,
    UserInputRequested,
)
from parsimony_agents.agent.failure.kinds import Action, Failure, FailureKind
from parsimony_agents.agent.failure.policy import DefaultPolicy, RecoveryPolicy
from parsimony_agents.agent.failure.suspension import compute_suspension_token
from parsimony_agents.agent.state import RunState, SuspensionRecord

_logger = logging.getLogger(__name__)

# Cap on the number of distinct kinds retained in ``state.lessons_learned`` — the
# renderer emits an XML block per lesson, so unlimited growth would balloon prompts.
_LESSONS_LEARNED_CAP = 5


class _AgentLike(Protocol):
    """Minimal agent surface the recovery funnel needs.

    Avoids importing the concrete :class:`Agent` class so :mod:`recovery` doesn't
    transitively depend on the loop. The real Agent will provide all of these.
    """

    suspension_secret: str
    policy: RecoveryPolicy


def _track_lesson(state: RunState, failure: Failure) -> None:
    """Append (or replace) the lessons_learned entry for this kind, capped at 5 distinct kinds.

    Most-recent occurrence of a given kind wins. The renderer reads this and emits
    ``<lessons_learned>`` inside the context block.
    """
    state.lessons_learned = [f for f in state.lessons_learned if f.kind != failure.kind]
    state.lessons_learned.append(failure)
    # Cap by *distinct kinds*; we already dedup above, so a length check suffices.
    if len(state.lessons_learned) > _LESSONS_LEARNED_CAP:
        # Drop the oldest. The deque trade-off isn't worth it for a 5-element list.
        state.lessons_learned = state.lessons_learned[-_LESSONS_LEARNED_CAP:]


def _build_suspension_record(
    state: RunState,
    failure: Failure | None,
    *,
    question: str,
    context: str | None,
    secret: str,
    originating_kind: FailureKind | None,
) -> SuspensionRecord:
    """Snapshot ``state`` into a JSON-serializable :class:`SuspensionRecord`.

    The token binds (run_id, session_id, nonce) under ``secret`` so the host
    cannot forge a resume. See ``failure.suspension`` for the wire format.
    """
    return SuspensionRecord(
        run_id=state.run_id,
        session_id=state.session_id,
        suspension_token=compute_suspension_token(
            run_id=state.run_id,
            session_id=state.session_id,
            secret=secret,
        ),
        messages=list(state.messages),
        iteration_count=state.iteration,
        tool_call_history=list(state.tool_call_history),
        minted_refs=list(state.turn.minted_refs),
        minted_live_names=dict(state.turn.minted_live_names),
        started_at=state.started_at,
        elapsed_seconds=state.elapsed_seconds(),
        pending_question=question,
        pending_question_context=context,
        originating_failure_kind=originating_kind,
        model_id=state.model_id,
        accumulated_reasoning=state.accumulated_reasoning,
        accumulated_reasoning_duration_s=state.accumulated_reasoning_duration_s,
        last_repeat_counts=dict(state.last_repeat_counts),
        cumulative_cost_usd=state.cumulative_cost_usd,
        cumulative_prompt_tokens=state.cumulative_prompt_tokens,
        cumulative_completion_tokens=state.cumulative_completion_tokens,
        lessons_learned=list(state.lessons_learned),
        failure_attempts=dict(state.failure_attempts),
    )


def _narrow_scope_instruction(failure: Failure) -> str:
    """Build the corrective prompt injected as ``state.pending_instruction``.

    Kind-specific. A ``no_progress`` strike means the LLM answered with prose and
    no tool call — the corrective prompt must NOT be steered toward "make
    progress", because that buries the fact that calling ``ask_user`` is itself a
    valid, correct response when the agent genuinely needs input from the user.
    The previous one-size-fits-all text ended with "pick the smallest piece of
    work … or return_unable", which pushed the agent to plough ahead even when it
    had really been trying to ask the user a question. Other narrow_scope kinds
    (``scope_too_large``, ``kernel_invalidated``) keep the shrink-the-step framing.
    """
    if failure.kind is FailureKind.no_progress:
        return (
            "Your previous turn was text only — no tool call — so the run could "
            "not advance. Respond now with a tool call, choosing the option that "
            "actually fits the situation (no default bias toward any one):\n"
            "- If a concrete next step is clear, take it (run code, read data, "
            "publish an artifact, …).\n"
            "- If you genuinely need information only the user can provide — an "
            "ambiguous request, a missing choice — call ask_user. That is the "
            "correct response, not a fallback or a failure.\n"
            "- If the task is already complete, call return_done.\n"
            "- If you are blocked and cannot proceed, call return_unable.\n"
            "Do not reply with prose alone again."
        )
    return (
        f"Your last attempt produced {failure.kind.value}: {failure.explanation} "
        "Narrow the next step: pick the single smallest piece of work that would "
        "make progress. If you cannot, call return_unable with a blockers list."
    )


def _ask_user_question_for(failure: Failure) -> tuple[str, str | None]:
    """Build a clarifying question + optional context block for an ``ask_user`` action."""
    if failure.kind is FailureKind.ambiguous_input:
        return (
            f"I need a clarification: {failure.explanation}",
            None,
        )
    if failure.kind is FailureKind.loop_detected:
        return (
            "I keep retrying the same approach and getting the same result. "
            "Could you share what specifically you want different?",
            failure.explanation,
        )
    if failure.kind is FailureKind.iteration_limit:
        return (
            "I have reached the step limit for this task. "
            "Should I continue from where I left off, or refocus the request?",
            failure.explanation,
        )
    if failure.kind is FailureKind.time_limit:
        return (
            "I have hit the time budget for this task. "
            "Want me to continue, or narrow it down to a sub-task?",
            failure.explanation,
        )
    return (
        f"I need a clarification before continuing: {failure.explanation}",
        None,
    )


async def handle_failure(
    failure: Failure,
    *,
    agent: _AgentLike,
    state: RunState,
) -> AsyncGenerator[AgentEvent, None]:
    """Dispatch the recovery action for ``failure`` and yield resulting events.

    Always increments ``state.failure_attempts[failure.kind]`` and tracks the
    failure in ``state.lessons_learned``. Whether the run terminates is signalled
    via ``state.done``; the caller's outer loop checks ``not state.done`` to decide.
    """
    policy = agent.policy if agent.policy is not None else DefaultPolicy()
    action = policy.decide(failure, state)

    _logger.info(
        "agent_failure failure_kind=%s action=%s iteration=%d attempts=%d run_id=%s",
        failure.kind.value,
        action.value,
        state.iteration,
        state.failure_attempts.get(failure.kind, 0),
        state.run_id,
    )

    # Record the attempt *after* the policy decision so ``decide`` sees the prior
    # count (e.g. the second-strike rule reads attempts==1 before bumping to 2).
    state.record_failure_attempt(failure.kind)
    _track_lesson(state, failure)

    if action is Action.retry:
        attempt = state.failure_attempts[failure.kind]
        delay = policy.backoff(failure.kind, attempt)
        if delay > 0:
            await asyncio.sleep(delay)
        yield AgentError(
            message=f"Recoverable failure ({failure.kind.value}): {failure.explanation}",
            failure=failure,
        )
        return

    if action is Action.narrow_scope:
        state.pending_instruction = _narrow_scope_instruction(failure)
        yield AgentError(
            message=f"Narrowing scope after {failure.kind.value}: {failure.explanation}",
            failure=failure,
        )
        return

    if action is Action.ask_user:
        question, context = _ask_user_question_for(failure)
        record = _build_suspension_record(
            state,
            failure,
            question=question,
            context=context,
            secret=agent.suspension_secret,
            originating_kind=failure.kind,
        )
        yield UserInputRequested(
            question=question,
            context=context,
            choices=None,
            suspension_record=record,
            originating_failure_kind=failure.kind.value,
        )
        state.done = True
        return

    if action is Action.handoff:
        yield Handoff(
            rationale=failure.explanation,
            blockers=list(failure.blockers),
            suggested_next_steps=[],
        )
        state.done = True
        return

    if action is Action.stop:
        yield PartialRunSummary(
            missing=list(failure.blockers),
            learned_facts=[],
            next_step_plan=None,
        )
        state.done = True
        return

    # Exhaustive: every Action is handled. Defensive raise so a new variant is
    # caught at runtime rather than silently swallowing the failure.
    raise RuntimeError(f"unhandled recovery action: {action!r}")


__all__ = ["handle_failure"]
