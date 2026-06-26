"""Failure taxonomy primitives.

Three types form the failure-handling spine:

- :class:`FailureKind` — the closed set of classifications detectors can produce.
- :class:`Action` — the recovery move :mod:`policy` chooses for a given failure.
- :class:`Failure` — the immutable value carrying a kind plus contextual detail.

:class:`FailureRaised` is the exception that bubbles a :class:`Failure` up through
the LLM chokepoint or tool execution boundary to the recovery funnel.

``Failure`` is implemented as a Pydantic dataclass (``frozen=True``) so it round-trips
through JSON inside the bigger Pydantic state models (``RunState``, ``SuspensionRecord``)
without needing custom serializers per field.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass


class FailureKind(StrEnum):
    """Closed enumeration of every failure mode the framework recognises.

    Categories (informal — used to drive precedence at the detector layer):

    - Transient/provider: :attr:`transient_provider`
    - Output-quality: :attr:`output_truncated`, :attr:`output_refused`
    - Input/scope: :attr:`ambiguous_input`, :attr:`scope_too_large`, :attr:`capability_gap`
    - Progress: :attr:`no_progress`, :attr:`loop_detected`
    - Tool/runtime: :attr:`tool_error`, :attr:`policy_violation`, :attr:`kernel_invalidated`
    - Budget exhaustion (hard stops): :attr:`iteration_limit`, :attr:`time_limit`
    """

    transient_provider = "transient_provider"
    output_truncated = "output_truncated"
    output_refused = "output_refused"
    ambiguous_input = "ambiguous_input"
    scope_too_large = "scope_too_large"
    capability_gap = "capability_gap"
    no_progress = "no_progress"
    loop_detected = "loop_detected"
    tool_error = "tool_error"
    policy_violation = "policy_violation"
    kernel_invalidated = "kernel_invalidated"
    iteration_limit = "iteration_limit"
    time_limit = "time_limit"


class Action(StrEnum):
    """Recovery move the policy dispatches for a :class:`Failure`."""

    retry = "retry"
    ask_user = "ask_user"
    narrow_scope = "narrow_scope"
    handoff = "handoff"
    stop = "stop"


# Default :class:`Action` per :class:`FailureKind`. The policy may override these
# based on retry budgets / attempt counters; this map is the static fallback so
# every Failure has a sensible suggested_action without policy plumbing.
_DEFAULT_ACTION_BY_KIND: Mapping[FailureKind, Action] = MappingProxyType(
    {
        FailureKind.transient_provider: Action.retry,
        FailureKind.output_truncated: Action.retry,
        FailureKind.output_refused: Action.handoff,
        FailureKind.ambiguous_input: Action.ask_user,
        FailureKind.scope_too_large: Action.narrow_scope,
        FailureKind.capability_gap: Action.handoff,
        FailureKind.no_progress: Action.narrow_scope,
        FailureKind.loop_detected: Action.ask_user,
        FailureKind.tool_error: Action.retry,
        FailureKind.policy_violation: Action.handoff,
        FailureKind.kernel_invalidated: Action.narrow_scope,
        FailureKind.iteration_limit: Action.ask_user,
        # time_limit grants one bounded "publish what you have" turn (narrow_scope),
        # then the narrow_scope second-strike hands off. See detectors.pre_step.
        FailureKind.time_limit: Action.narrow_scope,
    }
)


def default_action_for(kind: FailureKind) -> Action:
    """Return the default :class:`Action` for a :class:`FailureKind`."""

    return _DEFAULT_ACTION_BY_KIND[kind]


@pydantic_dataclass(frozen=True)
class Failure:
    """Immutable record of a failure detected somewhere in the agent loop.

    :param kind: The closed classification.
    :param explanation: Human-readable detail. Surfaces to the user via UI and to the LLM
        via ``pending_instruction`` / ``lessons_learned`` rendering.
    :param blockers: Optional list of structured blockers — used by handoff-tier failures
        (capability_gap, output_refused, policy_violation). Empty otherwise.
    :param suggested_action: Initial recovery move. ``None`` (the default) means use
        :func:`default_action_for` ``(kind)``; populated by ``__post_init__``.
        :mod:`policy` may override based on retry budgets / attempt counters.
    :param partial_data: Any partial result the failing operation produced (e.g. truncated
        output, partial tool result). Recovery may use this to enrich the next attempt.
    :param metadata: Kind-specific extras (e.g. ``repeat_count`` for ``loop_detected``,
        ``provider_error`` for ``transient_provider``).
    """

    kind: FailureKind
    explanation: str
    blockers: tuple[str, ...] = ()
    # ``None`` is the "use kind default" sentinel; after ``__post_init__`` this is
    # always an :class:`Action`. Typed as ``Action | None`` to make the sentinel
    # path explicit at the type level.
    suggested_action: Action | None = None
    partial_data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks normal assignment; use ``object.__setattr__``.
        if self.suggested_action is None:
            object.__setattr__(self, "suggested_action", _DEFAULT_ACTION_BY_KIND[self.kind])

        # Coerce iterable to tuple defensively so callers passing list[str] still
        # get an immutable container.
        if not isinstance(self.blockers, tuple):
            object.__setattr__(self, "blockers", tuple(self.blockers))

    def __hash__(self) -> int:
        # ``metadata`` is a dict (not hashable); hash on the kind + explanation +
        # blockers tuple. Two failures with the same kind + message + blockers count
        # as "the same" for set-membership semantics (used by lessons_learned dedup).
        return hash((self.kind, self.explanation, self.blockers))


class FailureRaised(Exception):
    """Raised by the LLM chokepoint or tool execution to surface a :class:`Failure`.

    The agent loop's outer ``try/except`` catches this and routes to the recovery
    funnel. Carries the :class:`Failure` value verbatim — no string serialisation.
    """

    def __init__(self, failure: Failure):
        super().__init__(f"{failure.kind}: {failure.explanation}")
        self.failure = failure


__all__ = [
    "Action",
    "Failure",
    "FailureKind",
    "FailureRaised",
    "default_action_for",
]
