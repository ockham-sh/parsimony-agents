"""Structured failure-handling taxonomy and recovery primitives.

This package owns the load-bearing types and the machinery around them:

- :class:`FailureKind` — a closed enum naming every failure mode the framework recognises.
- :class:`Action` — the recovery move the policy decides on.
- :class:`Failure` — the immutable value flowing between detectors, policy, and recovery.
- :mod:`detectors` — pure-function failure detectors run by the loop.
- :mod:`policy` — maps a :class:`Failure` to an :class:`Action`.
- :mod:`recovery` — the funnel that dispatches an :class:`Action` into agent events.
"""

# Eager imports: only the data types and pure-function detectors. These do NOT
# depend on :mod:`parsimony_agents.agent.events`, so they are safe to load at the
# package root.
from parsimony_agents.agent.failure.detectors import (
    accumulate_usage,
    loop_signature,
    post_llm,
    post_tool,
    pre_step,
    record_tool_call,
)
from parsimony_agents.agent.failure.kinds import (
    Action,
    Failure,
    FailureKind,
    FailureRaised,
    default_action_for,
)

# Lazy imports: ``policy`` is fine eagerly (no events dep), but ``recovery`` depends
# on :mod:`parsimony_agents.agent.events`, which imports :class:`Failure` from this
# package. Loading recovery here would create a cycle when ``events`` is loaded
# during package init. Keep recovery accessible at the root via ``__getattr__``.
from parsimony_agents.agent.failure.policy import DefaultPolicy, RecoveryPolicy
from parsimony_agents.agent.failure.suspension import (
    SuspensionExpired,
    SuspensionRequest,
    SuspensionTokenMismatch,
    compute_suspension_token,
    verify_suspension_token,
)
from parsimony_agents.agent.failure.termination import TerminationRequest


def __getattr__(name: str):
    if name == "handle_failure":
        from parsimony_agents.agent.failure.recovery import handle_failure

        return handle_failure
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Action",
    "DefaultPolicy",
    "Failure",
    "FailureKind",
    "FailureRaised",
    "RecoveryPolicy",
    "SuspensionExpired",
    "SuspensionRequest",
    "SuspensionTokenMismatch",
    "TerminationRequest",
    "accumulate_usage",
    "compute_suspension_token",
    "default_action_for",
    "handle_failure",
    "loop_signature",
    "post_llm",
    "post_tool",
    "pre_step",
    "record_tool_call",
    "verify_suspension_token",
]
