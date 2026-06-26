"""Recovery policy: maps a :class:`Failure` to an :class:`Action`.

The :class:`RecoveryPolicy` protocol lets hosts inject their own policy
(e.g. tighter retry budgets for cost-sensitive tiers, custom escalation rules)
without touching the recovery funnel.

:class:`DefaultPolicy` is the production policy. It implements:

- Per-kind retry budgets (transient_provider: 3, tool_error: 2, output_truncated: 1).
- Exponential backoff for ``transient_provider`` capped at 30s; zero backoff for
  other kinds.
- ``decide()`` starts from ``failure.suggested_action`` and promotes to ``handoff``
  when retry budgets are exhausted or when ``narrow_scope`` is hit twice for the
  same kind (the "second-strike" rule for ``no_progress`` per BRIEF §4.4).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from parsimony_agents.agent.failure.kinds import Action, Failure, FailureKind
from parsimony_agents.agent.state import RunState


@runtime_checkable
class RecoveryPolicy(Protocol):
    """Hosts may implement this protocol to inject a custom recovery policy."""

    def retry_budget(self, kind: FailureKind) -> int:
        """How many ``retry`` attempts are budgeted for this kind."""

    def backoff(self, kind: FailureKind, attempt: int) -> float:
        """Seconds to sleep before retry attempt ``attempt`` (1-indexed)."""

    def decide(self, failure: Failure, state: RunState) -> Action:
        """Final action for the failure; may override ``failure.suggested_action``."""


class DefaultPolicy:
    """Production default recovery policy. See module docstring for rules."""

    _RETRY_BUDGETS: dict[FailureKind, int] = {
        FailureKind.transient_provider: 3,
        FailureKind.tool_error: 2,
        FailureKind.output_truncated: 1,
    }

    _BACKOFF_CAP_S: float = 30.0

    def retry_budget(self, kind: FailureKind) -> int:
        return self._RETRY_BUDGETS.get(kind, 0)

    def backoff(self, kind: FailureKind, attempt: int) -> float:
        if kind is FailureKind.transient_provider:
            # 2 ** 1 = 2s, 2 ** 2 = 4s, 2 ** 3 = 8s, ... capped at 30s.
            return min(2.0 ** max(attempt, 1), self._BACKOFF_CAP_S)
        return 0.0

    def decide(self, failure: Failure, state: RunState) -> Action:
        action = failure.suggested_action or Action.stop
        prior_attempts = state.failure_attempts.get(failure.kind, 0)

        # Retry budget exhausted → promote to handoff so the user sees blockers
        # instead of a silent stop after N invisible retries.
        if action is Action.retry:
            budget = self.retry_budget(failure.kind)
            if prior_attempts >= budget:
                return Action.handoff

        # Second-strike for narrow_scope failures: the policy gave the agent one
        # chance to recover (e.g. text-no-tools, scope_too_large, kernel_invalidated)
        # by injecting pending_instruction. A second occurrence means narrowing isn't
        # working — hand off.
        if action is Action.narrow_scope and prior_attempts >= 1:
            return Action.handoff

        return action


__all__ = [
    "DefaultPolicy",
    "RecoveryPolicy",
]
