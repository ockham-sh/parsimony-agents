"""Phase 3 tests for ``parsimony_agents.agent.failure.policy``.

Verifies (PLAN Phase 3 done criteria):
- Retry budgets per kind.
- Exponential backoff for ``transient_provider`` capped at 30s.
- ``decide`` promotes ``retry`` → ``handoff`` when budget exhausted.
- Second-strike ``narrow_scope`` → ``handoff``.
"""

from __future__ import annotations

from parsimony_agents.agent.failure import Action, DefaultPolicy, Failure, FailureKind
from parsimony_agents.agent.failure.kinds import default_action_for
from parsimony_agents.agent.state import RunState


def test_every_failure_kind_has_a_default_action() -> None:
    """A new FailureKind without a default-action entry would KeyError at runtime."""
    for kind in FailureKind:
        assert isinstance(default_action_for(kind), Action)


def test_every_retry_default_kind_has_positive_budget() -> None:
    """Guard the footgun: a kind whose default action is ``retry`` but with no budget
    silently de-budgets to 0 → immediate handoff (zero retries). Any future
    retry-defaulting kind must also get a budget in DefaultPolicy."""
    policy = DefaultPolicy()
    retry_kinds = [k for k in FailureKind if default_action_for(k) is Action.retry]
    assert retry_kinds, "expected at least one retry-defaulting kind"
    for kind in retry_kinds:
        assert policy.retry_budget(kind) > 0, (
            f"{kind} defaults to Action.retry but has retry_budget 0 → immediate handoff; "
            "add it to DefaultPolicy._retry_budgets"
        )


def test_default_policy_retry_budget_transient_provider() -> None:
    """``transient_provider`` is retried up to 3 times."""
    assert DefaultPolicy().retry_budget(FailureKind.transient_provider) == 3


def test_default_policy_retry_budget_tool_error() -> None:
    assert DefaultPolicy().retry_budget(FailureKind.tool_error) == 2


def test_default_policy_retry_budget_unknown_kind_is_zero() -> None:
    """Any kind without an explicit retry budget defaults to 0 (no retry)."""
    assert DefaultPolicy().retry_budget(FailureKind.capability_gap) == 0


def test_default_policy_backoff_transient_provider_is_exponential_capped() -> None:
    """Backoff grows 2^attempt up to a 30s ceiling for ``transient_provider``."""
    policy = DefaultPolicy()
    assert policy.backoff(FailureKind.transient_provider, 1) == 2.0
    assert policy.backoff(FailureKind.transient_provider, 2) == 4.0
    assert policy.backoff(FailureKind.transient_provider, 3) == 8.0
    assert policy.backoff(FailureKind.transient_provider, 4) == 16.0
    assert policy.backoff(FailureKind.transient_provider, 5) == 30.0  # capped
    assert policy.backoff(FailureKind.transient_provider, 10) == 30.0


def test_default_policy_backoff_other_kinds_is_zero() -> None:
    """Non-transient kinds get zero backoff (synchronous retry)."""
    assert DefaultPolicy().backoff(FailureKind.tool_error, 1) == 0.0
    assert DefaultPolicy().backoff(FailureKind.output_truncated, 1) == 0.0


def test_decide_returns_suggested_action_when_no_attempts_yet() -> None:
    """First occurrence of a failure: ``decide`` honors ``failure.suggested_action``."""
    state = RunState(run_id="r1", session_id="s1")
    failure = Failure(kind=FailureKind.transient_provider, explanation="429")
    assert DefaultPolicy().decide(failure, state) is Action.retry


def test_decide_promotes_retry_to_handoff_when_budget_exhausted() -> None:
    """After ``retry_budget`` failures of the same kind, ``decide`` returns ``handoff``."""
    state = RunState(run_id="r1", session_id="s1", failure_attempts={FailureKind.transient_provider: 3})
    failure = Failure(kind=FailureKind.transient_provider, explanation="429")
    assert DefaultPolicy().decide(failure, state) is Action.handoff


def test_decide_narrow_scope_second_strike_is_handoff() -> None:
    """``narrow_scope`` after a prior attempt of the same kind escalates to ``handoff``."""
    state = RunState(run_id="r1", session_id="s1", failure_attempts={FailureKind.no_progress: 1})
    failure = Failure(kind=FailureKind.no_progress, explanation="text-only response")
    # Default suggested_action for no_progress is narrow_scope.
    assert DefaultPolicy().decide(failure, state) is Action.handoff


def test_decide_first_narrow_scope_stays_narrow_scope() -> None:
    state = RunState(run_id="r1", session_id="s1")  # no prior attempts
    failure = Failure(kind=FailureKind.no_progress, explanation="text-only response")
    assert DefaultPolicy().decide(failure, state) is Action.narrow_scope
