"""Phase 1 tests for ``parsimony_agents.agent.failure.kinds``.

Verifies (BRIEF §1.A, plan Phase 1 done criteria):
- Every :class:`FailureKind` has a default :class:`Action`.
- :class:`Failure` is hashable + immutable.
- Sentinel ``suggested_action=None`` resolves to the kind default; explicit value wins.
"""

from __future__ import annotations

import dataclasses

import pytest

from parsimony_agents.agent.failure import (
    Action,
    Failure,
    FailureKind,
    FailureRaised,
    default_action_for,
)


def test_every_failure_kind_has_default_action() -> None:
    """Every member of :class:`FailureKind` must have a default :class:`Action`."""
    for kind in FailureKind:
        action = default_action_for(kind)
        assert isinstance(action, Action)


def test_failure_sentinel_resolves_to_kind_default() -> None:
    """``suggested_action=None`` → ``__post_init__`` fills with the kind's default."""
    f = Failure(kind=FailureKind.transient_provider, explanation="boom")
    assert f.suggested_action is Action.retry

    f2 = Failure(kind=FailureKind.ambiguous_input, explanation="which one?")
    assert f2.suggested_action is Action.ask_user


def test_failure_explicit_action_wins_over_default() -> None:
    """An explicit ``suggested_action`` is preserved even when it differs from the default."""
    f = Failure(
        kind=FailureKind.transient_provider,
        explanation="boom",
        suggested_action=Action.handoff,
    )
    assert f.suggested_action is Action.handoff


def test_failure_is_frozen() -> None:
    """``frozen=True`` blocks attribute assignment after construction."""
    f = Failure(kind=FailureKind.tool_error, explanation="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.explanation = "y"


def test_failure_is_hashable() -> None:
    """:class:`Failure` is hashable; equal failures collapse in a set."""
    f1 = Failure(kind=FailureKind.no_progress, explanation="stalled", blockers=("a", "b"))
    f2 = Failure(kind=FailureKind.no_progress, explanation="stalled", blockers=("a", "b"))
    assert {f1, f2} == {f1}


def test_failure_blockers_coerced_to_tuple() -> None:
    """A list-of-strings ``blockers`` arg is coerced to an immutable tuple."""
    f = Failure(kind=FailureKind.capability_gap, explanation="x", blockers=["one", "two"])
    assert f.blockers == ("one", "two")
    assert isinstance(f.blockers, tuple)


def test_failure_raised_wraps_failure() -> None:
    """:class:`FailureRaised` carries the :class:`Failure` value verbatim."""
    f = Failure(kind=FailureKind.transient_provider, explanation="rate limited")
    exc = FailureRaised(f)
    assert exc.failure is f
    assert "transient_provider" in str(exc)
    assert "rate limited" in str(exc)
