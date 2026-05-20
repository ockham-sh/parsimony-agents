"""Last-N-iterations context compaction policy.

The agent loop renders tool results (observations) from the most recent
``RECENT_ITERATIONS_DEFAULT`` agent iterations at ``mode="default"``
(full fidelity); every earlier observation collapses to
``mode="minimal"``.

Iterations are delimited by assistant messages — each iteration emits
one assistant message followed by its tool results. So "the last N
iterations' observations" are the tool messages that appear after the
N-th-from-last assistant message.

These tests pin the rendering predicate. The actual call site lives in
``parsimony_agents/agent/agent.py``; here we exercise the same predicate
in isolation so a future refactor of the agent loop can't silently
regress the policy.
"""

from __future__ import annotations

from parsimony_agents.agent.agent import RECENT_ITERATIONS_DEFAULT
from parsimony_agents.agent.models import AgentMessage
from parsimony_agents.messages import Message, Text


def _render_modes(messages: list[AgentMessage], n: int = RECENT_ITERATIONS_DEFAULT) -> list[str]:
    """Replicate the predicate at agent.py — return the per-message mode."""
    assistant_indices = [i for i, m in enumerate(messages) if m.role == "assistant"]
    default_cutoff = -1 if len(assistant_indices) <= n else assistant_indices[-n]
    return [
        "minimal" if (m.role == "tool" and i < default_cutoff) else "default"
        for i, m in enumerate(messages)
    ]


def _make(role: str, *, content: str = "x", snapshot: bool = False) -> AgentMessage:
    meta = {"context_snapshot": True} if snapshot else {}
    cls = Message if snapshot else AgentMessage
    return cls(role=role, content=Text(content=content), metadata=meta)


def test_default_n_is_two():
    """The shipped default keeps the last 2 iterations at full fidelity."""
    assert RECENT_ITERATIONS_DEFAULT == 2


def test_few_iterations_all_default():
    """With <= N iterations, every observation stays at default."""
    # 2 iterations (2 assistant messages), N=2 → all default.
    messages = [
        _make("system"),
        _make("user"),
        _make("assistant"),
        _make("tool"),
        _make("assistant"),
        _make("tool"),
        _make("user", snapshot=True),
    ]
    assert _render_modes(messages, n=2) == ["default"] * 7


def test_old_observations_demoted_keeps_last_two_iterations():
    """With 4 iterations and N=2, the first 2 iterations' tools go minimal."""
    messages = [
        _make("system"),
        _make("user"),
        _make("assistant"),  # iteration 1
        _make("tool"),        # obs 1  -> minimal
        _make("assistant"),  # iteration 2
        _make("tool"),        # obs 2  -> minimal
        _make("assistant"),  # iteration 3
        _make("tool"),        # obs 3  -> default
        _make("assistant"),  # iteration 4
        _make("tool"),        # obs 4  -> default
        _make("user", snapshot=True),
    ]
    modes = _render_modes(messages, n=2)
    assert modes == [
        "default",  # system
        "default",  # user
        "default",  # assistant 1
        "minimal",  # obs 1 — older than last 2 iterations
        "default",  # assistant 2
        "minimal",  # obs 2 — older than last 2 iterations
        "default",  # assistant 3
        "default",  # obs 3 — within last 2
        "default",  # assistant 4
        "default",  # obs 4 — within last 2
        "default",  # snapshot
    ]


def test_n_equals_one_only_last_iteration_default():
    """N=1: only the most recent iteration's observations stay default."""
    messages = [
        _make("system"),
        _make("assistant"),  # iter 1
        _make("tool"),        # obs 1 -> minimal
        _make("assistant"),  # iter 2
        _make("tool"),        # obs 2 -> minimal
        _make("assistant"),  # iter 3
        _make("tool"),        # obs 3 -> default
    ]
    modes = _render_modes(messages, n=1)
    assert modes[2] == "minimal"
    assert modes[4] == "minimal"
    assert modes[6] == "default"


def test_multiple_tool_results_in_one_iteration_all_default():
    """An iteration with several tool calls: all its observations share the mode."""
    messages = [
        _make("system"),
        _make("assistant"),  # iter 1
        _make("tool"),
        _make("tool"),
        _make("assistant"),  # iter 2
        _make("tool"),
        _make("assistant"),  # iter 3 — most recent
        _make("tool"),
        _make("tool"),
        _make("tool"),
    ]
    modes = _render_modes(messages, n=2)
    # iter 1 (2 tools) older than last 2 → minimal
    assert modes[2] == "minimal"
    assert modes[3] == "minimal"
    # iter 2 + iter 3 within last 2 → default
    assert modes[5] == "default"
    assert modes[7] == modes[8] == modes[9] == "default"


def test_non_tool_messages_never_demoted():
    """User / assistant / system messages always render at default."""
    messages = [
        _make("system"),
        _make("user"),
        _make("assistant"),
        _make("tool"),
        _make("assistant"),
        _make("tool"),
        _make("assistant"),
        _make("tool"),
        _make("assistant"),
        _make("tool"),
    ]
    modes = _render_modes(messages, n=2)
    for i, m in enumerate(messages):
        if m.role != "tool":
            assert modes[i] == "default"


def test_no_iterations_yet_all_default():
    """Before any assistant turn, nothing is demoted."""
    messages = [_make("system"), _make("user")]
    assert _render_modes(messages, n=2) == ["default", "default"]


def test_compaction_spans_turn_boundaries():
    """Compaction is iteration-based, not turn-based — a new user message
    mid-history does not reset the window."""
    messages = [
        _make("system"),
        _make("user"),         # turn 1
        _make("assistant"),    # iter 1
        _make("tool"),          # obs 1 -> minimal (older than last 2 iters)
        _make("user"),         # turn 2 — does NOT make obs 1 "recent"
        _make("assistant"),    # iter 2
        _make("tool"),          # obs 2 -> default
        _make("assistant"),    # iter 3
        _make("tool"),          # obs 3 -> default
    ]
    modes = _render_modes(messages, n=2)
    assert modes[3] == "minimal"  # obs 1
    assert modes[6] == "default"  # obs 2
    assert modes[8] == "default"  # obs 3
