"""Context compaction policy — last-N-iterations observation compaction.

Only raw ``role="tool"`` observations are ever compacted: tool results from the
last :data:`RECENT_ITERATIONS_DEFAULT` agent iterations render at ``"default"``
(full fidelity), older ones collapse to ``"minimal"``. Agent iterations are
delimited by assistant messages (one assistant turn + its tool results each).
Assistant / user / system messages are never compacted — they carry the durable
reasoning thread, and exact system-tool text is load-bearing for recovery.

The pure renderer owns this policy: :func:`recent_iterations_cutoff` computes the
window boundary and :func:`infer_message_mode` applies it per message. These
tests pin both, so a refactor cannot silently regress to a flat message window
or start compacting the agent's reasoning messages.
"""

from __future__ import annotations

from parsimony_agents.agent.renderer import (
    RECENT_ITERATIONS_DEFAULT,
    infer_message_mode,
    recent_iterations_cutoff,
)


def _msgs(roles: list[str]) -> list[dict]:
    return [{"role": r} for r in roles]


def _modes(roles: list[str]) -> list[str]:
    """Render-mode per message, replicating ``render_for_llm``'s pass."""
    cutoff = recent_iterations_cutoff(_msgs(roles))
    last_tool_idx = max((i for i, r in enumerate(roles) if r == "tool"), default=-1)
    return [
        infer_message_mode(
            index=i,
            is_last_tool_message=(i == last_tool_idx),
            role=role,
            default_cutoff=cutoff,
        )
        for i, role in enumerate(roles)
    ]


def test_recent_iterations_default_is_two():
    """Compaction keeps the last 2 agent iterations' observations at full fidelity."""
    assert RECENT_ITERATIONS_DEFAULT == 2


def test_cutoff_zero_when_too_few_iterations():
    """<= N assistant messages → cutoff 0 → nothing compacts."""
    assert recent_iterations_cutoff(_msgs(["user", "assistant", "tool"])) == 0
    assert recent_iterations_cutoff(_msgs(["assistant", "tool", "assistant", "tool"])) == 0


def test_cutoff_is_nth_from_last_assistant():
    """With > N iterations, the cutoff is the N-th-from-last assistant index."""
    # assistant messages at indices 1, 3, 5; N=2 → cutoff = index of the
    # 2nd-from-last assistant = 3.
    roles = ["user", "assistant", "tool", "assistant", "tool", "assistant", "tool"]
    assert recent_iterations_cutoff(_msgs(roles)) == 3


def test_old_tool_observations_compact():
    """Tool results before the cutoff collapse to minimal; recent ones stay default."""
    roles = [
        "user",       # 0
        "assistant",  # 1 — iteration 1
        "tool",       # 2 — old observation → minimal
        "assistant",  # 3 — iteration 2 (cutoff)
        "tool",       # 4 — recent observation → default
        "assistant",  # 5 — iteration 3
        "tool",       # 6 — recent + last tool → default
    ]
    modes = _modes(roles)
    assert modes[2] == "minimal"
    assert modes[4] == "default"
    assert modes[6] == "default"


def test_assistant_messages_never_compacted():
    """Assistant reasoning messages always render default, however old."""
    roles = ["assistant", "tool"] * 6  # 6 iterations — well past the window
    modes = _modes(roles)
    assert [m for r, m in zip(roles, modes) if r == "assistant"] == ["default"] * 6


def test_user_and_system_messages_never_compacted():
    """User and system-tool messages always render default — never compacted."""
    user_first = ["user"] + ["assistant", "tool"] * 5
    system_first = ["system"] + ["assistant", "tool"] * 5
    assert _modes(user_first)[0] == "default"
    assert _modes(system_first)[0] == "default"


def test_last_tool_message_always_default():
    """The most-recent tool result stays default even outside the recent window."""
    assert (
        infer_message_mode(
            index=0, is_last_tool_message=True, role="tool", default_cutoff=99
        )
        == "default"
    )


def test_empty_history():
    """No messages → no modes, cutoff 0."""
    assert _modes([]) == []
    assert recent_iterations_cutoff([]) == 0
