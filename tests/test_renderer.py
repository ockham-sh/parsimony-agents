"""Phase 5 tests for ``parsimony_agents.agent.renderer``.

Verifies (PLAN Phase 5 done criteria):
- Snapshot deduplication (only most recent ``context_snapshot=True`` kept).
- Minimal vs default mode heuristic.
- pending_instruction injection as user message after system prompt.
- lessons_learned renders as XML, capped (the cap is enforced by the recovery
  funnel; the renderer just emits whatever lessons it's given).
- System-tool messages always render in default mode regardless of position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from parsimony_agents.agent.failure import Failure, FailureKind
from parsimony_agents.agent.renderer import (
    infer_message_mode,
    recent_iterations_cutoff,
    render_for_llm,
    render_lessons_learned,
    select_messages_to_render,
)
from parsimony_agents.agent.state import RunState

# ---------------------------------------------------------------------------
# Minimal message stub
# ---------------------------------------------------------------------------


@dataclass
class _StubMsg:
    """Minimal stub matching the ``_MessageLike`` Protocol the renderer expects."""

    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
    _captured_mode: Literal["default", "minimal"] | None = None

    def to_llm(self, mode: Literal["default", "minimal"]) -> list[dict[str, Any]]:
        # Capture the mode so tests can assert which messages got which mode.
        self._captured_mode = mode
        suffix = "" if mode == "default" else "[min]"
        return [{"role": self.role, "content": self.content + suffix}]


# ---------------------------------------------------------------------------
# Test 1: snapshot deduplication
# ---------------------------------------------------------------------------


def test_only_most_recent_context_snapshot_is_kept() -> None:
    """Older ``context_snapshot=True`` messages are filtered out."""
    msgs = [
        _StubMsg(role="user", content="first snapshot", metadata={"context_snapshot": True}),
        _StubMsg(role="user", content="real user input"),
        _StubMsg(role="user", content="second snapshot", metadata={"context_snapshot": True}),
        _StubMsg(role="assistant", content="reply"),
        _StubMsg(role="user", content="third snapshot", metadata={"context_snapshot": True}),
    ]
    out = select_messages_to_render(msgs)
    contents = [m.content for m in out]
    assert "first snapshot" not in contents
    assert "second snapshot" not in contents
    assert "third snapshot" in contents
    assert "real user input" in contents


def test_snapshot_dedup_no_op_when_zero_or_one_snapshot() -> None:
    msgs = [
        _StubMsg(role="user", content="real user input"),
        _StubMsg(role="assistant", content="reply"),
    ]
    assert select_messages_to_render(msgs) == msgs


# ---------------------------------------------------------------------------
# Test 2: minimal mode excludes session_state, default mode includes
# ---------------------------------------------------------------------------


def test_mode_heuristic_compacts_old_tool_observations() -> None:
    """Tool messages before the iteration cutoff render minimal; at/after → default."""
    assert infer_message_mode(
        index=2, is_last_tool_message=False, role="tool", default_cutoff=4,
    ) == "minimal"
    assert infer_message_mode(
        index=6, is_last_tool_message=False, role="tool", default_cutoff=4,
    ) == "default"


def test_mode_heuristic_non_tool_roles_never_compacted() -> None:
    """Assistant / user messages always render default — they carry the reasoning thread."""
    for role in ("assistant", "user"):
        assert infer_message_mode(
            index=0, is_last_tool_message=False, role=role, default_cutoff=99,
        ) == "default"


def test_mode_heuristic_last_tool_message_is_default() -> None:
    """The last role='tool' message is always default even before the iteration cutoff."""
    assert infer_message_mode(
        index=2, is_last_tool_message=True, role="tool", default_cutoff=99,
    ) == "default"


def test_mode_heuristic_system_role_always_default() -> None:
    """System-tool messages always render at full fidelity (exact error text matters)."""
    assert infer_message_mode(
        index=0, is_last_tool_message=False, role="system", default_cutoff=99,
    ) == "default"


def test_recent_iterations_cutoff_delimits_by_assistant() -> None:
    """Cutoff is the N-th-from-last assistant index; 0 when too few iterations."""
    few = [{"role": "user"}, {"role": "assistant"}, {"role": "tool"}]
    assert recent_iterations_cutoff(few) == 0
    many = [{"role": r} for r in ["assistant", "tool", "assistant", "tool", "assistant", "tool"]]
    # assistant indices 0, 2, 4; N=RECENT_ITERATIONS_DEFAULT=2 → cutoff = index 2.
    assert recent_iterations_cutoff(many) == many.index({"role": "assistant"}, 2)


# ---------------------------------------------------------------------------
# Test 3: pending_instruction renders after system prompt
# ---------------------------------------------------------------------------


def test_pending_instruction_renders_as_user_message_after_system() -> None:
    """When pending_instruction is set, it appears as a user message right after the system prompt."""
    state = RunState(
        run_id="r1",
        session_id="s1",
        pending_instruction="narrow your next step",
        messages=[_StubMsg(role="user", content="real input")],
    )
    out = render_for_llm(state, instructions="you are an agent")
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "narrow your next step"
    # Real user message is rendered after pending_instruction.
    assert any("real input" in m.get("content", "") for m in out[2:])


def test_no_pending_instruction_skips_the_injection() -> None:
    state = RunState(
        run_id="r1",
        session_id="s1",
        messages=[_StubMsg(role="user", content="real input")],
    )
    out = render_for_llm(state, instructions="you are an agent")
    # Only one user message (the real one); no pending_instruction injection.
    user_messages = [m for m in out if m["role"] == "user"]
    assert len(user_messages) == 1


# ---------------------------------------------------------------------------
# Test 4: lessons_learned XML
# ---------------------------------------------------------------------------


def test_lessons_learned_renders_as_xml_block() -> None:
    """Each Failure becomes a ``<failure ... />`` line; the block is wrapped."""
    lessons = [
        Failure(kind=FailureKind.tool_error, explanation="kernel died"),
        Failure(kind=FailureKind.no_progress, explanation="text only", blockers=("a", "b")),
    ]
    xml = render_lessons_learned(lessons)
    assert "<lessons_learned>" in xml
    assert "</lessons_learned>" in xml
    assert 'kind="tool_error"' in xml
    assert 'kind="no_progress"' in xml
    assert 'blockers="a, b"' in xml
    assert 'explanation="kernel died"' in xml


def test_lessons_learned_empty_returns_empty_string() -> None:
    assert render_lessons_learned([]) == ""


def test_render_for_llm_includes_lessons_learned_in_output() -> None:
    """When lessons exist they appear as the final user message (positional recency)."""
    state = RunState(
        run_id="r1",
        session_id="s1",
        messages=[_StubMsg(role="user", content="hi")],
        lessons_learned=[Failure(kind=FailureKind.tool_error, explanation="x")],
    )
    out = render_for_llm(state, instructions="i am an agent")
    last = out[-1]
    assert last["role"] == "user"
    assert "<lessons_learned>" in last["content"]


def test_render_for_llm_omits_lessons_block_when_empty() -> None:
    state = RunState(
        run_id="r1",
        session_id="s1",
        messages=[_StubMsg(role="user", content="hi")],
    )
    out = render_for_llm(state, instructions="i am an agent")
    assert not any("<lessons_learned>" in (m.get("content") or "") for m in out)


# ---------------------------------------------------------------------------
# Test 5: System-tool message always default mode
# ---------------------------------------------------------------------------


def test_system_tool_message_always_renders_default_mode() -> None:
    """``role='system'`` always gets default mode, even when outside the recency window."""
    msgs: list[_StubMsg] = [
        _StubMsg(role="system", content="system_tool_error"),
        *[_StubMsg(role="user", content=f"u{i}") for i in range(10)],
    ]
    state = RunState(run_id="r1", session_id="s1", messages=msgs)
    render_for_llm(state, instructions="i am an agent")
    # The system message at index 0 should have been called with mode="default".
    assert msgs[0]._captured_mode == "default"
    # User messages are never compacted — only raw tool observations are.
    assert msgs[1]._captured_mode == "default"


# ---------------------------------------------------------------------------
# Test 6: connectors_catalog inclusion / omission
# ---------------------------------------------------------------------------


def test_connectors_catalog_included_in_system_prompt_when_present() -> None:
    state = RunState(run_id="r1", session_id="s1")
    out = render_for_llm(
        state,
        instructions="you are an agent",
        capabilities_preamble="<available_connectors>postgres-x</available_connectors>",
    )
    assert "<available_connectors>postgres-x" in out[0]["content"]


def test_connectors_catalog_omitted_when_empty() -> None:
    state = RunState(run_id="r1", session_id="s1")
    out = render_for_llm(state, instructions="you are an agent")
    assert "<available_connectors>" not in out[0]["content"]


# ---------------------------------------------------------------------------
# Test 7: byte-stability across calls (cache hygiene)
# ---------------------------------------------------------------------------


def test_renderer_is_byte_stable_across_identical_calls() -> None:
    """Same input → same output. Pure function; provider caches stay hot."""
    state = RunState(
        run_id="r1",
        session_id="s1",
        messages=[
            _StubMsg(role="user", content="hi"),
            _StubMsg(role="assistant", content="hello"),
        ],
    )
    out1 = render_for_llm(state, instructions="agent")
    out2 = render_for_llm(state, instructions="agent")
    assert out1 == out2


# ---------------------------------------------------------------------------
# Test 8: tool message at index 0 with newer non-tool messages still gets default
# ---------------------------------------------------------------------------


def test_only_last_tool_message_gets_default_mode() -> None:
    """An old tool observation compacts; the recent one and the last tool stay default."""
    msgs: list[_StubMsg] = [
        _StubMsg(role="user", content="u0"),
        _StubMsg(role="assistant", content="a1"),    # idx 1 — iteration 1
        _StubMsg(role="tool", content="early-tool"),  # idx 2 — old observation
        _StubMsg(role="assistant", content="a2"),    # idx 3 — iteration 2 (cutoff)
        _StubMsg(role="tool", content="mid-tool"),    # idx 4 — recent observation
        _StubMsg(role="assistant", content="a3"),    # idx 5 — iteration 3
        _StubMsg(role="tool", content="late-tool"),   # idx 6 — the last tool
    ]
    state = RunState(run_id="r1", session_id="s1", messages=msgs)
    render_for_llm(state, instructions="agent")

    # Old tool observation (before the last-2-iterations cutoff) → minimal.
    assert msgs[2]._captured_mode == "minimal"
    # Recent tool observation and the last tool → default.
    assert msgs[4]._captured_mode == "default"
    assert msgs[6]._captured_mode == "default"
    # The agent's own reasoning messages are never compacted.
    assert msgs[1]._captured_mode == "default"
