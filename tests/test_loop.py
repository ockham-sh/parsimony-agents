"""Phase 7 integration tests for ``parsimony_agents.agent.loop``.

Drives the new ``run_loop`` end-to-end against a stubbed LLM. Each test patches
:func:`litellm.acompletion` / :func:`litellm.stream_chunk_builder` so the loop
sees a canned response sequence, then asserts the resulting event stream.

Covers (PLAN Phase 7 test criteria):
1. Happy path: text → tool call → ``return_done`` → loop exits.
2. Text-no-tools recovery: first turn text-only → ``pending_instruction`` injected.
3. Text-no-tools double strike: two consecutive text-only → ``Handoff``.
4. ``iteration_limit`` → ``UserInputRequested`` (action=ask_user via policy default).
5. ``return_unable`` → ``Handoff`` with blockers + rationale.
6. ``ask_user`` tool → ``UserInputRequested`` with valid suspension record.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.agent.events import (
    AgentError,
    Handoff,
    PartialRunSummary,
    TextDelta,
    ToolEvent,
    UserInputRequested,
)
from parsimony_agents.agent.failure import DefaultPolicy
from parsimony_agents.agent.loop import run_loop
from parsimony_agents.agent.state import RunState
from parsimony_agents.agent.termination_tools import TERMINATION_TOOLS
from parsimony_agents.tools import Tools


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async iterator wrapping a list of chunks. Used as the stream object."""

    def __init__(self, chunks: list[Any]):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _stream_chunk(
    *,
    content: str | None = None,
    tool_call: tuple[str, str, str] | None = None,  # (id, name, args_json)
) -> SimpleNamespace:
    """Build a single streaming delta chunk in litellm's shape."""
    tool_calls = None
    if tool_call is not None:
        tc_id, name, args = tool_call
        tool_calls = [
            SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args))
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, reasoning_content=None, tool_calls=tool_calls),
            )
        ]
    )


def _assembled(
    *,
    content: str = "",
    tool_calls: list[tuple[str, str, str]] | None = None,
    finish_reason: str = "tool_calls",
) -> SimpleNamespace:
    """Build the assembled response that ``litellm.stream_chunk_builder`` would return."""
    calls = []
    for tc_id, name, args in tool_calls or []:
        calls.append(
            SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=args))
        )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content="",
                    tool_calls=calls,
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


class _LLMScript:
    """Sequence of (stream_chunks, assembled_response) per LLM call.

    ``litellm.acompletion`` and ``litellm.stream_chunk_builder`` are patched to pull
    from this script in order.
    """

    def __init__(self, turns: list[tuple[list[Any], Any]]):
        self._turns = list(turns)
        self._stream_calls = 0

    async def acompletion(self, *_, **__) -> _FakeStream:
        if self._stream_calls >= len(self._turns):
            raise RuntimeError("LLMScript exhausted")
        chunks, _ = self._turns[self._stream_calls]
        self._stream_calls += 1
        return _FakeStream(chunks)

    def stream_chunk_builder(self, *_, **__) -> Any:
        # ``stream_chunk_builder`` is called once per call_llm completion, in lockstep
        # with the just-finished stream. ``_stream_calls`` is already incremented.
        _, assembled = self._turns[self._stream_calls - 1]
        return assembled


def _agent(
    *,
    tools: list | None = None,
    guardrails: AgentGuardrails | None = None,
) -> SimpleNamespace:
    """Build the minimum AgentLike for the loop."""
    return SimpleNamespace(
        guardrails=guardrails or AgentGuardrails(max_iterations=10),
        policy=DefaultPolicy(),
        suspension_secret="topsecret",
        model_config={"model": "claude-opus-4-7"},
        instructions="you are an agent",
        tools=Tools(list(tools or []) + TERMINATION_TOOLS),
    )


async def _drain(loop_agen) -> list:
    return [event async for event in loop_agen]


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_return_done_terminates_loop() -> None:
    """One LLM call calls return_done → loop exits with state.done = True."""
    script = _LLMScript(
        turns=[
            (
                [
                    _stream_chunk(content="Done!"),
                    _stream_chunk(tool_call=("call_1", "return_done", "")),
                ],
                _assembled(
                    content="Done!",
                    tool_calls=[("call_1", "return_done", '{"summary": "completed the task"}')],
                ),
            )
        ]
    )

    with (
        patch("litellm.acompletion", side_effect=script.acompletion),
        patch("litellm.stream_chunk_builder", side_effect=script.stream_chunk_builder),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        agent = _agent()
        state = RunState(run_id="r1", session_id="s1")
        events = await _drain(run_loop(agent, state))

    assert state.done is True
    # Expect at least one ToolEvent for return_done (started + completed).
    completed_tool_events = [e for e in events if isinstance(e, ToolEvent) and e.completed]
    assert any(e.tool_name == "return_done" for e in completed_tool_events)


# ---------------------------------------------------------------------------
# 2. Text-no-tools recovery (single strike)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_response_injects_pending_instruction_then_recovers() -> None:
    """First turn text-only → narrow_scope (pending_instruction injected); second turn return_done."""
    script = _LLMScript(
        turns=[
            (
                [_stream_chunk(content="Sure, I'll proceed.")],
                _assembled(content="Sure, I'll proceed.", tool_calls=None, finish_reason="stop"),
            ),
            (
                [_stream_chunk(tool_call=("call_2", "return_done", ""))],
                _assembled(
                    content="",
                    tool_calls=[("call_2", "return_done", '{"summary": "done now"}')],
                ),
            ),
        ]
    )

    with (
        patch("litellm.acompletion", side_effect=script.acompletion),
        patch("litellm.stream_chunk_builder", side_effect=script.stream_chunk_builder),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        agent = _agent()
        state = RunState(run_id="r1", session_id="s1")
        events = await _drain(run_loop(agent, state))

    # AgentError yielded from the first turn's narrow_scope recovery.
    assert any(isinstance(e, AgentError) for e in events)
    # State done after second turn's return_done.
    assert state.done is True


# ---------------------------------------------------------------------------
# 3. Text-no-tools double strike → Handoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_twice_escalates_to_handoff() -> None:
    """Two consecutive text-only responses → second escalates to Handoff."""
    script = _LLMScript(
        turns=[
            ([_stream_chunk(content="first")], _assembled(content="first", finish_reason="stop")),
            ([_stream_chunk(content="second")], _assembled(content="second", finish_reason="stop")),
        ]
    )

    with (
        patch("litellm.acompletion", side_effect=script.acompletion),
        patch("litellm.stream_chunk_builder", side_effect=script.stream_chunk_builder),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        agent = _agent()
        state = RunState(run_id="r1", session_id="s1")
        events = await _drain(run_loop(agent, state))

    assert any(isinstance(e, Handoff) for e in events)
    assert state.done is True


# ---------------------------------------------------------------------------
# 4. iteration_limit → UserInputRequested
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_iteration_limit_yields_user_input_requested() -> None:
    """state.iteration starting at max_iterations - 1 → after bump pre_step fires iteration_limit."""
    agent = _agent(guardrails=AgentGuardrails(max_iterations=1))
    state = RunState(run_id="r1", session_id="s1", iteration=0)
    # The loop increments to 1 = max_iterations, pre_step fires iteration_limit, default action ask_user.

    with patch("litellm.acompletion", side_effect=AssertionError("must not be called")):
        events = await _drain(run_loop(agent, state))

    user_inputs = [e for e in events if isinstance(e, UserInputRequested)]
    assert len(user_inputs) == 1
    assert user_inputs[0].originating_failure_kind == "iteration_limit"
    assert state.done is True


# ---------------------------------------------------------------------------
# 5. return_unable → Handoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_return_unable_yields_handoff_with_blockers() -> None:
    """An LLM that calls return_unable → Handoff with the structured blockers."""
    script = _LLMScript(
        turns=[
            (
                [_stream_chunk(tool_call=("call_3", "return_unable", ""))],
                _assembled(
                    content="",
                    tool_calls=[
                        (
                            "call_3",
                            "return_unable",
                            json.dumps(
                                {
                                    "blockers": ["missing SAP connector"],
                                    "rationale": "cannot reach the source system",
                                }
                            ),
                        )
                    ],
                ),
            )
        ]
    )

    with (
        patch("litellm.acompletion", side_effect=script.acompletion),
        patch("litellm.stream_chunk_builder", side_effect=script.stream_chunk_builder),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        agent = _agent()
        state = RunState(run_id="r1", session_id="s1")
        events = await _drain(run_loop(agent, state))

    handoffs = [e for e in events if isinstance(e, Handoff)]
    assert len(handoffs) == 1
    assert handoffs[0].blockers == ["missing SAP connector"]
    assert "source system" in handoffs[0].rationale
    assert state.done is True


# ---------------------------------------------------------------------------
# 6. ask_user → UserInputRequested with valid suspension record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_user_tool_suspends_with_valid_record() -> None:
    """ask_user → UserInputRequested with verifiable suspension token."""
    from parsimony_agents.agent.failure import verify_suspension_token

    script = _LLMScript(
        turns=[
            (
                [_stream_chunk(tool_call=("call_4", "ask_user", ""))],
                _assembled(
                    content="",
                    tool_calls=[
                        (
                            "call_4",
                            "ask_user",
                            json.dumps({"question": "which dataset?"}),
                        )
                    ],
                ),
            )
        ]
    )

    with (
        patch("litellm.acompletion", side_effect=script.acompletion),
        patch("litellm.stream_chunk_builder", side_effect=script.stream_chunk_builder),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        agent = _agent()
        state = RunState(run_id="r1", session_id="s1")
        events = await _drain(run_loop(agent, state))

    suspensions = [e for e in events if isinstance(e, UserInputRequested)]
    assert len(suspensions) == 1
    assert suspensions[0].question == "which dataset?"
    # The suspension record's token must verify under the same secret.
    assert verify_suspension_token(
        record=suspensions[0].suspension_record, secret="topsecret"
    ) is True
    assert state.done is True
