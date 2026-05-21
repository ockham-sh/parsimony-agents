"""Phase 4 tests for ``parsimony_agents.agent.llm``.

Verifies (PLAN Phase 4 done criteria):
- Successful call returns a fully-assembled :class:`LLMResponse`.
- ``RateLimitError`` → ``FailureRaised(transient_provider)`` with provider_error metadata.
- Streaming heartbeat fires ``FailureRaised(transient_provider)`` after silence.
- Cancellation during stream raises ``asyncio.CancelledError`` (not wrapped).
- Reasoning content is preserved on the final response.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.failure import FailureKind, FailureRaised
from parsimony_agents.agent.llm import (
    LLMComplete,
    LLMReasoningDelta,
    LLMResponse,
    LLMTextDelta,
    LLMToolCallStarted,
    call_llm,
)


# ---------------------------------------------------------------------------
# Helpers: stub litellm-style streaming response
# ---------------------------------------------------------------------------


def _chunk(
    *,
    content: str | None = None,
    reasoning: str | None = None,
    tool_call: tuple[str, str] | None = None,  # (name, tool_call_id)
) -> SimpleNamespace:
    """Build a single streaming chunk matching the litellm shape."""
    tool_calls = None
    if tool_call is not None:
        name, tc_id = tool_call
        tool_calls = [
            SimpleNamespace(id=tc_id, function=SimpleNamespace(name=name, arguments=""))
        ]
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _Stream:
    """Minimal async iterator wrapping a list of chunks (optionally with delays)."""

    def __init__(self, chunks: list[Any], delay_before_each: float = 0.0):
        self._chunks = list(chunks)
        self._delay = delay_before_each

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._chunks.pop(0)


def _assembled_response(
    *,
    content: str = "",
    reasoning_content: str = "",
    finish_reason: str = "stop",
    tool_calls: list | None = None,
) -> SimpleNamespace:
    """Build a fake assembled response that ``litellm.stream_chunk_builder`` would return."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls or [],
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )


# ---------------------------------------------------------------------------
# Test 1: success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_success_yields_deltas_then_complete() -> None:
    """Happy path: stream yields text deltas in order then an LLMComplete."""
    chunks = [
        _chunk(content="Hello "),
        _chunk(content="world"),
    ]
    fake_stream = _Stream(chunks)
    assembled = _assembled_response(content="Hello world")

    with (
        patch("litellm.acompletion", return_value=fake_stream),
        patch("litellm.stream_chunk_builder", return_value=assembled),
    ):
        signals = []
        async for sig in call_llm(
            messages=[{"role": "user", "content": "hi"}],
            model_config={"model": "anthropic/claude-opus-4-7"},
        ):
            signals.append(sig)

    text_signals = [s for s in signals if isinstance(s, LLMTextDelta)]
    assert [s.content for s in text_signals] == ["Hello ", "world"]
    complete = signals[-1]
    assert isinstance(complete, LLMComplete)
    assert isinstance(complete.response, LLMResponse)
    assert complete.response.content == "Hello world"
    assert complete.response.finish_reason == "stop"


# ---------------------------------------------------------------------------
# Test 2: litellm errors classified into FailureRaised(transient_provider)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_rate_limit_error_becomes_failure_raised() -> None:
    """``RateLimitError`` raised by litellm.acompletion → FailureRaised(transient_provider)."""

    class RateLimitError(Exception):
        pass

    async def _raise(*_, **__):
        raise RateLimitError("429 Too Many Requests")

    with patch("litellm.acompletion", side_effect=_raise):
        with pytest.raises(FailureRaised) as excinfo:
            async for _ in call_llm(
                messages=[{"role": "user", "content": "x"}],
                model_config={"model": "anthropic/claude-opus-4-7"},
            ):
                pass

    failure = excinfo.value.failure
    assert failure.kind == FailureKind.transient_provider
    assert failure.metadata["provider_error"] == "RateLimitError"
    assert "429" in failure.metadata["detail"]


@pytest.mark.asyncio
async def test_call_llm_authentication_error_classifies_as_capability_gap() -> None:
    """``AuthenticationError`` is not retryable → ``capability_gap``."""

    class AuthenticationError(Exception):
        pass

    async def _raise(*_, **__):
        raise AuthenticationError("invalid api key")

    with patch("litellm.acompletion", side_effect=_raise):
        with pytest.raises(FailureRaised) as excinfo:
            async for _ in call_llm(
                messages=[{"role": "user", "content": "x"}],
                model_config={"model": "anthropic/claude-opus-4-7"},
            ):
                pass

    failure = excinfo.value.failure
    assert failure.kind == FailureKind.capability_gap


# ---------------------------------------------------------------------------
# Test 3: streaming heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_heartbeat_fires_after_silence() -> None:
    """Slow stream (per-chunk delay > heartbeat_s) → FailureRaised(transient_provider)."""
    # 0.5s between each chunk; heartbeat at 0.05s → first chunk already trips.
    fake_stream = _Stream([_chunk(content="x")], delay_before_each=0.5)
    assembled = _assembled_response(content="x")

    with (
        patch("litellm.acompletion", return_value=fake_stream),
        patch("litellm.stream_chunk_builder", return_value=assembled),
    ):
        with pytest.raises(FailureRaised) as excinfo:
            async for _ in call_llm(
                messages=[{"role": "user", "content": "x"}],
                model_config={"model": "anthropic/claude-opus-4-7"},
                stream_heartbeat_s=0.05,
            ):
                pass

    failure = excinfo.value.failure
    assert failure.kind == FailureKind.transient_provider
    assert failure.metadata.get("reason") == "heartbeat_timeout"


# ---------------------------------------------------------------------------
# Test 4: cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_cancellation_raises_cancelled_error() -> None:
    """Setting the cancellation event mid-stream raises ``CancelledError`` (not wrapped)."""
    cancellation = CancellationRequest()
    chunks = [_chunk(content="part1"), _chunk(content="part2"), _chunk(content="part3")]
    fake_stream = _Stream(chunks)
    assembled = _assembled_response(content="part1part2part3")

    cancelled = False
    with (
        patch("litellm.acompletion", return_value=fake_stream),
        patch("litellm.stream_chunk_builder", return_value=assembled),
    ):
        try:
            async for sig in call_llm(
                messages=[{"role": "user", "content": "x"}],
                model_config={"model": "anthropic/claude-opus-4-7"},
                cancellation=cancellation,
            ):
                # Cancel mid-stream: after first text delta arrives.
                if isinstance(sig, LLMTextDelta):
                    cancellation.set()
        except asyncio.CancelledError:
            cancelled = True

    assert cancelled


@pytest.mark.asyncio
async def test_call_llm_cancellation_before_call_raises_immediately() -> None:
    """Pre-set cancellation → CancelledError before litellm.acompletion is even called."""
    cancellation = CancellationRequest()
    cancellation.set()
    with patch("litellm.acompletion", side_effect=AssertionError("must not be called")):
        with pytest.raises(asyncio.CancelledError):
            async for _ in call_llm(
                messages=[{"role": "user", "content": "x"}],
                model_config={"model": "anthropic/claude-opus-4-7"},
                cancellation=cancellation,
            ):
                pass


# ---------------------------------------------------------------------------
# Test 5: reasoning content preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_preserves_reasoning_content() -> None:
    """Reasoning deltas pass through and the assembled response carries them."""
    chunks = [
        _chunk(reasoning="thinking..."),
        _chunk(content="Final answer"),
    ]
    fake_stream = _Stream(chunks)
    assembled = _assembled_response(content="Final answer", reasoning_content="thinking...")

    with (
        patch("litellm.acompletion", return_value=fake_stream),
        patch("litellm.stream_chunk_builder", return_value=assembled),
    ):
        signals = []
        async for sig in call_llm(
            messages=[{"role": "user", "content": "x"}],
            model_config={"model": "anthropic/claude-opus-4-7"},
        ):
            signals.append(sig)

    reasoning_signals = [s for s in signals if isinstance(s, LLMReasoningDelta)]
    assert len(reasoning_signals) == 1
    assert reasoning_signals[0].content == "thinking..."

    complete = signals[-1]
    assert isinstance(complete, LLMComplete)
    assert complete.response.reasoning_content == "thinking..."
    assert complete.response.reasoning_duration_s >= 0.0


# ---------------------------------------------------------------------------
# Bonus: LLMToolCallStarted emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_emits_tool_call_started_once_per_id() -> None:
    """First chunk with a tool name → LLMToolCallStarted; further chunks for the same id don't re-emit."""
    chunks = [
        _chunk(tool_call=("read_data", "call_123")),
        _chunk(tool_call=("read_data", "call_123")),  # duplicate id, no re-emit
        _chunk(tool_call=("write_data", "call_456")),  # distinct id, fresh emit
    ]
    fake_stream = _Stream(chunks)
    assembled = _assembled_response()

    with (
        patch("litellm.acompletion", return_value=fake_stream),
        patch("litellm.stream_chunk_builder", return_value=assembled),
    ):
        signals = [
            s
            async for s in call_llm(
                messages=[{"role": "user", "content": "x"}],
                model_config={"model": "anthropic/claude-opus-4-7"},
            )
        ]

    starts = [s for s in signals if isinstance(s, LLMToolCallStarted)]
    assert len(starts) == 2
    assert starts[0].tool_name == "read_data"
    assert starts[0].tool_call_id == "call_123"
    assert starts[1].tool_name == "write_data"
