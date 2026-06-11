"""Single chokepoint for LLM calls.

One async generator — :func:`call_llm` — that streams a litellm completion call,
yielding internal :class:`LLMStreamSignal` events as chunks arrive, then finally
yielding an :class:`LLMComplete` carrying the assembled :class:`LLMResponse`.

Failures are bubbled as :exc:`FailureRaised` carrying a structured
:class:`~parsimony_agents.agent.failure.kinds.Failure`. Recovery is the loop's
responsibility — :func:`call_llm` does not retry. The previous monolithic loop's
inline retry block (``agent.py:722-829``) is replaced by the recovery funnel at
the loop level.

Failure classifications:

- ``litellm.RateLimitError``, ``litellm.InternalServerError``, ``litellm.Timeout``,
  ``litellm.APIConnectionError`` → ``Failure(kind=transient_provider)``, suggested
  action ``retry``. ``metadata["provider_error"]`` is the original exception class name.
- Streaming heartbeat exceeded (``stream_heartbeat_s`` of silence between chunks) →
  same ``transient_provider`` failure with ``metadata["reason"]="heartbeat_timeout"``.
- ``litellm.AuthenticationError``, ``litellm.BadRequestError`` → ``Failure(kind=
  capability_gap)`` with the original message. These are not retryable.
- ``asyncio.CancelledError`` is *not* caught — propagates so the loop can yield
  :class:`RunCancelled` (cancel takes precedence over suspend).

Merge point — LLM proxy session: when the proxy lands, reconcile
``response.usage_cost_usd`` from the proxy's ``X-Estimated-Cost-USD`` response header
(authoritative). Currently :func:`accumulate_usage` in :mod:`failure.detectors`
computes a best-effort estimate via ``litellm.completion_cost``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import litellm
from parsimony.transport import redact_sensitive_text

from parsimony_agents.agent.caching import apply_anthropic_cache_markers
from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.failure.kinds import Failure, FailureKind, FailureRaised

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public stream-signal hierarchy. Distinct from :mod:`agent.events` so the
# loop can translate internal LLM signals into transport-facing AgentEvents
# (the loop knows the message_id; ``call_llm`` does not).
# ---------------------------------------------------------------------------


@dataclass
class _LLMStreamSignalBase:
    type: str = ""


@dataclass
class LLMTextDelta(_LLMStreamSignalBase):
    type: Literal["llm_text_delta"] = "llm_text_delta"
    content: str = ""


@dataclass
class LLMReasoningDelta(_LLMStreamSignalBase):
    type: Literal["llm_reasoning_delta"] = "llm_reasoning_delta"
    content: str = ""


@dataclass
class LLMToolCallStarted(_LLMStreamSignalBase):
    """Emitted the first time a chunk reveals the name of a streaming tool call.

    The loop translates this into a ``ToolEvent(completed=False)`` so the UI can
    show "starting <tool>" before the tool actually executes. Tool args still
    accumulate inside ``LLMResponse.tool_calls`` until the stream ends.
    """

    type: Literal["llm_tool_call_started"] = "llm_tool_call_started"
    tool_name: str = ""
    tool_call_id: str = ""


@dataclass
class LLMComplete(_LLMStreamSignalBase):
    """Terminal signal carrying the assembled response. Always last."""

    type: Literal["llm_complete"] = "llm_complete"
    response: LLMResponse | None = None


LLMStreamSignal = LLMTextDelta | LLMReasoningDelta | LLMToolCallStarted | LLMComplete


@dataclass
class LLMResponse:
    """Assembled litellm completion. The loop reads :attr:`raw` for detectors."""

    raw: Any
    content: str = ""
    reasoning_content: str = ""
    reasoning_duration_s: float = 0.0
    # Convenience accessors so callers don't have to dig through ``raw.choices``.
    finish_reason: str | None = None
    tool_calls: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Failure classification for litellm exceptions
# ---------------------------------------------------------------------------


def _classify_litellm_exception(exc: BaseException) -> Failure:
    """Map a litellm exception to a :class:`Failure`.

    Returns ``Failure(kind=transient_provider)`` for retryable provider errors,
    ``Failure(kind=capability_gap)`` for permanent failures (auth, bad request).
    """
    cls_name = exc.__class__.__name__
    # litellm auth/bad-request messages (and their __cause__/__context__ chains)
    # routinely embed the request URL with key query params. Redact before this
    # text is logged or stored in recorder metadata (an audit viewer may surface
    # it). Mirrors tools.py, which redacts exception text the same way.
    detail = redact_sensitive_text(str(exc))
    transient_classes = {
        "RateLimitError",
        "InternalServerError",
        "Timeout",
        "APIConnectionError",
        "APIError",
        "ServiceUnavailableError",
    }
    permanent_classes = {
        "AuthenticationError",
        "BadRequestError",
        "ContentPolicyViolationError",
    }
    if cls_name in transient_classes:
        return Failure(
            kind=FailureKind.transient_provider,
            explanation="The AI model had a temporary problem responding.",
            metadata={"provider_error": cls_name, "detail": detail},
        )
    if cls_name in permanent_classes:
        # Permanent failures route straight to handoff (no retry), and the
        # explanation becomes the user-facing Handoff rationale (and lands in the
        # host's audit ledger). Name the error CLASS (safe) and a generic hint, but
        # NOT str(exc): litellm auth/bad-request messages routinely embed the request
        # payload, model/org identifiers, and redacted-but-present auth fragments.
        # The raw provider message is logged server-side and kept in metadata for
        # recorders only — never surfaced verbatim to the user.
        _logger.warning("permanent LLM failure class=%s exc=%s", cls_name, detail)
        return Failure(
            kind=FailureKind.capability_gap,
            explanation=(
                f"The AI provider rejected the request ({cls_name}). This usually "
                "means the model is misconfigured — e.g. a missing or invalid API "
                "key. Check the server logs for the provider's full message."
            ),
            metadata={"provider_error": cls_name, "detail": detail},
        )
    # Default: treat unknown LLM errors as transient (retryable) but log so we
    # can audit the classification later. Retry budget will eventually kick in.
    _logger.warning("unclassified LLM exception class=%s exc=%s", cls_name, detail)
    return Failure(
        kind=FailureKind.transient_provider,
        explanation="The AI model had a temporary problem responding.",
        metadata={"provider_error": cls_name, "unclassified": True, "detail": detail},
    )


# ---------------------------------------------------------------------------
# Streaming with heartbeat
# ---------------------------------------------------------------------------


async def _next_chunk_with_heartbeat(
    aiter: AsyncIterator[Any],
    *,
    heartbeat_s: float,
) -> Any:
    """Fetch the next stream chunk, raising :exc:`FailureRaised` if no chunk
    arrives within ``heartbeat_s`` seconds.

    ``StopAsyncIteration`` propagates as usual to signal end-of-stream.
    """
    try:
        return await asyncio.wait_for(aiter.__anext__(), timeout=heartbeat_s)
    except TimeoutError as exc:
        raise FailureRaised(
            Failure(
                kind=FailureKind.transient_provider,
                explanation=(f"The AI model stopped responding partway through (no output for {heartbeat_s:.0f}s)."),
                metadata={"reason": "heartbeat_timeout", "heartbeat_s": heartbeat_s},
            )
        ) from exc


# ---------------------------------------------------------------------------
# The chokepoint
# ---------------------------------------------------------------------------


async def call_llm(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    model_config: dict[str, Any],
    request_timeout_s: float = 60.0,
    stream_heartbeat_s: float = 20.0,
    cancellation: CancellationRequest | None = None,
) -> AsyncIterator[LLMStreamSignal]:
    """Stream a litellm completion call, yielding :class:`LLMStreamSignal` events.

    Always ends with :class:`LLMComplete` carrying the assembled
    :class:`LLMResponse` — even when the stream completes without any deltas.

    :raises FailureRaised: on classified litellm errors / heartbeat timeout.
    :raises asyncio.CancelledError: when ``cancellation`` is set during the stream.
        Pass-through (not wrapped) so the loop can distinguish cancel from failure.
    """
    if cancellation is not None and cancellation.is_set():
        # Cancel beats suspend / failure / anything else.
        raise asyncio.CancelledError("cancellation requested before LLM call")

    reasoning_start_time = time.monotonic()

    # Inject Anthropic ``cache_control`` breakpoints right before the call.
    # No-op on every non-Anthropic route; on Claude routes it caches the
    # system prompt, tool catalog, and stable history prefix so the volatile
    # per-iteration context snapshot is the only uncached tail.
    messages, tools = apply_anthropic_cache_markers(model_config.get("model"), messages, tools)

    try:
        response_stream = await litellm.acompletion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            request_timeout=request_timeout_s,
            stream=True,
            **model_config,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise FailureRaised(_classify_litellm_exception(exc)) from exc

    chunks: list[Any] = []
    seen_tool_call_starts: set[str] = set()  # to dedupe LLMToolCallStarted

    aiter = response_stream.__aiter__()

    try:
        while True:
            if cancellation is not None and cancellation.is_set():
                raise asyncio.CancelledError("cancellation requested during stream")

            try:
                chunk = await _next_chunk_with_heartbeat(aiter, heartbeat_s=stream_heartbeat_s)
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                raise
            except FailureRaised:
                raise
            except Exception as exc:
                raise FailureRaised(_classify_litellm_exception(exc)) from exc

            chunks.append(chunk)

            # Yield streaming signals. The loop bumps ``state.last_event_time_s``
            # on every yielded signal so the phase-boundary stall detector sees
            # progress while the model is mid-stream.
            try:
                delta = chunk.choices[0].delta
            except (AttributeError, IndexError):
                continue

            if content := getattr(delta, "content", None):
                yield LLMTextDelta(content=content)

            if reasoning := getattr(delta, "reasoning_content", None):
                yield LLMReasoningDelta(content=reasoning)

            # Tool calls in the stream show up incrementally. The first chunk
            # with .name is the "start" signal; subsequent chunks accumulate args.
            for tc_chunk in getattr(delta, "tool_calls", None) or []:
                fn = getattr(tc_chunk, "function", None)
                if fn is None:
                    continue
                name = getattr(fn, "name", None)
                tc_id = getattr(tc_chunk, "id", None) or ""
                if name and tc_id and tc_id not in seen_tool_call_starts:
                    seen_tool_call_starts.add(tc_id)
                    yield LLMToolCallStarted(tool_name=name, tool_call_id=tc_id)
    finally:
        # Best-effort: drain the iterator if we exit early. The stream's
        # underlying connection may be released by litellm itself; we don't
        # depend on this cleanup, just avoid leaking pending coroutines.
        close = getattr(response_stream, "aclose", None) or getattr(response_stream, "close", None)
        if close is not None:
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    # Assemble the final response and surface it as LLMComplete.
    try:
        assembled = litellm.stream_chunk_builder(chunks, messages=messages)
    except Exception as exc:
        raise FailureRaised(_classify_litellm_exception(exc)) from exc

    response = _materialize_response(assembled, reasoning_start_time)
    yield LLMComplete(response=response)


def _materialize_response(raw: Any, reasoning_start_time: float) -> LLMResponse:
    """Build :class:`LLMResponse` from the assembled litellm response object."""
    content = ""
    reasoning = ""
    finish_reason = None
    tool_calls: list[Any] = []
    try:
        choice = raw.choices[0]
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", "") or ""
            reasoning = getattr(message, "reasoning_content", "") or ""
            tool_calls = list(getattr(message, "tool_calls", None) or [])
        finish_reason = getattr(choice, "finish_reason", None)
    except (AttributeError, IndexError):
        pass

    duration = time.monotonic() - reasoning_start_time if reasoning else 0.0
    return LLMResponse(
        raw=raw,
        content=content,
        reasoning_content=reasoning,
        reasoning_duration_s=duration,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


__all__ = [
    "LLMComplete",
    "LLMReasoningDelta",
    "LLMResponse",
    "LLMStreamSignal",
    "LLMTextDelta",
    "LLMToolCallStarted",
    "call_llm",
]
