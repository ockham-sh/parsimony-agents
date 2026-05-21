"""Failure detectors and state-recording helpers.

Three detector functions, called by the loop:

- :func:`pre_step` — runs *before* each LLM call; budget exhaustion + phase-boundary stall.
- :func:`post_llm` — runs *after* each LLM response; ``finish_reason`` + loop detection.
- :func:`post_tool` — runs *after* each tool result; surfaces tool errors.

The three detectors are pure: they observe ``RunState`` and read fields off the
response/result, returning a :class:`Failure` or ``None`` without mutating anything.
The two state-recording helpers — :func:`record_tool_call` (appends loop-detection
signatures) and :func:`accumulate_usage` (accumulates cost / tokens) — *do* mutate
``RunState``; the loop calls them at well-defined points (``record_tool_call`` before
each tool invocation, ``accumulate_usage`` between ``post_llm`` and the next ``pre_step``).

Precedence at the same phase (per BRIEF §4.3): **hard-stops > quality issues > warnings**;
first-match-wins within each tier. Encoded in the function bodies, not via a separate scheduler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Protocol

from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.agent.failure.kinds import Failure, FailureKind
from parsimony_agents.agent.state import RunState

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duck-typed protocols. The detectors don't import concrete LLM response types
# (litellm shape, ToolResult shape) so they can be unit-tested with simple stubs.
# ---------------------------------------------------------------------------


class _MessageLike(Protocol):
    tool_calls: Any  # iterable or None


class _ChoiceLike(Protocol):
    message: _MessageLike
    finish_reason: str | None


class _UsageLike(Protocol):
    prompt_tokens: int
    completion_tokens: int


class _ResponseLike(Protocol):
    choices: list[_ChoiceLike]
    usage: _UsageLike | None


class _ToolCallLike(Protocol):
    name: str  # tool name (read directly off the call site)


# ---------------------------------------------------------------------------
# Loop signature
# ---------------------------------------------------------------------------


def loop_signature(tool_name: str, args: dict[str, Any]) -> str:
    """Stable signature for loop detection.

    Format: ``f"{tool_name}:{sha256(args_json)[:8]}"``. ``_ui_message`` is stripped
    so two calls that differ only in their human-readable prefix collapse into the
    same signature (the agent's "I'll try again with a different message" pattern).

    Keys are sorted for deterministic JSON; non-serializable values fall back to
    ``str()`` via ``default=str`` so the signature never crashes on exotic args.
    """
    args_clean = {k: v for k, v in args.items() if k != "_ui_message"}
    payload = json.dumps(args_clean, sort_keys=True, default=str)
    return f"{tool_name}:{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def record_tool_call(state: RunState, tool_name: str, args: dict[str, Any]) -> str:
    """Append the call's signature to ``state.tool_call_history`` and bump the per-signature counter.

    Returns the signature so callers (mostly the loop) can log it without re-hashing.
    """
    sig = loop_signature(tool_name, args)
    state.tool_call_history.append(sig)
    state.last_repeat_counts[sig] = state.last_repeat_counts.get(sig, 0) + 1
    return sig


# ---------------------------------------------------------------------------
# pre_step
# ---------------------------------------------------------------------------


def pre_step(state: RunState, guardrails: AgentGuardrails) -> Failure | None:
    """Pre-iteration checks. Returns the first failure under the precedence rule.

    Order (hard-stops > warnings):
    1. ``iteration_limit`` — iteration counter at/over ``max_iterations``.
    2. ``time_limit`` — elapsed wall-clock at/over ``max_execution_time_s``.
    3. ``no_progress`` — phase-boundary stall (more than ``stall_threshold_s``
       since the last yielded event).
    """
    if state.iteration >= guardrails.max_iterations:
        return Failure(
            kind=FailureKind.iteration_limit,
            explanation=(
                f"The run reached its limit of {guardrails.max_iterations} steps "
                "before the task was finished."
            ),
            metadata={"max_iterations": guardrails.max_iterations, "iteration": state.iteration},
        )

    elapsed = state.elapsed_seconds()
    if elapsed >= guardrails.max_execution_time_s:
        return Failure(
            kind=FailureKind.time_limit,
            explanation=(
                f"The run reached its {guardrails.max_execution_time_s:.0f}-second "
                "time limit before the task was finished."
            ),
            metadata={"max_execution_time_s": guardrails.max_execution_time_s, "elapsed_s": elapsed},
        )

    silence = time.monotonic() - state.last_event_time_s
    if silence > guardrails.stall_threshold_s:
        return Failure(
            kind=FailureKind.no_progress,
            explanation=f"The run stalled — no progress for {silence:.0f} seconds.",
            metadata={"silence_s": silence, "stall_threshold_s": guardrails.stall_threshold_s},
        )

    return None


# ---------------------------------------------------------------------------
# post_llm
# ---------------------------------------------------------------------------


def post_llm(
    response: _ResponseLike,
    state: RunState,
    guardrails: AgentGuardrails,
) -> Failure | None:
    """Post-LLM-call checks. Returns the first failure under the precedence rule.

    Order:
    1. ``output_truncated`` — ``finish_reason == "length"``.
    2. ``output_refused`` — ``finish_reason in {"content_filter", "refusal"}``.
    3. ``loop_detected`` — any tool_call signature in this response would push
       its per-signature counter to ``loop_hard_threshold`` once recorded.
       Soft-threshold breaches are logged but do not produce a Failure.

    The detector is non-mutating; it predicts whether the *next* call to
    :func:`record_tool_call` for each tool would trip the hard threshold,
    rather than recording first and reading back. This lets the loop choose
    whether to skip recording when a hard-fail is going to fire.
    """
    if not response.choices:
        return None

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        return Failure(
            kind=FailureKind.output_truncated,
            explanation="The AI model's response was cut off before it finished.",
            metadata={"finish_reason": "length"},
        )
    if finish_reason in ("content_filter", "refusal"):
        return Failure(
            kind=FailureKind.output_refused,
            explanation=(
                "The AI model declined to answer the request, "
                "most likely because of a safety filter."
            ),
            metadata={"finish_reason": finish_reason},
        )

    message = getattr(choice, "message", None)
    if message is None:
        return None
    tool_calls = getattr(message, "tool_calls", None) or []

    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if fn is None:
            continue
        name = getattr(fn, "name", None)
        if not name:
            continue
        args_str = getattr(fn, "arguments", None) or "{}"
        try:
            args = json.loads(args_str)
        except (ValueError, TypeError):
            args = {"_raw_args": str(args_str)}

        sig = loop_signature(name, args)
        # Project the would-be count: existing repeats + 1 (this call).
        projected = state.last_repeat_counts.get(sig, 0) + 1

        if projected >= guardrails.loop_hard_threshold:
            return Failure(
                kind=FailureKind.loop_detected,
                explanation=(
                    f"The agent repeated the same action ({name}) "
                    f"{projected} times without making progress."
                ),
                metadata={
                    "signature": sig,
                    "repeat_count": projected,
                    "tool_name": name,
                },
            )
        if projected >= guardrails.loop_soft_threshold:
            _logger.info(
                "loop_soft_threshold_hit signature=%s repeat_count=%d hard_threshold=%d",
                sig,
                projected,
                guardrails.loop_hard_threshold,
            )

    return None


# ---------------------------------------------------------------------------
# post_tool
# ---------------------------------------------------------------------------


def post_tool(result: Any, call: _ToolCallLike | None, state: RunState) -> Failure | None:
    """Post-tool-execution checks.

    A tool result can carry a failure in one of two ways. If ``result.failure``
    holds a structured :class:`Failure`, it is returned as-is. Otherwise, if
    ``result.exception_message`` is set, it is wrapped into a ``tool_error``
    :class:`Failure`. A clean result yields ``None``.
    """
    # Structured form — tool produced a Failure directly.
    new_failure = getattr(result, "failure", None)
    if isinstance(new_failure, Failure):
        return new_failure

    # Message-only form — wrap the exception_message into a tool_error Failure.
    exc_message = getattr(result, "exception_message", None)
    if exc_message:
        tool_name = getattr(call, "name", "unknown") if call is not None else "unknown"
        return Failure(
            kind=FailureKind.tool_error,
            explanation=str(exc_message),
            metadata={"tool_name": tool_name},
        )

    return None


# ---------------------------------------------------------------------------
# Helper: cost / token accumulator
# ---------------------------------------------------------------------------


def accumulate_usage(
    state: RunState,
    response: _ResponseLike,
    *,
    model: str | None = None,
) -> tuple[int, int, float]:
    """Accumulate this call's prompt/completion tokens + cost onto ``state``.

    Returns ``(prompt_tokens, completion_tokens, cost_usd)`` for the call so the
    caller can log per-call telemetry. Best-effort: if ``litellm.completion_cost``
    raises, cost is recorded as 0 (the call still completes; the proxy session
    will reconcile from authoritative response headers when it lands).

    Merge point — LLM proxy: the canonical cost source is the proxy's response
    header (e.g. ``X-Estimated-Cost-USD``). Once the proxy lands, callers should
    prefer the header value over ``litellm.completion_cost`` for ``cost_usd``.
    """
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    state.cumulative_prompt_tokens += prompt
    state.cumulative_completion_tokens += completion

    cost = 0.0
    if model is not None and usage is not None:
        try:
            import litellm  # noqa: PLC0415 — local import keeps detectors importable without litellm

            cost = float(
                litellm.completion_cost(
                    completion_response=response,
                    model=model,
                )
                or 0.0
            )
        except Exception as exc:
            _logger.debug("completion_cost failed; recording 0.0: %s", exc)
            cost = 0.0
    state.cumulative_cost_usd += cost
    return (prompt, completion, cost)


__all__ = [
    "accumulate_usage",
    "loop_signature",
    "post_llm",
    "post_tool",
    "pre_step",
    "record_tool_call",
]
