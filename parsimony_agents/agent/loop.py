"""The ReAct agent loop, built on the failure-handling spine.

:func:`run_loop` drives one agent run end-to-end as an :class:`AgentEvent`
stream. Each iteration runs three phases: a pre-step (``pre_step`` detectors),
an LLM call (the :func:`~parsimony_agents.agent.llm.call_llm` chokepoint), and
tool execution. ``Agent.run`` is a thin shim that builds a
:class:`~parsimony_agents.agent.state.RunState`, constructs a
``WorkspaceRunHooks`` object, and delegates here.

Workspace-specific behaviour (context-snapshot rebuild, rich tool dispatch,
``LLMCallCompleted`` / ``StateSnapshot`` emission, notebook stamping, ref
minting) is supplied through the optional hook protocol described below.
Library callers (and tests) pass a plain object satisfying :class:`AgentLike`;
when an object provides none of the optional hooks, the loop falls back to
built-in defaults.

.. code-block:: python

    from parsimony_agents.agent.loop import run_loop
    async for event in run_loop(agent, state):
        ...

Architecture (BRIEF §2):

- **One LLM chokepoint** (:func:`~parsimony_agents.agent.llm.call_llm`).
- **One funnel** for every failure (:func:`~parsimony_agents.agent.failure.recovery.handle_failure`).
- **Pure renderer** for state → messages (:func:`~parsimony_agents.agent.renderer.render_for_llm`).
- **Three detector phases** (``pre_step`` / ``post_llm`` / ``post_tool``) — no scattered checks.
- **Explicit termination** via ``return_done`` / ``return_unable`` / ``ask_user`` system tools
  (no implicit "agent stopped talking" exit).

Failure flow:

1. ``pre_step`` runs. If a Failure fires, route through ``handle_failure`` and ``continue``.
2. LLM call. ``FailureRaised`` from the chokepoint → ``handle_failure`` → ``continue``.
3. ``post_llm`` runs. Failure → ``handle_failure`` → ``continue``.
4. Tool execution. ``SuspensionRequest`` / ``TerminationRequest`` exit the loop cleanly.
   ``post_tool`` Failure → ``handle_failure``.
5. Loop until ``state.done`` (set by recovery or by the explicit termination tools).

Hook protocol (all optional — discovered via :func:`getattr`):

``render_messages(state) -> list[dict]``
    Override message rendering. Default: ``render_for_llm(state, instructions=agent.instructions)``.

``on_iteration_start(state) -> AsyncGenerator[AgentEvent, None]``
    Runs after ``pre_step``, before render. Workspace uses it to rebuild the
    per-iteration context snapshot. Default: no-op.

``translate_llm_signal(state, signal) -> AsyncGenerator[AgentEvent, None]``
    Translate one :class:`~parsimony_agents.agent.llm.LLMStreamSignal` into
    transport events. Called for *every* signal including ``LLMComplete`` (the
    hook is expected to ignore it). Default: maps text/reasoning/tool-call-started
    deltas with loop-minted message ids.

``on_llm_complete(state, response, *, latency_ms) -> AsyncGenerator[AgentEvent, None]``
    Runs after usage accumulation, before ``post_llm``. Workspace emits the
    legacy ``LLMCallCompleted`` event here. Default: no-op.

``append_assistant_message(state, response) -> None``
    Append the assistant turn to ``state.messages``. Synchronous. Runs whether or
    not ``post_llm`` flagged a failure (recovery needs the turn in the transcript).
    Default: appends a litellm-shaped dict.

``on_assistant_turn(state, response) -> AsyncGenerator[AgentEvent, None]``
    Runs only after ``post_llm`` passes, before tool dispatch. Workspace emits the
    consolidated (non-delta) reasoning/text flush here — skipped on a post_llm
    failure so truncated/looping output is not surfaced. Default: no-op.

``dispatch_tools(state, response, *, cancellation) -> AsyncGenerator[AgentEvent, None]``
    Execute the tool calls. May raise :exc:`SuspensionRequest` /
    :exc:`TerminationRequest` / :exc:`asyncio.CancelledError` — the loop catches
    all three. Default: :func:`_execute_tool_calls`.

``on_iteration_end(state) -> AsyncGenerator[AgentEvent, None]``
    Runs at the end of a completed iteration (and just before the loop exits on
    suspension / termination / cancellation). Workspace emits ``StateSnapshot``.
    Default: no-op.

``on_run_complete(state) -> AsyncGenerator[AgentEvent, None]``
    Runs once after the loop exits, for any reason. Workspace emits its residual
    reasoning flush, final telemetry, and a closing ``StateSnapshot``. Default: no-op.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from datetime import UTC
from typing import Any, Protocol
from uuid import uuid4

from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.config import AgentGuardrails
from parsimony_agents.agent.events import (
    AgentEvent,
    Handoff,
    ReasoningDelta,
    RunCancelled,
    TextDelta,
    ToolEvent,
    UserInputRequested,
)
from parsimony_agents.agent.failure import (
    Failure,
    FailureKind,
    FailureRaised,
    RecoveryPolicy,
    SuspensionExpired,
    SuspensionRequest,
    SuspensionTokenMismatch,
    TerminationRequest,
    accumulate_usage,
    handle_failure,
    post_llm,
    post_tool,
    pre_step,
    record_tool_call,
    verify_suspension_token,
)
from parsimony_agents.agent.failure.suspension import compute_suspension_token
from parsimony_agents.agent.llm import (
    LLMComplete,
    LLMReasoningDelta,
    LLMResponse,
    LLMStreamSignal,
    LLMTextDelta,
    LLMToolCallStarted,
    call_llm,
)
from parsimony_agents.agent.renderer import render_for_llm
from parsimony_agents.agent.state import RunState, SuspensionRecord
from parsimony_agents.tools import Tool, ToolResult, Tools

_logger = logging.getLogger(__name__)


# The instruction surfaced to the LLM when it produces a text-only response with
# no tool call. A text-only response is not a valid end-of-run — the agent must
# terminate explicitly via a termination tool. ``handle_failure`` turns this into
# a pending_instruction on the first strike and escalates to handoff on the second.
_NO_PROGRESS_EXPLANATION = (
    "The agent replied with text only and took no action, "
    "so the run could not move forward."
)


# ---------------------------------------------------------------------------
# Agent protocol (minimal surface the loop needs)
# ---------------------------------------------------------------------------


class AgentLike(Protocol):
    """Minimum agent surface :func:`run_loop` reads.

    :class:`~parsimony_agents.agent.agent.Agent` does not satisfy this protocol
    directly — it passes a per-run ``WorkspaceRunHooks`` object that proxies these
    attributes and adds the optional hook methods. For tests and library callers,
    any object with these attributes works.

    Optional hook methods (see the module docstring) are discovered via
    :func:`getattr`; an object providing none of them gets the built-in defaults.
    """

    guardrails: AgentGuardrails
    policy: RecoveryPolicy
    suspension_secret: str
    model_config: dict[str, Any]
    instructions: str | None
    tools: Tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _emit_hook(
    agent: Any, name: str, /, *args: Any, **kwargs: Any
) -> AsyncGenerator[AgentEvent, None]:
    """Invoke an optional async-generator hook, yielding its events.

    No-op when the hook is absent — this is what lets library callers pass a
    plain :class:`AgentLike` and still drive the loop.
    """
    hook = getattr(agent, name, None)
    if hook is None:
        return
    async for event in hook(*args, **kwargs):
        yield event


async def _default_translate_signal(
    agent: AgentLike,
    signal: LLMStreamSignal,
    *,
    text_message_id: str,
    reasoning_message_id: str,
) -> AsyncGenerator[AgentEvent, None]:
    """Built-in LLM-signal → AgentEvent translation (no workspace UI affordances)."""
    if isinstance(signal, LLMTextDelta):
        yield TextDelta(content=signal.content, message_id=text_message_id)
    elif isinstance(signal, LLMReasoningDelta):
        yield ReasoningDelta(content=signal.content, message_id=reasoning_message_id)
    elif isinstance(signal, LLMToolCallStarted):
        tool = agent.tools.get(signal.tool_name) if agent.tools else None
        yield ToolEvent(
            tool_name=signal.tool_name,
            tool_call_id=signal.tool_call_id,
            tool_type=(tool.tool_type if tool else "utility"),
            completed=False,
            result=None,
        )
    # LLMComplete carries the assembled response — captured by run_loop, not translated.


def _assistant_message_from(response: LLMResponse) -> dict[str, Any]:
    """Build the assistant-role message dict to append to ``state.messages``.

    Mirrors litellm's message shape so the next renderer pass produces a coherent
    transcript. ``tool_calls`` (when present) carry their original JSON-serializable
    payload via :meth:`model_dump` where possible.
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content,
    }
    if response.reasoning_content:
        message["reasoning_content"] = response.reasoning_content
    if response.tool_calls:
        normalized = []
        for tc in response.tool_calls:
            tc_id = getattr(tc, "id", None)
            fn = getattr(tc, "function", None)
            fn_name = getattr(fn, "name", None) if fn else None
            fn_args = getattr(fn, "arguments", "{}") if fn else "{}"
            normalized.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": fn_args},
                }
            )
        message["tool_calls"] = normalized
    return message


def _tool_result_message(*, tool_call_id: str, tool_name: str, result_text: str) -> dict[str, Any]:
    """Build the role=tool message that the LLM sees after a tool runs."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": result_text,
    }


def _parse_tool_args(raw: str) -> dict[str, Any]:
    """Parse litellm-supplied tool arguments JSON; tolerate malformed input."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {"_raw_args": raw}
    if isinstance(parsed, dict):
        return parsed
    return {"_value": parsed}


def _build_suspension_record_from_tool(
    state: RunState,
    *,
    question: str,
    context: str | None,
    secret: str,
    originating_failure_kind: str | None = None,
) -> SuspensionRecord:
    """Snapshot ``state`` into a :class:`SuspensionRecord` for an ``ask_user`` exit."""
    return SuspensionRecord(
        run_id=state.run_id,
        session_id=state.session_id,
        model_tier=state.model_tier,
        suspension_token=compute_suspension_token(
            run_id=state.run_id,
            session_id=state.session_id,
            secret=secret,
        ),
        messages=list(state.messages),
        iteration_count=state.iteration,
        tool_call_history=list(state.tool_call_history),
        minted_refs=list(state.turn.minted_refs),
        minted_live_names=dict(state.turn.minted_live_names),
        started_at=state.started_at,
        elapsed_seconds=state.elapsed_seconds(),
        pending_question=question,
        pending_question_context=context,
        originating_failure_kind=originating_failure_kind,
        accumulated_reasoning=state.accumulated_reasoning,
        accumulated_reasoning_duration_s=state.accumulated_reasoning_duration_s,
        last_repeat_counts=dict(state.last_repeat_counts),
        cumulative_cost_usd=state.cumulative_cost_usd,
        cumulative_prompt_tokens=state.cumulative_prompt_tokens,
        cumulative_completion_tokens=state.cumulative_completion_tokens,
        lessons_learned=list(state.lessons_learned),
        failure_attempts=dict(state.failure_attempts),
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


async def run_loop(
    agent: AgentLike,
    state: RunState,
    *,
    cancellation: CancellationRequest | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    """Drive the agent loop end-to-end against the failure-handling spine.

    Yields :class:`AgentEvent` instances. Sets ``state.done = True`` when the run
    concludes (whether by success, failure escalation to handoff, or budget hit).
    The caller's outer ``async for`` exits naturally.

    :param agent: Object satisfying :class:`AgentLike` — provides guardrails,
        policy, model config, instructions, tools, suspension secret. May also
        provide the optional hooks described in the module docstring.
    :param state: Mutable :class:`RunState`. Cumulative counters, ``failure_attempts``,
        and ``lessons_learned`` are updated in place.
    :param cancellation: Optional cancellation hook. If set mid-stream, the LLM
        chokepoint raises :exc:`asyncio.CancelledError`; the loop yields
        :class:`RunCancelled` and exits.

    :raises asyncio.CancelledError: Only when no cancellation hook is registered
        and the surrounding task is cancelled externally. The cancellation-hook
        path always swallows the exception and yields :class:`RunCancelled` instead.
    """
    render = getattr(agent, "render_messages", None)
    translate = getattr(agent, "translate_llm_signal", None)
    dispatch = getattr(agent, "dispatch_tools", None)
    append = getattr(agent, "append_assistant_message", None)
    tool_choice = getattr(agent, "tool_choice", "auto")

    while not state.done:
        state.iteration += 1
        state.turn = state.turn.__class__()  # fresh TurnSubstate for this iteration

        # --- Cancellation pre-check ---
        if cancellation is not None and cancellation.is_set():
            _logger.info(
                "Run cancelled",
                extra={"phase": "pre_iteration", "reason": cancellation.reason,
                       "iteration": state.iteration},
            )
            yield RunCancelled(message="cancelled before iteration", reason=cancellation.reason)
            state.done = True
            break

        # --- Pre-step detectors ---
        if (failure := pre_step(state, agent.guardrails)) is not None:
            async for event in handle_failure(failure, agent=agent, state=state):
                yield event
            continue

        # --- Iteration-start hook (workspace: rebuild context snapshot) ---
        async for event in _emit_hook(agent, "on_iteration_start", state):
            yield event

        # --- Render messages ---
        messages = render(state) if render is not None else render_for_llm(state, instructions=agent.instructions)

        # --- LLM call ---
        response: LLMResponse | None = None
        text_message_id = str(uuid4())
        reasoning_message_id = str(uuid4())
        t0 = time.perf_counter()

        try:
            async for sig in call_llm(
                messages=messages,
                tools=agent.tools.to_llm() if agent.tools else None,
                tool_choice=tool_choice,
                model_config=agent.model_config,
                request_timeout_s=agent.guardrails.llm_timeout_s,
                stream_heartbeat_s=agent.guardrails.stream_heartbeat_s,
                cancellation=cancellation,
            ):
                state.last_event_time_s = time.monotonic()
                if isinstance(sig, LLMComplete):
                    response = sig.response
                if translate is not None:
                    async for event in translate(state, sig):
                        yield event
                else:
                    async for event in _default_translate_signal(
                        agent, sig,
                        text_message_id=text_message_id,
                        reasoning_message_id=reasoning_message_id,
                    ):
                        yield event
        except FailureRaised as exc:
            async for event in handle_failure(exc.failure, agent=agent, state=state):
                yield event
            continue
        except asyncio.CancelledError:
            _reason = cancellation.reason if cancellation is not None else "user_request"
            _logger.info(
                "Run cancelled",
                extra={"phase": "llm_stream", "reason": _reason, "iteration": state.iteration},
            )
            yield RunCancelled(
                message="Generation was cancelled before the assistant message completed.",
                reason=_reason,
            )
            state.done = True
            break

        # Cancellation may have fired between LLM stream end and now.
        if cancellation is not None and cancellation.is_set():
            _logger.info(
                "Run cancelled",
                extra={"phase": "post_llm_stream", "reason": cancellation.reason,
                       "iteration": state.iteration},
            )
            yield RunCancelled(
                message="Generation was cancelled before the assistant message completed.",
                reason=cancellation.reason,
            )
            state.done = True
            break

        if response is None:
            # Defensive: call_llm always yields LLMComplete last, but if not we treat
            # it as a transient_provider failure so the loop can decide whether to retry.
            async for event in handle_failure(
                Failure(
                    kind=FailureKind.transient_provider,
                    explanation="The connection to the AI model ended before a full response arrived.",
                ),
                agent=agent,
                state=state,
            ):
                yield event
            continue

        # --- Accumulate usage ---
        accumulate_usage(state, response.raw, model=agent.model_config.get("model"))

        # --- LLM-complete hook (workspace: emit LLMCallCompleted) ---
        latency_ms = int((time.perf_counter() - t0) * 1000)
        async for event in _emit_hook(
            agent, "on_llm_complete", state, response, latency_ms=latency_ms
        ):
            yield event

        # --- Post-LLM detectors ---
        post_llm_failure = post_llm(response.raw, state, agent.guardrails)

        # --- Append assistant message to the transcript ---
        # Runs in both branches: recovery renders state.messages next iteration,
        # so it must see what the LLM said even when post_llm flags a failure.
        if append is not None:
            append(state, response)
        else:
            state.messages.append(_assistant_message_from(response))

        # If post_llm flagged loop_detected / output_truncated, route through the funnel.
        if post_llm_failure is not None:
            async for event in handle_failure(post_llm_failure, agent=agent, state=state):
                yield event
            continue

        # --- Assistant-turn hook (workspace: flush consolidated reasoning/text) ---
        async for event in _emit_hook(agent, "on_assistant_turn", state, response):
            yield event

        # --- No tool calls → no_progress (text-only response) ---
        if not response.tool_calls:
            _logger.info(
                "No tool calls in response — routing through no_progress recovery",
                extra={"iteration": state.iteration},
            )
            async for event in handle_failure(
                Failure(kind=FailureKind.no_progress, explanation=_NO_PROGRESS_EXPLANATION),
                agent=agent,
                state=state,
            ):
                yield event
            continue

        # --- Execute tool calls ---
        try:
            if dispatch is not None:
                async for event in dispatch(state, response, cancellation=cancellation):
                    yield event
            else:
                async for event in _execute_tool_calls(
                    tool_calls=response.tool_calls,
                    state=state,
                    agent=agent,
                ):
                    yield event
                    state.last_event_time_s = time.monotonic()
        except SuspensionRequest as suspension:
            # Soft suspension: build record, yield UserInputRequested, exit cleanly.
            record = _build_suspension_record_from_tool(
                state,
                question=suspension.question,
                context=suspension.context,
                secret=agent.suspension_secret,
                originating_failure_kind=suspension.originating_failure_kind,
            )
            yield UserInputRequested(
                question=suspension.question,
                context=suspension.context,
                choices=suspension.choices,
                suspension_record=record,
                originating_failure_kind=suspension.originating_failure_kind,
            )
            state.done = True
            async for event in _emit_hook(agent, "on_iteration_end", state):
                yield event
            break
        except TerminationRequest as termination:
            yield Handoff(
                rationale=termination.rationale,
                blockers=termination.blockers,
                suggested_next_steps=[],
            )
            state.done = True
            async for event in _emit_hook(agent, "on_iteration_end", state):
                yield event
            break
        except asyncio.CancelledError:
            _reason = cancellation.reason if cancellation is not None else "user_request"
            _logger.info(
                "Run cancelled",
                extra={"phase": "tool_execution", "reason": _reason,
                       "iteration": state.iteration},
            )
            yield RunCancelled(
                message="The run was cancelled while tools were executing.",
                reason=_reason,
            )
            state.done = True
            async for event in _emit_hook(agent, "on_iteration_end", state):
                yield event
            break

        # --- Iteration-end hook (workspace: emit StateSnapshot) ---
        async for event in _emit_hook(agent, "on_iteration_end", state):
            yield event

    # --- Run-complete hook (workspace: residual flush + telemetry + StateSnapshot) ---
    async for event in _emit_hook(agent, "on_run_complete", state):
        yield event


# ---------------------------------------------------------------------------
# Tool execution (default dispatch — used when the agent provides no
# ``dispatch_tools`` hook, i.e. library callers and tests)
# ---------------------------------------------------------------------------


async def _execute_tool_calls(
    *,
    tool_calls: list[Any],
    state: RunState,
    agent: AgentLike,
) -> AsyncGenerator[AgentEvent, None]:
    """Execute the tool calls returned by the LLM.

    For each call:
    1. Record the signature in ``state.tool_call_history`` (loop detection).
    2. Look up the tool. Unknown tool → :class:`Failure(kind=tool_error)` via post_tool path.
    3. Invoke. ``SuspensionRequest`` / ``TerminationRequest`` propagate up to ``run_loop``.
    4. Run :func:`post_tool` on the result. Failure → ``handle_failure``.
    5. Append a role=tool message to ``state.messages`` and yield a completed :class:`ToolEvent`.
    """
    for tc in tool_calls:
        tool_call_id = getattr(tc, "id", str(uuid4()))
        fn = getattr(tc, "function", None)
        tool_name = getattr(fn, "name", None) if fn else None
        raw_args = getattr(fn, "arguments", "{}") if fn else "{}"
        args = _parse_tool_args(raw_args)

        if not tool_name or tool_name not in agent.tools:
            failure = Failure(
                kind=FailureKind.tool_error,
                explanation=f"Unknown tool: {tool_name!r}.",
                metadata={"tool_name": tool_name or "unknown"},
            )
            async for event in handle_failure(failure, agent=agent, state=state):
                yield event
            # Append a synthetic tool message so the next render sees the failure inline.
            state.messages.append(
                _tool_result_message(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name or "unknown",
                    result_text=f"Tool not found: {tool_name}",
                )
            )
            continue

        # Loop-detection bookkeeping happens before invocation so a repeat that
        # raises (e.g. ask_user keeps asking the same question) still trips the
        # counter cleanly.
        record_tool_call(state, tool_name, args)

        tool: Tool = agent.tools[tool_name]
        # Strip the _ui_message arg before invocation — purely a UI affordance,
        # not a tool parameter.
        invocation_args = {k: v for k, v in args.items() if k != "_ui_message"}

        try:
            result = await tool(**invocation_args)
        except (SuspensionRequest, TerminationRequest):
            # Bubble: caller (run_loop) catches and translates to UserInputRequested / Handoff.
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Any unhandled exception inside a tool surfaces as a structured Failure.
            failure = Failure(
                kind=FailureKind.tool_error,
                explanation=str(exc),
                metadata={"tool_name": tool_name, "exception_class": exc.__class__.__name__},
            )
            async for event in handle_failure(failure, agent=agent, state=state):
                yield event
            state.messages.append(
                _tool_result_message(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result_text=f"Tool raised: {exc}",
                )
            )
            continue

        if not isinstance(result, ToolResult):
            result = ToolResult.from_data(result)

        # post_tool detector: structured failures inside the result get routed.
        if (failure := post_tool(result, tool, state)) is not None:
            async for event in handle_failure(failure, agent=agent, state=state):
                yield event

        # Always append a tool-message + emit a completed ToolEvent so the transcript stays linear.
        result_text = (
            (result.failure.explanation if result.failure else None)
            or result.exception_message
            or (json.dumps(result.data, default=str) if result.data is not None else "")
        )
        state.messages.append(
            _tool_result_message(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result_text=result_text,
            )
        )
        yield ToolEvent(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_type=tool.tool_type,
            completed=True,
            result=result.data if result.ok else None,
        )

        # Explicit termination via return_done: set state.done = True.
        # (return_unable raises TerminationRequest from inside the function call; it
        # short-circuits to the outer except clause and never reaches here.)
        if tool.tool_type == "system" and tool_name == "return_done" and result.ok:
            state.done = True


async def resume_run(
    agent: AgentLike,
    suspension: SuspensionRecord,
    user_reply: str,
    *,
    files: Any | None = None,
    code_executor: Any | None = None,
    cancellation: CancellationRequest | None = None,
    max_suspension_age_s: float | None = 24 * 3600.0,
) -> AsyncGenerator[AgentEvent, None]:
    """Resume a suspended run with a user reply.

    Validates the suspension token, checks staleness, rebuilds :class:`RunState`
    from the record, appends the user's reply as the next message, then re-enters
    :func:`run_loop`. Runtime services (``files``, ``code_executor``, ``cancellation``)
    are re-injected via kwargs — they cannot be persisted across the suspend boundary.

    :raises SuspensionTokenMismatch: presented record's token fails HMAC verification.
    :raises SuspensionExpired: record is older than ``max_suspension_age_s`` (default 24h).
    :raises ValueError: ``user_reply`` is empty.

    Cancellation precedence: if ``cancellation`` is set before resume, the loop
    yields :class:`RunCancelled` immediately on its first cancellation check.
    """
    from datetime import datetime

    if not user_reply or not user_reply.strip():
        raise ValueError("resume_run requires a non-empty user_reply")

    if not verify_suspension_token(record=suspension, secret=agent.suspension_secret):
        raise SuspensionTokenMismatch(
            f"suspension token failed verification for run_id={suspension.run_id!r}"
        )

    if max_suspension_age_s is not None:
        age = (datetime.now(UTC) - suspension.suspended_at).total_seconds()
        if age > max_suspension_age_s:
            raise SuspensionExpired(
                f"suspension is {age:.0f}s old (max {max_suspension_age_s:.0f}s)"
            )

    state = RunState.from_suspension(
        suspension,
        files=files,
        code_executor=code_executor,
        cancellation=cancellation,
    )

    # Append the user's reply as the next user message. The renderer will surface it
    # as the most-recent input on the next iteration. No special marker — the LLM
    # treats it as a normal user message (BRIEF §4.2).
    state.messages.append({"role": "user", "content": user_reply.strip()})

    async for event in run_loop(agent, state, cancellation=cancellation):
        yield event


__all__ = ["AgentLike", "resume_run", "run_loop"]
