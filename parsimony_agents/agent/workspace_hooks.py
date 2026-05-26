"""Per-run hook object that adapts :class:`~parsimony_agents.agent.agent.Agent`
to the :class:`~parsimony_agents.agent.loop.AgentLike` protocol.

``Agent.run`` constructs one ``WorkspaceRunHooks`` instance per run and delegates
to :func:`~parsimony_agents.agent.loop.run_loop`. ``run_loop`` discovers the
optional hook methods on this object via :func:`getattr` and calls them at the
appropriate points in the loop.

The hook methods carry the workspace-specific behaviour the generic loop does
not provide: rebuilding the per-iteration context snapshot, rich tool dispatch
(concurrent batches, notebook stamping, ref minting), and emitting the
``LLMCallCompleted`` / ``StateSnapshot`` events the workspace transport expects.
Termination tools (``ask_user`` / ``return_unable``) are handled by re-raising
the captured ``SuspensionRequest`` / ``TerminationRequest`` so ``run_loop``
builds the ``UserInputRequested`` / ``Handoff`` event.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from opentelemetry import trace

from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.events import (
    AgentError,
    RunCancelled,
    StateSnapshot,
    ToolEvent,
)
from parsimony_agents.agent.events import (
    LLMCallCompleted as LLMCallCompletedEvent,
)
from parsimony_agents.agent.events import (
    ReasoningDelta as ReasoningDeltaEvent,
)
from parsimony_agents.agent.events import (
    TextDelta as TextDeltaEvent,
)
from parsimony_agents.agent.events import (
    ToolResultObserved as ToolResultObservedEvent,
)
from parsimony_agents.agent.helpers import TurnState
from parsimony_agents.agent.models import AgentContext, AgentMessage
from parsimony_agents.agent.outputs import (
    SystemToolMessage,
    UtilityToolOutput,
)
from parsimony_agents.agent.tracing import trace_tool_execution
from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.identity import (
    ArtifactRef,
    LiveNameCollisionError,
    notebook_content_sha,
    notebook_logical_id,
)
from parsimony_agents.messages import Message, Text
from parsimony_agents.notebook import ScriptPreview, stamp_fetch_log_to_script
from parsimony_agents.tools import ToolMethod, Tools

logger = logging.getLogger("parsimony_agents")
error_logger = logging.getLogger("parsimony_agents.errors")

# Tool message for cooperative cancellation; keeps one tool output per tool call id.
CANCELLED_TOOL_TEXT = "Cancelled by user before the tool completed."

# Framework control-flow tools. These do NOT receive the workspace
# AgentContext (which the workspace tools all take as ``context: AgentContext``);
# their LLM-visible ``context`` parameter means a clarification-hint string.
_TERMINATION_TOOL_NAMES = frozenset({"return_done", "return_unable", "ask_user"})

# Read-only system tools that never touch the CodeExecutor; safe to run concurrently.
# All other tool names require sequential execution in one batch to avoid re-entrant
# kernel or workspace races (see ``_tool_batch_allows_concurrent``).
_TOOL_NAMES_SAFE_CONCURRENT: frozenset[str] = frozenset(
    {
        "list_files",
        "read_artifact",
        "read_data",
        "read_file",
    }
)


def _tool_batch_allows_concurrent(
    tool_names: list[str],
    tools: Tools,
) -> bool:
    if not tool_names:
        return True
    for name in tool_names:
        if name not in tools:
            return False
        if tools[name].tool_type == "return":
            return False
    return all(n in _TOOL_NAMES_SAFE_CONCURRENT for n in tool_names)


def _append_tool_msg_and_observe(
    ctx: AgentContext,
    *,
    content: Any,
    name: str,
    tool_call_id: str,
) -> ToolResultObservedEvent:
    """Append a role='tool' AgentMessage to ``ctx.messages`` and return the
    ``ToolResultObserved`` event carrying the exact content the LLM will
    read on its next iteration (the output of
    ``AgentMessage.to_llm()["content"]``).

    When every block is plain text — the common case — the blocks are
    concatenated into a single string for storage. Multi-modal results
    (e.g. image blocks) are kept as the original list-of-blocks form so
    structure isn't lost.

    Caller is expected to ``yield`` the returned event into the agent's
    event stream so the eval recorder can attach it to the turn record.
    """
    msg = AgentMessage(role="tool", content=content, name=name, tool_call_id=tool_call_id)
    ctx.messages.append(msg)
    blocks = msg.to_llm().get("content", [])
    if blocks and all(isinstance(b, dict) and b.get("type") == "text" for b in blocks):
        llm_content: list[dict[str, Any]] | str = "".join(b.get("text", "") for b in blocks)
    else:
        llm_content = blocks
    return ToolResultObservedEvent(
        tool_call_id=tool_call_id,
        tool_name=name,
        llm_content=llm_content,
    )


class WorkspaceRunHooks:
    """Per-run adapter satisfying :class:`AgentLike` plus the optional hooks.

    Constructed once per :meth:`Agent.run` call. Proxies the protocol
    attributes from the :class:`Agent` and holds the per-run mutable state that
    used to live as closure variables inside the monolithic loop body.
    """

    def __init__(
        self,
        *,
        agent: Any,
        ctx: AgentContext,
        turn_state: TurnState,
        cancellation: CancellationRequest | None,
        tool_choice: str,
        start_time: float,
        agent_span: Any,
    ) -> None:
        self.agent = agent
        self.ctx = ctx
        self.turn_state = turn_state
        self.cancellation = cancellation
        self.tool_choice = tool_choice
        self.start_time = start_time
        self.agent_span = agent_span

        # --- AgentLike protocol surface (proxied from the Agent) ----------
        self.guardrails = agent.guardrails
        self.policy = agent.policy
        self.suspension_secret = agent.suspension_secret
        self.model_config = agent.model_config
        self.instructions = agent.instructions
        self.tools = agent.system_tools

        # --- Cross-iteration mutable bits (were closure vars in Agent.run) -
        self._reasoning_message_id: str = str(uuid4())
        self._text_message_id: str = str(uuid4())
        self._accumulated_reasoning: str = ""
        self._accumulated_duration: float = 0.0
        self._reasoning_start_time: float = 0.0
        self._last_tool_internal_error: str | None = None

    def _append_message(self, state: Any, msg: Any) -> None:
        """Append ``msg`` to both the workspace ``ctx`` and the spine ``state``."""
        self.ctx.messages.append(msg)
        state.messages.append(msg)

    # ------------------------------------------------------------------
    # Hook 2: on_iteration_start
    # ------------------------------------------------------------------
    async def on_iteration_start(self, state: Any) -> AsyncIterator[Any]:
        self._text_message_id = str(uuid4())
        self._reasoning_start_time = time.time()

        # --- Rebuild context snapshot, mirror into state.messages ------
        # No ``connectors=``: the catalog is rendered separately into the
        # cache-stable prefix (see ``render_messages`` / ``__init__``); the
        # per-iteration snapshot carries only volatile turn state.
        iter_snapshot = await self.ctx.to_snapshot(
            minted_refs=self.turn_state.minted_refs,
            minted_live_names=self.turn_state.minted_live_names,
        )
        self.ctx.messages = [
            m for m in self.ctx.messages if m.metadata.get("context_snapshot", False) is False
        ]
        self.ctx.messages.append(
            Message(
                role="user",
                content=iter_snapshot,
                metadata={"context_snapshot": True},
            )
        )
        state.messages = list(self.ctx.messages)
        return
        if False:  # pragma: no cover - makes this an async generator
            yield

    # ------------------------------------------------------------------
    # Hook 1: render_messages
    # ------------------------------------------------------------------
    def render_messages(self, state: Any) -> list[dict]:
        # Render-mode selection (recent tool observations at "default", older
        # ones "minimal") lives in ``render_for_llm`` / ``infer_message_mode``.
        # The system prompt is already ``state.messages[0]`` and the connector
        # catalog is its own stable prefix message (see ``_inject_connector_catalog``
        # in agent.py); ``instructions=None`` avoids a duplicate system layer.
        from parsimony_agents.agent.renderer import render_for_llm

        return render_for_llm(state, instructions=None)

    # ------------------------------------------------------------------
    # Hook 3: translate_llm_signal
    # ------------------------------------------------------------------
    async def translate_llm_signal(self, state: Any, sig: Any) -> AsyncIterator[Any]:
        from parsimony_agents.agent.llm import (
            LLMReasoningDelta,
            LLMTextDelta,
            LLMToolCallStarted,
        )

        if isinstance(sig, LLMTextDelta):
            yield TextDeltaEvent(
                content=sig.content,
                message_id=self._text_message_id,
                delta=True,
            )
        elif isinstance(sig, LLMReasoningDelta):
            yield ReasoningDeltaEvent(
                content=sig.content,
                message_id=self._reasoning_message_id,
                delta=True,
            )
        elif isinstance(sig, LLMToolCallStarted) and sig.tool_name in self.tools:
            t = self.tools[sig.tool_name]
            if t.tool_type == "system" and t.ui_message is not None:
                yield ToolEvent(
                    tool_name=sig.tool_name,
                    tool_call_id=sig.tool_call_id,
                    tool_type="system",
                    completed=False,
                    result=SystemToolMessage(message=t.ui_message),
                    ui_message=t.ui_message,
                )
        # LLMComplete carries the assembled response — captured by run_loop.

    # ------------------------------------------------------------------
    # Hook 4: on_llm_complete
    # ------------------------------------------------------------------
    async def on_llm_complete(
        self, state: Any, response: Any, *, latency_ms: int
    ) -> AsyncIterator[Any]:
        # --- Emit LLMCallCompleted (legacy event, reconstructed) --------
        _assembled_message = (
            response.raw.choices[0].message
            if response.raw and getattr(response.raw, "choices", None)
            else None
        )
        _usage_obj = getattr(response.raw, "usage", None) if response.raw else None
        _usage_dict: dict[str, Any] | None = None
        if _usage_obj is not None and hasattr(_usage_obj, "model_dump"):
            try:
                _usage_dict = _usage_obj.model_dump()
            except Exception:
                _usage_dict = None

        # Per-call USD cost → usage["cost_usd"] for eval recorders. Spine-level
        # ``accumulate_usage`` already maintains the *cumulative* cost on
        # ``state``; this is the orthogonal *per-call* figure on the event.
        # Fallback chain: litellm hidden ``response_cost`` → ``completion_cost``
        # → an inline ``cost`` already present on the usage dict. Best-effort —
        # any failure simply omits the key.
        _cost_usd: float | None = None
        try:
            _hidden = getattr(response.raw, "_hidden_params", None) or {}
            _cost_usd = _hidden.get("response_cost")
        except Exception:
            _cost_usd = None
        if _cost_usd is None and response.raw is not None:
            try:
                import litellm  # noqa: PLC0415

                _cost_usd = litellm.completion_cost(completion_response=response.raw) or None
            except Exception:
                _cost_usd = None
        if _cost_usd is None and _usage_dict and "cost" in _usage_dict:
            _cost_usd = _usage_dict["cost"]
        if _cost_usd is not None and _usage_dict is not None:
            _usage_dict["cost_usd"] = _cost_usd

        _tool_calls_payload: list[dict[str, Any]] = []
        for _tc in getattr(_assembled_message, "tool_calls", None) or []:
            _raw_args = _tc.function.arguments or "{}"
            try:
                _args = json.loads(_raw_args)
            except json.JSONDecodeError as _exc:
                _args = {"_raw": _raw_args, "_decode_error": str(_exc)}
            _tool_calls_payload.append(
                {"id": _tc.id, "name": _tc.function.name, "args": _args}
            )

        yield LLMCallCompletedEvent(
            iteration=state.iteration,
            response_text=(
                (getattr(_assembled_message, "content", None) or "")
                if _assembled_message
                else ""
            ),
            reasoning_text=(
                getattr(_assembled_message, "reasoning_content", None)
                if _assembled_message
                else None
            ),
            tool_calls=_tool_calls_payload,
            usage=_usage_dict,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Hook 5: append_assistant_message
    # ------------------------------------------------------------------
    def append_assistant_message(self, state: Any, response: Any) -> None:
        # --- Append the assistant response to ctx + state ---------------
        response_message = Message.from_litellm(response.raw.choices[0].message)
        response_message = AgentMessage(**response_message.model_dump())
        _new_tool_calls = list(response_message.tool_calls or [])
        response_message.tool_calls = _new_tool_calls if _new_tool_calls else None

        if _new_tool_calls:
            logger.info(
                "Tool calls",
                extra={
                    "iteration": state.iteration,
                    "tool_names": [tc.function.name for tc in _new_tool_calls],
                },
            )

        self._append_message(state, response_message)

    # ------------------------------------------------------------------
    # Hook 6: on_assistant_turn
    # ------------------------------------------------------------------
    async def on_assistant_turn(self, state: Any, response: Any) -> AsyncIterator[Any]:
        # --- Reasoning + text deltas (final flushes) --------------------
        if turn_reasoning := response.reasoning_content:
            self._accumulated_reasoning += turn_reasoning
            self._accumulated_duration += time.time() - self._reasoning_start_time

        is_silent = not response.content and all(
            getattr(self.tools.get(tc.function.name), "tool_type", None) == "system"
            and getattr(self.tools.get(tc.function.name), "ui_message", None) is None
            for tc in (response.tool_calls or [])
        )

        if self._accumulated_reasoning and not is_silent:
            yield ReasoningDeltaEvent(
                content=self._accumulated_reasoning,
                message_id=self._reasoning_message_id,
                title=f"Thought for {self._accumulated_duration:.1f} seconds",
                delta=False,
            )
            self._accumulated_reasoning = ""
            self._accumulated_duration = 0.0
            self._reasoning_message_id = str(uuid4())

        if text_message := response.content:
            yield TextDeltaEvent(
                content=text_message,
                message_id=self._text_message_id,
                delta=False,
            )

    # ------------------------------------------------------------------
    # Hook 7: dispatch_tools
    # ------------------------------------------------------------------
    async def dispatch_tools(
        self, state: Any, response: Any, *, cancellation: CancellationRequest | None
    ) -> AsyncIterator[Any]:
        from datetime import datetime as _dt  # noqa: F401

        from parsimony_agents.agent.failure.suspension import (
            SuspensionRequest as _SuspensionRequest,
        )
        from parsimony_agents.agent.failure.termination import (
            TerminationRequest as _TerminationRequest,
        )

        ctx = self.ctx
        tools = self.tools
        turn_state = self.turn_state
        tool_calls = response.tool_calls

        # ── Stage 1: validate, yield loading UI chunks, build coroutines
        def _build_coroutine(
            tool: ToolMethod,
            args_with_ctx: dict,
            name: str,
            t_type: str,
            raw_args: dict,
            timeout_s: float,
        ):
            @trace_tool_execution(name, t_type, logger, error_logger, timeout_s)
            async def _execute():
                return await tool(**args_with_ctx)

            return _execute(tool_args=raw_args)

        tool_executions: list[tuple] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            if tool_name not in tools:
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=f"Unknown tool: {tool_name}",
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
                continue

            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=(
                        f"Tool {tool_name} received malformed JSON arguments and "
                        f"could not be executed: {exc}. Re-issue the call with "
                        "valid JSON arguments."
                    ),
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
                continue
            llm_ui_message = tool_args.pop("_ui_message", None)

            if tool_name == "dry_execute_code" and not (
                isinstance(llm_ui_message, str) and llm_ui_message.strip()
            ):
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=(
                        "dry_execute_code requires a non-empty _ui_message: one short, "
                        "plain-language, past-tense line describing what this run does "
                        "for the user (e.g. 'Previewed a rolling mean')."
                    ),
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
                continue

            # Spine-native loop detection: record_tool_call appends to
            # state.tool_call_history + bumps last_repeat_counts. Hard-thresh
            # repeats are already routed via post_llm above (handle_failure).
            # Here we only retain the legacy soft-warn behaviour.
            from parsimony_agents.agent.failure import record_tool_call

            sig_str = record_tool_call(state, tool_name, tool_args)
            repeat_count = state.last_repeat_counts[sig_str] - 1  # prior count

            tool_timeout_s = self.agent._tool_timeout_seconds(tool_name, tool_args)

            if repeat_count >= self.guardrails.loop_soft_threshold:
                logger.warning(
                    f"Loop suspected: {tool_name} called {repeat_count + 1} times with same args",
                    extra={"tool_name": tool_name, "repeat_count": repeat_count + 1},
                )

            tool_type = tools[tool_name].tool_type

            if tool_type == "code":
                notebook_path = self.agent._resolve_code_tool_path(tool_args)
                if notebook_path is None:
                    obs = _append_tool_msg_and_observe(
                        ctx,
                        content=(
                            "return_notebook and edit_notebook require a non-empty 'path'. "
                            "The path is the notebook's identity address "
                            "(e.g. 'notebooks/inflation_analysis.py' → slug 'inflation_analysis'); "
                            "reuse to add a revision, or pick a fresh path under 'notebooks/' to create one."
                        ),
                        name=tool_name,
                        tool_call_id=tool_call.id,
                    )
                    state.messages = list(ctx.messages)
                    yield obs
                    continue

            loading_label = tools[tool_name].ui_message
            if tool_type == "code":
                if tool_name == "return_notebook" and tool_args.get("execute") is True:
                    loading_label = "Writing and running notebook"
                elif tool_name == "edit_notebook" and tool_args.get("execute") is True:
                    loading_label = "Editing and running notebook"
                if tool_name in ("return_notebook", "edit_notebook"):
                    preview = ScriptPreview(
                        path=notebook_path, code=tool_args.get("code", "") or ""
                    )
                else:
                    preview = ScriptPreview(path=notebook_path, code="")
                preview.ui_message = loading_label
                yield ToolEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    tool_type="code",
                    completed=False,
                    result=preview,
                    ui_message=loading_label,
                )
            elif tool_type == "utility":
                yield ToolEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    tool_type="utility",
                    completed=False,
                    result=UtilityToolOutput(
                        ui_message=loading_label or f"Executing {tool_name}",
                        metadata=self.agent._utility_tool_metadata(
                            tool_name=tool_name,
                            tool_description=tools[tool_name].ui_description
                            or tools[tool_name].description,
                            tool_args=tool_args,
                        ),
                    ),
                )

            # Termination tools (ask_user / return_done /
            # return_unable) are framework control-flow tools and use the
            # LLM-visible ``context`` parameter for a clarification-hint
            # string. They must NOT receive ``context=AgentContext``, which
            # the workspace tools (return_notebook, write_file, etc.) all
            # take as their workspace handle. Without this guard the
            # workspace injection clobbers ``ask_user``'s ``context`` arg
            # and ``context.strip()`` fails.
            invocation_args = dict(tool_args) if tool_name in _TERMINATION_TOOL_NAMES else {**tool_args, "context": ctx}

            tool_executions.append(
                (
                    tool_call,
                    sig_str,
                    repeat_count,
                    _build_coroutine(
                        tools[tool_name],
                        invocation_args,
                        tool_name,
                        tool_type,
                        tool_args,
                        tool_timeout_s,
                    ),
                )
            )

        if tool_executions and cancellation and cancellation.is_set():
            for tool_call, _sig_str, _repeat_count, _ in tool_executions:
                tool_name = tool_call.function.name
                if tool_name not in tools:
                    continue
                tool_args = json.loads(tool_call.function.arguments)
                tool_args.pop("_ui_message", None)
                for tev in self.agent._emit_cancelled_tool_events(
                    tools, tool_name, tool_args, tool_call
                ):
                    yield tev
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=CANCELLED_TOOL_TEXT,
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
            logger.info(
                "Run cancelled",
                extra={"phase": "pre_tool_batch", "reason": cancellation.reason,
                       "iteration": state.iteration},
            )
            yield RunCancelled(
                message="The run was cancelled before the remaining tools could finish.",
                reason=cancellation.reason,
            )
            state.done = True
            return

        # ── Stage 2: execute coroutines
        batch_names = [t.function.name for t, _, _, _ in tool_executions]
        concurrent_batch = _tool_batch_allows_concurrent(batch_names, tools)
        raw_results = await self.agent._run_tool_coros_with_cancellation(
            tool_executions, cancellation, concurrent_batch
        )

        # ── Stage 3: flush results sequentially into shared state
        # Track whether any tool in this batch issued an explicit termination
        # request so the surrounding loop exits after the batch finishes.
        pending_termination: tuple[str, Any] | None = None

        for (tool_call, _sig_str, repeat_count, _), raw_result in zip(
            tool_executions, raw_results, strict=True
        ):
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            llm_ui_message = tool_args.pop("_ui_message", None)

            if isinstance(raw_result, asyncio.CancelledError):
                for tev in self.agent._emit_cancelled_tool_events(
                    tools, tool_name, tool_args, tool_call
                ):
                    yield tev
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=CANCELLED_TOOL_TEXT,
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
                continue

            # ``ask_user`` / ``return_unable`` raise control-flow
            # exceptions inside the tool function; the cancellation-aware
            # gather captured them as result values. Translate them to the
            # corresponding spine events and end the run after the batch is
            # drained.
            if isinstance(raw_result, _SuspensionRequest):
                pending_termination = ("suspend", raw_result)
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=raw_result.question,
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
                continue

            if isinstance(raw_result, _TerminationRequest):
                pending_termination = ("handoff", raw_result)
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=raw_result.rationale,
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
                continue

            if isinstance(raw_result, Exception):
                obs = _append_tool_msg_and_observe(
                    ctx,
                    content=str(raw_result),
                    name=tool_name,
                    tool_call_id=tool_call.id,
                )
                state.messages = list(ctx.messages)
                yield obs
                continue

            tool_result = raw_result

            if not tool_result.success:
                self._last_tool_internal_error = (
                    f"Tool {tool_name} got an internal error. Trace ID: "
                    f"{trace.get_current_span().get_span_context().trace_id}"
                )

            tool_call_output: Any = None

            match tools[tool_name].tool_type:
                case "utility":
                    ui_message = tools[tool_name].ui_message or f"Executing {tool_name}"
                    ui_message_completed = (
                        llm_ui_message or tools[tool_name].ui_message_completed
                    )

                    if tool_result.data:
                        tool_call_output = tool_result.data
                        if ui_message_completed and not tool_call_output.ui_message_completed:
                            tool_call_output.ui_message_completed = ui_message_completed
                        yield ToolEvent(
                            tool_name=tool_name,
                            tool_call_id=tool_call.id,
                            tool_type="utility",
                            completed=True,
                            result=tool_call_output,
                            ui_message_completed=ui_message_completed,
                        )
                    else:
                        tool_call_output = tool_result.exception_message
                        output = UtilityToolOutput(
                            ui_message=ui_message,
                            ui_message_completed=ui_message_completed,
                            metadata=self.agent._utility_tool_metadata(
                                tool_name=tool_name,
                                tool_description=tools[tool_name].ui_description
                                or tools[tool_name].description,
                                tool_args=tool_args,
                            ),
                            content=Text(content=f"Error: {tool_result.exception_message}"),
                        )
                        yield ToolEvent(
                            tool_name=tool_name,
                            tool_call_id=tool_call.id,
                            tool_type="utility",
                            completed=True,
                            result=output,
                            ui_message_completed=ui_message_completed,
                        )

                case "return":
                    return_loading = tools[tool_name].ui_message
                    if tool_result.data:
                        tool_call_output = tool_result.data
                        return_types = (Dataset, Chart, Report)
                        if isinstance(tool_call_output, return_types):
                            yield ToolEvent(
                                tool_name=tool_name,
                                tool_call_id=tool_call.id,
                                tool_type="return",
                                completed=True,
                                result=tool_call_output,
                                ui_message=return_loading,
                                ui_message_completed=llm_ui_message,
                            )
                            if tool_call_output.logical_id and tool_call_output.content_sha:
                                turn_state.minted_refs.append(
                                    ArtifactRef(
                                        kind=tool_call_output.type,
                                        logical_id=tool_call_output.logical_id,
                                        content_sha=tool_call_output.content_sha,
                                    )
                                )
                                ln = getattr(tool_call_output, "live_name", None)
                                if ln:
                                    turn_state.minted_live_names[
                                        f"{tool_call_output.type}:{tool_call_output.logical_id}"
                                    ] = ln
                    else:
                        tool_call_output = tool_result.exception_message

                case "code":
                    tool_call_output = (
                        tool_result.data if tool_result.data else tool_result.exception_message
                    )
                    if tool_result.success:
                        notebook_path = self.agent._resolve_code_tool_path(tool_args)
                        if notebook_path is None:
                            raise ValueError("code tools require a non-empty 'path'.")
                        ran_kernel = (
                            tool_name in ("return_notebook", "edit_notebook")
                            and tool_args.get("execute") is True
                        )
                        script = await self.agent._notebook_script_after_tool(
                            tool_name=tool_name,
                            notebook_path=notebook_path,
                            tool_args=tool_args,
                            context=ctx,
                        )
                        if ran_kernel:
                            ko = tool_result.data
                            if not isinstance(ko, KernelOutput):
                                raise TypeError(
                                    f"{tool_name} with kernel run did not return KernelOutput"
                                )
                            script.output = ko
                            script.data_objects = stamp_fetch_log_to_script(ko)
                        notebook = script
                        preview = script.to_preview()
                        if tool_name == "return_notebook":
                            preview.ui_message = (llm_ui_message or "").strip() or None
                        else:
                            preview.ui_message = None
                        yield ToolEvent(
                            tool_name=tool_name,
                            tool_call_id=tool_call.id,
                            tool_type="code",
                            completed=True,
                            result={"notebook": notebook, "preview": preview},
                            ui_message_completed=llm_ui_message,
                            also_executed=ran_kernel,
                        )
                        if tool_name in ("return_notebook", "edit_notebook"):
                            try:
                                nb_ref = await self.agent._notebook_ref_for(
                                    script.code, notebook_path, ctx
                                )
                            except LiveNameCollisionError as exc:
                                nb_ref = ArtifactRef(
                                    kind="notebook",
                                    logical_id=exc.existing_logical_id,
                                    content_sha=notebook_content_sha(script.code),
                                )
                            turn_state.minted_refs.append(nb_ref)
                            try:
                                nb_live_name = notebook_logical_id(notebook_path)
                            except ValueError:
                                nb_live_name = None
                            if nb_live_name:
                                turn_state.minted_live_names[
                                    f"notebook:{nb_ref.logical_id}"
                                ] = nb_live_name

                case "system":
                    tool_call_output = tool_result.data
                    system_ui = (
                        llm_ui_message
                        or tools[tool_name].ui_message_completed
                        or tools[tool_name].ui_message
                    )
                    if system_ui is not None:
                        yield ToolEvent(
                            tool_name=tool_name,
                            tool_call_id=tool_call.id,
                            tool_type="system",
                            completed=True,
                            result=SystemToolMessage(message=system_ui),
                            ui_message_completed=llm_ui_message,
                        )
                    # Explicit-success termination.
                    # ``return_done`` returns a SystemToolOutput; the loop
                    # ends after this batch's StateSnapshot is yielded.
                    if tool_name == "return_done" and tool_result.success:
                        state.done = True

                case _:
                    raise ValueError(f"Invalid tool type: {tools[tool_name].tool_type}")

            if repeat_count >= self.guardrails.loop_soft_threshold:
                tool_call_output = [
                    {
                        "type": "text",
                        "text": (
                            "Warning: You have called the same tool with the same "
                            "arguments multiple times. Consider trying a different "
                            "approach.\n\n"
                        ),
                    },
                    *Message._normalize_content(tool_call_output),
                ]

            obs = _append_tool_msg_and_observe(
                ctx,
                content=tool_call_output,
                name=tool_name,
                tool_call_id=tool_call.id,
            )
            state.messages = list(ctx.messages)
            yield obs

        if any(isinstance(r, asyncio.CancelledError) for r in raw_results) and not state.done:
            cancel_reason = (
                cancellation.reason if cancellation is not None else "user_request"
            )
            logger.info(
                "Run cancelled",
                extra={"phase": "tool_execution", "reason": cancel_reason,
                       "iteration": state.iteration},
            )
            yield RunCancelled(
                message="The run was cancelled while tools were executing.",
                reason=cancel_reason,
            )
            state.done = True

        # Post-batch explicit termination dispatch.
        # If a tool in this batch called ``ask_user`` or ``return_unable``,
        # re-raise the captured control-flow exception so ``run_loop`` builds
        # the corresponding ``UserInputRequested`` / ``Handoff`` event itself.
        if pending_termination is not None:
            _kind, _exc = pending_termination
            raise _exc

    # ------------------------------------------------------------------
    # Hook 8: on_iteration_end
    # ------------------------------------------------------------------
    async def on_iteration_end(self, state: Any) -> AsyncIterator[Any]:
        yield StateSnapshot(context=self.ctx.model_copy(deep=False))

    # ------------------------------------------------------------------
    # Hook 9: on_run_complete
    # ------------------------------------------------------------------
    async def on_run_complete(self, state: Any) -> AsyncIterator[Any]:
        # --- Post-loop: residual reasoning flush + final telemetry ---------
        if self._accumulated_reasoning:
            yield ReasoningDeltaEvent(
                content=self._accumulated_reasoning,
                message_id=self._reasoning_message_id,
                title=f"Thought for {self._accumulated_duration:.1f} seconds",
                delta=False,
            )

        total_time = time.time() - self.start_time
        logger.info(
            f"Agent run completed in {total_time:.3f}s after {state.iteration} iterations",
            extra={"duration_s": total_time, "iterations": state.iteration},
        )
        self.agent._record_agent_metrics(self.agent_span, total_time, state.iteration)

        # "Agent gave up with a lingering tool error" final report. The
        # workspace-side complement to spine failures: surfaces a tool-level
        # crash the LLM tried to ignore. (Could later be routed via
        # Failure(kind=tool_error) through handle_failure instead.)
        if not state.done and self._last_tool_internal_error:
            yield AgentError(
                message=self._last_tool_internal_error,
                recoverable=False,
                error_type="tool_error",
            )

        yield StateSnapshot(context=self.ctx.model_copy(deep=False))


__all__ = ["WorkspaceRunHooks"]
