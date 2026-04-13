from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import litellm
import pandas as pd
from litellm.exceptions import APIError, InternalServerError, RateLimitError, ServiceUnavailableError, Timeout
from opentelemetry import trace
from pydantic import TypeAdapter

from parsimony_agents.agent.config import AgentConfig, AgentGuardrails, FileStore
from parsimony_agents.agent.events import (
    AgentError,
    StateSnapshot,
    ToolEvent,
)
from parsimony_agents.agent.events import (
    ReasoningDelta as ReasoningDeltaEvent,
)
from parsimony_agents.agent.events import (
    TextDelta as TextDeltaEvent,
)
from parsimony_agents.agent.helpers import (
    TurnState,
)
from parsimony_agents.agent.helpers import (
    parse_cell_ref as _parse_cell_ref,
)
from parsimony_agents.agent.helpers import (
    system_error as _system_error,
)
from parsimony_agents.agent.models import (
    AgentContext,
    AgentMessage,
    ReturnedChartState,
    ReturnedDatasetState,
)
from parsimony_agents.agent.outputs import SystemToolMessage, SystemToolOutput, UtilityToolOutput
from parsimony_agents.agent.tracing import trace_tool_execution
from parsimony_agents.artifacts import (
    Chart,
    Dataset,
)
from parsimony_agents.execution import (
    DataFrameObject,
    ExceptionObject,
    FigureObject,
    PrimitiveObject,
    StringPaginator,
)
from parsimony_agents.execution.executor import BaseCodeExecutor
from parsimony_agents.execution.factory import OutputFactory as FrameworkOutputFactory
from parsimony_agents.messages import Message, Text, blocks_to_text
from parsimony_agents.notebook import Script
from parsimony_agents.rag.keyword_store import get_or_create_session_keyword_store
from parsimony_agents.rag.vector_store import get_or_create_session_vector_store
from parsimony_agents.tools import Tools, toolmethod
from parsimony_agents.variable import Variable, VariableStore
from parsimony_agents.views import get_llm_view_defaults

logger = logging.getLogger("parsimony_agents")
error_logger = logging.getLogger("parsimony_agents.errors")

litellm.REPEATED_STREAMING_CHUNK_LIMIT = 100

# ---------------------------------------------------------------------------
# Agent defaults
# ---------------------------------------------------------------------------
_DRY_EXECUTE_DEFAULT_TIMEOUT_S: float = 120.0  # Default sandbox timeout for dry_execute_code  # TODO: Monitor how many repeated chunks appear naturally before hitting the limit


def _serialize_and_hash_object(obj: Any) -> int:
    return hash(json.dumps(obj, sort_keys=True))




@dataclass
class AgentResult:
    """Structured result from :meth:`Agent.ask`.

    Collects the streaming events from a single ``run()`` call into an
    easy-to-inspect object.  Each field stores the full framework object
    so the display layer can access all metadata.
    """

    text: str = ""
    """Concatenated assistant text (all ``TextDelta`` content)."""

    datasets: dict[str, Dataset] = field(default_factory=dict)
    """Returned :class:`Dataset` objects keyed by variable name."""

    charts: dict[str, Chart] = field(default_factory=dict)
    """Returned :class:`Chart` objects keyed by chart variable name."""

    code: dict[str, Script] = field(default_factory=dict)
    """Returned :class:`Script` objects keyed by notebook name (execution order preserved)."""

    context: AgentContext | None = None
    """Final :class:`AgentContext` — use for multi-turn continuation or inspection."""

    events: list[Any] = field(default_factory=list)
    """Full event log (every ``AgentEvent`` yielded during the run)."""

    @property
    def ok(self) -> bool:
        """``True`` if no error events occurred."""
        return not any(getattr(e, "type", None) == "error" for e in self.events)

    def _collect(self, event: Any) -> None:
        """Accumulate a single event into this result (called by :meth:`Agent.ask`)."""
        self.events.append(event)
        etype = getattr(event, "type", None)
        if etype == "text_delta":
            self.text += event.content
        elif etype == "state_snapshot":
            self.context = event.context
            if self.context is not None:
                for nb_name, nb in getattr(self.context, "notebooks", {}).items():
                    self.code[nb_name] = nb
        elif etype == "tool_event" and getattr(event, "completed", False):
            result = getattr(event, "result", None)
            if result is None:
                return
            if isinstance(result, Dataset):
                var_name = getattr(result, "variable_name", None)
                if var_name:
                    self.datasets[var_name] = result
            elif isinstance(result, Chart):
                chart_name = getattr(result, "chart_variable_name", None)
                if chart_name:
                    self.charts[chart_name] = result


class Agent:
    """Data analysis agent: LLM loop, tools, and code execution (yields AgentEvent).

    **Quick start (OSS users):**

    .. code-block:: python

        from parsimony_agents import Agent
        from parsimony.connectors.fred import CONNECTORS as FRED

        agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind_deps(api_key="..."))
        result = await agent.ask("Show me US GDP trends")
        print(result.text, result.datasets)

    **Power usage (product / full control):**

    Pass explicit ``model_config``, ``instructions``, ``code_executor``, and
    ``output_factory`` for complete control over the agent configuration.
    """

    RETURN_TOOLS = ("return_dataset", "return_chart")
    CODE_TOOL_NAMES = {"code_set", "code_edit", "dry_execute_code"}

    def __init__(
        self,
        *,
        # --- Convenience params (OSS front door) ---
        model: str | None = None,
        api_key: str | None = None,
        connectors: Any | None = None,
        # --- Expert bundle (product / power usage) ---
        config: AgentConfig | None = None,
        # --- Explicit params (override config or use standalone) ---
        model_config: dict[str, Any] | None = None,
        instructions: str | None = None,
        code_executor: BaseCodeExecutor | None = None,
        output_factory: FrameworkOutputFactory | None = None,
        guardrails: AgentGuardrails | None = None,
        session_id: str | None = None,
        file_store: FileStore | None = None,
    ):
        from parsimony_agents.agent.prompts import DEFAULT_DATA_ANALYSIS_PROMPT
        from parsimony_agents.execution.executor import CodeExecutor as _LocalExecutor

        # Unpack AgentConfig bundle (individual kwargs always take precedence)
        if config is not None:
            if model_config is None:
                model_config = config.model_config
            if instructions is None:
                instructions = config.instructions
            if code_executor is None:
                code_executor = config.code_executor
            if output_factory is None:
                output_factory = config.output_factory
            if guardrails is None:
                guardrails = config.guardrails
            if session_id is None:
                session_id = config.session_id
            if file_store is None:
                file_store = config.file_store
        if guardrails is None:
            guardrails = AgentGuardrails()

        # Resolve model_config: explicit > built from model= convenience param
        if model_config is not None:
            resolved_config: dict[str, Any] = model_config
        elif model is not None:
            resolved_config = {"model": model, **({"api_key": api_key} if api_key else {})}
        else:
            raise TypeError(
                "Agent requires either model_config={...} or model='model-name'"
            )

        # Resolve instructions: explicit > default prompt; always append connector catalog
        if instructions is not None:
            resolved_instructions = instructions
        else:
            resolved_instructions = DEFAULT_DATA_ANALYSIS_PROMPT
        if connectors is not None:
            resolved_instructions += connectors.to_llm()

        # Resolve output_factory first (executor depends on it)
        output_factory = (
            output_factory
            or getattr(code_executor, "_output_factory", None)
            or FrameworkOutputFactory(local_dir=tempfile.mkdtemp(prefix="parsimony_agent_"))
        )

        # Resolve code_executor: explicit > local in-process executor
        resolved_executor = code_executor or _LocalExecutor(
            cwd=str(output_factory._local_dir),
            output_factory=output_factory,
        )

        self.instructions = resolved_instructions
        self.session_id = session_id or str(uuid4())
        self.file_store = file_store
        self._connectors = connectors
        self.code_executor = resolved_executor
        self._output_factory = output_factory

        self.figures = []

        self.system_tools = Tools(
            [
                self.code_set,
                self.code_edit,
                self.dry_execute_code,
                self.return_dataset,
                self.return_chart,
                self.output_read,
                self.output_search,
                self.get_context,
            ]
        )

        self.model_config = resolved_config

        self.guardrails = guardrails

        self._CODE_EDIT_TOOL_NAMES = {"code_set", "code_edit"}

    def _record_agent_metrics(self, agent_span: Any, total_time: float, iteration_count: int) -> None:
        if agent_span and agent_span.is_recording():
            agent_span.set_attribute("agent.total_duration_s", total_time)
            agent_span.set_attribute("agent.final_iterations", iteration_count)
            agent_span.set_attribute("agent.model", self.model_config.get("model", "unknown"))

    def _tool_timeout_seconds(self, tool_name: str, raw_args: dict[str, Any]) -> float:
        global_cap = self.guardrails.tool_timeout_s
        if tool_name != "dry_execute_code":
            return global_cap
        raw_timeout = raw_args.get("timeout_seconds", _DRY_EXECUTE_DEFAULT_TIMEOUT_S)
        try:
            requested = float(raw_timeout)
        except (TypeError, ValueError):
            requested = _DRY_EXECUTE_DEFAULT_TIMEOUT_S
        if requested <= 0:
            requested = _DRY_EXECUTE_DEFAULT_TIMEOUT_S
        return min(requested, global_cap)

    @staticmethod
    def _utility_tool_metadata(
        *,
        tool_name: str,
        tool_description: str,
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "source": tool_name,
            "source_description": tool_description,
            **tool_args,
        }

    async def _handle_llm_error(
        self,
        last_exception: Exception,
        text_message_id: str,
        turn_state: TurnState,
        agent_span: Any,
    ) -> AsyncGenerator[Any, None]:
        """Classify an LLM exception and yield the appropriate error/text events.

        Handles RateLimitError, Timeout, ServiceUnavailableError, APIError, and
        all other unexpected exceptions with user-facing messages and span recording.
        """
        section = "final_response" if turn_state.final_response_started else "analysis"

        if isinstance(last_exception, RateLimitError):
            error_logger.error("Rate limit exceeded: %s", last_exception, exc_info=True)
            yield AgentError(
                message="Rate limit exceeded",
                recoverable=False,
                error_type="rate_limit",
                section=section,
            )
            yield TextDeltaEvent(
                content=(
                    "We're currently in beta and experiencing high demand. "
                    "The AI model has hit its rate limit, please wait a moment and try again. "
                    "This is expected during peak usage and will be resolved as we scale."
                ),
                message_id=text_message_id,
                delta=False,
                section="final_response",
            )
        elif isinstance(last_exception, Timeout):
            error_logger.error("Request timeout: %s", last_exception, exc_info=True)
            yield AgentError(
                message="Request timeout",
                recoverable=False,
                error_type="timeout",
                section=section,
            )
            yield TextDeltaEvent(
                content=(
                    "The AI model took too long to respond, please wait a moment and try again. "
                    "We're currently in beta and this will be resolved as we improve the service."
                ),
                message_id=text_message_id,
                delta=False,
                section="final_response",
            )
        elif isinstance(last_exception, ServiceUnavailableError) or (
            isinstance(last_exception, APIError) and "unavailable" in str(last_exception).lower()
        ):
            error_logger.error("Model unavailable: %s", last_exception, exc_info=True)
            yield AgentError(
                message="Model unavailable",
                recoverable=False,
                error_type="unavailable",
                section=section,
            )
            yield TextDeltaEvent(
                content=(
                    "The selected AI model is currently unavailable. "
                    "Please try again in a moment or select a different model."
                ),
                message_id=text_message_id,
                delta=False,
                section="final_response",
            )
        else:
            error_logger.error(
                "LLM error (%s): %s", type(last_exception).__name__, last_exception, exc_info=True
            )
            yield TextDeltaEvent(
                content="I'm sorry, but an error occurred while trying to communicate with the AI model, and your request cannot proceed.",
                message_id=text_message_id,
                delta=False,
                section=section,
            )

        if agent_span and agent_span.is_recording():
            agent_span.record_exception(last_exception)


    async def ask(
        self,
        message: str | Text,
        *,
        ctx: AgentContext | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the agent and collect all results into an :class:`AgentResult`.

        This is the simple API — equivalent to consuming ``run()`` and
        collecting events, but returns a structured result instead of
        requiring event-loop handling.

        .. code-block:: python

            result = await agent.ask("Show me US GDP trends")
            print(result.text)        # assistant's text response
            print(result.datasets)    # {"us_gdp": DataFrame}
            print(result.code)        # {"main": "import pandas as ..."}
            assert result.ok
        """
        result = AgentResult()
        async for event in self.run(message, ctx=ctx, **kwargs):
            result._collect(event)
        return result

    async def _setup_connectors(self) -> None:
        if self._connectors is None:
            return
        await self.code_executor.set_connectors(self._connectors)

    async def run(
        self,
        user_message: str | Text,
        *,
        ctx: AgentContext | None = None,
        tool_choice: str = "auto",
    ) -> AsyncGenerator[Any, None]:
        """Stream agent events for a single user turn.

        Yields :class:`AgentEvent` subclass instances (TextDelta, ToolEvent,
        StateSnapshot, etc.) as they are produced.  For simple one-shot usage
        prefer :meth:`ask` which collects the stream into an :class:`AgentResult`.
        """
        if isinstance(user_message, str):
            user_message = Text(content=user_message)

        agent_span = trace.get_current_span()

        logger.info("Agent run started", extra={"prompt_preview": user_message.content[:1000]})
        start_time = time.time()

        if isinstance(self.instructions, str):
            system_message = AgentMessage(role="system", content=Text(content=self.instructions.rstrip()))
        else:
            system_message = AgentMessage(role="system", content=self.instructions)

        if not ctx:
            ctx = AgentContext(messages=[system_message], session_id=self.session_id)
            yield StateSnapshot(context=ctx, section="analysis")
        else:
            ctx.messages[0] = system_message

        if self.session_id and self.file_store is not None:
            ctx.files = self.file_store
            ctx.vector_store = get_or_create_session_vector_store(self.session_id)
            ctx.keyword_store = get_or_create_session_keyword_store(self.session_id)

            await self.code_executor.set_cwd(str(ctx.files.get_files_dir()), session_id=self.session_id)

        remote_v = await self.code_executor.get_sandbox_state_version()
        skip_initial_replay = (
            ctx is not None
            and remote_v is not None
            and remote_v == ctx.state_version
        )

        if not skip_initial_replay:
            await self.code_executor.push_state(ctx.data_context)

            await self._setup_connectors()

            await ctx.execute_notebooks(code_executor=self.code_executor, update_outputs=True)

            await self.code_executor.set_sandbox_state_version(ctx.state_version)
        else:
            # Warm executor: namespace already matches ctx.state_version.
            pass

        turn_state = TurnState()

        iteration_count = 0
        tool_call_history = []
        last_tool_internal_error = None  # Track last tool internalerror to report at end if agent gives up
        
        # Check if this is a continuation request
        is_continuation = user_message.content.strip().lower() == "continue"
        if is_continuation:
            # Reset start time for continuation (use default limits)
            start_time = time.time()
            logger.info("Continuation requested - resetting timer", extra={
                "max_iters": self.guardrails.max_iterations,
                "max_execution_time": self.guardrails.max_execution_time_s,
            })
        else:
            if ctx:

                # add user message context

                ctx.messages.append(AgentMessage(role="user", content=user_message))

        iteration = 0

        tool_choice = "auto"

        reasoning_message_id = str(uuid4())
        accumulated_reasoning = ""
        accumulated_duration = 0.0

        context_snapshot = await ctx.to_snapshot()

        # filter previous context snapshots
        ctx.messages = [m for m in ctx.messages if m.metadata.get("context_snapshot", False) is False]

        ctx.messages.append(Message(role="user", content=context_snapshot, metadata={"context_snapshot": True}))
        
        while not turn_state.stopped:

            iteration += 1

            # Guard: max execution time
            elapsed = time.time() - start_time
            if elapsed > self.guardrails.max_execution_time_s:
                error_msg = f"Reached max agent execution time ({self.guardrails.max_execution_time_s}s)."
                logger.warning(error_msg, extra={"elapsed_s": elapsed, "max_execution_time_s": self.guardrails.max_execution_time_s})
                if agent_span and agent_span.is_recording():
                    agent_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                yield AgentError(
                    message="Time limit reached.",
                    recoverable=True,
                    error_type="time_limit",
                    section="final_response",
                )
                break

            # Guard: max iterations per request
            if iteration_count >= self.guardrails.max_iterations:
                error_msg = f"Reached max agent iterations ({self.guardrails.max_iterations})."
                logger.warning(error_msg, extra={"iteration_count": iteration_count, "max_iterations": self.guardrails.max_iterations})
                if agent_span and agent_span.is_recording():
                    agent_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                yield AgentError(
                    message="Iteration limit reached.",
                    recoverable=True,
                    error_type="iteration_limit",
                    section="final_response",
                )
                break

            tools = self.system_tools.copy()


            # Most recent tool result uses "default" mode, older ones use "minimal" #TODO: REVISE THIS LOGIC

            last_tool_idx = next((i for i in range(len(ctx.messages) - 1, -1, -1) if ctx.messages[i].role == "tool"), -1)
            llm_messages = [
                m.to_llm(mode="default" if (i >= (len(ctx.messages) - 4) or i == last_tool_idx) else "minimal")
                for i, m in enumerate(ctx.messages)
            ]

            # appending of the user message happens after because 
            # 1) the user message needs to persist in the messages, but the context is temporary
            # 2) the user message should be the last message in the list of messages


            response = None
            last_exception = None
            for attempt in range(self.guardrails.llm_max_retries + 1):

                try:
                    text_message_id = str(uuid4())  # Single messageId for all text chunks
                    reasoning_start_time = time.time()  # Track when reasoning started

                    request_model_config = dict(self.model_config)

                    try:
                        with open("ignored/llm_messages.json", "w") as f:
                            json.dump(llm_messages, f)
                    except OSError:
                        pass

                    logger.info("Agent thinking", extra={"iteration": iteration_count, "attempt": attempt + 1})
                    response = await litellm.acompletion(
                        messages=llm_messages,
                        tools=tools.to_llm(),
                        tool_choice=tool_choice,
                        request_timeout=self.guardrails.llm_timeout_s,
                        stream=True,
                        **request_model_config
                    )

                    # Process streaming response
                    chunks = []
                    async for chunk in response:
                        chunks.append(chunk)
                        if chunk.choices[0].delta.content:
                            yield TextDeltaEvent(
                                content=chunk.choices[0].delta.content,
                                message_id=text_message_id,
                                delta=True,
                                section="final_response" if turn_state.final_response_started else "analysis",
                            )
                        if reasoning_content := getattr(chunk.choices[0].delta, "reasoning_content", None):
                            yield ReasoningDeltaEvent(
                                content=reasoning_content,
                                message_id=reasoning_message_id,
                                delta=True,
                                section="final_response" if turn_state.final_response_started else "analysis",
                            )

                        for tool_call_chunk in (chunk.choices[0].delta.tool_calls or []):

                            if not (name := tool_call_chunk.function.name):
                                continue
                            
                            tool_call_id = tool_call_chunk.id

                            if name not in tools:
                                ctx.messages.append(
                                    AgentMessage(role="tool", content=f"Unknown tool: {name}", name=name, tool_call_id=tool_call_id)
                                )
                                continue

                            tool_type = tools[name].tool_type
                            loading_label = tools[name].ui_message
                            match tool_type:
                                case "utility":
                                    yield ToolEvent(
                                        tool_name=name,
                                        tool_call_id=tool_call_id,
                                        tool_type="utility",
                                        completed=False,
                                        result=UtilityToolOutput(
                                            ui_message=loading_label or f"Executing {name}",
                                        ),
                                        ui_message=loading_label or f"Executing {name}",
                                        section="final_response" if turn_state.final_response_started else "analysis",
                                    )
                                case "system":
                                    if loading_label is not None:
                                        yield ToolEvent(
                                            tool_name=name,
                                            tool_call_id=tool_call_id,
                                            tool_type="system",
                                            completed=False,
                                            result=SystemToolMessage(message=loading_label),
                                            ui_message=loading_label,
                                            section="final_response" if turn_state.final_response_started else "analysis",
                                        )
                                case _:
                                    pass


                    response = litellm.stream_chunk_builder(chunks, messages=llm_messages)

                    last_exception = None
                    break
                except (InternalServerError, RateLimitError, Timeout) as e:
                    last_exception = e
                    # Backoff then retry if attempts remain
                    if attempt < self.guardrails.llm_max_retries:
                        await asyncio.sleep(min(2 ** attempt, 8))
                    else:
                        break
                except Exception as e:
                    # Catch all other exceptions (e.g., APIConnectionError, TypeError, etc.)
                    last_exception = e
                    # Log for debugging but don't retry on unexpected errors
                    error_logger.error("Unexpected error calling LLM: %s", e, exc_info=True)
                    break

            if last_exception is not None:
                async for event in self._handle_llm_error(
                    last_exception, text_message_id, turn_state, agent_span
                ):
                    yield event
                break

            if response is None or len(response.choices) == 0:
                logger.warning("No response from LLM", extra={})
                continue

            response_message = Message.from_litellm(response.choices[0].message)
            response_message = AgentMessage(**response_message.model_dump())

            _new_tool_calls = []

            _has_return_tool_call = False

            for tool_call in (response_message.tool_calls or []):
                _new_tool_calls.append(tool_call)
                if tool_call.function.name in self._CODE_EDIT_TOOL_NAMES:
                    # Notebook/script will be re-executed by the tool implementation.
                    pass
                if tool_call.function.name in self.RETURN_TOOLS:  # type: ignore[comparison-overlap]
                    _has_return_tool_call = True
                    break 

            if len(_new_tool_calls) == 0:
                turn_state.stopped = True
                turn_state.final_response_started = True
            elif len(_new_tool_calls) > 1 and _has_return_tool_call:
                # filter the speculative return tool call
                _new_tool_calls = _new_tool_calls[:-1]

            response_message.tool_calls = _new_tool_calls if _new_tool_calls else None

            # Log tool calls if any
            if _new_tool_calls:
                logger.info("Tool calls", extra={
                    "iteration": iteration_count,
                    "tool_names": [tc.function.name for tc in _new_tool_calls]
                })

            ctx.messages.append(response_message)

            iteration_count += 1

            if turn_reasoning := getattr(response.choices[0].message, "reasoning_content", None):
                accumulated_reasoning += turn_reasoning
                accumulated_duration += (time.time() - reasoning_start_time)

            is_silent = not response_message.content and all(
                getattr(tools.get(tc.function.name), "tool_type", None) == "system" and 
                getattr(tools.get(tc.function.name), "ui_message", None) is None
                for tc in (response_message.tool_calls or [])
            )

            if accumulated_reasoning and not is_silent:
                yield ReasoningDeltaEvent(
                    content=accumulated_reasoning,
                    message_id=reasoning_message_id,
                    title=f"Thought for {accumulated_duration:.1f} seconds",
                    delta=False,
                    section="final_response" if turn_state.final_response_started else "analysis",
                )
                accumulated_reasoning = ""
                accumulated_duration = 0.0
                reasoning_message_id = str(uuid4())


            if text_message := response_message.content:
                yield TextDeltaEvent(
                    content=text_message,
                    message_id=text_message_id,
                    delta=False,
                    section="final_response" if turn_state.final_response_started else "analysis",
                )


            if not (tool_calls := getattr(response_message, "tool_calls", None)):

                logger.info("Agent stopped", extra={"iteration_count": iteration_count, "finish_reason": "stop"})
                # the consecutiveness has been broken, so we need to create a new version

                break
                    

            
            else:

                # Build a traced coroutine for a single tool call.
                # Defined as a factory so each coroutine captures its arguments by value,
                # avoiding the classic Python closure-over-loop-variable pitfall.
                def _build_coroutine(
                    name: str,
                    args_with_ctx: dict,
                    t_type: str,
                    raw_args: dict,
                    timeout_s: float,
                ):
                    @trace_tool_execution(name, t_type, logger, error_logger, timeout_s)
                    async def _execute():
                        return await tools[name](**args_with_ctx)
                    return _execute(tool_args=raw_args)

                # ── Phase 1: validate, yield "started" UI chunks, build coroutines ──────
                # Each entry: (tool_call, call_signature, coroutine).
                tool_executions: list[tuple] = []

                for tool_call in tool_calls:
                    tool_name = tool_call.function.name

                    if tool_name not in tools:
                        ctx.messages.append(
                            AgentMessage(role="tool", content=f"Unknown tool: {tool_name}", name=tool_name, tool_call_id=tool_call.id)
                        )
                        continue

                    tool_args = json.loads(tool_call.function.arguments)

                    # Extract LLM-provided UI description before hashing/execution.
                    llm_ui_message = tool_args.pop("_ui_message", None)

                    # Loop detection: same tool + same args called repeatedly.
                    args_hash = _serialize_and_hash_object(tool_args)
                    call_signature = (tool_name, args_hash)
                    tool_timeout_s = self._tool_timeout_seconds(tool_name, tool_args)
                    repeat_count = sum(1 for sig in tool_call_history if sig == call_signature)

                    if repeat_count == 6:
                        logger.warning("Loop detected: %s called %d times with same args", tool_name, repeat_count + 1, extra={
                            "tool_name": tool_name,
                            "repeat_count": repeat_count + 1
                        })
                        yield AgentError(
                            message="Internal Error",
                            recoverable=False,
                            error_type="loop_detection",
                            section="final_response" if turn_state.final_response_started else "analysis",
                        )
                        ctx.messages.append(
                            AgentMessage(role="tool", content="Loop detected: You have called the same tool with the same arguments multiple times. Consider trying a different approach.", name=tool_name, tool_call_id=tool_call.id)
                        )
                        turn_state.stopped = True
                        continue
                    elif repeat_count >= 2:
                        logger.warning("Loop suspected: %s called %d times with same args", tool_name, repeat_count + 1, extra={
                            "tool_name": tool_name,
                            "repeat_count": repeat_count + 1
                        })

                    tool_type = tools[tool_name].tool_type

                    # Code tools require a non-empty notebook_name. Fail fast before execution.
                    if tool_type == "code":
                        notebook_name = tool_args.get("notebook_name")
                        if not isinstance(notebook_name, str) or not notebook_name.strip():
                            ctx.messages.append(
                                AgentMessage(
                                    role="tool",
                                    content="code_set and code_edit require a non-empty notebook_name. Provide a descriptive name (e.g. 'inflation_analysis', 'returns_pipeline').",
                                    name=tool_name,
                                    tool_call_id=tool_call.id,
                                )
                            )
                            continue

                    # Yield the "started" UI chunk for code tools here; other tool types
                    # already emit their "started" chunk during LLM response streaming.
                    # ui_message = Tool definition label (loading state).
                    # ui_message_completed = LLM _ui_message (completed state, set later).
                    loading_label = tools[tool_name].ui_message
                    if tool_type == "code":
                        notebook_name = tool_args.get("notebook_name")
                        if isinstance(notebook_name, str) and notebook_name.strip():
                            notebook = ctx.get_or_create_notebook(notebook_name)
                            preview = notebook.to_preview()
                            preview.ui_message = loading_label
                            yield ToolEvent(
                                tool_name=tool_name,
                                tool_call_id=tool_call.id,
                                tool_type="code",
                                completed=False,
                                result=preview,
                                ui_message=loading_label,
                                section="final_response" if turn_state.final_response_started else "analysis",
                            )

                    tool_executions.append((
                        tool_call,
                        call_signature,
                        repeat_count,
                        _build_coroutine(
                            tool_name,
                            {**tool_args, "context": ctx},
                            tool_type,
                            tool_args,
                            tool_timeout_s,
                        ),
                    ))

                # ── Phase 2: execute all tool coroutines concurrently ────────────────────
                # For single-tool turns this degrades to a plain await with zero overhead.
                # For multi-tool retrieval turns (utility) this parallelises
                # external I/O and per-tool LLM summary calls.
                raw_results = await asyncio.gather(
                    *(coro for _, _, _, coro in tool_executions),
                    return_exceptions=True,
                )

                # ── Phase 3: flush results sequentially into shared state ────────────────
                # Results are committed in the original tool-call order so the LLM sees a
                # well-formed message sequence, and ctx mutations stay serialised.
                for (tool_call, call_signature, repeat_count, _), raw_result in zip(tool_executions, raw_results):
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    # Re-extract _ui_message (already stripped during Phase 1,
                    # but we re-parse from raw arguments here).
                    llm_ui_message = tool_args.pop("_ui_message", None)

                    if isinstance(raw_result, asyncio.CancelledError):
                        raise raw_result

                    # Guard: asyncio.gather(return_exceptions=True) can return any
                    # BaseException subclass, not just Exception.  Re-raise signals
                    # that indicate process-level conditions (KeyboardInterrupt, SystemExit,
                    # GeneratorExit) so they propagate normally; convert all other
                    # exceptions into a tool error message without accessing result
                    # attributes that only exist on valid tool results.
                    if isinstance(raw_result, BaseException) and not isinstance(raw_result, Exception):
                        logger.error(
                            "Tool %s raised non-Exception BaseException: %r",
                            tool_name,
                            raw_result,
                        )
                        raise raw_result

                    if isinstance(raw_result, Exception):
                        logger.warning(
                            "Tool %s raised an exception: %r",
                            tool_name,
                            raw_result,
                        )
                        ctx.messages.append(
                            AgentMessage(role="tool", content=str(raw_result), name=tool_name, tool_call_id=tool_call.id)
                        )
                        continue

                    tool_result = raw_result

                    if not tool_result.success:
                        last_tool_internal_error = f"Tool {tool_name} got an internal error. Trace ID: {trace.get_current_span().get_span_context().trace_id}"

                    tool_call_output: Any = None

                    match tools[tool_name].tool_type:
                        case "utility":
                            # Static label for loading state; LLM message for completed state.
                            ui_message = tools[tool_name].ui_message or f"Executing {tool_name}"
                            ui_message_completed = llm_ui_message or tools[tool_name].ui_message_completed

                            if tool_result.data:
                                tool_call_output = tool_result.data

                                if (new_obj := tool_call_output.content) is not None and isinstance(new_obj, Variable):
                                    ctx.bump_state_version()
                                    ctx.data_context.extend([new_obj])
                                    push_store = VariableStore(variables={new_obj.name: new_obj})
                                    await self.code_executor.push_state(push_store)
                                    await self.code_executor.set_sandbox_state_version(ctx.state_version)

                                # Propagate into the output object (consumed by UI).
                                if ui_message_completed and not tool_call_output.ui_message_completed:
                                    tool_call_output.ui_message_completed = ui_message_completed

                                yield ToolEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call.id,
                                    tool_type="utility",
                                    completed=True,
                                    result=tool_call_output,
                                    ui_message_completed=ui_message_completed,
                                    section="final_response" if turn_state.final_response_started else "analysis",
                                )
                            else:
                                tool_call_output = tool_result.exception_message
                                output = UtilityToolOutput(
                                    ui_message=ui_message,
                                    ui_message_completed=ui_message_completed,
                                    metadata=self._utility_tool_metadata(
                                        tool_name=tool_name,
                                        tool_description=tools[tool_name].ui_description or tools[tool_name].description,
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
                                    section="final_response" if turn_state.final_response_started else "analysis",
                                )

                        case "return":
                            return_loading = tools[tool_name].ui_message
                            if tool_result.data:
                                tool_call_output = tool_result.data
                                return_types = (Dataset, Chart)
                                if isinstance(tool_call_output, return_types):
                                    turn_state.final_response_started = True
                                    yield ToolEvent(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call.id,
                                        tool_type="return",
                                        completed=True,
                                        result=tool_call_output,
                                        ui_message=return_loading,
                                        ui_message_completed=llm_ui_message,
                                        section="final_response",
                                    )
                            else:
                                tool_call_output = tool_result.exception_message

                        case "code":
                            tool_call_output = tool_result.data if tool_result.data else tool_result.exception_message
                            if tool_result.success:
                                notebook_name = tool_args.get("notebook_name")
                                if not isinstance(notebook_name, str) or not notebook_name.strip():
                                    raise ValueError("code tools require a non-empty notebook_name.")
                                notebook = ctx.get_or_create_notebook(notebook_name)

                                preview = notebook.to_preview()
                                preview.ui_message = llm_ui_message

                                yield ToolEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call.id,
                                    tool_type="code",
                                    completed=True,
                                    result={"notebook": notebook, "preview": preview},
                                    ui_message_completed=llm_ui_message,
                                    section="final_response" if turn_state.final_response_started else "analysis",
                                )

                                turn_state.edited_notebook_names.add(notebook.id)

                        case "system":
                            tool_call_output = tool_result.data
                            system_ui = llm_ui_message or tools[tool_name].ui_message_completed or tools[tool_name].ui_message
                            if system_ui is not None:
                                yield ToolEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call.id,
                                    tool_type="system",
                                    completed=True,
                                    result=SystemToolMessage(message=system_ui),
                                    ui_message_completed=llm_ui_message,
                                    section="final_response" if turn_state.final_response_started else "analysis",
                                )

                        case _:
                            raise ValueError(f"Invalid tool type: {tools[tool_name].tool_type}")

                    # If the agent is repeating itself, prepend the warning into the single
                    # tool result message rather than emitting a second message for the same
                    # tool_call_id (which violates the LLM API contract).
                    if repeat_count >= 2:
                        tool_call_output = [
                            {"type": "text", "text": "Warning: You have called the same tool with the same arguments multiple times. Consider trying a different approach.\n\n"},
                            *Message._normalize_content(tool_call_output),
                        ]

                    ctx.messages.append(
                        AgentMessage(role="tool", content=tool_call_output, name=tool_name, tool_call_id=tool_call.id)
                    )
                    tool_call_history.append(call_signature)

                # Persist tool results immediately so a client disconnect mid-turn can recover.
                yield StateSnapshot(context=ctx.model_copy(deep=False), section="analysis")


        if accumulated_reasoning:
            yield ReasoningDeltaEvent(
                content=accumulated_reasoning,
                message_id=reasoning_message_id,
                title=f"Thought for {accumulated_duration:.1f} seconds",
                delta=False,
                section="final_response" if turn_state.final_response_started else "analysis",
            )

        # Log agent run completion (after while loop ends)
        total_time = time.time() - start_time
        logger.info(
            "Agent run completed in %.3fs after %d iterations",
            total_time,
            iteration_count,
            extra={
                "duration_s": total_time,
                "iterations": iteration_count,
            },
        )
        
        # Record final metrics on span (span created at router level)
        self._record_agent_metrics(agent_span, total_time, iteration_count)

        # If agent gave up without successfully returning, yield the last tool error
        if not turn_state.stopped and last_tool_internal_error:
            yield AgentError(
                message=last_tool_internal_error,
                recoverable=False,
                error_type="tool_error",
                section="final_response" if turn_state.final_response_started else "analysis",
            )

        if turn_state.edited_notebook_names:
            for notebook_name in turn_state.edited_notebook_names:
                ctx.get_or_create_notebook(notebook_name).increment_version()

        yield StateSnapshot(context=ctx.model_copy(deep=False), section="analysis")

    @toolmethod(
        name="get_context",
        description="Get the current context of the agent.",
        parameters_schema={},
        tool_type="system",
        ui_message=None
    )
    async def get_context(self, *, context: AgentContext) -> AgentContext:
        """Return a serializable snapshot of the current agent context (tool)."""
        context_snapshot = await context.to_snapshot()
        return context_snapshot


    @toolmethod(
        name="output_read",
        description="""Read specific pages of a long output by variable name (up to 5 pages per call). 
        This tool is used to inspect, page through, and extract detailed content from large outputs.
        It can be used on its own when page locations are known, and works best in combination 
        with `output_search` to explore search hits in full context or continue sequential review.

        For DataFrames, cells may be truncated (shown with "..."). To read the full content of a 
        specific cell and paginate through it, use variable_name="df[row,col]" with 0-indexed row 
        and column name or index (e.g. df[5,2] or df[5,description]). Pages then refer to character 
        chunks within that cell, same as for primitives.
        """,
        parameters_schema={
            "type": "object",
            "properties": { 
                "variable_name": { "type": "string", "description": "The variable name (e.g. 'df') or cell ref (e.g. 'df[row,col]') to display pages from." },
                "pages": { "type": "array", "description": "The pages to read. Maximum of 5 pages.", "items": { "type": "integer" }, "minItems": 1, "maxItems": 5}
            },
            "required": ["variable_name", "pages"],
            "additionalProperties": False
        },
        tool_type="system",
        ui_message="Reading output...",
        ui_message_completed="Read output"
    )
    async def output_read(self, *, variable_name: str, pages: list[int], context: AgentContext) -> SystemToolOutput:
        """Page through a long variable or cell value (tool, max 5 pages per call)."""
        pages = pages[:5]
        display_pages = [p - 1 for p in pages]

        cell_ref = _parse_cell_ref(variable_name)
        if cell_ref is not None:
            base_name, row, col = cell_ref
            prim, err = await self._get_cell_as_primitive(cell_ref, context, from_data_context=False)
            if err:
                return _system_error(err)
            page_chars = get_llm_view_defaults("primitive")["default"].page_chars
            paginator = StringPaginator(prim.value, chars_per_page=page_chars)
            page_blocks = "\n".join(paginator.iter_pages(display_pages))
            header = f'type="primitive"\nCell [{row},{col}] from {base_name}'
            text = f"{header}\n{page_blocks}" if page_blocks else f"{header}\n(Empty cell.)"
            return SystemToolOutput(content=Text(content=text))

        obj_kernel = await self.code_executor.eval(variable_name)
        output = obj_kernel.outputs[0]
        text = blocks_to_text(output.to_llm(overrides={"display_pages": display_pages}))
        return SystemToolOutput(content=Text(content=text))


    @toolmethod(
        name="output_search",
        description="""Perform a hybrid (keyword and semantic) search across all indexed content, including long dataframes and text assets. 
        This tool is used to locate specific terms, codes, values, or headers within large variables / utility tool outputs.
        Results include page numbers and excerpts, enabling targeted follow-up with `output_read` for full context or sequential review.
        For DataFrames, you can also search within a specific cell using variable_name='df[row,col]' (0-indexed); the cell is indexed on demand for full-text search.
        """,
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (natural language or specific terms)."
                },
                "variable_name": {
                    "type": "string",
                    "description": "Specific variable or filename to search."
                },
                "top_k": {
                    "type": "integer",
                    "description": "The number of results to return.",
                    "default": 5
                }
            },
            "required": ["query", "variable_name"],
            "additionalProperties": False
        },
        tool_type="system",
        ui_message="Reading output...",
        ui_message_completed="Read output"
    )
    async def output_search(self, *, query: str, variable_name: str | None = None, top_k: int = 5, context: AgentContext) -> SystemToolOutput:
        """Search outputs using hybrid search (keyword + semantic)."""
        from parsimony_agents.rag import hybrid_search

        if variable_name:
            indexed = await self._ensure_indexed(variable_name, context)
            if not indexed:
                return _system_error(f"Could not index '{variable_name}' for search")

        try:
            results = await hybrid_search(
                query=query,
                keyword_store=context.keyword_store,
                vector_store=context.vector_store,
                identifier=variable_name,
                k=top_k
            )
        except Exception as e:
            return _system_error(f"Search failed: {str(e)}")

        response_text = self._format_no_search_results(query, variable_name) if not results else self._format_search_results(query, results)

        return SystemToolOutput(content=Text(content=response_text))
    
    async def _ensure_indexed(self, variable_name: str, context: AgentContext) -> bool:
        """Ensure a variable (or cell ref) is indexed in both stores (lazy indexing)."""
        keyword_indexed = context.keyword_store.is_indexed(variable_name) if context.keyword_store else True
        vector_indexed = context.vector_store.is_indexed(variable_name) if context.vector_store else True

        if keyword_indexed and vector_indexed:
            return True

        # Cell ref: variable_name="df[row,col]" - index cell content as primitive
        cell_ref = _parse_cell_ref(variable_name)
        if cell_ref is not None:
            return await self._ensure_cell_indexed(cell_ref, variable_name, context)

        if variable_name not in context.data_context:
            return False
        var = context.data_context[variable_name]
        output = var.output

        if output is not None:
            if output.type in ("dataframe", "primitive"):
                tasks = []
                if context.keyword_store and not keyword_indexed:
                    tasks.append(context.keyword_store.index_output(output, variable_name))
                if context.vector_store and not vector_indexed:
                    tasks.append(context.vector_store.index_output(output, variable_name))
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    return all(r is True or not isinstance(r, Exception) for r in results)

        return False

    async def _get_cell_as_primitive(
        self,
        cell_ref: tuple[str, int, str],
        context: AgentContext,
        *,
        from_data_context: bool,
    ) -> tuple[PrimitiveObject | None, str | None]:
        """
        Resolve a cell ref to a PrimitiveObject. Returns (prim, None) on success, (None, error_msg) on failure.
        from_data_context: True = use context.data_context (for indexing), False = use code_executor.eval (for read).
        """
        base_name, row, col = cell_ref
        df: pd.DataFrame | None = None

        if from_data_context:
            if base_name not in context.data_context:
                return (None, f"Variable '{base_name}' not in data context.")
            output = context.data_context[base_name].output
            if output is None or not isinstance(output, DataFrameObject):
                return (None, f"'{base_name}' is not a DataFrame.")
            df = output.value
        else:
            obj_kernel = await self.code_executor.eval(base_name)
            if not obj_kernel.outputs:
                return (None, f"Variable '{base_name}' not found or has no output.")
            output = obj_kernel.outputs[0]
            if isinstance(output, ExceptionObject):
                return (None, f"Error evaluating '{base_name}': {output.value}")
            if not isinstance(output, DataFrameObject):
                return (None, f"'{base_name}' is not a DataFrame; cell ref requires a DataFrame variable.")
            df = output.value

        try:
            col_idx = int(col) if str(col).lstrip("-").isdigit() else df.columns.get_loc(col)
        except (KeyError, ValueError) as e:
            return (None, f"Invalid column '{col}' for cell ref: {e}")
        if row < 0 or row >= len(df):
            return (None, f"Row {row} out of range (0..{len(df) - 1}) for '{base_name}'.")
        try:
            cell_val = df.iloc[row, col_idx]
        except IndexError as e:
            return (None, f"Cell [{row},{col}] out of range: {e}")
        cell_text = "<NULL>" if pd.isna(cell_val) else str(cell_val)
        return (PrimitiveObject(value=cell_text), None)

    async def _ensure_cell_indexed(
        self,
        cell_ref: tuple[str, int, str],
        identifier: str,
        context: AgentContext,
    ) -> bool:
        """Index a DataFrame cell as a primitive for search (lazy, on demand)."""
        prim, _ = await self._get_cell_as_primitive(cell_ref, context, from_data_context=True)
        if prim is None:
            return False
        tasks = []
        if context.keyword_store:
            tasks.append(context.keyword_store.index_output(prim, identifier))
        if context.vector_store:
            tasks.append(context.vector_store.index_output(prim, identifier))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return all(r is True or not isinstance(r, Exception) for r in results)
        return False
    
    
    def _format_search_results(self, query: str, results: list) -> str:
        """Format search results."""
        lines: list[str] = []
        for result in results:
            page = result.metadata.get("page", 1)
            lines.append(f"Page {page}:\n{result.content}\n")
        return "\n".join(lines).rstrip()
    
    def _format_no_search_results(self, query: str, variable_name: str | None) -> str:
        """Format response when no relevant results are found."""
        if variable_name:
            return f"No matching content for '{query}' in '{variable_name}'."
        return f"No matching content for '{query}'."




    @toolmethod(
        name="dry_execute_code",
        description="Execute temporary Python code without modifying notebook code or session state. Use for quick inspections, ad-hoc calculations, or exploratory checks.",
        parameters_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Temporary Python code to execute. Results are shown but not persisted to any notebook."
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Optional execution timeout in seconds. Defaults to 120; capped by the global tool timeout."
                }
            },
            "required": ["code"],
            "additionalProperties": False
        },
        tool_type="utility",
        ui_message="Running exploratory code",
        ui_description="Execute temporary code without modifying notebooks.",
    )
    async def dry_execute_code(
        self,
        *,
        code: str,
        timeout_seconds: float = _DRY_EXECUTE_DEFAULT_TIMEOUT_S,
        context: AgentContext,
    ) -> UtilityToolOutput:
        """Execute exploratory code in a sandboxed copy without modifying notebooks (tool)."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        effective_timeout = min(timeout_seconds, self.guardrails.tool_timeout_s)

        metadata = self._utility_tool_metadata(
            tool_name="dry_execute_code",
            tool_description="Execute temporary code without modifying notebooks.",
            tool_args={
                "code": code,
                "timeout_seconds": timeout_seconds,
                "effective_timeout_seconds": effective_timeout,
            },
        )

        # No need to manually copy locals - the executor handles the authoritative
        # push automatically.
        # Use dry_run=True to ensure code execution is sandboxed within the actor
        kernel_output = await self.code_executor.execute(
            code,
            dry_run=True,
            timeout_seconds=effective_timeout,
        )

        # Attach lightweight metadata to the kernel output
        kernel_output.metadata = metadata

        return UtilityToolOutput(
            metadata=metadata,
            content=kernel_output,
            ui_message="Executing temporary code"
        )

    @toolmethod(
        name="code_set",
        description="Replace the entire analysis script. This overwrites previous code and re-executes the full script.",
        parameters_schema={
            "type": "object",
            "properties": {
                "notebook_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Required. A stable, descriptive name for the notebook (e.g. 'gdp_retrieval', 'gdp_transform', 'gdp_validation'). Must not be empty.",
                },
                "code": {"type": "string", "description": "The full Python script to set."}
            },
            "required": ["notebook_name", "code"],
            "additionalProperties": False,
        },
        tool_type="code",
        ui_message="Writing notebook",
    )
    async def code_set(self, *, notebook_name: str, code: str, context: AgentContext) -> str:
        """Overwrite a notebook's full code and re-execute it (tool)."""
        notebook = context.get_or_create_notebook(notebook_name)
        context.active_notebook_name = notebook.id
        context.bump_state_version()
        notebook.code_set(code=code)
        await self._reexecute_context_notebooks(context=context)
        self._stamp_data_objects(notebook)
        self._invalidate_notebook_dependent_state(context=context, notebook_ref=notebook.id)
        kernel_output = notebook.output

        for figure in kernel_output.get_figures():
            self.figures.append(figure)

        return kernel_output

    @toolmethod(
        name="code_edit",
        description="Edit the analysis script by replacing exactly one occurrence of old_str with new_str. Use sufficiently long and specific context to ensure uniqueness. Re-executes the full script after applying the edit.",
        parameters_schema={
            "type": "object",
            "properties": {
                "notebook_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Required. A stable, descriptive name for the notebook (e.g. 'gdp_retrieval', 'gdp_transform', 'gdp_validation'). Must not be empty.",
                },
                "old_str": {"type": "string", "description": "Exact substring to replace (must occur exactly once)."},
                "new_str": {"type": "string", "description": "Replacement text."},
            },
            "required": ["notebook_name", "old_str", "new_str"],
            "additionalProperties": False,
        },
        tool_type="code",
        ui_message="Editing notebook",
    )
    async def code_edit(self, *, notebook_name: str, old_str: str, new_str: str, context: AgentContext) -> str:
        """Apply a targeted string replacement to a notebook and re-execute it (tool)."""
        if old_str == "":
            return await self.code_set(notebook_name=notebook_name, code=new_str, context=context)

        notebook = context.get_or_create_notebook(notebook_name)
        context.active_notebook_name = notebook.id
        context.bump_state_version()
        notebook.code_edit(old_str=old_str, new_str=new_str)
        await self._reexecute_context_notebooks(context=context)
        self._stamp_data_objects(notebook)
        self._invalidate_notebook_dependent_state(context=context, notebook_ref=notebook.id)
        kernel_output = notebook.output

        for figure in kernel_output.get_figures():
            self.figures.append(figure)

        return kernel_output

    def _stamp_data_objects(self, notebook: Any) -> None:
        """Copy :attr:`KernelOutput.fetch_log` onto the notebook for the UI."""
        from parsimony_agents.execution.outputs import FetchLogEntry

        ko = notebook.output
        if not ko or not ko.fetch_log:
            notebook.data_objects = []
            return

        seen: set[str] = set()
        deduped: list[FetchLogEntry] = []
        for item in ko.fetch_log:
            entry = item if isinstance(item, FetchLogEntry) else FetchLogEntry.model_validate(item)
            key = f"{entry.source}:{json.dumps(entry.params, sort_keys=True, default=str)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        notebook.data_objects = deduped

    async def _reexecute_context_notebooks(self, *, context: AgentContext) -> None:
        await self.code_executor.replace_state(context.data_context)
        await context.execute_notebooks(code_executor=self.code_executor, update_outputs=True)
        await self.code_executor.set_sandbox_state_version(context.state_version)

    def _invalidate_notebook_dependent_state(self, *, context: AgentContext, notebook_ref: str) -> None:
        derived_variable_names = [
            name
            for name, var in context.data_context.variables.items()
            if var.notebook_ref == notebook_ref
        ]
        for variable_name in derived_variable_names:
            context.data_context.variables.pop(variable_name, None)

        returned = context.returned_dataset
        if returned is not None and notebook_ref in returned.notebook_refs:
            context.returned_datasets.pop(returned.artifact_id, None)
            if context.active_returned_dataset_id == returned.artifact_id:
                context.active_returned_dataset_id = None
                context.returned_dataset = None
            context.mark_charts_stale_for_dataset(
                dataset_artifact_id=returned.artifact_id,
                latest_version=returned.version + 1,
            )
            return
        for artifact_id, returned_chart in list(context.returned_charts.items()):
            if returned_chart.chart_notebook_ref != notebook_ref:
                continue
            updated_chart = returned_chart.model_copy(
                update={
                    "latest_source_dataset_version": max(
                        returned_chart.latest_source_dataset_version,
                        returned_chart.source_dataset_version + 1,
                    ),
                    "is_stale": True,
                }
            )
            context.returned_charts[artifact_id] = updated_chart
            if context.active_returned_chart_id == artifact_id:
                context.returned_chart = updated_chart


    @toolmethod(
        name="return_dataset",
        description="Return exactly one validated dataset. dataset_variable_name must be a plain variable name. Provide notebook refs for the stages used in the pipeline.",
        parameters_schema={
            "type": "object",
            "properties": {
                "dataset_variable_name": {
                    "type": "string",
                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                    "description": "Plain notebook variable name for the final pandas DataFrame or Series to return. Do not pass expressions, slices, or indexing such as df.head(20) or df[0:20].",
                },
                "title": {
                    "type": "string",
                    "description": "Short, informative display title for the dataset. Focus on what the data represents: domain, entities, time period. Avoid generic suffixes like 'Data' or 'Dataset'. Avoid process qualifiers like 'cleaned', 'processed', or 'merged' unless essential for disambiguation.",
                },
                "notebook_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of notebook names used to produce this dataset, in execution order. Include all notebooks whose code contributes to the final dataset (e.g. retrieval, transformation, validation notebooks).",
                },
                "description": {
                    "type": "string",
                    "description": "One concise sentence describing what this dataset contains and what it is useful for.",
                },
                "notes": {
                    "type": "array",
                    "description": "Important decisions, assumptions, caveats, and validation-relevant context required to interpret and trust the returned dataset.",
                    "items": {"type": "string"},
                },
                "tags": {
                    "type": "array",
                    "description": "Optional context tags you add beyond the data object's tags. Data object tags (from fetch summaries or source datasets) describe what the data is. Your tags describe context or intent—e.g. 'quarterly-review', 'peer-comparison', 'baseline-scenario'. Lowercase, short labels. They are merged with the data object tags.",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "dataset_variable_name",
                "title",
                "description",
                "notes",
            ],
            "additionalProperties": False,
        },
        tool_type="return",
        ui_message="Returning dataset",
    )
    async def return_dataset(
            self,
            *,
            context: AgentContext,
            dataset_variable_name: str,
            title: str,
            description: str,
            notes: list[str],
            tags: list[str] | None = None,
            notebook_refs: list[str] | None = None,
        ) -> Dataset:
        """Declare a DataFrame variable as the session's primary returned dataset (tool)."""
        dataset_variable_name = self._require_plain_variable_name(
            value=dataset_variable_name,
            parameter_name="dataset_variable_name",
        )
        title = title.strip()
        if not title:
            raise ValueError("title must be a non-empty string.")
        notes = TypeAdapter(list[str]).validate_python(notes)
        clean_refs = [(r or "").strip() for r in (notebook_refs or []) if (r or "").strip()]
        self._validate_return_dataset_refs(
            context=context,
            dataset_variable_name=dataset_variable_name,
            notebook_refs=clean_refs,
        )

        # Find the first notebook that produces the dataset variable (contains the variable name in its code)
        producing_ref = None
        for ref in clean_refs:
            if ref in context.notebooks and dataset_variable_name in context.notebooks[ref].code:
                producing_ref = ref
                break
        variable = await self._resolve_returned_dataset(
            context=context,
            dataset_variable_name=dataset_variable_name,
            producing_notebook_ref=producing_ref,
        )
        if variable.output is None:
            raise ValueError(f"Dataset '{dataset_variable_name}' has no tabular payload to return.")

        source_dataset_variable_names = list(variable.source_datasets)
        if not source_dataset_variable_names:
            source_dataset_variable_names = [dataset_variable_name]

        # Collect tags from agent args
        final_tags: list[str] = []
        for t in TypeAdapter(list[str]).validate_python(tags or []):
            s = str(t).strip()
            if s and s not in final_tags:
                final_tags.append(s)

        existing = context.get_returned_dataset()
        if existing and existing.dataset_variable_name != dataset_variable_name:
            raise ValueError(
                f"Session already has a returned dataset '{existing.dataset_variable_name}'. Reuse that dataset or update it in place."
            )

        artifact_id = existing.artifact_id if existing is not None else ""
        version = existing.version if existing is not None else 1
        returned_state = ReturnedDatasetState(
            artifact_id=artifact_id,
            version=version,
            dataset_variable_name=dataset_variable_name,
            title=title,
            description=description,
            notes=notes,
            source_dataset_variable_names=source_dataset_variable_names,
            notebook_refs=clean_refs,
        )
        context.set_returned_dataset(returned_state)

        return Dataset(
            artifact_id=returned_state.artifact_id,
            version=returned_state.version,
            variable_name=dataset_variable_name,
            variable_preview=variable.to_frontend_dict(),
            title=title,
            description=description,
            notes=notes,
            tags=final_tags,
            notebook_refs=clean_refs,
        )

    @toolmethod(
        name="return_chart",
        description="Return one optional chart primitive for an already returned dataset.",
        parameters_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short human-readable title for the chart, shown in the UI.",
                },
                "source_dataset_variable_name": {
                    "type": "string",
                    "description": "Variable name of the already returned dataset this chart visualizes.",
                },
                "chart_variable_name": {
                    "type": "string",
                    "description": "Variable name that resolves to an Altair chart built from the source dataset.",
                },
                "chart_notebook_ref": {
                    "type": "string",
                    "description": "Notebook ref for the visualization stage.",
                },
                "description": {
                    "type": "string",
                    "description": "One concise sentence describing what the chart helps the user see.",
                },
                "notes": {
                    "type": "array",
                    "description": "Important caveats, encoding decisions, and interpretation notes for the chart.",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "source_dataset_variable_name",
                "chart_variable_name",
                "chart_notebook_ref",
                "description",
                "notes",
            ],
            "additionalProperties": False,
        },
        tool_type="return",
        ui_message="Returning chart",
    )
    async def return_chart(
        self,
        *,
        context: AgentContext,
        title: str = "",
        source_dataset_variable_name: str,
        chart_variable_name: str,
        chart_notebook_ref: str,
        description: str,
        notes: list[str],
    ) -> Chart:
        """Declare an Altair chart variable as the session's primary returned chart (tool)."""
        source_dataset_variable_name = self._require_plain_variable_name(
            value=source_dataset_variable_name,
            parameter_name="source_dataset_variable_name",
        )
        chart_variable_name = self._require_plain_variable_name(
            value=chart_variable_name,
            parameter_name="chart_variable_name",
        )
        returned_state = context.get_returned_dataset()
        if returned_state is None:
            raise ValueError("No returned dataset is available. Call return_dataset before return_chart.")
        if returned_state.dataset_variable_name != source_dataset_variable_name:
            raise ValueError(
                "Chart source must match the returned dataset variable "
                f"'{returned_state.dataset_variable_name}'."
            )
        notes = TypeAdapter(list[str]).validate_python(notes)
        chart_ref = (chart_notebook_ref or "").strip()
        if not chart_ref:
            raise ValueError("chart_notebook_ref must be a non-empty string.")
        self._validate_return_chart_refs(
            context=context,
            source_dataset_variable_name=source_dataset_variable_name,
            chart_variable_name=chart_variable_name,
            chart_notebook_ref=chart_ref,
        )

        fig_obj = await self.code_executor.get(chart_variable_name)
        if not isinstance(fig_obj, FigureObject):
            raise ValueError("chart_variable_name must resolve to an Altair chart.")
        if fig_obj.name is None:
            fig_obj.name = chart_variable_name

        existing_chart = context.get_returned_chart()
        artifact_id = (
            existing_chart.artifact_id
            if existing_chart is not None
            and existing_chart.source_dataset_artifact_id == returned_state.artifact_id
            and existing_chart.chart_variable_name == chart_variable_name
            and existing_chart.chart_notebook_ref == chart_ref
            else ""
        )
        version = existing_chart.version if artifact_id else 1
        now = datetime.now(UTC)
        returned_chart = ReturnedChartState(
            artifact_id=artifact_id,
            version=version,
            title=(title or "").strip(),
            source_dataset_artifact_id=returned_state.artifact_id,
            source_dataset_variable_name=source_dataset_variable_name,
            source_dataset_version=returned_state.version,
            latest_source_dataset_version=returned_state.version,
            is_stale=False,
            chart_variable_name=chart_variable_name,
            chart_notebook_ref=chart_ref,
            description=description,
            notes=notes,
            last_refreshed_at=now,
        )
        context.set_returned_chart(returned_chart)

        return Chart(
            artifact_id=returned_chart.artifact_id,
            version=returned_chart.version,
            title=(title or "").strip(),
            source_dataset_artifact_id=returned_state.artifact_id,
            source_dataset_variable_name=source_dataset_variable_name,
            source_dataset_version=returned_state.version,
            latest_source_dataset_version=returned_state.version,
            is_stale=False,
            chart_variable_name=chart_variable_name,
            figure=fig_obj,
            chart_notebook_ref=chart_ref,
            description=description,
            notes=notes,
            last_refreshed_at=now,
        )

    @staticmethod
    def _require_plain_variable_name(*, value: str, parameter_name: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError(f"{parameter_name} must be a non-empty string.")
        if not normalized.isidentifier():
            raise ValueError(
                f"{parameter_name} must be a plain variable name, not an expression or slice. "
                f"Got '{value}'."
            )
        return normalized

    def _validate_return_dataset_refs(
        self,
        *,
        context: AgentContext,
        dataset_variable_name: str,
        notebook_refs: list[str],
    ) -> None:
        # No notebook refs at all — direct return from data context
        if not notebook_refs:
            if dataset_variable_name not in context.data_context:
                raise ValueError(
                    "When returning source data as-is without notebooks, the dataset variable must already exist "
                    f"in the data context (e.g. from a fetch). Variable '{dataset_variable_name}' was not found."
                )
            return

        # Validate each notebook ref exists and has no errors
        for ref in notebook_refs:
            if ref not in context.notebooks:
                raise ValueError(f"Notebook '{ref}' was not found.")
            notebook = context.notebooks[ref]
            if notebook.has_errors():
                raise ValueError(f"Notebook '{ref}' has execution errors.")

    def _validate_return_chart_refs(
        self,
        *,
        context: AgentContext,
        source_dataset_variable_name: str,
        chart_variable_name: str,
        chart_notebook_ref: str,
    ) -> None:
        if chart_notebook_ref not in context.notebooks:
            raise ValueError(f"Chart notebook '{chart_notebook_ref}' was not found.")
        chart_notebook = context.notebooks[chart_notebook_ref]
        if chart_notebook.has_errors():
            raise ValueError(f"Chart notebook '{chart_notebook_ref}' has execution errors.")
        if source_dataset_variable_name not in chart_notebook.code:
            raise ValueError(
                f"Chart notebook '{chart_notebook_ref}' must reference dataset '{source_dataset_variable_name}'."
            )
        if chart_variable_name not in chart_notebook.code:
            raise ValueError(
                f"Chart notebook '{chart_notebook_ref}' must define chart variable '{chart_variable_name}'."
            )

    async def _resolve_returned_dataset(
        self,
        *,
        context: AgentContext,
        dataset_variable_name: str,
        producing_notebook_ref: str,
    ) -> Variable:
        if dataset_variable_name in context.data_context:
            existing = context.data_context[dataset_variable_name]
            if not existing.is_tabular or existing.output is None:
                raise ValueError(
                    f"Variable '{dataset_variable_name}' must resolve to a pandas DataFrame or Series."
                )
            if not producing_notebook_ref:
                return existing
            if existing.source in {"dataset", "artifact"}:
                artifact_var = await self._materialize_artifact(
                    context=context,
                    variable_name=dataset_variable_name,
                    producing_notebook_ref=producing_notebook_ref,
                )
                if not artifact_var.is_tabular or artifact_var.output is None:
                    raise ValueError(
                        f"Variable '{dataset_variable_name}' must resolve to a pandas DataFrame or Series."
                    )
            else:
                source_datasets = self._extract_source_datasets(
                    context=context,
                    notebook_name=producing_notebook_ref,
                    exclude_variable_name=dataset_variable_name,
                )
                artifact_var = Variable(
                    name=existing.name,
                    output=existing.output,
                    source="dataset",
                    source_description="Dataset prepared from an existing variable via the notebook pipeline.",
                    source_datasets=source_datasets or [dataset_variable_name],
                    notebook_ref=producing_notebook_ref or None,
                    hidden=existing.hidden,
                )
            context.data_context[dataset_variable_name] = artifact_var
            return artifact_var

        artifact_var = await self._materialize_artifact(
            context=context,
            variable_name=dataset_variable_name,
            producing_notebook_ref=producing_notebook_ref,
        )
        if not artifact_var.is_tabular or artifact_var.output is None:
            raise ValueError(
                f"Variable '{dataset_variable_name}' must resolve to a pandas DataFrame or Series."
            )
        context.data_context.extend([artifact_var])
        return artifact_var

    async def refresh_returned_dataset(
        self,
        *,
        context: AgentContext,
        artifact_id: str | None = None,
    ) -> Dataset:
        """Re-execute notebooks and bump the version of the returned dataset.

        Used to propagate upstream data changes to an already-returned artifact.
        """
        returned_state = context.get_returned_dataset(artifact_id)
        if returned_state is None:
            raise ValueError("No returned dataset is available to refresh.")
        if not returned_state.title:
            raise ValueError("The returned dataset is missing refreshable title metadata.")

        context.bump_state_version()
        await self.code_executor.replace_state(context.data_context)
        await context.execute_notebooks(code_executor=self.code_executor, update_outputs=True)
        await self.code_executor.set_sandbox_state_version(context.state_version)

        refreshed = await self.return_dataset(
            context=context,
            dataset_variable_name=returned_state.dataset_variable_name,
            title=returned_state.title,
            description=returned_state.description,
            notes=returned_state.notes,
            notebook_refs=returned_state.notebook_refs,
        )
        if not refreshed.success or refreshed.data is None:
            raise ValueError(refreshed.exception_message or "Dataset refresh failed.")
        refreshed_state = context.get_returned_dataset()
        if refreshed_state is None:
            raise ValueError("Dataset refresh completed without refresh state.")
        next_version = returned_state.version + 1
        refreshed_state = refreshed_state.model_copy(
            update={
                "artifact_id": returned_state.artifact_id,
                "version": next_version,
            }
        )
        context.set_returned_dataset(refreshed_state)
        context.mark_charts_stale_for_dataset(
            dataset_artifact_id=returned_state.artifact_id,
            latest_version=next_version,
        )
        refreshed_artifact = refreshed.data.model_copy(
            update={
                "artifact_id": returned_state.artifact_id,
                "version": next_version,
            }
        )
        return refreshed_artifact

    async def refresh_returned_chart(
        self,
        *,
        context: AgentContext,
        artifact_id: str | None = None,
    ) -> Chart:
        """Re-render the returned chart from updated notebook state.

        Should be called after the source dataset has been refreshed.
        """
        returned_chart = context.get_returned_chart(artifact_id)
        if returned_chart is None:
            raise ValueError("No returned chart is available to refresh.")
        returned_state = context.get_returned_dataset(returned_chart.source_dataset_artifact_id)
        if returned_state is None:
            raise ValueError("No returned dataset is available to refresh the chart against.")

        chart_ref = (returned_chart.chart_notebook_ref or "").strip()
        if not chart_ref:
            raise ValueError("The returned chart is missing refreshable notebook metadata.")

        if returned_chart.source_dataset_artifact_id != returned_state.artifact_id:
            raise ValueError(
                "Chart source must match the returned dataset artifact "
                f"'{returned_state.artifact_id}'."
            )

        self._validate_return_chart_refs(
            context=context,
            source_dataset_variable_name=returned_chart.source_dataset_variable_name,
            chart_variable_name=returned_chart.chart_variable_name,
            chart_notebook_ref=chart_ref,
        )

        context.bump_state_version()
        await self.code_executor.replace_state(context.data_context)
        context.active_notebook_name = chart_ref
        chart_notebook = context.notebooks[chart_ref]
        await chart_notebook.execute(code_executor=self.code_executor, update_outputs=True)
        await self.code_executor.set_sandbox_state_version(context.state_version)

        fig_obj = await self.code_executor.get(returned_chart.chart_variable_name)
        if not isinstance(fig_obj, FigureObject):
            raise ValueError("chart_variable_name must resolve to an Altair chart.")
        if fig_obj.name is None:
            fig_obj.name = returned_chart.chart_variable_name

        now = datetime.now(UTC)
        next_version = returned_chart.version + 1
        updated_chart_state = returned_chart.model_copy(
            update={
                "version": next_version,
                "source_dataset_artifact_id": returned_state.artifact_id,
                "source_dataset_variable_name": returned_state.dataset_variable_name,
                "source_dataset_version": returned_state.version,
                "latest_source_dataset_version": returned_state.version,
                "is_stale": False,
                "last_refreshed_at": now,
            }
        )
        context.set_returned_chart(updated_chart_state)

        return Chart(
            artifact_id=updated_chart_state.artifact_id,
            version=updated_chart_state.version,
            title=updated_chart_state.title,
            source_dataset_artifact_id=updated_chart_state.source_dataset_artifact_id,
            source_dataset_variable_name=updated_chart_state.source_dataset_variable_name,
            source_dataset_version=updated_chart_state.source_dataset_version,
            latest_source_dataset_version=updated_chart_state.latest_source_dataset_version,
            is_stale=False,
            chart_variable_name=updated_chart_state.chart_variable_name,
            figure=fig_obj,
            chart_notebook_ref=chart_ref,
            description=updated_chart_state.description,
            notes=updated_chart_state.notes,
            last_refreshed_at=now,
        )

    def _extract_source_datasets(
        self,
        *,
        context: AgentContext,
        notebook_name: str,
        exclude_variable_name: str | None = None,
    ) -> list[str]:
        if notebook_name not in context.notebooks:
            return []
        notebook_code = context.notebooks[notebook_name].code
        source_variables = []
        for variable_name in context.data_context.variables.keys():
            if exclude_variable_name is not None and variable_name == exclude_variable_name:
                continue
            if re.search(rf"\b{re.escape(variable_name)}\b", notebook_code):
                source_variables.append(variable_name)
        return source_variables

    async def _materialize_artifact(
        self,
        *,
        context: AgentContext,
        variable_name: str,
        producing_notebook_ref: str | None = None,
    ) -> Variable:
        value = await self.code_executor.get(variable_name)
        notebook_ref = producing_notebook_ref or context.active_notebook_name
        source_datasets = self._extract_source_datasets(
            context=context,
            notebook_name=notebook_ref,
            exclude_variable_name=variable_name,
        )

        if isinstance(value, DataFrameObject):
            materialized_df = value.value.copy(deep=True)
            output = self._output_factory.from_value(
                materialized_df,
                ref=context._get_ref_name(key=variable_name, subdir="dataset"),
            )
            return Variable(
                name=variable_name,
                output=output,
                source="artifact",
                source_description="Dataset prepared in the notebook pipeline.",
                source_datasets=source_datasets,
                notebook_ref=notebook_ref,
            )
        if isinstance(value, PrimitiveObject):
            return Variable(
                name=variable_name,
                output=value,
                source="artifact",
                source_description="Dataset prepared in the notebook pipeline.",
                source_datasets=source_datasets,
                notebook_ref=notebook_ref,
            )
        if isinstance(value, ExceptionObject):
            raise ValueError(f"Variable '{variable_name}' failed during evaluation: {value.value}")
        raise ValueError(
            f"Variable '{variable_name}' is not a supported data object type (expected DataFrame, Series, or primitive; got {type(value)})."
        )
