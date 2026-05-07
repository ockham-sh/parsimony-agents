from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import tempfile
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import litellm
import pandas as pd
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from opentelemetry import trace
from parsimony.connector import Connectors
from parsimony.result import Provenance, Result
from pydantic import TypeAdapter

from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.config import AgentGuardrails, FileStore
from parsimony_agents.agent.events import (
    AgentError,
    RunCancelled,
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
from parsimony_agents.agent.outputs import (
    ArtifactLlmResult,
    SystemToolMessage,
    SystemToolOutput,
    UtilityToolOutput,
)
from parsimony_agents.agent.tracing import trace_tool_execution
from parsimony_agents.artifacts import (
    Chart,
    Dataset,
    snapshot_path,
)
from parsimony_agents.execution import (
    DataFrameObject,
    ExceptionObject,
    FigureObject,
    PrimitiveObject,
    StringPaginator,
)
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.execution.executor import BaseCodeExecutor
from parsimony_agents.execution.factory import OutputFactory as FrameworkOutputFactory
from parsimony_agents.execution.parquet_helpers import parquet_summary
from parsimony_agents.messages import Message, Text, blocks_to_text

# Tool message for cooperative cancellation; keeps one tool output per tool call id.
CANCELLED_TOOL_TEXT = "Cancelled by user before the tool completed."
from parsimony_agents.notebook import Script, ScriptPreview, stamp_fetch_log_to_script
from parsimony_agents.notebook_io import deserialize_notebook, serialize_notebook
from parsimony_agents.rag.keyword_store import get_or_create_session_keyword_store
from parsimony_agents.rag.vector_store import get_or_create_session_vector_store
from parsimony_agents.tools import ToolMethod, Tools, toolmethod
from parsimony_agents.views import get_llm_view_defaults

logger = logging.getLogger("parsimony_agents")
error_logger = logging.getLogger("parsimony_agents.errors")

litellm.REPEATED_STREAMING_CHUNK_LIMIT = 100  # TODO: Monitor how many repeated chunks appear naturally before hitting the limit

# Read-only system tools that never touch the CodeExecutor; safe to run concurrently.
# All other tool names require sequential execution in one batch to avoid re-entrant
# kernel or workspace races (see ``_tool_batch_allows_concurrent``).
_TOOL_NAMES_SAFE_CONCURRENT: frozenset[str] = frozenset(
    {
        "get_context",
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


def _serialize_and_hash_object(obj: Any) -> int:
    return hash(json.dumps(obj, sort_keys=True))


async def _resolve_sources_from_variables(
    code_executor: Any, var_names: list[str]
) -> list[Provenance]:
    """Read each named kernel variable's ``.provenance`` and return the list."""
    if not var_names:
        return []

    names_literal = repr(list(var_names))
    expr = (
        "__import__('json').dumps(["
        "(lambda _v, _n: {"
        "'name': _n, "
        "'provenance': (_v.provenance.model_dump(mode='json') "
        "if hasattr(_v, 'provenance') and _v is not None else None)"
        "})(globals().get(_n), _n) "
        f"for _n in {names_literal}"
        "])"
    )
    ko = await code_executor.eval(expr)
    if not ko.outputs:
        raise RuntimeError("eval returned no output while resolving sources_from_variables.")
    payload_obj = ko.outputs[0]
    if getattr(payload_obj, "type", None) == "exception":
        raise RuntimeError(
            f"eval failed while resolving sources_from_variables: {payload_obj.value!r}"
        )
    raw_value = getattr(payload_obj, "value", None)
    if not isinstance(raw_value, str):
        raise RuntimeError(
            f"eval returned unexpected output type for sources_from_variables: {type(payload_obj).__name__}"
        )
    entries = json.loads(raw_value)

    provenances: list[Provenance] = []
    for entry in entries:
        name = entry["name"]
        prov_dict = entry.get("provenance")
        if prov_dict is None:
            raise ValueError(
                f"sources_from_variables: '{name}' is not in the kernel or "
                "does not hold a connector Result. Pass variable names that "
                "received the return value of a connector fetch (e.g. "
                "``raw = await connectors[\"fred_fetch\"](...)``)."
            )
        provenances.append(Provenance.model_validate(prov_dict))
    return provenances


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
    """Returned :class:`Dataset` objects keyed by ``artifact_id``."""

    charts: dict[str, Chart] = field(default_factory=dict)
    """Returned :class:`Chart` objects keyed by ``artifact_id``."""

    code: dict[str, Script] = field(default_factory=dict)
    """Returned :class:`Script` objects keyed by notebook path (execution order preserved)."""

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
        elif etype == "tool_event" and getattr(event, "completed", False):
            result = getattr(event, "result", None)
            if result is None:
                return
            if isinstance(result, Dataset) and result.artifact_id:
                self.datasets[result.artifact_id] = result
            elif isinstance(result, Chart) and result.artifact_id:
                self.charts[result.artifact_id] = result


class Agent:
    """Data analysis agent: LLM loop, tools, and code execution (yields AgentEvent).

    **Quick start (OSS users):**

    .. code-block:: python

        from parsimony_agents import Agent
        from parsimony_fred import CONNECTORS as FRED

        agent = Agent(model="claude-sonnet-4-6", connectors=FRED.bind(api_key="..."))
        result = await agent.ask("Show me US GDP trends")
        print(result.text, result.datasets)

    **Power usage (product / full control):**

    Pass explicit ``model_config``, ``instructions``, ``code_executor``, and
    ``output_factory`` for complete control over the agent configuration.
    """

    RETURN_TOOLS = ("return_dataset", "return_chart")
    CODE_TOOL_NAMES = {"code_set", "code_edit", "dry_execute_code", "execute"}

    def __init__(
        self,
        *,
        # --- Convenience params (OSS front door) ---
        model: str | None = None,
        api_key: str | None = None,
        connectors: Any | None = None,
        # --- Explicit params (product / power usage) ---
        model_config: dict[str, Any] | None = None,
        instructions: str | None = None,
        code_executor: BaseCodeExecutor | None = None,
        output_factory: FrameworkOutputFactory | None = None,
        guardrails: AgentGuardrails = AgentGuardrails(),
        session_id: str | None = None,
        file_store: FileStore | None = None,
        read_artifact_fn: Callable[[str, dict[str, Any]], Awaitable[ArtifactLlmResult]] | None = None,
    ):
        from parsimony_agents.agent.prompts import DEFAULT_DATA_ANALYSIS_PROMPT
        from parsimony_agents.execution.executor import CodeExecutor as _LocalExecutor

        # Resolve model_config: explicit > built from model= convenience param
        if model_config is not None:
            resolved_config: dict[str, Any] = model_config
        elif model is not None:
            resolved_config = {"model": model, **({"api_key": api_key} if api_key else {})}
        else:
            raise TypeError(
                "Agent requires either model_config={...} or model='model-name'"
            )

        # Resolve instructions: explicit > default prompt. The connector catalog
        # is *not* appended here — connectors live in the executor namespace and
        # are advertised per-turn via AgentContextSnapshot.connectors_catalog,
        # so the system prompt stays stable and cache-friendly.
        resolved_instructions = instructions if instructions is not None else DEFAULT_DATA_ANALYSIS_PROMPT
        if connectors is not None and not isinstance(connectors, (Connectors, Mapping)):
            raise TypeError(
                "connectors must be a Connectors or Mapping[str, Connectors]; "
                f"got {type(connectors).__name__}"
            )

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
        self._read_artifact_fn = read_artifact_fn

        _system_tool_methods: list[ToolMethod] = [
            self.code_set,
            self.code_edit,
            self.dry_execute_code,
            self.write_file,
            self.edit_file,
            self.execute_workspace_tool,
            self.read_file,
            self.read_data,
        ]
        if read_artifact_fn is not None:
            _system_tool_methods.append(self.read_artifact)
        _system_tool_methods.extend(
            [
                self.list_files,
                self.run_notebook,
                self.restart_kernel,
                self.return_dataset,
                self.return_chart,
                self.output_read,
                self.output_search,
                self.get_context,
            ]
        )
        self.system_tools = Tools(_system_tool_methods)

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
        raw_timeout = raw_args.get("timeout_seconds", 120.0)
        try:
            requested = float(raw_timeout)
        except (TypeError, ValueError):
            requested = 120.0
        if requested <= 0:
            requested = 120.0
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

    @staticmethod
    def _resolve_code_tool_path(tool_args: dict[str, Any]) -> str | None:
        """Pull the canonical ``path`` parameter from a code tool's args.

        Returns the trimmed path, or ``None`` when ``path`` is absent or
        blank — the loop treats that as a contract error and replies with a
        tool message instructing the agent to provide ``path``.
        """
        raw_path = tool_args.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            return raw_path.strip()
        return None

    async def _handle_llm_error(
        self,
        last_exception: Exception,
        text_message_id: str,
        turn_state: TurnState,
        agent_span: Any,
    ) -> AsyncIterator[AgentError | TextDeltaEvent]:
        """Classify an LLM exception into a typed ``AgentError`` plus a
        user-facing ``TextDeltaEvent``, and record it on the active span.

        The branches are exhaustive — the trailing ``else`` is the catch-all
        for unexpected provider errors. Each branch emits exactly the same
        two-event shape so the caller always yields a uniform error frame.
        """
        section = "final_response" if turn_state.final_response_started else "analysis"
        model_name = self.model_config.get("model", "the configured model")

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
        elif isinstance(last_exception, AuthenticationError):
            error_logger.error("LLM authentication failed: %s", last_exception, exc_info=True)
            detail = str(last_exception).splitlines()[0] if str(last_exception) else ""
            yield AgentError(
                message=f"Authentication failed for {model_name}: {detail}",
                recoverable=False,
                error_type="authentication",
                section=section,
            )
            yield TextDeltaEvent(
                content=(
                    f"Authentication failed for `{model_name}`. "
                    "Check that the required API key environment variable is set "
                    "(e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) "
                    "and is valid for the selected model provider."
                ),
                message_id=text_message_id,
                delta=False,
                section="final_response",
            )
        elif isinstance(last_exception, (BadRequestError, NotFoundError)):
            error_logger.error("LLM bad request: %s", last_exception, exc_info=True)
            detail = str(last_exception).splitlines()[0] if str(last_exception) else ""
            yield AgentError(
                message=f"Invalid request to {model_name}: {detail}",
                recoverable=False,
                error_type="bad_request",
                section=section,
            )
            yield TextDeltaEvent(
                content=(
                    f"The request to `{model_name}` was rejected by the provider. "
                    "This usually means the model name is invalid, unavailable in your region, "
                    f"or the request payload is malformed. Provider said: {detail}"
                ),
                message_id=text_message_id,
                delta=False,
                section="final_response",
            )
        elif isinstance(last_exception, APIConnectionError):
            error_logger.error("LLM connection error: %s", last_exception, exc_info=True)
            yield AgentError(
                message=f"Could not reach the model provider: {last_exception}",
                recoverable=False,
                error_type="connection",
                section=section,
            )
            yield TextDeltaEvent(
                content=(
                    "Could not connect to the AI model provider. "
                    "Check your network connection and try again."
                ),
                message_id=text_message_id,
                delta=False,
                section="final_response",
            )
        else:
            error_logger.error(
                "LLM error (%s): %s", type(last_exception).__name__, last_exception, exc_info=True
            )
            detail = str(last_exception).splitlines()[0] if str(last_exception) else type(last_exception).__name__
            yield AgentError(
                message=f"LLM call failed ({type(last_exception).__name__}): {detail}",
                recoverable=False,
                error_type="llm_error",
                section=section,
            )
            yield TextDeltaEvent(
                content=(
                    f"An error occurred while communicating with the AI model "
                    f"({type(last_exception).__name__}): {detail}"
                ),
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
        cancellation: CancellationRequest | None = None,
    ) -> AsyncGenerator[Any, None]:
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

        await self._setup_connectors()

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

        context_snapshot = await ctx.to_snapshot(connectors=self._connectors)

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
                    user_broke_stream = False
                    async for chunk in response:
                        if cancellation and cancellation.is_set():
                            user_broke_stream = True
                            break
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
                            # Utility tools: the card appears with full metadata ("In") once
                            # args are complete (Phase 1 planning, below), not here during
                            # streaming when args are only partially known. This produces two
                            # clean file writes: args-known → completed, matching the two
                            # visible chunks the UI needs (In, then In+Out).
                            if tool_type == "system" and loading_label is not None:
                                yield ToolEvent(
                                    tool_name=name,
                                    tool_call_id=tool_call_id,
                                    tool_type="system",
                                    completed=False,
                                    result=SystemToolMessage(message=loading_label),
                                    ui_message=loading_label,
                                    section="final_response" if turn_state.final_response_started else "analysis",
                                )

                    if user_broke_stream and cancellation is not None:
                        yield RunCancelled(
                            message="Generation was cancelled before the assistant message completed.",
                            reason=cancellation.reason,
                            section="final_response" if turn_state.final_response_started else "analysis",
                        )
                        turn_state.stopped = True
                        last_exception = None
                        break

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
                    error_logger.error(
                        f"Unexpected error calling LLM: {str(e)}",
                        exc_info=True
                    )
                    break

            if turn_state.stopped:
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

            for tool_call in (response_message.tool_calls or []):
                _new_tool_calls.append(tool_call)
                if tool_call.function.name in self._CODE_EDIT_TOOL_NAMES:
                    # Notebook may run in the same call when ``execute`` is true.
                    pass

            if len(_new_tool_calls) == 0:
                turn_state.stopped = True
                turn_state.final_response_started = True

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

                    if tool_name == "dry_execute_code" and not (
                        isinstance(llm_ui_message, str) and llm_ui_message.strip()
                    ):
                        ctx.messages.append(
                            AgentMessage(
                                role="tool",
                                content=(
                                    "dry_execute_code requires a non-empty _ui_message: one short, plain-language, "
                                    "past-tense line describing what this run does for the user (e.g. 'Previewed a rolling mean')."
                                ),
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )
                        continue

                    # Loop detection: same tool + same args called repeatedly.
                    args_hash = _serialize_and_hash_object(tool_args)
                    call_signature = (tool_name, args_hash)
                    tool_timeout_s = self._tool_timeout_seconds(tool_name, tool_args)
                    repeat_count = sum(1 for sig in tool_call_history if sig == call_signature)

                    if repeat_count == 6:
                        logger.warning(f"Loop detected: {tool_name} called {repeat_count + 1} times with same args", extra={
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
                        logger.warning(f"Loop suspected: {tool_name} called {repeat_count + 1} times with same args", extra={
                            "tool_name": tool_name,
                            "repeat_count": repeat_count + 1
                        })

                    tool_type = tools[tool_name].tool_type

                    if tool_type == "code":
                        notebook_path = self._resolve_code_tool_path(tool_args)
                        if notebook_path is None:
                            ctx.messages.append(
                                AgentMessage(
                                    role="tool",
                                    content="code_set and code_edit require a non-empty 'path'. Provide a workspace path like 'notebooks/inflation_analysis.py'.",
                                    name=tool_name,
                                    tool_call_id=tool_call.id,
                                )
                            )
                            continue

                    loading_label = tools[tool_name].ui_message
                    if tool_type == "code":
                        if tool_name == "code_set" and tool_args.get("execute") is True:
                            loading_label = "Writing and running notebook"
                        elif tool_name == "code_edit" and tool_args.get("execute") is True:
                            loading_label = "Editing and running notebook"
                        if tool_name in ("code_set", "code_edit"):
                            preview = ScriptPreview(
                                path=notebook_path,
                                code=tool_args.get("code", "") or "",
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
                            section="final_response" if turn_state.final_response_started else "analysis",
                        )
                    elif tool_type == "utility":
                        # Emit a second file write with the full tool args in metadata,
                        # before execution starts. The loading event (Phase 1 streaming)
                        # already wrote the file with just ui_message; this overwrites it
                        # with metadata so the frontend can display "In [·]: <code>" while
                        # the tool is running, before the output ("Out") is known.
                        yield ToolEvent(
                            tool_name=tool_name,
                            tool_call_id=tool_call.id,
                            tool_type="utility",
                            completed=False,
                            result=UtilityToolOutput(
                                ui_message=loading_label or f"Executing {tool_name}",
                                metadata=self._utility_tool_metadata(
                                    tool_name=tool_name,
                                    tool_description=tools[tool_name].ui_description or tools[tool_name].description,
                                    tool_args=tool_args,
                                ),
                            ),
                            section="final_response" if turn_state.final_response_started else "analysis",
                        )

                    tool_executions.append((
                        tool_call,
                        call_signature,
                        repeat_count,
                        _build_coroutine(
                            tools[tool_name],
                            {**tool_args, "context": ctx},
                            tool_name,
                            tool_type,
                            tool_args,
                            tool_timeout_s,
                        ),
                    ))

                if tool_executions and cancellation and cancellation.is_set():
                    for tool_call, call_signature, repeat_count, _ in tool_executions:
                        tool_name = tool_call.function.name
                        if tool_name not in tools:
                            continue
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_args.pop("_ui_message", None)
                        for tev in self._emit_cancelled_tool_events(
                            tools, tool_name, tool_args, tool_call, turn_state
                        ):
                            yield tev
                        ctx.messages.append(
                            AgentMessage(
                                role="tool",
                                content=CANCELLED_TOOL_TEXT,
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )
                        tool_call_history.append(call_signature)
                    yield RunCancelled(
                        message="The run was cancelled before the remaining tools could finish.",
                        reason=cancellation.reason,  # cancellation is not None here
                        section="final_response" if turn_state.final_response_started else "analysis",
                    )
                    turn_state.stopped = True
                    yield StateSnapshot(context=ctx.model_copy(deep=False), section="analysis")
                    break

                # ── Phase 2: execute tool coroutines ───────────────────────────────────────
                # Batches of read-only tools may run concurrently. Any batch with a
                # return tool, a kernel/executor tool, a workspace-mutating tool, etc.
                # runs sequentially in tool-call order. See
                # ``_TOOL_NAMES_SAFE_CONCURRENT`` and ``_tool_batch_allows_concurrent``.
                batch_names = [t.function.name for t, _, _, _ in tool_executions]
                concurrent_batch = _tool_batch_allows_concurrent(batch_names, tools)
                raw_results = await self._run_tool_coros_with_cancellation(
                    tool_executions, cancellation, concurrent_batch
                )

                # ── Phase 3: flush results sequentially into shared state ────────────────
                # Results are committed in the original tool-call order so the LLM sees a
                # well-formed message sequence, and ctx mutations stay serialised.
                for (tool_call, call_signature, repeat_count, _), raw_result in zip(
                    tool_executions, raw_results, strict=True
                ):
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    # Re-extract _ui_message (already stripped during Phase 1,
                    # but we re-parse from raw arguments here).
                    llm_ui_message = tool_args.pop("_ui_message", None)

                    if isinstance(raw_result, asyncio.CancelledError):
                        for tev in self._emit_cancelled_tool_events(
                            tools, tool_name, tool_args, tool_call, turn_state
                        ):
                            yield tev
                        ctx.messages.append(
                            AgentMessage(
                                role="tool",
                                content=CANCELLED_TOOL_TEXT,
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )
                        tool_call_history.append(call_signature)
                        continue

                    if isinstance(raw_result, Exception):
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
                                    turn_state.stopped = True
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
                                notebook_path = self._resolve_code_tool_path(tool_args)
                                if notebook_path is None:
                                    raise ValueError("code tools require a non-empty 'path'.")
                                ran_kernel = tool_name == "run_notebook" or (
                                    tool_name in ("code_set", "code_edit") and tool_args.get("execute") is True
                                )
                                also_executed = tool_name in ("code_set", "code_edit") and tool_args.get(
                                    "execute"
                                ) is True
                                if ran_kernel:
                                    ko = tool_result.data
                                    if not isinstance(ko, KernelOutput):
                                        raise TypeError(
                                            f"{tool_name} with kernel run did not return KernelOutput"
                                        )
                                    raw = await self.code_executor.read_workspace_file(notebook_path)
                                    script = deserialize_notebook(raw, path=notebook_path)
                                    script.output = ko
                                    script.data_objects = stamp_fetch_log_to_script(ko)
                                    notebook = script
                                    preview = script.to_preview()
                                else:
                                    # code_set or code_edit — re-read the written file (no execution)
                                    raw = await self.code_executor.read_workspace_file(notebook_path)
                                    script = deserialize_notebook(raw, path=notebook_path)
                                    notebook = script
                                    preview = script.to_preview()
                                if tool_name == "code_set":
                                    preview.ui_message = (llm_ui_message or "").strip() or None
                                else:
                                    # run_notebook, code_edit: no optional detail in the file-ref line on the wire.
                                    preview.ui_message = None

                                yield ToolEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call.id,
                                    tool_type="code",
                                    completed=True,
                                    result={"notebook": notebook, "preview": preview},
                                    ui_message_completed=llm_ui_message,
                                    also_executed=also_executed,
                                    section="final_response" if turn_state.final_response_started else "analysis",
                                )

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

                if any(isinstance(r, asyncio.CancelledError) for r in raw_results) and not turn_state.stopped:
                    cancel_reason = cancellation.reason if cancellation is not None else "user_request"
                    yield RunCancelled(
                        message="The run was cancelled while tools were executing.",
                        reason=cancel_reason,
                        section="final_response" if turn_state.final_response_started else "analysis",
                    )
                    turn_state.stopped = True

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
            f"Agent run completed in {total_time:.3f}s after {iteration_count} iterations",
            extra={
                "duration_s": total_time,
                "iterations": iteration_count
            }
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

        yield StateSnapshot(context=ctx.model_copy(deep=False), section="analysis")

    async def _run_tool_coros_with_cancellation(
        self,
        tool_executions: list,
        cancellation: CancellationRequest | None,
        concurrent_batch: bool,
    ) -> list[object]:
        if not tool_executions:
            return []
        if cancellation and cancellation.is_set():
            return [asyncio.CancelledError() for _ in tool_executions]
        coros = [te[3] for te in tool_executions]
        if not concurrent_batch:
            out: list[object] = []
            for c in coros:
                if cancellation and cancellation.is_set():
                    return out + [asyncio.CancelledError() for _ in coros[len(out) :]]
                try:
                    r = await c
                    out.append(r)
                except BaseException as exc:  # noqa: BLE001
                    out.append(exc)
            return out
        t_tasks = [asyncio.create_task(c) for c in coros]

        async def _gather_tool_tasks() -> list[object]:
            return list(await asyncio.gather(*t_tasks, return_exceptions=True))

        t_gather = asyncio.create_task(_gather_tool_tasks())
        if cancellation is None:
            return list(await t_gather)
        t_cancel = asyncio.create_task(cancellation.event.wait())
        d, _ = await asyncio.wait({t_gather, t_cancel}, return_when=asyncio.FIRST_COMPLETED)
        if t_cancel in d:
            t_gather.cancel()
            for tt in t_tasks:
                if not tt.done():
                    tt.cancel()
            for tt in t_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await tt
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t_gather
            return [asyncio.CancelledError() for _ in coros]
        t_cancel.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t_cancel
        return list(await t_gather)

    def _emit_cancelled_tool_events(
        self,
        tools: Tools,
        tool_name: str,
        tool_args: dict,
        tool_call: Any,
        turn_state: TurnState,
    ) -> list[ToolEvent]:
        ttype = tools[tool_name].tool_type
        section = "final_response" if turn_state.final_response_started else "analysis"
        msg = CANCELLED_TOOL_TEXT
        if ttype == "utility":
            ui = tools[tool_name].ui_message or f"Executing {tool_name}"
            uic = None  # completed message when cancelling (optional)
            uo = UtilityToolOutput(
                ui_message=ui,
                ui_message_completed=uic,
                metadata=self._utility_tool_metadata(
                    tool_name=tool_name,
                    tool_description=tools[tool_name].ui_description or tools[tool_name].description,
                    tool_args=tool_args,
                ),
                content=Text(content=msg),
            )
            return [
                ToolEvent(
                    tool_name=tool_name,
                    tool_call_id=tool_call.id,
                    tool_type="utility",
                    completed=True,
                    result=uo,
                    ui_message_completed=uic,
                    section=section,
                )
            ]
        if ttype == "system":
            if tools[tool_name].ui_message is not None or tools[tool_name].ui_message_completed is not None:
                return [
                    ToolEvent(
                        tool_name=tool_name,
                        tool_call_id=tool_call.id,
                        tool_type="system",
                        completed=True,
                        result=SystemToolMessage(
                            message=msg,
                            tool_name=tool_name,
                            tool_args=tool_args,
                        ),
                        section=section,
                    )
                ]
        return []

    @toolmethod(
        name="get_context",
        description="Get the current context of the agent.",
        parameters_schema={},
        tool_type="system",
        ui_message=None
    )
    async def get_context(self, *, context: AgentContext) -> AgentContext:
        context_snapshot = await context.to_snapshot(connectors=self._connectors)
        return context_snapshot


    @toolmethod(
        name="output_read",
        description="""[LIVE KERNEL ONLY] Read pages from a value already in the Python kernel namespace
        (e.g. a large DataFrame or primitive) — not for workspace files. For persisted .parquet, charts,
        notebooks, and .output.json on disk, use `read_artifact` with the appropriate `view` and `locator`.

        Up to 5 pages per call. Combine with `output_search` to jump to search hits. For large DataFrames,
        use variable_name="df[row,col]" for cell character pagination (0-indexed row and column name).
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
        pages = pages[:5]
        display_pages = [p - 1 for p in pages]

        cell_ref = _parse_cell_ref(variable_name)
        if cell_ref is not None:
            base_name, row, col = cell_ref
            prim, err = await self._get_cell_as_primitive(cell_ref, context)
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
        description="""[LIVE KERNEL] Search within large in-kernel values (e.g. DataFrames) and utility output buffers.
        Use `read_artifact` (not this tool) to inspect persisted workspace files. Results include page
        numbers for `output_read`. For DataFrames, variable_name can be 'df[row,col]' for a single cell.
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

        output = await self.code_executor.get(variable_name)
        if output is None or output.type not in ("dataframe", "primitive"):
            return False
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
    ) -> tuple[PrimitiveObject | None, str | None]:
        """Resolve a cell ref to a PrimitiveObject (kernel is source of truth)."""
        _ = context
        base_name, row, col = cell_ref
        df: pd.DataFrame | None = None
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
        prim, _ = await self._get_cell_as_primitive(cell_ref, context)
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
        description=(
            "Execute temporary Python code without modifying notebook code or session state. "
            "Use for quick inspections, ad-hoc calculations, or exploratory checks. "
            "Requires _ui_message: a short plain-language, past-tense line for the user describing what this run does."
        ),
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
                },
                "_ui_message": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Required. One line for the user (not the LLM): past tense, what this temporary run does or shows. "
                        "E.g. 'Checked CPI year-over-year growth'."
                    ),
                },
            },
            "required": ["code", "_ui_message"],
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
        timeout_seconds: float = 120.0,
        context: AgentContext,
    ) -> UtilityToolOutput:
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

    def _workspace_root(self) -> Path:
        cwd = getattr(self.code_executor, "cwd", None)
        if not cwd:
            raise RuntimeError("Code executor has no working directory set.")
        return Path(cwd)

    @toolmethod(
        name="write_file",
        description="Write or overwrite a text file in the workspace (UTF-8). Does not execute the file.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path (e.g. main.py, charts/x.vl.json)."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        tool_type="utility",
        ui_message="Writing file",
    )
    async def write_file(self, *, path: str, content: str, context: AgentContext) -> str:
        existed = (self._workspace_root() / path).exists()
        await self.code_executor.write_workspace_file(path, content.encode("utf-8"))
        nlines = len(content.splitlines())
        action = "modified" if existed else "created"
        return f"{action.capitalize()} {path} ({nlines} lines)."

    @toolmethod(
        name="edit_file",
        description="Replace exactly one occurrence of old_str with new_str in a text file.",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file."},
                "old_str": {"type": "string", "description": "Exact substring to replace (must occur exactly once)."},
                "new_str": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_str", "new_str"],
            "additionalProperties": False,
        },
        tool_type="utility",
        ui_message="Editing file",
    )
    async def edit_file(self, *, path: str, old_str: str, new_str: str, context: AgentContext) -> str:
        raw = await self.code_executor.read_workspace_file(path)
        text = raw.decode("utf-8")
        if old_str == "":
            return await self.write_file(path=path, content=new_str, context=context)
        n = text.count(old_str)
        if n == 0:
            raise ValueError("old_str not found in file.")
        if n > 1:
            raise ValueError("old_str occurs multiple times; provide a more specific target.")
        new_text = text.replace(old_str, new_str, 1)
        await self.code_executor.write_workspace_file(path, new_text.encode("utf-8"))
        nlines = len(new_text.splitlines())
        return f"Modified {path} ({nlines} lines)."

    @toolmethod(
        name="execute",
        description="Execute a Python script file in a fresh namespace (connectors available as client).",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the .py file to run."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        tool_type="utility",
        ui_message="Executing script",
    )
    async def execute_workspace_tool(self, *, path: str, context: AgentContext) -> UtilityToolOutput:
        raw = await self.code_executor.read_workspace_file(path)
        code = raw.decode("utf-8")
        effective_timeout = self.guardrails.tool_timeout_s
        metadata = self._utility_tool_metadata(
            tool_name="execute",
            tool_description="Execute a workspace Python script.",
            tool_args={"path": path, "effective_timeout_seconds": effective_timeout},
        )
        kernel_output = await self.code_executor.execute_workspace(
            code,
            dry_run=False,
            timeout_seconds=effective_timeout,
        )
        kernel_output.metadata = metadata
        return UtilityToolOutput(
            metadata=metadata,
            content=kernel_output,
            ui_message=f"Executed {path}",
        )

    @toolmethod(
        name="read_file",
        description="Read a text file as raw UTF-8. Prefer `read_artifact` for .parquet, .vl.json, and .py "
        "notebooks. Optional legacy Parquet head: `read_data`.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative file path."}},
            "required": ["path"],
            "additionalProperties": False,
        },
        tool_type="system",
        ui_message="Reading file",
        ui_message_completed="Read file",
    )
    async def read_file(self, *, path: str, context: AgentContext) -> SystemToolOutput:
        raw = await self.code_executor.read_workspace_file(path)
        text = raw.decode("utf-8")
        return SystemToolOutput(content=Text(content=text))

    @toolmethod(
        name="read_data",
        description="Read schema, provenance, and a short head of a .parquet (compact XML). Prefer "
        "`read_artifact` (view=outline|page) for registry projections and row paging; use this for a "
        "one-shot legacy preview.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path ending in .parquet"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        tool_type="system",
        ui_message="Reading data file",
        ui_message_completed="Read data file",
    )
    async def read_data(self, *, path: str, context: AgentContext) -> SystemToolOutput:
        if not path.endswith(".parquet"):
            raise ValueError("read_data only supports .parquet files.")
        full = self._workspace_root() / path
        result = Result.from_parquet(full)
        df = result.df
        head = df.head(5)
        schema_line = " | ".join(f"{c.name} ({c.role.value}, {c.dtype})" for c in result.columns)
        prov = result.provenance
        lines = [
            f'<data_file path="{path}">',
            "  <schema>",
            f"    {schema_line}",
            "  </schema>",
            "  <provenance>",
            f"    source: {prov.source or ''}",
            f"    params: {json.dumps(prov.params or {}, sort_keys=True)}",
            f"    fetched_at: {prov.fetched_at.isoformat() if prov.fetched_at else ''}",
            "  </provenance>",
            f"  <summary>{len(df)} rows x {len(result.columns)} columns</summary>",
            "  <head>",
            head.to_string(),
            "  </head>",
            "</data_file>",
        ]
        return SystemToolOutput(content=Text(content="\n".join(lines)))

    @toolmethod(
        name="read_artifact",
        description=(
            "Read a workspace artifact via the kind registry. Use list_files to discover paths. "
            "view=summary (default) is a compact one-screen read; outline exposes structure; "
            "page requires a locator (e.g. rows offset/limit for Parquet, section_index for .py). "
            "view=full is the most useful complete view: notebook code, dataset outline + row paging "
            "guidance, chart as a rendered image (Vega-Lite JSON only with locator {kind: chart_spec}). "
            "Prefer this over read_file for .parquet, .vl.json, .py notebooks, and .output.json. "
            "mode=summary|full is a legacy alias when view is omitted (maps to view)."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "view": {
                    "type": "string",
                    "description": "summary (default) | outline | page | full",
                    "enum": ["summary", "outline", "page", "full"],
                },
                "mode": {
                    "type": "string",
                    "description": "Deprecated. Use view. summary (default) or full if view is omitted.",
                    "enum": ["summary", "full"],
                },
                "locator": {
                    "type": "object",
                    "description": "Optional pagination locator, e.g. {kind: rows, offset, limit} or {kind: section, section_index: 0}.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        tool_type="system",
        ui_message="Reading artifact",
        ui_message_completed="Read artifact",
    )
    async def read_artifact(
        self,
        *,
        path: str,
        view: str | None = None,
        mode: str | None = None,
        locator: dict | None = None,
        context: AgentContext,
    ) -> SystemToolOutput:
        if self._read_artifact_fn is None:
            raise RuntimeError("read_artifact is not enabled for this agent configuration")
        options: dict[str, Any] = {
            "view": view,
            "mode": mode,
            "locator": locator,
        }
        result = await self._read_artifact_fn(path, options)
        if result.kernel_output is not None:
            return SystemToolOutput(content=result.kernel_output)
        return SystemToolOutput(content=Text(content=result.text))

    @toolmethod(
        name="list_files",
        description="Lightweight directory discovery: list paths and sizes. Use read_artifact for .parquet, "
        ".vl.json, and notebooks, not for bulk inspection here.",
        parameters_schema={
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Optional subdirectory prefix (e.g. data/).",
                    "default": "",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
        tool_type="system",
        ui_message="Listing files",
        ui_message_completed="Listed files",
    )
    async def list_files(self, *, prefix: str = "", context: AgentContext) -> SystemToolOutput:
        rows = await self.code_executor.list_workspace_files(prefix)
        lines: list[str] = ["<workspace_files>"]
        root = self._workspace_root()
        for rel, size_b in rows:
            line = f"  {rel} ({size_b} bytes)"
            if rel.endswith(".parquet"):
                line += f" — {parquet_summary(root / rel)}"
            lines.append(line)
        lines.append("</workspace_files>")
        return SystemToolOutput(content=Text(content="\n".join(lines)))

    @toolmethod(
        name="run_notebook",
        description=(
            "[CODE CELLS TOOL] Execute an existing workspace .py notebook in the current "
            "kernel namespace (additive — like running a cell in Jupyter). Variables it "
            "produces are immediately available for subsequent tool calls. Use this to "
            "restore prior computation after a kernel restart, or to load a dependency "
            "notebook before writing new code."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path to the .py notebook file.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        tool_type="code",
        ui_message="Running notebook",
        ui_message_completed="Ran notebook",
    )
    async def run_notebook(self, *, path: str, context: AgentContext) -> KernelOutput:
        raw = await self.code_executor.read_workspace_file(path)
        script = deserialize_notebook(raw, path=path)
        return await self.code_executor.execute(script.code)

    @toolmethod(
        name="restart_kernel",
        description=(
            "[SYSTEM TOOL] Clear the Python kernel namespace. All variables are lost; "
            "workspace files and notebooks are preserved. Use before starting a new "
            "independent analysis or when you want to verify a notebook runs cleanly "
            "from scratch."
        ),
        parameters_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        tool_type="system",
        ui_message="Restarting kernel",
        ui_message_completed="Kernel restarted",
    )
    async def restart_kernel(self, *, context: AgentContext) -> SystemToolOutput:
        await self.code_executor.clear_namespace()
        return SystemToolOutput(content=Text(content="Kernel restarted. Namespace cleared."))

    @toolmethod(
        name="code_set",
        description=(
            "Replace the entire analysis notebook .py file on disk. "
            "By default does not execute the kernel — use ``execute``: true to write and run in one step, "
            "or call ``run_notebook`` after writing when you only need execution. "
            "Always start ``code`` with a one-line module docstring (triple-quoted) written "
            "for a non-technical reader. State what the notebook produces and briefly note any "
            "methodological choice that materially affects interpretation. "
            "Optional _ui_message: short plain-language line for the user after '>' in the file-ref row."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Required. Workspace path, e.g. 'notebooks/inflation_analysis.py'. The path is "
                        "the notebook's identity; reusing it overwrites; a new path creates a new notebook."
                    ),
                },
                "code": {
                    "type": "string",
                    "description": (
                        "Full Python source. Must begin with a triple-quoted module docstring that "
                        "describes what this notebook produces for a non-technical reader and notes "
                        "material methodological choices. Use ``# # Heading`` comments for major "
                        "sections and ``# plain comment`` comment blocks before each code block to "
                        "explain the step's intent and any choice that materially affects coverage, "
                        "comparability, or interpretation. Do not narrate routine code mechanics."
                    ),
                },
                "execute": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, run the notebook in the kernel immediately after writing (same as "
                        "``run_notebook`` on the new file). If false, only persist the file."
                    ),
                },
            },
            "required": ["path", "code"],
            "additionalProperties": False,
        },
        tool_type="code",
        ui_message="Writing notebook",
    )
    async def code_set(
        self, *, path: str, code: str, context: AgentContext, execute: bool = False
    ) -> str | KernelOutput:
        normalized = path.strip()
        if not normalized:
            raise ValueError("path must be a non-empty string.")
        script = Script(path=normalized, code=code)
        raw = serialize_notebook(script)
        await self.code_executor.write_workspace_file(normalized, raw)
        if execute:
            raw2 = await self.code_executor.read_workspace_file(normalized)
            loaded = deserialize_notebook(raw2, path=normalized)
            return await self.code_executor.execute(loaded.code)
        return f"Wrote {normalized} (not executed; use run_notebook to execute)."

    @toolmethod(
        name="code_edit",
        description=(
            "Edit the notebook on disk: replace one occurrence of old_str with new_str. "
            "Use sufficiently long and specific context to ensure uniqueness. "
            "By default does not execute — set ``execute``: true to run the updated file in the kernel "
            "in the same call, or call ``run_notebook`` afterward."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Required. Workspace path to edit, e.g. 'notebooks/inflation_analysis.py'."
                    ),
                },
                "old_str": {
                    "type": "string",
                    "description": (
                        "Substring to replace (must occur exactly once), or empty string to replace the whole file."
                    ),
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text, or the full new file when old_str is empty.",
                },
                "execute": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "If true, run the notebook in the kernel immediately after saving (same as "
                        "``run_notebook`` on the updated file)."
                    ),
                },
            },
            "required": ["path", "old_str", "new_str"],
            "additionalProperties": False,
        },
        tool_type="code",
        ui_message="Editing notebook",
    )
    async def code_edit(
        self,
        *,
        path: str,
        old_str: str,
        new_str: str,
        context: AgentContext,
        execute: bool = False,
    ) -> str | KernelOutput:
        if old_str == "":
            return await self.code_set(path=path, code=new_str, context=context, execute=execute)
        raw = await self.code_executor.read_workspace_file(path)
        script = deserialize_notebook(raw, path=path)
        n = script.code.count(old_str)
        if n == 0:
            raise ValueError("old_str not found in script body.")
        if n > 1:
            raise ValueError("old_str occurs multiple times; provide a more specific target.")
        script.code = script.code.replace(old_str, new_str, 1)
        out = serialize_notebook(script)
        await self.code_executor.write_workspace_file(path, out)
        if execute:
            raw2 = await self.code_executor.read_workspace_file(path)
            loaded = deserialize_notebook(raw2, path=path)
            return await self.code_executor.execute(loaded.code)
        return f"Modified {path} (not executed; use run_notebook to execute)."

    @toolmethod(
        name="return_dataset",
        description=(
            "Return exactly one validated dataset when the user should receive the table. "
            "``dataset_variable_name`` must be a plain notebook variable name. "
            "``sources_from_variables`` is the list of variable names that hold the "
            "connector ``Result`` return values that fed this dataset (e.g. "
            "``raw = await connectors[\"fred_fetch\"](...)``); pass an empty list only "
            "when no upstream connector fetches contributed (purely-computed datasets). "
            "Provide notebook refs for the stages used in the pipeline. The framework "
            "picks the persistence location and embeds curation metadata in the on-disk file. "
            "Charts can be delivered with return_chart alone."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "dataset_variable_name": {
                    "type": "string",
                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                    "description": "Plain notebook variable name for the final pandas DataFrame or Series to return. Do not pass expressions, slices, or indexing such as df.head(20) or df[0:20].",
                },
                "sources_from_variables": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                    },
                    "description": (
                        "Variable names that hold the connector Result return values "
                        "fed into this dataset (the variables on the LHS of "
                        "``await connectors[...](...)`` calls). The framework resolves "
                        "each to its provenance — source, params, fetched_at, "
                        "data_object_path — and stamps the list as Dataset.sources. "
                        "Pass [] only for purely-computed datasets with no upstream "
                        "connector fetches."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Short, informative display title for the dataset. Focus on what the data represents: domain, entities, time period. Avoid generic suffixes like 'Data' or 'Dataset'. Avoid process qualifiers like 'cleaned', 'processed', or 'merged' unless essential for disambiguation.",
                },
                "notebook_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of notebook paths (e.g. 'notebooks/sp500_fetch.py') used to produce this dataset, in execution order.",
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
                "sources_from_variables",
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
            sources_from_variables: list[str],
            title: str,
            description: str,
            notes: list[str],
            tags: list[str] | None = None,
            notebook_refs: list[str] | None = None,
        ) -> Dataset:
        dataset_variable_name = self._require_plain_variable_name(
            value=dataset_variable_name,
            parameter_name="dataset_variable_name",
        )
        title = title.strip()
        if not title:
            raise ValueError("title must be a non-empty string.")
        notes = TypeAdapter(list[str]).validate_python(notes)
        clean_refs = [(r or "").strip() for r in (notebook_refs or []) if (r or "").strip()]
        clean_source_vars = [
            self._require_plain_variable_name(
                value=name,
                parameter_name="sources_from_variables",
            )
            for name in TypeAdapter(list[str]).validate_python(sources_from_variables)
        ]
        await self._validate_return_dataset_refs(
            context=context,
            dataset_variable_name=dataset_variable_name,
            notebook_refs=clean_refs,
        )

        out_obj = await self.code_executor.get(dataset_variable_name)
        if out_obj is None:
            raise ValueError(
                f"Variable '{dataset_variable_name}' is not in the kernel. "
                "Run the relevant notebook with run_notebook (or use dry_execute_code) first."
            )
        if not isinstance(out_obj, DataFrameObject):
            raise TypeError(
                f"Dataset '{dataset_variable_name}' must resolve to a pandas DataFrame; "
                f"got {type(out_obj).__name__}."
            )
        source_dataset_variable_names = [dataset_variable_name]

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

        sources = await _resolve_sources_from_variables(
            self.code_executor, clean_source_vars
        )
        dataset = Dataset(
            artifact_id=returned_state.artifact_id,
            version=returned_state.version,
            title=title,
            description=description,
            tags=final_tags,
            notes=notes,
            notebook_refs=clean_refs,
            sources=sources,
        )
        return dataset.with_payload(out_obj)

    @toolmethod(
        name="return_chart",
        description=(
            "Return one optional chart primitive built from a clean dataframe in the kernel. "
            "The source dataframe does not need to have been returned with return_dataset. "
            "``sources_from_variables`` is the list of variable names that hold the "
            "connector ``Result`` return values that fed the chart's source data; "
            "pass an empty list only when no upstream connector fetches contributed. "
            "The framework picks the persistence location and embeds curation "
            "metadata in the on-disk Vega-Lite file."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short human-readable title for the chart, shown in the UI.",
                },
                "source_dataset_variable_name": {
                    "type": "string",
                    "description": (
                        "Name of the kernel variable holding the pandas DataFrame the chart is based on. "
                        "It does not need to have been returned with return_dataset."
                    ),
                },
                "chart_variable_name": {
                    "type": "string",
                    "description": "Variable name that resolves to an Altair chart built from the source dataset.",
                },
                "chart_notebook_ref": {
                    "type": "string",
                    "description": "Notebook path for the visualization stage (e.g. 'notebooks/viz.py').",
                },
                "sources_from_variables": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                    },
                    "description": (
                        "Variable names that hold the connector Result return values "
                        "fed into the chart's source data (the variables on the LHS of "
                        "``await connectors[...](...)`` calls). Pass [] only for charts "
                        "with no upstream connector fetches."
                    ),
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
                "title",
                "source_dataset_variable_name",
                "chart_variable_name",
                "chart_notebook_ref",
                "sources_from_variables",
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
        title: str,
        source_dataset_variable_name: str,
        chart_variable_name: str,
        chart_notebook_ref: str,
        sources_from_variables: list[str],
        description: str,
        notes: list[str],
    ) -> Chart:
        title = title.strip()
        if not title:
            raise ValueError("title must be a non-empty string.")
        source_dataset_variable_name = self._require_plain_variable_name(
            value=source_dataset_variable_name,
            parameter_name="source_dataset_variable_name",
        )
        chart_variable_name = self._require_plain_variable_name(
            value=chart_variable_name,
            parameter_name="chart_variable_name",
        )
        notes = TypeAdapter(list[str]).validate_python(notes)
        chart_ref = (chart_notebook_ref or "").strip()
        if not chart_ref:
            raise ValueError("chart_notebook_ref must be a non-empty string.")

        source_obj = await self.code_executor.get(source_dataset_variable_name)
        if source_obj is None:
            raise ValueError(
                f"Variable '{source_dataset_variable_name}' is not in the kernel. "
                "If session_state lists it, use it directly; otherwise run the notebook that creates it."
            )
        if not isinstance(source_obj, DataFrameObject):
            raise TypeError(
                f"source_dataset_variable_name '{source_dataset_variable_name}' must resolve to a "
                f"pandas DataFrame; got {type(source_obj).__name__}."
            )

        await self._validate_return_chart_refs(
            context=context,
            source_dataset_variable_name=source_dataset_variable_name,
            chart_variable_name=chart_variable_name,
            chart_notebook_ref=chart_ref,
        )

        fig_obj = await self.code_executor.get(chart_variable_name)
        if fig_obj is None:
            raise ValueError(
                f"Variable '{chart_variable_name}' is not in the kernel. "
                "If session_state lists it, use it directly; otherwise run the notebook that creates it."
            )
        if not isinstance(fig_obj, FigureObject):
            raise TypeError(
                f"chart_variable_name '{chart_variable_name}' must resolve to an Altair chart; "
                f"got {type(fig_obj).__name__}."
            )
        if fig_obj.name is None:
            fig_obj.name = chart_variable_name

        returned_dataset = context.get_returned_dataset()
        if (
            returned_dataset is not None
            and returned_dataset.dataset_variable_name == source_dataset_variable_name
        ):
            source_dataset_path_value = snapshot_path(
                artifact_id=returned_dataset.artifact_id,
                version=returned_dataset.version,
                kind="dataset",
                title=returned_dataset.title or "",
            )
        else:
            source_dataset_path_value = ""
        existing_chart = context.get_returned_chart()
        artifact_id = (
            existing_chart.artifact_id
            if existing_chart is not None
            and existing_chart.source_dataset_path == source_dataset_path_value
            and existing_chart.chart_variable_name == chart_variable_name
            and existing_chart.chart_notebook_ref == chart_ref
            else ""
        )
        version = existing_chart.version if artifact_id else 1
        returned_chart = ReturnedChartState(
            artifact_id=artifact_id,
            version=version,
            title=title,
            source_dataset_path=source_dataset_path_value,
            source_dataset_variable_name=source_dataset_variable_name,
            chart_variable_name=chart_variable_name,
            chart_notebook_ref=chart_ref,
            description=description,
            notes=notes,
        )
        context.set_returned_chart(returned_chart)

        clean_chart_source_vars = [
            self._require_plain_variable_name(
                value=name,
                parameter_name="sources_from_variables",
            )
            for name in TypeAdapter(list[str]).validate_python(sources_from_variables)
        ]
        chart_sources = await _resolve_sources_from_variables(
            self.code_executor, clean_chart_source_vars
        )
        chart = Chart(
            artifact_id=returned_chart.artifact_id,
            version=returned_chart.version,
            title=title,
            description=description,
            source_dataset_path=source_dataset_path_value,
            chart_notebook_ref=chart_ref,
            notes=notes,
            sources=chart_sources,
        )
        # Hand the live FigureObject to the streaming dispatcher; it writes
        # the Vega-Lite snapshot and computes the card preview.
        return chart.with_payload(fig_obj)

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

    async def _validate_return_dataset_refs(
        self,
        *,
        context: AgentContext,
        dataset_variable_name: str,
        notebook_refs: list[str],
    ) -> None:
        if not notebook_refs:
            out = await self.code_executor.get(dataset_variable_name)
            if out is None:
                raise ValueError(
                    f"Variable '{dataset_variable_name}' is not in the kernel. "
                    "When not using notebook_refs, the variable must exist in the current namespace."
                )
            return

        for ref in notebook_refs:
            try:
                await self.code_executor.read_workspace_file(ref)
            except FileNotFoundError as e:  # pragma: no cover
                raise ValueError(f"Notebook file '{ref}' not found in the workspace.") from e
            except Exception as e:  # pragma: no cover
                raise ValueError(f"Could not read notebook file '{ref}': {e}") from e

    async def _validate_return_chart_refs(
        self,
        *,
        context: AgentContext,
        source_dataset_variable_name: str,
        chart_variable_name: str,
        chart_notebook_ref: str,
    ) -> None:
        _ = context
        try:
            raw = await self.code_executor.read_workspace_file(chart_notebook_ref)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Chart notebook '{chart_notebook_ref}' not found or unreadable: {e}") from e
        code = deserialize_notebook(raw, path=chart_notebook_ref).code
        if source_dataset_variable_name not in code:
            raise ValueError(
                f"Chart notebook '{chart_notebook_ref}' should reference the dataset variable '{source_dataset_variable_name}'."
            )
        if chart_variable_name not in code:
            raise ValueError(
                f"Chart notebook '{chart_notebook_ref}' should define chart variable '{chart_variable_name}'."
            )
