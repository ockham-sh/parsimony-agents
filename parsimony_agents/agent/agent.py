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
)
from parsimony_agents.agent.outputs import (
    ArtifactLlmResult,
    SystemToolMessage,
    SystemToolOutput,
    UtilityToolOutput,
)
from parsimony_agents.agent.tracing import trace_tool_execution
from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.identity import (
    ArtifactRef,
    LiveNameCollisionError,
    chart_logical_id,
    dataset_logical_id,
    notebook_content_sha,
    notebook_logical_id,
    report_logical_id,
)
from parsimony_agents.agent.seen_refs import extract_seen_live_names
from parsimony_agents.refresh import embedded_refs_from_markdown, refresh_artifact
from parsimony_agents.agent.xml_render import escape_attr, escape_text
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
from parsimony_agents.notebook_io import (
    deserialize_notebook,
    last_content_sha_from_log,
    read_latest_notebook,
    serialize_notebook,
)
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


def _format_list_artifacts(
    items: list[dict[str, Any]],
    *,
    query: str | None,
    kind: str | None,
) -> str:
    """Render the ``list_artifacts`` result as a compact text block.

    Empty result: friendly message including the filter that was tried,
    so the agent can tell "nothing matched my topic" apart from "the
    workspace is empty". Non-empty: one line per artifact, ``[kind]
    live_name — title/summary``, ordered most-recent-first. The
    live_name is what the agent feeds into ``read_artifact(live_name=…,
    kind=…)``.
    """
    if not items:
        parts: list[str] = []
        if query:
            parts.append(f"query={query!r}")
        if kind:
            parts.append(f"kind={kind!r}")
        scope = (" matching " + ", ".join(parts)) if parts else ""
        return f"No artifacts found in this workspace{scope}."
    lines = [
        f"{len(items)} artifact(s) (most recent first). "
        "Inspect with read_artifact(live_name=…, kind=…):"
    ]
    for item in items:
        kind_label = item.get("kind", "?")
        live_name = item.get("live_name", "?")
        summary = (item.get("summary") or "").strip()
        suffix = f" — {summary}" if summary else ""
        lines.append(f"  [{kind_label}] {live_name}{suffix}")
    return "\n".join(lines)


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
    """Returned :class:`Dataset` objects keyed by ``logical_id``."""

    charts: dict[str, Chart] = field(default_factory=dict)
    """Returned :class:`Chart` objects keyed by ``logical_id``."""

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
            if isinstance(result, Dataset) and result.logical_id:
                self.datasets[result.logical_id] = result
            elif isinstance(result, Chart) and result.logical_id:
                self.charts[result.logical_id] = result


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

    RETURN_TOOLS = ("return_dataset", "return_chart", "return_report")
    CODE_TOOL_NAMES = {"return_notebook", "edit_notebook", "dry_execute_code"}

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
        read_artifact_fn: Callable[
            [str, str, dict[str, Any]], Awaitable[ArtifactLlmResult]
        ] | None = None,
        list_artifacts_fn: Callable[
            [str | None, str | None, int], Awaitable[list[dict[str, Any]]]
        ] | None = None,
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
        self._list_artifacts_fn = list_artifacts_fn

        _system_tool_methods: list[ToolMethod] = [
            self.return_notebook,
            self.edit_notebook,
            self.dry_execute_code,
            self.write_file,
            self.edit_file,
            self.read_file,
            self.read_data,
        ]
        if read_artifact_fn is not None:
            _system_tool_methods.append(self.read_artifact)
        if list_artifacts_fn is not None:
            _system_tool_methods.append(self.list_artifacts)
        _system_tool_methods.extend(
            [
                self.list_files,
                self.restart_kernel,
                self.return_dataset,
                self.return_chart,
                self.return_report,
                self.edit_report,
                self.refresh,
                self.output_read,
                self.output_search,
            ]
        )
        self.system_tools = Tools(_system_tool_methods)

        self.model_config = resolved_config

        self.guardrails = guardrails

        self._CODE_EDIT_TOOL_NAMES = {"return_notebook", "edit_notebook"}

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
        agent_span: Any,
    ) -> AsyncIterator[AgentError | TextDeltaEvent]:
        """Classify an LLM exception into a typed ``AgentError`` plus a
        user-facing ``TextDeltaEvent``, and record it on the active span.

        The branches are exhaustive — the trailing ``else`` is the catch-all
        for unexpected provider errors. Each branch emits exactly the same
        two-event shape so the caller always yields a uniform error frame.
        """
        model_name = self.model_config.get("model", "the configured model")

        if isinstance(last_exception, RateLimitError):
            error_logger.error("Rate limit exceeded: %s", last_exception, exc_info=True)
            yield AgentError(
                message="Rate limit exceeded",
                recoverable=False,
                error_type="rate_limit",
            )
            yield TextDeltaEvent(
                content=(
                    "We're currently in beta and experiencing high demand. "
                    "The AI model has hit its rate limit, please wait a moment and try again. "
                    "This is expected during peak usage and will be resolved as we scale."
                ),
                message_id=text_message_id,
                delta=False,
            )
        elif isinstance(last_exception, Timeout):
            error_logger.error("Request timeout: %s", last_exception, exc_info=True)
            yield AgentError(
                message="Request timeout",
                recoverable=False,
                error_type="timeout",
            )
            yield TextDeltaEvent(
                content=(
                    "The AI model took too long to respond, please wait a moment and try again. "
                    "We're currently in beta and this will be resolved as we improve the service."
                ),
                message_id=text_message_id,
                delta=False,
            )
        elif isinstance(last_exception, ServiceUnavailableError) or (
            isinstance(last_exception, APIError) and "unavailable" in str(last_exception).lower()
        ):
            error_logger.error("Model unavailable: %s", last_exception, exc_info=True)
            yield AgentError(
                message="Model unavailable",
                recoverable=False,
                error_type="unavailable",
            )
            yield TextDeltaEvent(
                content=(
                    "The selected AI model is currently unavailable. "
                    "Please try again in a moment or select a different model."
                ),
                message_id=text_message_id,
                delta=False,
            )
        elif isinstance(last_exception, AuthenticationError):
            error_logger.error("LLM authentication failed: %s", last_exception, exc_info=True)
            detail = str(last_exception).splitlines()[0] if str(last_exception) else ""
            yield AgentError(
                message=f"Authentication failed for {model_name}: {detail}",
                recoverable=False,
                error_type="authentication",
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
            )
        elif isinstance(last_exception, (BadRequestError, NotFoundError)):
            error_logger.error("LLM bad request: %s", last_exception, exc_info=True)
            detail = str(last_exception).splitlines()[0] if str(last_exception) else ""
            yield AgentError(
                message=f"Invalid request to {model_name}: {detail}",
                recoverable=False,
                error_type="bad_request",
            )
            yield TextDeltaEvent(
                content=(
                    f"The request to `{model_name}` was rejected by the provider. "
                    "This usually means the model name is invalid, unavailable in your region, "
                    f"or the request payload is malformed. Provider said: {detail}"
                ),
                message_id=text_message_id,
                delta=False,
            )
        elif isinstance(last_exception, APIConnectionError):
            error_logger.error("LLM connection error: %s", last_exception, exc_info=True)
            yield AgentError(
                message=f"Could not reach the model provider: {last_exception}",
                recoverable=False,
                error_type="connection",
            )
            yield TextDeltaEvent(
                content=(
                    "Could not connect to the AI model provider. "
                    "Check your network connection and try again."
                ),
                message_id=text_message_id,
                delta=False,
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
            )
            yield TextDeltaEvent(
                content=(
                    f"An error occurred while communicating with the AI model "
                    f"({type(last_exception).__name__}): {detail}"
                ),
                message_id=text_message_id,
                delta=False,
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
            yield StateSnapshot(context=ctx)
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

        # The context snapshot is rebuilt at the START of every iteration (see
        # the loop body) so the rendered ``<turn_artifacts>`` block always
        # reflects this turn's freshly-minted refs. We seed an initial snapshot
        # here so the message list has one before iteration begins.
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
                )
                break

            tools = self.system_tools.copy()

            # Rebuild the context snapshot every iteration so the rendered
            # <turn_artifacts> block reflects this turn's latest minted refs.
            # Cheap: to_snapshot just lists files and renders the connector
            # catalog. Replacing the previous snapshot keeps message history
            # bounded; the LLM only ever sees the current state, never a
            # mid-turn-stale view.
            iter_snapshot = await ctx.to_snapshot(
                connectors=self._connectors,
                minted_refs=turn_state.minted_refs,
                minted_live_names=turn_state.minted_live_names,
            )
            ctx.messages = [
                m for m in ctx.messages if m.metadata.get("context_snapshot", False) is False
            ]
            ctx.messages.append(
                Message(
                    role="user",
                    content=iter_snapshot,
                    metadata={"context_snapshot": True},
                )
            )


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
                            )
                        if reasoning_content := getattr(chunk.choices[0].delta, "reasoning_content", None):
                            yield ReasoningDeltaEvent(
                                content=reasoning_content,
                                message_id=reasoning_message_id,
                                delta=True,
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
                                )

                    if user_broke_stream and cancellation is not None:
                        yield RunCancelled(
                            message="Generation was cancelled before the assistant message completed.",
                            reason=cancellation.reason,
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
                    last_exception, text_message_id, agent_span
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
                )
                accumulated_reasoning = ""
                accumulated_duration = 0.0
                reasoning_message_id = str(uuid4())


            if text_message := response_message.content:
                yield TextDeltaEvent(
                    content=text_message,
                    message_id=text_message_id,
                    delta=False,
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
                                    content="return_notebook and edit_notebook require a non-empty 'path'. The path is the notebook's identity address (e.g. 'notebooks/inflation_analysis.py' → slug 'inflation_analysis'); reuse to add a revision, or pick a fresh path under 'notebooks/' to create one.",
                                    name=tool_name,
                                    tool_call_id=tool_call.id,
                                )
                            )
                            continue

                    loading_label = tools[tool_name].ui_message
                    if tool_type == "code":
                        if tool_name == "return_notebook" and tool_args.get("execute") is True:
                            loading_label = "Writing and running notebook"
                        elif tool_name == "edit_notebook" and tool_args.get("execute") is True:
                            loading_label = "Editing and running notebook"
                        if tool_name in ("return_notebook", "edit_notebook"):
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
                            tools, tool_name, tool_args, tool_call
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
                    )
                    turn_state.stopped = True
                    yield StateSnapshot(context=ctx.model_copy(deep=False))
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
                            tools, tool_name, tool_args, tool_call
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
                                )

                        case "return":
                            return_loading = tools[tool_name].ui_message
                            if tool_result.data:
                                tool_call_output = tool_result.data
                                return_types = (Dataset, Chart, Report)
                                if isinstance(tool_call_output, return_types):
                                    # Flip the streaming UI to delivery mode but DO NOT stop
                                    # the loop: a single user request may need multiple
                                    # deliverables (dataset + chart + report). The agent
                                    # iterates freely until it emits no more tool calls
                                    # (natural termination at the no-tool-calls branch above)
                                    # or hits ``max_iterations``. Republish is idempotent under
                                    # content-addressing, so this is safe.
                                    yield ToolEvent(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call.id,
                                        tool_type="return",
                                        completed=True,
                                        result=tool_call_output,
                                        ui_message=return_loading,
                                        ui_message_completed=llm_ui_message,
                                    )
                                    # After the yield, the streaming layer has persisted
                                    # the artifact and mutated ``content_sha``. Append to
                                    # the turn's ref ledger so the next iteration's
                                    # <turn_artifacts> block carries the freshly-minted
                                    # triplet — the agent never has to scan back through
                                    # tool messages to find it (Task 15).
                                    if tool_call_output.logical_id and tool_call_output.content_sha:
                                        turn_state.minted_refs.append(
                                            ArtifactRef(
                                                kind=tool_call_output.type,
                                                logical_id=tool_call_output.logical_id,
                                                content_sha=tool_call_output.content_sha,
                                            )
                                        )
                                        # Carry the agent-typed slug forward so the
                                        # next iteration's <turn_artifacts> row carries
                                        # live_name=... — that attribute is what the
                                        # seen-set extractor keys on. Without it, the
                                        # very next return_* (e.g. return_chart after
                                        # return_dataset) raises LiveNameCollisionError
                                        # against our own iter-just-finished mint.
                                        ln = getattr(tool_call_output, "live_name", None)
                                        if ln:
                                            turn_state.minted_live_names[
                                                f"{tool_call_output.type}:{tool_call_output.logical_id}"
                                            ] = ln
                            else:
                                tool_call_output = tool_result.exception_message

                        case "code":
                            tool_call_output = tool_result.data if tool_result.data else tool_result.exception_message
                            if tool_result.success:
                                notebook_path = self._resolve_code_tool_path(tool_args)
                                if notebook_path is None:
                                    raise ValueError("code tools require a non-empty 'path'.")
                                ran_kernel = (
                                    tool_name in ("return_notebook", "edit_notebook")
                                    and tool_args.get("execute") is True
                                )
                                # Derive the canonical post-turn notebook source without
                                # touching a transient working copy (the new model has
                                # none — bytes live solely under
                                # ``.ockham/notebooks/<lid>/<csha>.py``). For
                                # return_notebook, the new source IS ``tool_args.code``.
                                # For edit_notebook, we re-apply the edit against the
                                # latest snapshot (matches what the tool method itself
                                # did).
                                script = await self._notebook_script_after_tool(
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
                                    # edit_notebook: no optional detail in the file-ref line on the wire.
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
                                # Track the notebook ref in the turn ledger
                                # (Task 15). The ref's content_sha is computed
                                # synchronously from the script bytes; no
                                # streaming-layer wait needed.
                                if tool_name in ("return_notebook", "edit_notebook"):
                                    try:
                                        nb_ref = await self._notebook_ref_for(
                                            script.code, notebook_path, ctx
                                        )
                                    except LiveNameCollisionError as exc:
                                        # Post-success mint: the host resolver
                                        # scans curations including the one
                                        # we just wrote. Its live_name
                                        # matches our path but the seen-set
                                        # — derived from messages — does not
                                        # yet carry the freshly-minted pair.
                                        # The early-validation already
                                        # cleared cross-terminal collisions,
                                        # so trust the resolver's
                                        # ``existing_logical_id`` here.
                                        nb_ref = ArtifactRef(
                                            kind="notebook",
                                            logical_id=exc.existing_logical_id,
                                            content_sha=notebook_content_sha(
                                                script.code
                                            ),
                                        )
                                    turn_state.minted_refs.append(nb_ref)
                                    # The agent-typed slug is the path basename
                                    # (``notebooks/<slug>.py``). That is what
                                    # the next iteration's seen-set extractor
                                    # must see — otherwise the post-mint
                                    # ``return_dataset`` / ``return_chart``
                                    # call that resolves the producing
                                    # notebook via the host resolver raises
                                    # ``LiveNameCollisionError`` against this
                                    # terminal's own write.
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
                            system_ui = llm_ui_message or tools[tool_name].ui_message_completed or tools[tool_name].ui_message
                            if system_ui is not None:
                                yield ToolEvent(
                                    tool_name=tool_name,
                                    tool_call_id=tool_call.id,
                                    tool_type="system",
                                    completed=True,
                                    result=SystemToolMessage(message=system_ui),
                                    ui_message_completed=llm_ui_message,
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
                    )
                    turn_state.stopped = True

                # Persist tool results immediately so a client disconnect mid-turn can recover.
                yield StateSnapshot(context=ctx.model_copy(deep=False))


        if accumulated_reasoning:
            yield ReasoningDeltaEvent(
                content=accumulated_reasoning,
                message_id=reasoning_message_id,
                title=f"Thought for {accumulated_duration:.1f} seconds",
                delta=False,
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
            )

        yield StateSnapshot(context=ctx.model_copy(deep=False))

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
    ) -> list[ToolEvent]:
        ttype = tools[tool_name].tool_type
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
                    )
                ]
        return []

    @toolmethod(
        name="output_read",
        description=(
            "Read pages from an in-kernel value (DataFrame or primitive). For persisted "
            "files use read_artifact. variable_name='df[row,col]' paginates a single cell."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "variable_name": {"type": "string", "description": "Kernel variable, or 'df[row,col]' cell ref."},
                "pages": {"type": "array", "description": "Up to 5 pages.", "items": {"type": "integer"}, "minItems": 1, "maxItems": 5},
            },
            "required": ["variable_name", "pages"],
            "additionalProperties": False,
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
        description=(
            "Search within an in-kernel value. Returns hits with page numbers for output_read. "
            "Use read_artifact for persisted workspace files."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "variable_name": {"type": "string", "description": "Kernel variable or 'df[row,col]' cell ref."},
                "top_k": {"type": "integer", "description": "Result count.", "default": 5},
            },
            "required": ["query", "variable_name"],
            "additionalProperties": False,
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
            "Run scratch Python; stdout/display() land in the conversation, kernel state is preserved, "
            "no notebook is published. _ui_message is a short past-tense line shown to the user."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "timeout_seconds": {"type": "number", "description": "Default 120s, capped by tool timeout."},
                "_ui_message": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Past-tense user-facing line, e.g. 'Checked CPI year-over-year growth'.",
                },
            },
            "required": ["code", "_ui_message"],
            "additionalProperties": False,
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
        # Use dry_run=True to ensure code execution is sandboxed within the actor.
        # ``seen_live_names`` flows through so ``load_dataset`` inside the cell
        # honours the calling terminal's gate.
        kernel_output = await self.code_executor.execute(
            code,
            dry_run=True,
            timeout_seconds=effective_timeout,
            seen_live_names=extract_seen_live_names(context.messages),
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
        name="read_file",
        description="Raw UTF-8 read for unregistered text files. Prefer read_artifact for typed kinds.",
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
        description="Compact .parquet preview (schema + provenance + head). Prefer read_artifact for paging.",
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
        # Connector-supplied: column names, dtypes, source, params. Escape every interpolation.
        schema_line = " | ".join(
            f"{escape_text(c.name)} ({escape_text(c.role.value)}, {escape_text(c.dtype)})"
            for c in result.columns
        )
        prov = result.provenance
        lines = [
            f'<data_file path="{escape_attr(path)}">',
            "  <schema>",
            f"    {schema_line}",
            "  </schema>",
            "  <provenance>",
            f"    source: {escape_text(prov.source or '')}",
            f"    params: {escape_text(json.dumps(prov.params or {}, sort_keys=True))}",
            f"    fetched_at: {escape_text(prov.fetched_at.isoformat() if prov.fetched_at else '')}",
            "  </provenance>",
            f"  <summary>{len(df)} rows x {len(result.columns)} columns</summary>",
            "  <head>",
            escape_text(head.to_string()),
            "  </head>",
            "</data_file>",
        ]
        return SystemToolOutput(content=Text(content="\n".join(lines)))

    @toolmethod(
        name="read_artifact",
        description=(
            "Read a typed workspace artifact (notebook / dataset / chart / report) "
            "by its live_name. Use this after list_artifacts surfaces a sibling-"
            "terminal artifact you want to compose with — the read brings it into "
            "your context, after which return_* / load_dataset / refresh / "
            "edit_report against the same live_name + kind no longer raises "
            "LiveNameCollisionError. view=summary (default) | outline | page | "
            "full. page requires locator (e.g. {kind: rows, offset, limit} for "
            "datasets, {kind: section, section_index} for notebooks). full renders "
            "charts as images."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "live_name": {
                    "type": "string",
                    "description": (
                        "Workspace slug, exactly as it appears in "
                        "<turn_artifacts> or in a list_artifacts row."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": ["notebook", "dataset", "chart", "report"],
                    "description": "Artifact kind.",
                },
                "view": {
                    "type": "string",
                    "description": "summary | outline | page | full",
                    "enum": ["summary", "outline", "page", "full"],
                },
                "mode": {
                    "type": "string",
                    "description": "Deprecated alias for view (summary | full).",
                    "enum": ["summary", "full"],
                },
                "locator": {
                    "type": "object",
                    "description": "Pagination locator, required for view=page.",
                },
            },
            "required": ["live_name", "kind"],
            "additionalProperties": False,
        },
        tool_type="system",
        ui_message="Reading artifact",
        ui_message_completed="Read artifact",
    )
    async def read_artifact(
        self,
        *,
        live_name: str,
        kind: str,
        view: str | None = None,
        mode: str | None = None,
        locator: dict | None = None,
        context: AgentContext,  # noqa: ARG002
    ) -> SystemToolOutput:
        if self._read_artifact_fn is None:
            raise RuntimeError(
                "read_artifact is not enabled for this agent configuration"
            )
        options: dict[str, Any] = {
            "view": view,
            "mode": mode,
            "locator": locator,
        }
        result = await self._read_artifact_fn(live_name, kind, options)
        if result.kernel_output is not None:
            return SystemToolOutput(content=result.kernel_output)
        return SystemToolOutput(content=Text(content=result.text))

    @toolmethod(
        name="list_artifacts",
        description=(
            "Discover artifacts already in this workspace by topical keyword. "
            "Use this BEFORE fetching data from any external source (connector, "
            "web search, API) when the user references a topic you have not yet "
            "worked with in this conversation. Cross-terminal: this lists "
            "artifacts produced by sibling terminal sessions too. Returns up to "
            "`limit` matches ordered by recency, each as "
            "{live_name, kind, title, summary}. Inspect one with "
            "read_artifact(live_name=..., kind=...)."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Topical keyword drawn from the user's request "
                        "(case-insensitive substring on name/title/description/tags). "
                        "Empty or missing returns all artifacts."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": ["notebook", "dataset", "chart", "report", "data_object"],
                    "description": "Optional filter to a single artifact kind.",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max items to return (1-100).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        tool_type="system",
        ui_message="Listing artifacts",
        ui_message_completed="Listed artifacts",
    )
    async def list_artifacts(
        self,
        *,
        query: str | None = None,
        kind: str | None = None,
        limit: int = 20,
        context: AgentContext,  # noqa: ARG002
    ) -> SystemToolOutput:
        if self._list_artifacts_fn is None:
            raise RuntimeError(
                "list_artifacts is not enabled for this agent configuration"
            )
        bounded_limit = max(1, min(100, int(limit)))
        items = await self._list_artifacts_fn(query, kind, bounded_limit)
        text = _format_list_artifacts(items, query=query, kind=kind)
        return SystemToolOutput(content=Text(content=text))

    @toolmethod(
        name="list_files",
        description=(
            "Discover unregistered workspace files (user-dropped CSV/JSON, raw text). "
            "Do NOT use this to verify typed artifacts (notebook/dataset/chart/report) — "
            "those are listed every iteration in <session_state>.<turn_artifacts>. "
            "Use read_artifact to inspect contents."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "prefix": {"type": "string", "description": "Optional subdirectory prefix.", "default": ""},
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
        name="restart_kernel",
        description=(
            "Clear the kernel namespace. Variables are lost; workspace files persist. "
            "Use before a new analysis or to verify a notebook runs cleanly from scratch."
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
        name="return_notebook",
        description=(
            "Publish a notebook revision (full source). Reuse a path to add a revision under the same "
            "logical_id; pick a fresh path under notebooks/ to create one. execute=true runs it after "
            "publishing. Authoring style: start with a one-line triple-quoted docstring written for a "
            "non-technical reader (what the notebook produces, methodological choices that materially "
            "affect interpretation). Use '# # Heading' comments for sections and '# comment' blocks "
            "before each code block to explain intent and any decision that affects coverage, "
            "comparability, or interpretation. Do not narrate routine mechanics."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Virtual notebook path, e.g. notebooks/<your_notebook>.py.",
                },
                "code": {
                    "type": "string",
                    "description": "Full Python source. See description for required docstring + comment structure.",
                },
                "execute": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, run in the kernel after publishing.",
                },
            },
            "required": ["path", "code"],
            "additionalProperties": False,
        },
        tool_type="code",
        ui_message="Writing notebook",
    )
    async def return_notebook(
        self, *, path: str, code: str, context: AgentContext, execute: bool = False
    ) -> str | KernelOutput:
        normalized = path.strip()
        if not normalized:
            raise ValueError("path must be a non-empty string.")
        # Early-validation: surface cross-terminal collisions BEFORE any
        # kernel side effect. The host-bound resolver compares the path's
        # live_name to existing curations and raises
        # :class:`LiveNameCollisionError` when the artifact belongs to a
        # sibling terminal the agent has never read. ``ValueError`` from
        # path-shape edge cases (standalone tests, legacy flat paths) is
        # swallowed — there is nothing to gate against.
        try:
            await self._resolve_notebook_logical_id(normalized, context)
        except ValueError:
            pass
        # Canonicalize newlines via in-memory round-trip — same shape the
        # snapshot store will write, so ``content_sha`` matches what the
        # streaming layer's persist step computes from the ScriptPreview.
        canonical = deserialize_notebook(
            serialize_notebook(Script(path=normalized, code=code)), path=normalized
        ).code
        if execute:
            # The notebook is the producer for every kernel variable it
            # assigns — open a producer-scoped run so the variable origin
            # ledger gets populated. This is what makes "publish a
            # dataset" automatic-lineage: the agent never types refs.
            ko = await self.code_executor.execute(
                canonical,
                producer_notebook_path=normalized,
                seen_live_names=extract_seen_live_names(context.messages),
            )
            return await self._stamp_notebook_ref(ko, canonical, normalized, context)
        return f"Published {normalized} (not executed; pass execute=true to run)."

    @toolmethod(
        name="edit_notebook",
        description=(
            "Surgical edit of an existing notebook: replace one occurrence of old_str with new_str. "
            "old_str='' replaces the whole file (equivalent to return_notebook). execute=true runs "
            "the result. Authoring style: see return_notebook."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Existing notebook path (resolves to its logical_id).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Substring to replace (must occur exactly once); empty replaces whole file.",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "execute": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, run in the kernel after publishing.",
                },
            },
            "required": ["path", "old_str", "new_str"],
            "additionalProperties": False,
        },
        tool_type="code",
        ui_message="Editing notebook",
    )
    async def edit_notebook(
        self,
        *,
        path: str,
        old_str: str,
        new_str: str,
        context: AgentContext,
        execute: bool = False,
    ) -> str | KernelOutput:
        if old_str == "":
            return await self.return_notebook(path=path, code=new_str, context=context, execute=execute)
        logical_id = await self._resolve_notebook_logical_id(path, context)
        raw, _csha = await read_latest_notebook(self.code_executor, logical_id=logical_id)
        script = deserialize_notebook(raw, path=path)
        n = script.code.count(old_str)
        if n == 0:
            raise ValueError("old_str not found in script body.")
        if n > 1:
            raise ValueError("old_str occurs multiple times; provide a more specific target.")
        script.code = script.code.replace(old_str, new_str, 1)
        if execute:
            ko = await self.code_executor.execute(
                script.code,
                producer_notebook_path=path,
                seen_live_names=extract_seen_live_names(context.messages),
            )
            return await self._stamp_notebook_ref(ko, script.code, path, context)
        return f"Modified {path} (not executed; pass execute=true to run)."

    async def _notebook_script_after_tool(
        self,
        *,
        tool_name: str,
        notebook_path: str,
        tool_args: dict[str, Any],
        context: AgentContext,
    ) -> Script:
        """Rebuild the canonical post-turn :class:`Script` for a code tool call.

        Mirror of what the tool method produced, derived from inputs only —
        no disk read of a transient working copy (notebooks live solely
        under ``.ockham/notebooks/<lid>/<csha>.py``):

        - ``return_notebook``: ``tool_args["code"]`` round-tripped through
          the notebook serializer (canonical newlines).
        - ``edit_notebook`` with empty ``old_str``: behaves as
          ``return_notebook`` over ``new_str``; matches the fallback in
          :meth:`edit_notebook`.
        - ``edit_notebook`` with non-empty ``old_str``: re-read the latest
          snapshot and re-apply the substring edit. Within a single turn
          no other writes can have happened, so this reproduces what the
          tool method itself executed.
        """
        if tool_name == "return_notebook":
            raw_code = tool_args.get("code", "") or ""
            return deserialize_notebook(
                serialize_notebook(Script(path=notebook_path, code=raw_code)),
                path=notebook_path,
            )
        if tool_name == "edit_notebook":
            old_str = tool_args.get("old_str", "")
            new_str = tool_args.get("new_str", "")
            if old_str == "":
                return deserialize_notebook(
                    serialize_notebook(Script(path=notebook_path, code=new_str)),
                    path=notebook_path,
                )
            logical_id = await self._resolve_notebook_logical_id(notebook_path, context)
            raw, _csha = await read_latest_notebook(self.code_executor, logical_id=logical_id)
            script = deserialize_notebook(raw, path=notebook_path)
            script.code = script.code.replace(old_str, new_str, 1)
            return script
        raise ValueError(f"unsupported code tool: {tool_name!r}")

    @staticmethod
    async def _resolve_notebook_logical_id(
        working_copy_path: str, context: AgentContext
    ) -> str:
        """Resolve a notebook user-visible path to its ``logical_id``.

        1. If the workspace host bound ``context.notebook_logical_id_resolver``,
           use it. The host scans existing curations so a notebook whose
           ``live_name`` was renamed via curation keeps its original
           logical_id when the agent next publishes via ``return_notebook``
           under the new path.
        2. Else, derive directly from the path
           (``notebook_logical_id(working_copy_path)``) — the slug-based
           default for fresh notebooks.

        Raises :class:`ValueError` from :func:`notebook_logical_id` for
        paths outside ``notebooks/`` (standalone tests, legacy flat
        usage) — callers that want a content-sha fallback must catch
        it themselves.
        """
        resolver = context.notebook_logical_id_resolver
        if resolver is not None:
            return await resolver(working_copy_path)
        return notebook_logical_id(working_copy_path)

    @staticmethod
    async def _notebook_ref_for(
        code: str, working_copy_path: str, context: AgentContext
    ) -> ArtifactRef:
        """Canonical notebook ref for *code* at *working_copy_path*.

        ``content_sha = notebook_content_sha(code)`` is universal.
        ``logical_id`` resolution defers to
        :meth:`_resolve_notebook_logical_id`. Falls back to
        ``logical_id == content_sha`` for malformed paths so standalone
        tests / legacy flat usage keep working.
        """
        csha = notebook_content_sha(code)
        try:
            lid = await Agent._resolve_notebook_logical_id(working_copy_path, context)
        except ValueError:
            lid = csha
        return ArtifactRef(kind="notebook", logical_id=lid, content_sha=csha)

    @classmethod
    async def _stamp_notebook_ref(
        cls,
        ko: KernelOutput,
        code: str,
        working_copy_path: str,
        context: AgentContext,
    ) -> KernelOutput:
        """Stamp ``KernelOutput.metadata['notebook_ref']`` so to_llm surfaces it.

        Run after every kernel run kicked off by a code tool
        (``return_notebook`` or ``edit_notebook`` with ``execute=True``).
        The ref matches what the streaming layer's ``FileRefChunk`` emits
        — the LLM sees identical refs from both surfaces.
        """
        ref = await cls._notebook_ref_for(code, working_copy_path, context)
        ko.metadata = {**(ko.metadata or {}), "notebook_ref": ref.to_dict()}
        return ko

    @toolmethod(
        name="return_dataset",
        description=(
            "Publish a DataFrame deliverable. Pass the kernel variable name + human "
            "metadata; the framework derives lineage from the variable's producing "
            "notebook run (no ref triplets, no source arrays — those are inferred)."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "dataset_variable_name": {
                    "type": "string",
                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                    "description": "Plain variable name of the final DataFrame.",
                },
                "title": {"type": "string", "description": "Short display title."},
                "description": {"type": "string", "description": "One sentence on contents."},
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Decisions, assumptions, caveats.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional short tags.",
                },
                "live_name": {
                    "type": "string",
                    "description": "File-tree slug (no extension); the handle other notebooks load by.",
                },
            },
            "required": [
                "dataset_variable_name",
                "title",
                "description",
                "notes",
                "live_name",
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
        live_name: str,
        tags: list[str] | None = None,
    ) -> Dataset:
        dataset_variable_name = self._require_plain_variable_name(
            value=dataset_variable_name, parameter_name="dataset_variable_name"
        )
        title = title.strip()
        if not title:
            raise ValueError("title must be a non-empty string.")
        live_name = (live_name or "").strip()
        if not live_name:
            raise ValueError(
                "live_name must be a non-empty workspace slug. It is the handle "
                "other notebooks will load this dataset by via load_dataset('<slug>')."
            )
        notes = TypeAdapter(list[str]).validate_python(notes)

        out_obj = await self.code_executor.get(dataset_variable_name)
        if out_obj is None:
            raise ValueError(
                f"Variable '{dataset_variable_name}' is not in the kernel. "
                "Publish the producing notebook with return_notebook(execute=true) "
                "(or refresh a downstream artifact) to repopulate state."
            )
        if not isinstance(out_obj, DataFrameObject):
            raise TypeError(
                f"Dataset '{dataset_variable_name}' must resolve to a pandas DataFrame; "
                f"got {type(out_obj).__name__}."
            )

        nb_refs, src_refs = await self._lineage_for_variable(
            dataset_variable_name, context=context
        )

        final_tags: list[str] = []
        for t in TypeAdapter(list[str]).validate_python(tags or []):
            s = str(t).strip()
            if s and s not in final_tags:
                final_tags.append(s)

        lid = dataset_logical_id(
            notebook_refs=nb_refs,
            variable_name=dataset_variable_name,
            source_refs=src_refs,
        )
        dataset = Dataset(
            logical_id=lid,
            title=title,
            description=description,
            tags=final_tags,
            notes=notes,
            notebook_refs=nb_refs,
            source_refs=src_refs,
            variable_name=dataset_variable_name,
            live_name=live_name,
        )
        return dataset.with_payload(out_obj)

    @toolmethod(
        name="return_chart",
        description=(
            "Publish an Altair chart deliverable. Pass the kernel variable name + human "
            "metadata; the framework derives lineage from the variable's producing run. "
            "Visual contract: no title/subtitle in the chart spec (use the title "
            "parameter); legend orient='top|bottom|left|right'; size 640x400 fixed; "
            "explicit altair encodings (:Q :N :O :T); aggregate to ≤5000 points; "
            "tooltips yes, zoom/pan no; dark theme background #0d0d0d, primary text "
            "#f1f5f9, accent #3b82f6; positive/negative pair #10b981/#f87171."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short display title (≤50 chars; no styling)."},
                "chart_variable_name": {
                    "type": "string",
                    "pattern": "^[A-Za-z_][A-Za-z0-9_]*$",
                    "description": "Plain variable name resolving to an Altair chart.",
                },
                "description": {"type": "string", "description": "One sentence on what the chart shows."},
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Caveats, encoding decisions, interpretation notes.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional short tags.",
                },
                "live_name": {
                    "type": "string",
                    "description": "File-tree slug (no extension).",
                },
            },
            "required": [
                "title",
                "chart_variable_name",
                "description",
                "notes",
                "live_name",
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
        chart_variable_name: str,
        description: str,
        notes: list[str],
        live_name: str,
        tags: list[str] | None = None,
    ) -> Chart:
        title = title.strip()
        if not title:
            raise ValueError("title must be a non-empty string.")
        chart_variable_name = self._require_plain_variable_name(
            value=chart_variable_name, parameter_name="chart_variable_name"
        )
        live_name = (live_name or "").strip()
        if not live_name:
            raise ValueError("live_name must be a non-empty workspace slug.")
        notes = TypeAdapter(list[str]).validate_python(notes)

        fig_obj = await self.code_executor.get(chart_variable_name)
        if fig_obj is None:
            raise ValueError(
                f"Variable '{chart_variable_name}' is not in the kernel. "
                "Run the notebook that creates it first."
            )
        if not isinstance(fig_obj, FigureObject):
            raise TypeError(
                f"chart_variable_name '{chart_variable_name}' must resolve to an Altair chart; "
                f"got {type(fig_obj).__name__}."
            )
        if fig_obj.name is None:
            fig_obj.name = chart_variable_name

        nb_refs, src_refs, ds_refs = await self._chart_lineage_for_variable(
            chart_variable_name, context=context
        )
        if not nb_refs:
            raise ValueError(
                f"Chart variable {chart_variable_name!r} has no producing notebook "
                "in this kernel. Publish a notebook that assigns it via "
                "return_notebook(execute=true) first."
            )
        nb_ref = nb_refs[0]

        final_tags: list[str] = []
        for t in TypeAdapter(list[str]).validate_python(tags or []):
            s = str(t).strip()
            if s and s not in final_tags:
                final_tags.append(s)

        lid = chart_logical_id(
            notebook_ref=nb_ref,
            chart_variable_name=chart_variable_name,
            source_dataset_refs=ds_refs,
            source_refs=src_refs,
        )
        chart = Chart(
            logical_id=lid,
            title=title,
            description=description,
            tags=final_tags,
            notes=notes,
            notebook_ref=nb_ref,
            source_dataset_refs=ds_refs,
            source_refs=src_refs,
            variable_name=chart_variable_name,
            live_name=live_name,
        )
        return chart.with_payload(fig_obj)

    @toolmethod(
        name="return_report",
        description=(
            "Publish a markdown report. ``markdown`` is the full body — leading "
            "'# Title', prose, and embedded artifacts as "
            "``![](file://./.ockham/<kind>s/<logical_id>/<content_sha>.<ext>)``. "
            "The framework parses those paths and treats them as the report's "
            "embedded_refs; you do not pass a separate lineage list."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Display title (matches leading '# Title')."},
                "markdown": {"type": "string", "description": "Full markdown body with embedded refs."},
                "description": {"type": "string", "description": "One sentence on what the report covers."},
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Assumptions, caveats, context.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional short tags.",
                },
                "live_name": {
                    "type": "string",
                    "description": "File-tree slug (no extension).",
                },
            },
            "required": ["title", "markdown", "description", "notes", "live_name"],
            "additionalProperties": False,
        },
        tool_type="return",
        ui_message="Publishing report",
    )
    async def return_report(
        self,
        *,
        context: AgentContext,  # noqa: ARG002
        title: str,
        markdown: str,
        description: str,
        notes: list[str],
        live_name: str,
        tags: list[str] | None = None,
    ) -> Report:
        title = title.strip()
        if not title:
            raise ValueError("title must be a non-empty string.")
        if not markdown.strip():
            raise ValueError("markdown must be a non-empty string.")
        live_name = (live_name or "").strip()
        if not live_name:
            raise ValueError("live_name must be a non-empty workspace slug.")
        notes = TypeAdapter(list[str]).validate_python(notes)
        # Parse the embedded refs from the markdown body — the path IS the ref.
        # Reject the report if any reference does not resolve on disk.
        #
        # TODO(report-embed-by-live_name): swap this surface to live_name —
        # the lid/csha in the path is the last hash leak in the agent surface,
        # and the agent has been observed typing `latest` here in the hope it
        # would resolve (it does not). Keep snapshots reproducible by
        # persisting a frozen live_name → ArtifactRef pin map alongside the
        # report curation (analogue of dataset/chart source_refs); the
        # renderer resolves embeds against that pin map, so refresh produces a
        # new report snapshot pointing at the new version while old snapshots
        # stay byte-stable. Also update ``embedded_refs_from_markdown`` and
        # ``_validate_refs_resolve`` once the surface flips.
        emb = embedded_refs_from_markdown(markdown)
        await self._validate_refs_resolve(emb)
        final_tags: list[str] = []
        for t in TypeAdapter(list[str]).validate_python(tags or []):
            s = str(t).strip()
            if s and s not in final_tags:
                final_tags.append(s)
        return Report(
            logical_id=report_logical_id(embedded_refs=emb, title=title),
            title=title,
            description=description,
            notes=notes,
            tags=final_tags,
            markdown=markdown,
            embedded_refs=emb,
            live_name=live_name,
        )

    @toolmethod(
        name="edit_report",
        description=(
            "Surgical edit of an existing report: replace one occurrence of old_str "
            "with new_str against the latest snapshot. Identify the report by its "
            "workspace slug (live_name). logical_id is preserved; embedded refs "
            "are re-extracted from the new markdown body."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "live_name": {
                    "type": "string",
                    "description": "The report's workspace slug (live_name).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Substring to replace (must occur exactly once).",
                },
                "new_str": {"type": "string", "description": "Replacement text."},
            },
            "required": ["live_name", "old_str", "new_str"],
            "additionalProperties": False,
        },
        tool_type="return",
        ui_message="Editing report",
    )
    async def edit_report(
        self,
        *,
        context: AgentContext,
        live_name: str,
        old_str: str,
        new_str: str,
    ) -> Report:
        live_name = (live_name or "").strip()
        if not live_name:
            raise ValueError("edit_report: live_name must be non-empty.")
        if old_str == "":
            raise ValueError(
                "edit_report: old_str must be a non-empty substring; full-body "
                "rewrites should go through return_report."
            )

        seen = extract_seen_live_names(context.messages)
        target_lid = await self._resolve_artifact_slug(
            live_name, kind="report", seen_live_names=seen
        )

        log_path = f".ockham/reports/{target_lid}/log.jsonl"
        try:
            raw_log = await self.code_executor.read_workspace_file(log_path)
        except FileNotFoundError as e:
            raise ValueError(
                f"edit_report: report {live_name!r} has no log.jsonl "
                "— it has not been published yet."
            ) from e
        last_csha = last_content_sha_from_log(raw_log)
        if last_csha is None:
            raise ValueError(
                f"edit_report: report {live_name!r} log.jsonl has no usable entries."
            )

        snapshot_path = f".ockham/reports/{target_lid}/{last_csha}.report.md"
        raw = await self.code_executor.read_workspace_file(snapshot_path)
        markdown = raw.decode("utf-8")
        n = markdown.count(old_str)
        if n == 0:
            raise ValueError("edit_report: old_str not found in report markdown.")
        if n > 1:
            raise ValueError(
                "edit_report: old_str occurs multiple times; provide a more specific target."
            )
        new_markdown = markdown.replace(old_str, new_str, 1)

        new_embedded = embedded_refs_from_markdown(new_markdown)
        await self._validate_refs_resolve(new_embedded)

        cur_path = f".ockham/reports/{target_lid}/curation.json"
        try:
            raw_cur = await self.code_executor.read_workspace_file(cur_path)
            curation = json.loads(raw_cur.decode("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            curation = {}

        return Report(
            logical_id=target_lid,
            title=str(curation.get("title", "") or ""),
            description=str(curation.get("description", "") or ""),
            notes=list(curation.get("notes") or []),
            tags=list(curation.get("tags") or []),
            markdown=new_markdown,
            embedded_refs=new_embedded,
            live_name=curation.get("live_name") if isinstance(curation.get("live_name"), str) else live_name,
        )

    @toolmethod(
        name="refresh",
        description=(
            "Re-derive an existing dataset / chart / report from latest upstream. "
            "Identify the target by its workspace slug (live_name). Walks lineage, "
            "re-runs producing notebooks, appends a new content_sha under the same "
            "logical_id. Idempotent when no upstream byte changed."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "live_name": {
                    "type": "string",
                    "description": "The artifact's workspace slug (live_name).",
                },
            },
            "required": ["live_name"],
            "additionalProperties": False,
        },
        tool_type="return",
        ui_message="Refreshing",
    )
    async def refresh(
        self,
        *,
        context: AgentContext,
        live_name: str,
    ) -> Dataset | Chart | Report:
        live_name = (live_name or "").strip()
        if not live_name:
            raise ValueError("refresh: live_name must be non-empty.")
        seen = extract_seen_live_names(context.messages)
        target_ref = await self._resolve_slug_to_latest_ref(
            live_name, seen_live_names=seen
        )

        new_ref = await refresh_artifact(target_ref, executor=self.code_executor)

        # Read the freshly persisted snapshot back into a typed model so
        # the streaming layer can emit the standard pill card and
        # idempotently re-persist (same content_sha → same path → no-op).
        blob = await self.code_executor.read_workspace_file(new_ref.workspace_file_path)
        if new_ref.kind == "dataset":
            from parsimony_agents.dataset_io import deserialize_dataset

            _result, dataset = deserialize_dataset(blob)
            payload = await self.code_executor.get(dataset.variable_name)
            if payload is None:
                raise ValueError(
                    f"refresh: dataset variable {dataset.variable_name!r} "
                    "is not in the kernel after refresh."
                )
            return dataset.with_payload(payload)
        if new_ref.kind == "chart":
            from parsimony_agents.chart_io import deserialize_chart

            chart, _spec = deserialize_chart(blob)
            payload = await self.code_executor.get(chart.variable_name)
            if payload is None:
                raise ValueError(
                    f"refresh: chart variable {chart.variable_name!r} is "
                    "not in the kernel after refresh."
                )
            return chart.with_payload(payload)
        if new_ref.kind == "report":
            # Reports have no kernel payload — read curation + bytes
            # directly. The streaming layer's persist path handles the
            # rest idempotently.
            return await self._reload_report_for_refresh(new_ref, blob)
        raise AssertionError(f"refresh: unreachable kind {new_ref.kind!r}")

    async def _reload_report_for_refresh(
        self, ref: ArtifactRef, blob: bytes
    ) -> Report:
        """Reconstruct a :class:`Report` model from disk after refresh.

        Reports have no in-kernel payload — snapshot bytes ARE the
        markdown source. Curation lives in the sibling ``curation.json``;
        embedded refs are recovered by parsing the markdown's
        ``.ockham/<kind>s/<lid>/<csha>.<ext>`` paths (the same shape
        :func:`refresh.embedded_refs_from_markdown` produces).
        """
        from parsimony_agents.refresh import embedded_refs_from_markdown

        markdown = blob.decode("utf-8")
        cur_path = f".ockham/reports/{ref.logical_id}/curation.json"
        try:
            raw_cur = await self.code_executor.read_workspace_file(cur_path)
            cur = json.loads(raw_cur.decode("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            cur = {}
        return Report(
            logical_id=ref.logical_id,
            content_sha=ref.content_sha,
            title=cur.get("title", "") or "",
            description=cur.get("description", "") or "",
            tags=list(cur.get("tags") or []),
            notes=list(cur.get("notes") or []),
            live_name=cur.get("live_name"),
            markdown=markdown,
            embedded_refs=embedded_refs_from_markdown(markdown),
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

    async def _validate_refs_resolve(self, refs: list[ArtifactRef]) -> None:
        """Each ref must resolve to bytes on disk.

        Used for ``return_report``: every embedded ``.ockham/...`` path
        in the markdown body must point at a real snapshot. Refresh and
        return_dataset/chart derive their refs from the framework (the
        run scope and the snapshot store), so they never hit this path.
        """
        for ref in refs:
            try:
                await self.code_executor.read_workspace_file(ref.workspace_file_path)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Embedded ref {ref.kind}:{ref.logical_id}:{ref.content_sha} does not "
                    f"resolve ({ref.workspace_file_path!r} not found). Reports embed "
                    "artifacts via ![](file://./.ockham/<kind>s/<lid>/<csha>.<ext>) — "
                    "the path must be one of the rows in <turn_artifacts>."
                ) from e

    async def _lineage_for_variable(
        self,
        variable_name: str,
        *,
        context: AgentContext,
    ) -> tuple[list[ArtifactRef], list[ArtifactRef]]:
        """Return ``(notebook_refs, source_refs)`` for a dataset variable.

        Pulls the variable's :class:`VariableOrigin` from the kernel's
        :class:`OriginLedger`. The producing notebook is the singular
        ``notebook_refs`` entry; ``source_refs`` is the union of the
        run's observed connector fetches and load_dataset events.

        The producing notebook must have a persisted snapshot (i.e. it
        was published via ``return_notebook``) — otherwise this is the
        "published artifact must come from a published recipe" rule
        firing as a hard error rather than soft prose.
        """
        origin = await self.code_executor.get_origin(variable_name)
        if origin is None:
            raise ValueError(
                f"Variable '{variable_name}' has no producing notebook on record. "
                "It was likely assigned in dry_execute_code or before any notebook ran. "
                "Publish a notebook that assigns it via return_notebook(execute=true) "
                "first."
            )
        notebook_ref = await self._notebook_ref_for_published_path(
            origin.notebook_path, context=context
        )
        source_refs = list(origin.load_refs) + list(origin.fetch_refs)
        return [notebook_ref], source_refs

    async def _chart_lineage_for_variable(
        self,
        variable_name: str,
        *,
        context: AgentContext,
    ) -> tuple[list[ArtifactRef], list[ArtifactRef], list[ArtifactRef]]:
        """Return ``(notebook_refs, source_refs, source_dataset_refs)`` for a chart variable.

        Same model as :meth:`_lineage_for_variable` but partitions
        ``source_refs`` into dataset (load) edges and data_object (fetch)
        edges — charts carry them on separate fields.
        """
        origin = await self.code_executor.get_origin(variable_name)
        if origin is None:
            raise ValueError(
                f"Variable '{variable_name}' has no producing notebook on record."
            )
        notebook_ref = await self._notebook_ref_for_published_path(
            origin.notebook_path, context=context
        )
        return [notebook_ref], list(origin.fetch_refs), list(origin.load_refs)

    async def _notebook_ref_for_published_path(
        self, working_copy_path: str, *, context: AgentContext
    ) -> ArtifactRef:
        """Resolve a notebook path to its latest persisted :class:`ArtifactRef`.

        The notebook MUST have a ``log.jsonl`` — i.e. the agent must
        have published it via ``return_notebook`` at least once. Without
        that, this is the "published artifact must come from a published
        recipe" check failing loud: scratch-cell or unpublished notebook
        bytes cannot be cited as the producer of a published deliverable.
        """
        logical_id = await Agent._resolve_notebook_logical_id(working_copy_path, context)
        try:
            _raw, latest_csha = await read_latest_notebook(
                self.code_executor, logical_id=logical_id
            )
        except FileNotFoundError as e:
            raise ValueError(
                f"Notebook {working_copy_path!r} has not been published yet. "
                "Published deliverables (datasets/charts) must come from a "
                "published recipe — call return_notebook on this path first."
            ) from e
        return ArtifactRef(kind="notebook", logical_id=logical_id, content_sha=latest_csha)

    async def _resolve_artifact_slug(
        self,
        live_name: str,
        *,
        kind: str | None = None,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> str:
        """Resolve a workspace slug to a typed artifact's ``logical_id``.

        Scans ``.ockham/<kind>s/*/curation.json`` for a unique
        ``live_name`` match. When ``kind`` is given, only that kind is
        searched. Raises :class:`ValueError` on miss or ambiguity with a
        message that names the right discovery surface.

        Cross-terminal gate: when ``seen_live_names`` is provided and the
        match's ``(kind, live_name)`` pair is not in it, raise
        :class:`LiveNameCollisionError` — the artifact belongs to a
        sibling terminal and the agent must ``read_artifact`` it first.
        Passing ``None`` skips the gate for programmatic callers.
        """
        kinds = (kind,) if kind else ("dataset", "chart", "report")
        matches: list[tuple[str, str]] = []  # (kind, logical_id)
        for k in kinds:
            try:
                _ = await self.code_executor.list_workspace_files(f".ockham/{k}s")
            except FileNotFoundError:
                continue
            entries = await self.code_executor.list_workspace_files(f".ockham/{k}s")
            seen: set[str] = set()
            for rel, _size in entries:
                # ``rel`` ≈ ``.ockham/datasets/<lid>/curation.json`` or snapshot.
                parts = rel.split("/")
                if len(parts) < 4 or parts[-1] != "curation.json":
                    continue
                lid = parts[2]
                if lid in seen:
                    continue
                seen.add(lid)
                try:
                    raw = await self.code_executor.read_workspace_file(rel)
                except FileNotFoundError:
                    continue
                try:
                    data = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(data, dict):
                    continue
                if data.get("kind") != k:
                    continue
                if data.get("live_name") == live_name:
                    matches.append((k, lid))
        if not matches:
            kind_label = kind or "dataset/chart/report"
            raise ValueError(
                f"No {kind_label} has live_name {live_name!r}. Use the slug "
                "shown in <turn_artifacts>."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Slug {live_name!r} matches multiple artifacts. Rename one "
                "via curation before referring to it."
            )
        matched_kind, matched_lid = matches[0]
        if (
            seen_live_names is not None
            and (matched_kind, live_name) not in seen_live_names
        ):
            raise LiveNameCollisionError(
                live_name=live_name,
                existing_logical_id=matched_lid,
                kind=matched_kind,
            )
        return matched_lid

    async def _resolve_slug_to_latest_ref(
        self,
        live_name: str,
        *,
        seen_live_names: set[tuple[str, str]] | None = None,
    ) -> ArtifactRef:
        """Resolve a slug to the latest :class:`ArtifactRef` across kinds.

        Used by :meth:`refresh`. Determines the kind from the curation
        match, then walks ``log.jsonl`` for the latest ``content_sha``.

        ``seen_live_names`` is forwarded to :meth:`_resolve_artifact_slug`;
        a :class:`LiveNameCollisionError` from that call propagates
        unchanged (the caller catches it and surfaces the recovery
        instruction to the LLM).
        """
        kinds = ("dataset", "chart", "report")
        for k in kinds:
            try:
                lid = await self._resolve_artifact_slug(
                    live_name, kind=k, seen_live_names=seen_live_names
                )
            except ValueError:
                continue
            # LiveNameCollisionError NOT swallowed here — surface the
            # cross-terminal failure to the agent's tool framework.
            log_path = f".ockham/{k}s/{lid}/log.jsonl"
            try:
                raw_log = await self.code_executor.read_workspace_file(log_path)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Artifact {live_name!r} has no log.jsonl."
                ) from e
            last_csha = last_content_sha_from_log(raw_log)
            if not last_csha:
                raise ValueError(
                    f"Artifact {live_name!r} log.jsonl is empty."
                )
            return ArtifactRef(kind=k, logical_id=lid, content_sha=last_csha)
        raise ValueError(
            f"No published artifact has live_name {live_name!r}."
        )
