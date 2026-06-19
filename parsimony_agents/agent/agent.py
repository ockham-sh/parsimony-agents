from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from opentelemetry import trace
from parsimony.connector import Connectors
from parsimony.result import Result
from pydantic import TypeAdapter

from parsimony_agents.agent.cancellation import CancellationRequest
from parsimony_agents.agent.config import AgentGuardrails, FileStore
from parsimony_agents.agent.events import (
    StateSnapshot,
    ToolEvent,
)
from parsimony_agents.agent.helpers import (
    TurnState,
    render_connector_catalog,
    render_connector_skills,
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
from parsimony_agents.agent.seen_refs import extract_seen_live_names
from parsimony_agents.agent.xml_render import escape_attr, escape_text
from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.execution import (
    DataFrameObject,
    FigureObject,
)
from parsimony_agents.execution.executor import BaseCodeExecutor
from parsimony_agents.execution.factory import OutputFactory as FrameworkOutputFactory
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.execution.parquet_helpers import parquet_summary
from parsimony_agents.identity import (
    ArtifactRef,
    LiveNameCollisionError,
    SnapshotKind,
    chart_logical_id,
    dataset_logical_id,
    notebook_content_sha,
    notebook_logical_id,
    report_logical_id,
)
from parsimony_agents.messages import Text
from parsimony_agents.notebook import Script
from parsimony_agents.notebook_io import (
    deserialize_notebook,
    last_content_sha_from_log,
    read_latest_notebook,
    serialize_notebook,
)
from parsimony_agents.refresh import (
    embedded_refs_from_markdown,
    extract_embed_keys_from_markdown,
    refresh_artifact,
)
from parsimony_agents.tools import ToolMethod, Tools, toolmethod

if TYPE_CHECKING:
    from parsimony_agents.agent.state import SuspensionRecord

# Tool message for cooperative cancellation; keeps one tool output per tool call id.
# Used by ``_emit_cancelled_tool_events`` below; the loop body that also referenced
# it moved to ``workspace_hooks.py`` (which keeps its own copy).
CANCELLED_TOOL_TEXT = "Cancelled by user before the tool completed."

logger = logging.getLogger("parsimony_agents")
error_logger = logging.getLogger("parsimony_agents.errors")


litellm.REPEATED_STREAMING_CHUNK_LIMIT = 100


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
    lines = [f"{len(items)} artifact(s) (most recent first). Inspect with read_artifact(live_name=…, kind=…):"]
    for item in items:
        kind_label = item.get("kind", "?")
        live_name = item.get("live_name", "?")
        summary = (item.get("summary") or "").strip()
        suffix = f" — {summary}" if summary else ""
        lines.append(f"  [{kind_label}] {live_name}{suffix}")
    return "\n".join(lines)


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

    reports: dict[str, Report] = field(default_factory=dict)
    """Returned :class:`Report` objects keyed by ``logical_id``."""

    context: AgentContext | None = None
    """Final :class:`AgentContext` — use for multi-turn continuation or inspection."""

    events: list[Any] = field(default_factory=list)
    """Full event log (every ``AgentEvent`` yielded during the run)."""

    @property
    def ok(self) -> bool:
        """``True`` if the run finished without an error or terminal failure.

        ``handoff`` and ``partial_run_summary`` are non-interactive terminal
        failures (the agent gave up, or ran out of budget). They carry no
        ``error`` event, so they must be checked explicitly — otherwise a run
        that handed off on a missing API key or an unrecoverable provider error
        would report ``ok``.
        """
        failed = {"error", "handoff", "partial_run_summary"}
        return not any(getattr(e, "type", None) in failed for e in self.events)

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
            elif isinstance(result, Report) and result.logical_id:
                self.reports[result.logical_id] = result


def _inject_connector_catalog(ctx: AgentContext, connectors: Any) -> None:
    """Place the connector catalog as a stable ``role="user"`` message at ``ctx.messages[1]``.

    The catalog is static for the whole session. As a fixed message right after
    the system prompt it sits inside the provider's cached prefix — billed once
    per session — instead of riding the volatile per-iteration ``<session_state>``
    snapshot, where its ~15-20k tokens would be re-sent uncached every iteration.

    Filtered-then-reinserted so a connector rebind between turns refreshes it; the
    content is otherwise byte-identical, which is what keeps the cached prefix
    stable. No-op when there are no connectors.
    """
    ctx.messages = [m for m in ctx.messages if not m.metadata.get("connectors_catalog", False)]
    catalog = render_connector_catalog(connectors)
    if catalog:
        ctx.messages.insert(
            1,
            AgentMessage(
                role="user",
                content=Text(content=f"<available_connectors>\n{catalog}\n</available_connectors>"),
                metadata={"connectors_catalog": True},
            ),
        )


def _inject_connector_skills(ctx: AgentContext, connectors: Any) -> None:
    """Place the connector skills as a stable ``role="user"`` message in the cached prefix.

    Like the connector catalog (see :func:`_inject_connector_catalog`), the skills are static
    for the whole session, so they sit in the provider's cached prefix — billed once — rather
    than the volatile per-iteration snapshot. Inserted right after the catalog message (or the
    system prompt when there are no connectors) so ordering, and therefore the cached prefix,
    stays byte-stable. Filtered-then-reinserted so a connector rebind refreshes it. No-op when
    no bound bundle carries a skill.
    """
    ctx.messages = [m for m in ctx.messages if not m.metadata.get("connector_skills", False)]
    skills_text = render_connector_skills(connectors)
    if not skills_text:
        return
    after_catalog = next(
        (i for i, m in enumerate(ctx.messages) if m.metadata.get("connectors_catalog", False)),
        0,
    )
    ctx.messages.insert(
        after_catalog + 1,
        AgentMessage(
            role="user",
            content=Text(content=f"<connector_skills>\n{skills_text}\n</connector_skills>"),
            metadata={"connector_skills": True},
        ),
    )


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
        guardrails: AgentGuardrails | None = None,
        session_id: str | None = None,
        file_store: FileStore | None = None,
        # Opaque host model identifier. Not interpreted by the agent — carried
        # into SuspensionRecord so Agent.resume can rebuild on the same model.
        # The host resolves model_id → model_config separately.
        model_id: str | None = None,
        # --- Failure-handling spine ---
        # ``policy`` drives :func:`handle_failure` retry/backoff/handoff decisions
        # (see :class:`DefaultPolicy`). ``suspension_secret`` is the HMAC key
        # used to sign :class:`SuspensionRecord` tokens — when omitted, the
        # session_id is reused (acceptable for in-process v1; v2 cross-process
        # resume will require a real shared secret).
        policy: Any | None = None,
        suspension_secret: str | None = None,
        read_artifact_fn: Callable[[str, str, dict[str, Any]], Awaitable[ArtifactLlmResult]] | None = None,
        list_artifacts_fn: Callable[[str | None, str | None, int], Awaitable[list[dict[str, Any]]]] | None = None,
    ):
        from parsimony_agents.agent.prompts import DEFAULT_DATA_ANALYSIS_PROMPT
        from parsimony_agents.execution.executor import CodeExecutor as _LocalExecutor

        # Resolve model_config: explicit > built from model= convenience param
        if model_config is not None:
            resolved_config: dict[str, Any] = model_config
        elif model is not None:
            resolved_config = {"model": model, **({"api_key": api_key} if api_key else {})}
        else:
            raise TypeError("Agent requires either model_config={...} or model='model-name'")

        # Resolve instructions: explicit > default prompt. The connector catalog
        # is *not* appended here — connectors live in the executor namespace and
        # are advertised per-turn via AgentContextSnapshot.connectors_catalog,
        # so the system prompt stays stable and cache-friendly.
        resolved_instructions = instructions if instructions is not None else DEFAULT_DATA_ANALYSIS_PROMPT
        if connectors is not None and not isinstance(connectors, (Connectors, Mapping)):
            raise TypeError(
                f"connectors must be a Connectors or Mapping[str, Connectors]; got {type(connectors).__name__}"
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
        self.model_id = model_id
        self.file_store = file_store
        self._connectors = connectors
        self.code_executor = resolved_executor
        self._output_factory = output_factory

        self.figures = []

        # Standalone artifact discovery. A workspace host (terminal) injects
        # read/list fns and populates session_state itself. The OSS front door
        # has no host, so default both to the local ``.ockham/`` tree the
        # executor already writes — otherwise list_artifacts / read_artifact are
        # unregistered and <turn_artifacts> is empty, and the agent (instructed
        # to reuse existing artifacts) loops on a follow-up turn with nothing to
        # discover. ``_local_discovery`` then also drives session_state rebuild
        # in :meth:`run` each turn.
        # All-or-nothing: local discovery engages only when the host injects
        # neither fn. A host that provides its own surface must provide both
        # (read + list) plus its own session_state — we never half-fill, which
        # would give a local list_artifacts with no <turn_artifacts>.
        self._local_discovery = read_artifact_fn is None and list_artifacts_fn is None
        if self._local_discovery:
            from parsimony_agents.agent.local_store import (
                list_local_artifacts,
                read_local_artifact,
            )

            def _local_dir() -> Path:
                return Path(getattr(resolved_executor, "cwd", None) or ".")

            async def read_artifact_fn(live_name, kind, options):  # type: ignore[misc]
                return read_local_artifact(_local_dir(), live_name, kind, options)

            async def list_artifacts_fn(query, kind, limit):  # type: ignore[misc]
                return list_local_artifacts(_local_dir(), query, kind, limit)

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
            ]
        )

        # Explicit-termination tools. ``ask_user`` lets the agent pause for
        # clarification; ``return_done`` / ``return_unable`` are the explicit
        # end-of-run signals — a text-only response is not a valid stop.
        # Imported lazily so the spine stays out of the module-load cycle.
        from parsimony_agents.agent.termination_tools import TERMINATION_TOOLS

        self.system_tools = Tools(list(_system_tool_methods) + list(TERMINATION_TOOLS))

        self.model_config = resolved_config

        self.guardrails = guardrails or AgentGuardrails()

        # Failure-handling spine attributes; resolved lazily to avoid importing
        # the spine at module load time (recovery transitively imports events,
        # which imports Failure from this package).
        from parsimony_agents.agent.failure import DefaultPolicy as _DefaultPolicy

        self.policy = policy if policy is not None else _DefaultPolicy()
        self.suspension_secret = suspension_secret if suspension_secret is not None else self.session_id
        # AgentLike protocol completeness: ``run_loop`` reads ``agent.tools``
        # while the workspace code reads ``agent.system_tools`` — alias the two.
        self.tools = self.system_tools

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
            print(result.datasets)    # Dataset objects keyed by logical_id
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
        # Lazy imports for the failure-handling spine: keeps the Agent module
        # cheap to import and breaks the cycle between recovery (which imports
        # events) and this module (which exports the Agent).
        from datetime import datetime

        from parsimony_agents.agent.loop import run_loop
        from parsimony_agents.agent.state import RunState
        from parsimony_agents.agent.workspace_hooks import WorkspaceRunHooks

        if isinstance(user_message, str):
            user_message = Text(content=user_message)

        agent_span = trace.get_current_span()
        logger.info("Agent run started", extra={"prompt_preview": user_message.content[:1000]})
        start_time = time.time()

        # --- System prompt + workspace ctx setup (unchanged) ----------------
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
            await self.code_executor.set_cwd(str(ctx.files.get_files_dir()), session_id=self.session_id)

        await self._setup_connectors()

        # Connector catalog + skills → stable cached-prefix messages (see helper docstrings).
        _inject_connector_catalog(ctx, self._connectors)
        _inject_connector_skills(ctx, self._connectors)

        # Standalone: rebuild session_state from the local .ockham/ tree at the
        # start of every turn so <turn_artifacts> lists prior-turn work the agent
        # can reuse. Within a turn, freshness comes from minted_refs fusion in
        # the iteration-start hook; this only seeds turn-start discovery. A host
        # populates session_state itself, so we leave it untouched there.
        if self._local_discovery:
            from parsimony_agents.agent.local_store import build_local_session_state

            local_dir = Path(getattr(self.code_executor, "cwd", None) or ".")
            ctx.session_state = build_local_session_state(self.code_executor, local_dir)
            ctx.local_discovery = True

        # --- Legacy workspace turn state (still drives ref minting) --------
        turn_state = TurnState()

        # --- Failure-handling spine state (mirrors ctx.messages 1:1) -------
        # ``state.messages`` is what pre_step / post_llm / render_for_llm read;
        # ``ctx.messages`` is what every existing workspace helper / event
        # consumer reads. Kept identical by routing every append through
        # ``WorkspaceRunHooks._append_message`` (and the seed appends below).
        state = RunState(
            run_id=str(uuid4()),
            session_id=self.session_id,
            model_id=self.model_id,
            messages=list(ctx.messages),
            started_at=datetime.now(UTC),
        )

        # Continuation request: reset wall-clock budget but keep transcript.
        is_continuation = user_message.content.strip().lower() == "continue"
        if is_continuation:
            start_time = time.time()
            logger.info(
                "Continuation requested - resetting timer",
                extra={
                    "max_iters": self.guardrails.max_iterations,
                    "max_execution_time": self.guardrails.max_execution_time_s,
                },
            )
        else:
            user_msg = AgentMessage(role="user", content=user_message)
            ctx.messages.append(user_msg)
            state.messages.append(user_msg)

        # Seed an initial context snapshot so the first iteration has one.
        # No ``connectors=``: the catalog is its own stable prefix message
        # (see ``_inject_connector_catalog``); the snapshot stays volatile-only.
        context_snapshot = await ctx.to_snapshot()
        ctx.messages = [m for m in ctx.messages if m.metadata.get("context_snapshot", False) is False]
        ctx.messages.append(AgentMessage(role="user", content=context_snapshot, metadata={"context_snapshot": True}))
        state.messages = list(ctx.messages)

        hooks = WorkspaceRunHooks(
            agent=self,
            ctx=ctx,
            turn_state=turn_state,
            cancellation=cancellation,
            tool_choice=tool_choice,
            start_time=start_time,
            agent_span=agent_span,
        )
        async for event in run_loop(hooks, state, cancellation=cancellation):
            yield event

    async def resume(
        self,
        suspension: SuspensionRecord,
        user_reply: str,
        *,
        cancellation: CancellationRequest | None = None,
        max_suspension_age_s: float | None = 24 * 3600.0,
        configure_ctx: Callable[[AgentContext], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[Any, None]:
        """Resume a run that suspended via ``ask_user``.

        Validates the suspension token + staleness, rebuilds the workspace
        :class:`AgentContext` and :class:`RunState` from the record, appends
        ``user_reply`` as the next user message, and re-enters the loop through
        :class:`WorkspaceRunHooks` — the same path :meth:`run` uses, so a resumed
        run gets context snapshots, rich tool dispatch, and event emission
        identically to a fresh run.

        ``configure_ctx`` re-applies the host's ctx seams to the rebuilt context.
        On a fresh turn the host sets these (``report_validator``,
        ``notebook_logical_id_resolver``, ``session_state``, …) on the ``ctx`` it
        passes to :meth:`run`; resume rebuilds ``ctx`` from the record, which is
        ``exclude=True`` for the runtime-only seams, so without this hook every
        host seam would silently revert to ``None`` on resume (a report authored on
        a resumed turn would skip the trust-boundary validator, etc.). The callback
        runs on the rebuilt ctx before the first iteration, so any seam the host
        adds is covered without changing this signature again.

        :raises SuspensionTokenMismatch: the record's token fails HMAC verification.
        :raises SuspensionExpired: the record is older than ``max_suspension_age_s``.
        :raises ValueError: ``user_reply`` is empty.
        """
        from datetime import datetime

        from parsimony_agents.agent.failure import (
            SuspensionExpired,
            SuspensionTokenMismatch,
            verify_suspension_token,
        )
        from parsimony_agents.agent.loop import run_loop
        from parsimony_agents.agent.state import RunState
        from parsimony_agents.agent.workspace_hooks import WorkspaceRunHooks

        if not user_reply or not user_reply.strip():
            raise ValueError("resume requires a non-empty user_reply")

        if not verify_suspension_token(record=suspension, secret=self.suspension_secret):
            raise SuspensionTokenMismatch(f"suspension token failed verification for run_id={suspension.run_id!r}")

        if max_suspension_age_s is not None:
            age = (datetime.now(UTC) - suspension.suspended_at).total_seconds()
            if age > max_suspension_age_s:
                raise SuspensionExpired(f"suspension is {age:.0f}s old (max {max_suspension_age_s:.0f}s)")

        agent_span = trace.get_current_span()
        logger.info(
            "Agent resume started",
            extra={"run_id": suspension.run_id, "reply_preview": user_reply[:1000]},
        )
        start_time = time.time()

        # --- System prompt + workspace ctx rebuilt from the record ----------
        if isinstance(self.instructions, str):
            system_message = AgentMessage(role="system", content=Text(content=self.instructions.rstrip()))
        else:
            system_message = AgentMessage(role="system", content=self.instructions)

        # AgentContext validates ``messages`` against AgentMessage at construction,
        # but a restored transcript legitimately mixes AgentMessage and Message
        # (context-snapshot rows). Build minimally, then assign by mutation — the
        # same sidestep ``run`` relies on (assignment is not re-validated).
        ctx = AgentContext(messages=[system_message], session_id=self.session_id)
        if suspension.messages:
            restored = list(suspension.messages)
            restored[0] = system_message  # refresh the system prompt
            ctx.messages = restored

        if self.session_id and self.file_store is not None:
            ctx.files = self.file_store
            await self.code_executor.set_cwd(str(ctx.files.get_files_dir()), session_id=self.session_id)

        await self._setup_connectors()

        # Connector catalog + skills → stable cached-prefix messages (see helper docstrings).
        _inject_connector_catalog(ctx, self._connectors)
        _inject_connector_skills(ctx, self._connectors)

        # Standalone: rebuild session_state from the local .ockham/ tree so the
        # resumed turn sees prior artifacts in <turn_artifacts> (mirrors run();
        # otherwise a resumed standalone run loops the same way an un-fixed
        # follow-up turn did). A host populates session_state itself.
        if self._local_discovery:
            from parsimony_agents.agent.local_store import build_local_session_state

            local_dir = Path(getattr(self.code_executor, "cwd", None) or ".")
            ctx.session_state = build_local_session_state(self.code_executor, local_dir)
            ctx.local_discovery = True

        # Re-apply the host's runtime-only ctx seams (report_validator,
        # notebook_logical_id_resolver, session_state, …) onto the rebuilt ctx.
        # None of them are carried in the suspension record (report_validator and
        # notebook_logical_id_resolver are exclude=True; session_state simply is
        # not a SuspensionRecord field), so without this the host loses every seam
        # on resume. Runs after the standalone local_discovery block so a host
        # (which leaves local_discovery False) is unaffected by it.
        if configure_ctx is not None:
            await configure_ctx(ctx)

        # --- Turn state carries forward refs minted before suspension -------
        turn_state = TurnState(
            minted_refs=list(suspension.minted_refs),
            minted_live_names=dict(suspension.minted_live_names),
        )

        # --- Spine state rebuilt from the record ----------------------------
        state = RunState.from_suspension(suspension, cancellation=cancellation)
        state.messages = list(ctx.messages)

        # --- Append the user's reply as a normal user message (BRIEF §4.2) --
        reply_msg = AgentMessage(role="user", content=Text(content=user_reply.strip()))
        ctx.messages.append(reply_msg)
        state.messages.append(reply_msg)

        # --- Seed an initial context snapshot so iteration 1 has one --------
        # No ``connectors=``: the catalog is its own stable prefix message
        # (see ``_inject_connector_catalog``); the snapshot stays volatile-only.
        context_snapshot = await ctx.to_snapshot()
        ctx.messages = [m for m in ctx.messages if m.metadata.get("context_snapshot", False) is False]
        ctx.messages.append(AgentMessage(role="user", content=context_snapshot, metadata={"context_snapshot": True}))
        state.messages = list(ctx.messages)

        hooks = WorkspaceRunHooks(
            agent=self,
            ctx=ctx,
            turn_state=turn_state,
            cancellation=cancellation,
            tool_choice="auto",
            start_time=start_time,
            agent_span=agent_span,
        )
        yield StateSnapshot(context=ctx)
        async for event in run_loop(hooks, state, cancellation=cancellation):
            yield event

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
        if ttype == "system" and (
            tools[tool_name].ui_message is not None or tools[tool_name].ui_message_completed is not None
        ):
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
        name="dry_execute_code",
        description=(
            "Run scratch Python against a throwaway copy of the kernel: stdout/display() land in the "
            "conversation and you can read existing variables, but assignments here do NOT persist and "
            "no notebook is published. To keep or later search a result, produce it in a real cell. "
            "_ui_message is a short past-tense line shown to the user."
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

        return UtilityToolOutput(metadata=metadata, content=kernel_output, ui_message="Executing temporary code")

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
            f"{escape_text(c.name)} ({escape_text(c.role.value)}, {escape_text(c.dtype)})" for c in result.columns
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
                        "Workspace slug, exactly as it appears in <turn_artifacts> or in a list_artifacts row."
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
            raise RuntimeError("read_artifact is not enabled for this agent configuration")
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
                    "enum": ["notebook", "dataset", "chart", "report"],
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
            raise RuntimeError("list_artifacts is not enabled for this agent configuration")
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
        with contextlib.suppress(ValueError):
            await self._resolve_notebook_logical_id(normalized, context)
        # Canonicalize newlines via in-memory round-trip — same shape the
        # snapshot store will write, so ``content_sha`` matches what the
        # streaming layer's persist step computes from the ScriptPreview.
        canonical = deserialize_notebook(serialize_notebook(Script(path=normalized, code=code)), path=normalized).code
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
            ko = await self._stamp_notebook_ref(ko, canonical, normalized, context)
            return ko
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
            ko = await self._stamp_notebook_ref(ko, script.code, path, context)
            return ko
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
    async def _resolve_notebook_logical_id(working_copy_path: str, context: AgentContext) -> str:
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
    async def _notebook_ref_for(code: str, working_copy_path: str, context: AgentContext) -> ArtifactRef:
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
                f"Dataset '{dataset_variable_name}' must resolve to a pandas DataFrame; got {type(out_obj).__name__}."
            )

        nb_refs, src_refs = await self._lineage_for_variable(dataset_variable_name, context=context)

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
                f"Variable '{chart_variable_name}' is not in the kernel. Run the notebook that creates it first."
            )
        if not isinstance(fig_obj, FigureObject):
            raise TypeError(
                f"chart_variable_name '{chart_variable_name}' must resolve to an Altair chart; "
                f"got {type(fig_obj).__name__}."
            )
        if fig_obj.name is None:
            fig_obj.name = chart_variable_name

        nb_refs, src_refs, ds_refs = await self._chart_lineage_for_variable(chart_variable_name, context=context)
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
            "Publish a markdown report rendered by Quarto. The framework parses embedded "
            "artifact URIs from the body, freezes a pin map alongside the snapshot so old "
            "reports stay byte-stable when an embedded artifact is later renamed, and "
            "renders each requested format. Title and subtitle render at the top of "
            "documents and as the cover slide of decks — they live in the parsimony "
            "metadata, NOT as a leading '# Title' heading in the body. Don't write Quarto "
            "YAML, themes, or shortcodes — the framework owns rendering. See section F of "
            "the system prompt for tables-vs-charts and slide-deck composition guidance."
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Display title rendered at the top of documents and on the cover "
                        "slide of decks. Do NOT also include it as a leading '# Title' "
                        "heading in markdown — that would duplicate it."
                    ),
                },
                "subtitle": {
                    "type": "string",
                    "description": (
                        "Optional one-line subtitle rendered below the title in documents "
                        "and on the cover slide in decks. Omit when there is nothing to "
                        "add — empty subtitles render no extra line."
                    ),
                },
                "markdown": {
                    "type": "string",
                    "description": (
                        "Body content only (no leading '# Title' — that comes from the "
                        "title parameter). Prose plus embedded artifacts referenced by "
                        "live_name. Embed syntax: ![alt](file://./charts/<live_name>.vl.json) "
                        "for charts, ![alt](file://./data/<live_name>.parquet) for datasets "
                        "(renders as a table). Only charts and datasets are embeddable — "
                        "notebooks, data_objects, and other reports are not."
                    ),
                },
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
                    "description": "File-tree slug (no extension); the handle other tools reference this report by.",
                },
                "formats": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["html", "pdf", "pptx", "dashboard", "revealjs"]},
                    "description": (
                        "Output formats Quarto should produce. Defaults to ['html','pdf']. "
                        "Slide formats (pptx, revealjs) slice the body on H2 boundaries — "
                        "see section F of the system prompt for deck composition."
                    ),
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
        subtitle: str | None = None,
        tags: list[str] | None = None,
        formats: list[str] | None = None,
    ) -> Report:
        from parsimony_agents.report_format import DEFAULT_FORMATS, VALID_FORMATS

        title = title.strip()
        if not title:
            raise ValueError("title must be a non-empty string.")
        if not markdown.strip():
            raise ValueError("markdown must be a non-empty string.")
        live_name = (live_name or "").strip()
        if not live_name:
            raise ValueError("live_name must be a non-empty workspace slug.")
        subtitle_value = (subtitle or "").strip()
        notes = TypeAdapter(list[str]).validate_python(notes)
        # Body embeds artifacts by live_name. We walk every URI in the
        # body, look up the artifact's latest snapshot via curation, and
        # build a frozen ``live_name -> ArtifactRef`` pin map that
        # travels with the snapshot bytes. The renderer resolves embeds
        # against THIS map (not current curation), so renaming an
        # embedded artifact after publication never silently mutates an
        # old render. Closes TODO(report-embed-by-live_name).
        embed_keys = extract_embed_keys_from_markdown(markdown)
        pin_map = await self._build_report_pin_map(embed_keys)
        emb = embedded_refs_from_markdown(markdown, pin_map)
        await self._validate_refs_resolve(emb)
        final_tags: list[str] = []
        for t in TypeAdapter(list[str]).validate_python(tags or []):
            s = str(t).strip()
            if s and s not in final_tags:
                final_tags.append(s)
        chosen_formats: list[str]
        if formats is None or not formats:
            chosen_formats = list(DEFAULT_FORMATS)
        else:
            seen: set[str] = set()
            chosen_formats = []
            for f in TypeAdapter(list[str]).validate_python(formats):
                s = f.strip()
                if not s or s in seen:
                    continue
                if s not in VALID_FORMATS:
                    raise ValueError(f"return_report: unknown format {s!r}; pick from {sorted(VALID_FORMATS)}.")
                seen.add(s)
                chosen_formats.append(s)
            if not chosen_formats:
                chosen_formats = list(DEFAULT_FORMATS)
        return Report(
            logical_id=report_logical_id(embedded_refs=emb, title=title),
            title=title,
            subtitle=subtitle_value,
            description=description,
            notes=notes,
            tags=final_tags,
            markdown=markdown,
            live_name_pins=pin_map,
            live_name=live_name,
            formats=chosen_formats,
        )

    @toolmethod(
        name="edit_report",
        description=(
            "Surgical edit of an existing report: replace one occurrence of old_str "
            "with new_str against the latest snapshot. Identify the report by its "
            "workspace slug (live_name). logical_id and the persisted formats list "
            "are preserved; embedded refs are re-extracted from the new markdown body. "
            "To switch output formats, re-publish via return_report with the new formats list."
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
        from parsimony_agents.report_format import parse_snapshot

        live_name = (live_name or "").strip()
        if not live_name:
            raise ValueError("edit_report: live_name must be non-empty.")
        if old_str == "":
            raise ValueError(
                "edit_report: old_str must be a non-empty substring; full-body "
                "rewrites should go through return_report."
            )

        seen = extract_seen_live_names(context.messages)
        target_lid = await self._resolve_artifact_slug(live_name, kind="report", seen_live_names=seen)

        log_path = f".ockham/reports/{target_lid}/log.jsonl"
        try:
            raw_log = await self.code_executor.read_workspace_file(log_path)
        except FileNotFoundError as e:
            raise ValueError(
                f"edit_report: report {live_name!r} has no log.jsonl — it has not been published yet."
            ) from e
        last_csha = last_content_sha_from_log(raw_log)
        if last_csha is None:
            raise ValueError(f"edit_report: report {live_name!r} log.jsonl has no usable entries.")

        snapshot_path = f".ockham/reports/{target_lid}/{last_csha}.qmd"
        raw = await self.code_executor.read_workspace_file(snapshot_path)
        # Read/edit symmetry: old_str matches against the body the agent
        # authored, not the persisted bytes (which carry YAML
        # frontmatter). Parse the snapshot first, edit the body alone,
        # recompose at persist time. Title, subtitle, and pin map all
        # travel through unchanged — edit_report is body-only; use
        # return_report to change title/subtitle or embeds.
        snap = parse_snapshot(raw.decode("utf-8"))
        n = snap.body.count(old_str)
        if n == 0:
            raise ValueError("edit_report: old_str not found in report markdown.")
        if n > 1:
            raise ValueError("edit_report: old_str occurs multiple times; provide a more specific target.")
        new_markdown = snap.body.replace(old_str, new_str, 1)

        # Resolve embeds against the PRESERVED pin map. If the edit
        # added a new live_name URI that's not in the pin map this
        # raises — by design, since the snapshot's pin map is frozen.
        new_embedded = embedded_refs_from_markdown(new_markdown, snap.pins)
        await self._validate_refs_resolve(new_embedded)

        cur_path = f".ockham/reports/{target_lid}/curation.json"
        try:
            raw_cur = await self.code_executor.read_workspace_file(cur_path)
            curation = json.loads(raw_cur.decode("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            curation = {}

        return Report(
            logical_id=target_lid,
            title=snap.title,
            subtitle=snap.subtitle,
            description=str(curation.get("description", "") or ""),
            notes=list(curation.get("notes") or []),
            tags=list(curation.get("tags") or []),
            markdown=new_markdown,
            live_name_pins=snap.pins,  # frozen — edits never re-pin
            live_name=curation.get("live_name") if isinstance(curation.get("live_name"), str) else live_name,
            formats=snap.formats,
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
        target_ref = await self._resolve_slug_to_latest_ref(live_name, seen_live_names=seen)

        new_ref = await refresh_artifact(
            target_ref,
            executor=self.code_executor,
            report_validator=getattr(context, "report_validator", None),
        )

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
                    f"refresh: dataset variable {dataset.variable_name!r} is not in the kernel after refresh."
                )
            if not isinstance(payload, DataFrameObject):
                raise ValueError(
                    f"refresh: dataset variable {dataset.variable_name!r} "
                    f"has unexpected payload type {type(payload).__name__}."
                )
            return dataset.with_payload(payload)
        if new_ref.kind == "chart":
            from parsimony_agents.chart_io import deserialize_chart

            chart, _spec = deserialize_chart(blob)
            payload = await self.code_executor.get(chart.variable_name)
            if payload is None:
                raise ValueError(f"refresh: chart variable {chart.variable_name!r} is not in the kernel after refresh.")
            if not isinstance(payload, FigureObject):
                raise ValueError(
                    f"refresh: chart variable {chart.variable_name!r} "
                    f"has unexpected payload type {type(payload).__name__}."
                )
            return chart.with_payload(payload)
        if new_ref.kind == "report":
            # Reports have no kernel payload — read curation + bytes
            # directly. The streaming layer's persist path handles the
            # rest idempotently.
            return await self._reload_report_for_refresh(new_ref, blob)
        raise AssertionError(f"refresh: unreachable kind {new_ref.kind!r}")

    async def _reload_report_for_refresh(self, ref: ArtifactRef, blob: bytes) -> Report:
        """Reconstruct a :class:`Report` model from disk after refresh.

        Reports have no in-kernel payload — snapshot bytes ARE the
        source document. The YAML frontmatter carries the pin map; the
        body's ``file://./charts|data/<live_name>.<ext>`` URIs resolve
        against it.
        """
        from parsimony_agents.report_format import parse_snapshot

        snap = parse_snapshot(blob.decode("utf-8"))
        cur_path = f".ockham/reports/{ref.logical_id}/curation.json"
        try:
            raw_cur = await self.code_executor.read_workspace_file(cur_path)
            cur = json.loads(raw_cur.decode("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            cur = {}
        return Report(
            logical_id=ref.logical_id,
            content_sha=ref.content_sha,
            title=snap.title,
            subtitle=snap.subtitle,
            description=cur.get("description", "") or "",
            tags=list(cur.get("tags") or []),
            notes=list(cur.get("notes") or []),
            live_name=cur.get("live_name"),
            markdown=snap.body,
            live_name_pins=snap.pins,
            formats=snap.formats,
        )

    @staticmethod
    def _require_plain_variable_name(*, value: str, parameter_name: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError(f"{parameter_name} must be a non-empty string.")
        if not normalized.isidentifier():
            raise ValueError(
                f"{parameter_name} must be a plain variable name, not an expression or slice. Got '{value}'."
            )
        return normalized

    async def _validate_refs_resolve(self, refs: list[ArtifactRef]) -> None:
        """Each ref must resolve to bytes on disk.

        Used by ``return_report`` / ``edit_report``: every ArtifactRef
        the body resolves to (via the pin map) must point at a real
        snapshot on disk. Refresh and return_dataset/chart derive their
        refs from the framework (the run scope + snapshot store), so
        they never hit this path.
        """
        for ref in refs:
            try:
                await self.code_executor.read_workspace_file(ref.workspace_file_path)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Embedded ref {ref.kind}:{ref.logical_id}:{ref.content_sha} does not "
                    f"resolve ({ref.workspace_file_path!r} not found). Reports embed "
                    "artifacts via ![](file://./charts/<live_name>.vl.json) or "
                    "![](file://./data/<live_name>.parquet) — the live_name must already "
                    "be a published chart or dataset in this workspace."
                ) from e

    async def _build_report_pin_map(self, embed_keys: list[tuple[SnapshotKind, str]]) -> dict[str, ArtifactRef]:
        """Resolve each ``(kind, live_name)`` to its latest published snapshot.

        Walks ``.ockham/<kind>s/*/curation.json`` for the first match
        whose ``live_name`` equals the requested slug AND whose
        ``log.jsonl`` has at least one persisted snapshot. Returns the
        ``live_name -> ArtifactRef`` mapping that travels with the
        snapshot bytes — the renderer never re-resolves through
        curation, so this map alone determines what an old report
        renders to.

        Raises ``ValueError`` when a live_name has no matching artifact
        on disk; the agent must publish the embedded chart/dataset
        before publishing the report.
        """
        pin_map: dict[str, ArtifactRef] = {}
        seen_live_names: set[str] = set()
        for kind, live_name in embed_keys:
            if live_name in seen_live_names:
                continue  # already pinned via earlier (kind, live_name) tuple
            seen_live_names.add(live_name)
            ref = await self._resolve_live_name_to_latest_ref(kind=kind, live_name=live_name)
            if ref is None:
                raise ValueError(
                    f"return_report: embedded {kind} live_name {live_name!r} has no "
                    f"published snapshot in this workspace. Publish the {kind} via "
                    f"return_{kind} before embedding it in a report."
                )
            pin_map[live_name] = ref
        return pin_map

    async def _resolve_live_name_to_latest_ref(self, *, kind: SnapshotKind, live_name: str) -> ArtifactRef | None:
        """Look up the latest snapshot for ``(kind, live_name)`` via curation.

        Scans ``.ockham/<kind>s/*/curation.json`` for a curation whose
        ``live_name`` matches; reads the sibling ``log.jsonl`` for the
        most recent ``content_sha``. Returns ``None`` if no published
        artifact in this workspace matches.
        """
        listing_path = f".ockham/{kind}s"
        try:
            entries = await self.code_executor.list_workspace_files(listing_path)
        except FileNotFoundError:
            return None
        lids: set[str] = set()
        for path, _size in entries:
            parts = path.split("/")
            if len(parts) >= 3 and parts[0] == ".ockham" and parts[1] == f"{kind}s":
                lids.add(parts[2])
        for lid in lids:
            cur_path = f".ockham/{kind}s/{lid}/curation.json"
            try:
                raw_cur = await self.code_executor.read_workspace_file(cur_path)
                cur = json.loads(raw_cur.decode("utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(cur, dict) or cur.get("live_name") != live_name:
                continue
            log_path = f".ockham/{kind}s/{lid}/log.jsonl"
            try:
                raw_log = await self.code_executor.read_workspace_file(log_path)
            except FileNotFoundError:
                continue
            last_csha = last_content_sha_from_log(raw_log)
            if not last_csha:
                continue
            return ArtifactRef(kind=kind, logical_id=lid, content_sha=last_csha)
        return None

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
        notebook_ref = await self._notebook_ref_for_published_path(origin.notebook_path, context=context)
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
            raise ValueError(f"Variable '{variable_name}' has no producing notebook on record.")
        notebook_ref = await self._notebook_ref_for_published_path(origin.notebook_path, context=context)
        return [notebook_ref], list(origin.fetch_refs), list(origin.load_refs)

    async def _notebook_ref_for_published_path(self, working_copy_path: str, *, context: AgentContext) -> ArtifactRef:
        """Resolve a notebook path to its latest persisted :class:`ArtifactRef`.

        The notebook MUST have a ``log.jsonl`` — i.e. the agent must
        have published it via ``return_notebook`` at least once. Without
        that, this is the "published artifact must come from a published
        recipe" check failing loud: scratch-cell or unpublished notebook
        bytes cannot be cited as the producer of a published deliverable.
        """
        logical_id = await Agent._resolve_notebook_logical_id(working_copy_path, context)
        try:
            _raw, latest_csha = await read_latest_notebook(self.code_executor, logical_id=logical_id)
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
        kinds: tuple[SnapshotKind, ...] = (kind,) if kind else ("dataset", "chart", "report")
        matches: list[tuple[SnapshotKind, str]] = []  # (kind, logical_id)
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
            raise ValueError(f"No {kind_label} has live_name {live_name!r}. Use the slug shown in <turn_artifacts>.")
        if len(matches) > 1:
            raise ValueError(
                f"Slug {live_name!r} matches multiple artifacts. Rename one via curation before referring to it."
            )
        matched_kind, matched_lid = matches[0]
        if seen_live_names is not None and (matched_kind, live_name) not in seen_live_names:
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
                lid = await self._resolve_artifact_slug(live_name, kind=k, seen_live_names=seen_live_names)
            except ValueError:
                continue
            # LiveNameCollisionError NOT swallowed here — surface the
            # cross-terminal failure to the agent's tool framework.
            log_path = f".ockham/{k}s/{lid}/log.jsonl"
            try:
                raw_log = await self.code_executor.read_workspace_file(log_path)
            except FileNotFoundError as e:
                raise ValueError(f"Artifact {live_name!r} has no log.jsonl.") from e
            last_csha = last_content_sha_from_log(raw_log)
            if not last_csha:
                raise ValueError(f"Artifact {live_name!r} log.jsonl is empty.")
            return ArtifactRef(kind=k, logical_id=lid, content_sha=last_csha)
        raise ValueError(f"No published artifact has live_name {live_name!r}.")
