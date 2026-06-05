"""Agent session state and message models (no FastAPI / SSE dependencies).

The legacy ``ReturnedDatasetState`` / ``ReturnedChartState`` slots have
been removed (§5.8 item A): with content-addressed identity,
match-and-reuse is automatic — the same logical inputs always hash to
the same path, so no per-session bookkeeping is required to detect a
re-publish.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

import altair as alt
import numpy
import pandas
import scipy
import statsmodels
from pydantic import Field

from parsimony_agents.agent.outputs import SystemToolMessage, SystemToolOutput, UtilityToolOutput
from parsimony_agents.agent.session_state import SessionState
from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.execution import KernelOutput
from parsimony_agents.identity import ArtifactRef
from parsimony_agents.messages import Message, MessageContent, Reasoning, Text
from parsimony_agents.notebook import Script


class AgentContextSnapshot(MessageContent):
    type: Literal["agent_context_snapshot"] = "agent_context_snapshot"
    #: Pre-rendered catalog of connectors bound into the executor this turn,
    #: as produced by :func:`parsimony_agents.agent.helpers.render_connector_catalog`.
    #: Empty string means no connectors are bound and the corresponding XML
    #: block is omitted from :meth:`to_llm`.
    connectors_catalog: str = ""
    #: Optional kernel + workspace artifact hints (filled by the host in workspace mode).
    session_state: SessionState | None = None
    #: Refs minted by ``return_*`` / ``edit_*`` / ``refresh`` during the
    #: current turn (populated from ``TurnState.minted_refs`` each iteration).
    #: Fused with ``session_state.workspace_artifacts`` to render a single
    #: always-current ``<turn_artifacts>`` block.
    minted_refs: list[ArtifactRef] = Field(default_factory=list)
    #: ``f"{kind}:{logical_id}"`` → ``live_name`` for the same refs in
    #: :attr:`minted_refs`. Carries the agent-typed workspace slug so the
    #: rendered ``<artifact ... live_name="..."/>`` row appears in the next
    #: iteration's prompt, letting the seen-set extractor recognise this
    #: terminal's own writes (otherwise a freshly-minted artifact reads as
    #: a sibling-terminal collision on the very next ``return_*`` call).
    minted_live_names: dict[str, str] = Field(default_factory=dict)
    #: ``(kind, live_name)`` pairs the calling terminal has interacted with
    #: in the current conversation. Used by :meth:`SessionState.to_llm_text`
    #: to filter cross-turn workspace artifacts to this terminal's seen-set
    #: — sibling-terminal artifacts are hidden until the agent calls
    #: ``list_artifacts`` / ``read_artifact``. Serialised as a list of
    #: two-element lists for JSON compatibility; rehydrated on use.
    seen_live_names_pairs: list[tuple[str, str]] = Field(default_factory=list)

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        chunks.append(
            {
                "type": "text",
                "text": '<context role="system">\n',
            }
        )

        chunks.extend(
            [
                {
                    "type": "text",
                    "text": f"Current datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                }
            ]
        )

        _parts = [
            "\n<modules>",
            ", ".join(
                [
                    f"pandas {pandas.__version__}",
                    f"numpy {numpy.__version__}",
                    f"scipy {scipy.__version__}",
                    f"statsmodels {statsmodels.__version__}",
                    f"altair {alt.__version__}",
                ]
            ),
            "</modules>",
            "\n",
        ]

        chunks.append(
            {
                "type": "text",
                "text": "\n".join(_parts),
            }
        )

        if self.connectors_catalog:
            chunks.append(
                {
                    "type": "text",
                    "text": (f"<available_connectors>\n{self.connectors_catalog}\n</available_connectors>\n"),
                }
            )

        if self.session_state is not None and mode != "minimal":
            seen = {tuple(p) for p in self.seen_live_names_pairs if len(p) == 2}
            chunks.append(
                {
                    "type": "text",
                    "text": self.session_state.to_llm_text(
                        minted_refs=self.minted_refs or None,
                        minted_live_names=self.minted_live_names or None,
                        seen_live_names=seen,
                    ),
                }
            )

        chunks.append(
            {
                "type": "text",
                "text": "\n</context>\n",
            }
        )

        return chunks


AgentMessageContent = Annotated[
    Chart
    | Dataset
    | Report
    | Script
    | AgentContextSnapshot
    | UtilityToolOutput
    | SystemToolOutput
    | SystemToolMessage
    | KernelOutput
    | Reasoning
    | Text
    | str
    | list[dict[str, Any]],
    Field(union_mode="smart"),
]


class AgentMessage(Message):
    content: AgentMessageContent | None = Field(default=None, description="Content of the message")


class AgentContext(MessageContent):
    session_id: str
    messages: list[AgentMessage] = Field(default_factory=list)

    # Session-scoped services (runtime only, not serialized).
    # Runtime types: FileStore, SessionVectorStore, SessionKeywordStore.
    files: Any | None = Field(default=None, exclude=True)
    vector_store: Any | None = Field(default=None, exclude=True)
    keyword_store: Any | None = Field(default=None, exclude=True)

    #: Filled by the host before :meth:`to_snapshot` in workspace mode.
    session_state: SessionState | None = None
    #: Resolves a notebook working-copy path → its current ``logical_id``.
    #: The host injects this so the agent's emitted refs honour user-side
    #: renames: a notebook renamed in the UI keeps its original logical_id
    #: even when ``return_notebook`` later targets the new path. The resolver
    #: scans existing curations (no allocation, no flock — slug is
    #: deterministic from the first creation path).
    #:
    #: When ``None`` (parsimony-agents standalone, no workspace host),
    #: the agent falls back to deriving logical_id from the path directly
    #: (``notebook_logical_id``).
    notebook_logical_id_resolver: Any | None = Field(default=None, exclude=True)

    #: Optional host-injected report validator implementing the
    #: :class:`~parsimony_agents.execution.artifact_store.ReportValidator` protocol
    #: (``(body, *, pin_map_keys) -> None``, raising on unsafe content: active HTML,
    #: executable fences, out-of-allowlist refs). ``persist_artifact`` calls it
    #: BEFORE writing a ``return_report`` / refresh snapshot, so unsafe bytes never
    #: reach the workspace tree and the agent gets a self-correct error. The
    #: standalone agent leaves this None (the author is the user, who reads their
    #: own output); a workspace host (terminal) injects its validator so the
    #: persisted snapshot is trusted by every read/render path.
    #:
    #: Typed ``Any`` (not the Protocol) only because it is an ``exclude=True``
    #: runtime-only field — like the other injected services on this model — so it
    #: never enters Pydantic's schema; the real type is enforced at the
    #: ``persist_artifact(report_validator=...)`` boundary.
    report_validator: Any | None = Field(default=None, exclude=True)

    #: Single-terminal standalone mode. When True, the cross-terminal seen-set
    #: filter is meaningless (every ``.ockham/`` artifact belongs to this one
    #: agent), so ``to_snapshot`` pre-seeds the seen-set with the agent's own
    #: ``session_state`` artifacts. Without this, a follow-up turn that does not
    #: carry forward the prior turn's messages (fresh ctx / one-shot ``ask``)
    #: would have an empty seen-set and the filter would drop every
    #: disk-discovered artifact from ``<turn_artifacts>`` — re-triggering the
    #: reuse loop the persistence fix is meant to end. The host leaves this
    #: False so its cross-terminal hiding stays intact.
    local_discovery: bool = Field(default=False, exclude=True)

    async def to_snapshot(
        self,
        *,
        connectors: Any = None,
        minted_refs: list[ArtifactRef] | None = None,
        minted_live_names: dict[str, str] | None = None,
    ) -> AgentContextSnapshot:
        from parsimony_agents.agent.helpers import render_connector_catalog
        from parsimony_agents.agent.seen_refs import extract_seen_live_names

        # Compute the calling terminal's seen-set from the live message
        # graph. The snapshot stores it as a JSON-friendly list of pairs;
        # :meth:`AgentContextSnapshot.to_llm` rehydrates it back into a
        # set before passing to the session_state renderer.
        seen = set(extract_seen_live_names(self.messages))
        if self.local_discovery and self.session_state is not None:
            # Standalone: the agent's own on-disk artifacts are, by definition,
            # ones it has interacted with — admit them past the filter.
            for a in self.session_state.workspace_artifacts:
                if a.live_name:
                    seen.add((a.kind, a.live_name))
        seen_pairs = sorted(seen)
        return AgentContextSnapshot(
            connectors_catalog=render_connector_catalog(connectors),
            session_state=self.session_state,
            minted_refs=list(minted_refs or []),
            minted_live_names=dict(minted_live_names or {}),
            seen_live_names_pairs=list(seen_pairs),
        )

    def _get_ref_name(self, key: str = str(uuid4())[:8], subdir: str = "artifacts") -> str:
        return f"{self.session_id}/{subdir}/{key}"


__all__ = [
    "AgentContext",
    "AgentContextSnapshot",
    "AgentMessage",
    "AgentMessageContent",
    "SessionState",
]
