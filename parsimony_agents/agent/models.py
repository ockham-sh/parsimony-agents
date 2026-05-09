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
from pydantic import Field, model_validator

from parsimony_agents.agent.outputs import SystemToolMessage, SystemToolOutput, UtilityToolOutput
from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.execution import KernelOutput
from parsimony_agents.identity import ArtifactRef
from parsimony_agents.messages import Message, MessageContent, Reasoning, Text
from parsimony_agents.notebook import Script

from parsimony_agents.agent.session_state import SessionState


def _build_workspace_tree(files: list[tuple[str, int]]) -> str:
    """Render a sorted (path, size_bytes) list as an indented directory tree.

    Example output::

        notebooks/
        ├── gdp_retrieval.py   4.2 KB
        └── inflation.py       2.1 KB
        data/
        └── raw.csv           12.0 KB
    """
    from collections import defaultdict

    def _fmt_size(n: int) -> str:
        if n >= 1024 * 1024:
            return f"{n / (1024 * 1024):.1f} MB"
        if n >= 1024:
            return f"{n / 1024:.1f} KB"
        return f"{n} B"

    # Build a nested dict tree: {dir: {subdir: {...}, filename: size_bytes}}
    tree: dict = {}
    for path, size in sorted(files):
        parts = path.replace("\\", "/").split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = size

    lines: list[str] = []

    def _render(node: dict, prefix: str) -> None:
        entries = sorted(node.items())
        for idx, (name, value) in enumerate(entries):
            is_last = idx == len(entries) - 1
            connector = "└── " if is_last else "├── "
            if isinstance(value, dict):
                lines.append(f"{prefix}{connector}{name}/")
                extension = "    " if is_last else "│   "
                _render(value, prefix + extension)
            else:
                size_str = _fmt_size(value)
                lines.append(f"{prefix}{connector}{name}   {size_str}")

    # Top-level entries: dirs get their name + "/" on their own line, then recurse
    top_entries = sorted(tree.items())
    for idx, (name, value) in enumerate(top_entries):
        if isinstance(value, dict):
            lines.append(f"{name}/")
            _render(value, "")
        else:
            size_str = _fmt_size(value)
            lines.append(f"{name}   {size_str}")

    return "\n".join(lines)


class AgentContextSnapshot(MessageContent):
    type: Literal["agent_context_snapshot"] = "agent_context_snapshot"
    #: Workspace files as (relative_path, size_bytes) pairs, sorted alphabetically.
    #: Populated from the materialized workspace directory before each turn.
    files_list: list[tuple[str, int]] = Field(default_factory=list)
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

        if self.files_list:
            tree_str = _build_workspace_tree(self.files_list)
            chunks.append(
                {
                    "type": "text",
                    "text": f"<workspace>\n{tree_str}\n</workspace>\n",
                }
            )

        _parts = [
            "\n"
            "<modules>",
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
                    "text": (
                        "<available_connectors>\n"
                        f"{self.connectors_catalog}\n"
                        "</available_connectors>\n"
                    ),
                }
            )

        if self.session_state is not None and mode != "minimal":
            chunks.append(
                {
                    "type": "text",
                    "text": self.session_state.to_llm_text(minted_refs=self.minted_refs or None),
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
    Chart | Dataset | Report | Script | AgentContextSnapshot | UtilityToolOutput | SystemToolOutput | SystemToolMessage | KernelOutput | Reasoning | Text | str | list[dict[str, Any]],
    Field(union_mode="smart"),
]


class AgentMessage(Message):
    content: AgentMessageContent | None = Field(default=None, description="Content of the message")


class AgentContext(MessageContent):
    session_id: str
    messages: list[AgentMessage] = Field(default_factory=list)

    # Session-scoped services (runtime only, not serialized). Runtime types: FileStore, SessionVectorStore, SessionKeywordStore.
    files: Any | None = Field(default=None, exclude=True)
    vector_store: Any | None = Field(default=None, exclude=True)
    keyword_store: Any | None = Field(default=None, exclude=True)

    #: Workspace files as (relative_path, size_bytes) pairs, injected by router before each turn.
    files_list: list[tuple[str, int]] = Field(default_factory=list)
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

    async def to_snapshot(
        self,
        *,
        connectors: Any = None,
        minted_refs: list[ArtifactRef] | None = None,
    ) -> AgentContextSnapshot:
        from parsimony_agents.agent.helpers import render_connector_catalog

        # Prefer the injected file list (workspace mode); fall back to FileStore for
        # session-mode agents that still use file_store.
        if self.files_list:
            files_list: list[tuple[str, int]] = list(self.files_list)
        elif self.files is not None:
            raw: list[str] = await self.files.list_files()
            files_list = [(p, 0) for p in raw]
        else:
            files_list = []

        return AgentContextSnapshot(
            files_list=files_list,
            connectors_catalog=render_connector_catalog(connectors),
            session_state=self.session_state,
            minted_refs=list(minted_refs or []),
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
