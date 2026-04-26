"""Agent session state and message models (no FastAPI / SSE dependencies)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

import altair as alt
import numpy
import pandas
import scipy
import statsmodels
from pydantic import BaseModel, Field, model_validator

from parsimony_agents.agent.outputs import SystemToolMessage, SystemToolOutput, UtilityToolOutput
from parsimony_agents.artifacts import Chart, Dataset
from parsimony_agents.execution import KernelOutput
from parsimony_agents.messages import Message, MessageContent, Reasoning, Text
from parsimony_agents.notebook import Script

from parsimony_agents.agent.session_state import SessionState


class ReturnedDatasetState(BaseModel):
    """Session-only bookkeeping for a dataset the agent has returned.

    Holds the executor variable name and a snapshot of the curation envelope. Does
    not own a workspace path: snapshots are framework-managed under
    ``.ockham/cards/`` keyed by ``artifact_id`` + ``version``.
    """

    artifact_id: str = ""
    version: int = 1
    dataset_variable_name: str
    title: str | None = None
    description: str = ""
    notes: list[str] = Field(default_factory=list)
    source_dataset_variable_names: list[str] = Field(default_factory=list)
    notebook_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_identity(self) -> ReturnedDatasetState:
        if not self.artifact_id:
            self.artifact_id = str(uuid.uuid4())
        if self.version < 1:
            self.version = 1
        return self


class ReturnedChartState(BaseModel):
    """Session-only bookkeeping for a chart the agent has returned.

    Holds the executor variable names (chart + source dataset) and a
    snapshot of the curation envelope. When a matching dataset was returned
    in the same session, ``source_dataset_path`` is the workspace-relative
    dataset snapshot (see :func:`parsimony_agents.artifacts.snapshot_path`);
    for chart-only deliverables it may be empty.
    """

    artifact_id: str = ""
    version: int = 1
    title: str = ""
    source_dataset_path: str = ""
    source_dataset_variable_name: str
    chart_variable_name: str
    chart_notebook_ref: str | None = None
    description: str = ""
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def populate_identity(self) -> ReturnedChartState:
        if not self.artifact_id:
            self.artifact_id = str(uuid.uuid4())
        if self.version < 1:
            self.version = 1
        return self


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
                    "text": self.session_state.to_llm_text(),
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
    Chart | Dataset | Script | AgentContextSnapshot | UtilityToolOutput | SystemToolOutput | SystemToolMessage | KernelOutput | Reasoning | Text | str | list[dict[str, Any]],
    Field(union_mode="smart"),
]


class AgentMessage(Message):
    content: AgentMessageContent | None = Field(default=None, description="Content of the message")


class AgentContext(MessageContent):
    session_id: str
    messages: list[AgentMessage] = Field(default_factory=list)
    returned_datasets: dict[str, ReturnedDatasetState] = Field(default_factory=dict)
    returned_charts: dict[str, ReturnedChartState] = Field(default_factory=dict)
    active_returned_dataset_id: str | None = Field(default=None)
    active_returned_chart_id: str | None = Field(default=None)
    returned_dataset: ReturnedDatasetState | None = Field(default=None)
    returned_chart: ReturnedChartState | None = Field(default=None)

    # Session-scoped services (runtime only, not serialized). Runtime types: FileStore, SessionVectorStore, SessionKeywordStore.
    files: Any | None = Field(default=None, exclude=True)
    vector_store: Any | None = Field(default=None, exclude=True)
    keyword_store: Any | None = Field(default=None, exclude=True)

    #: Workspace files as (relative_path, size_bytes) pairs, injected by router before each turn.
    files_list: list[tuple[str, int]] = Field(default_factory=list)
    #: Filled by the host before :meth:`to_snapshot` in workspace mode.
    session_state: SessionState | None = None

    def get_returned_dataset(self, artifact_id: str | None = None) -> ReturnedDatasetState | None:
        target_id = artifact_id or self.active_returned_dataset_id
        if target_id and target_id in self.returned_datasets:
            return self.returned_datasets[target_id]
        return self.returned_dataset

    def set_returned_dataset(self, state: ReturnedDatasetState) -> ReturnedDatasetState:
        self.returned_datasets[state.artifact_id] = state
        self.active_returned_dataset_id = state.artifact_id
        self.returned_dataset = state
        return state

    def get_returned_chart(self, artifact_id: str | None = None) -> ReturnedChartState | None:
        target_id = artifact_id or self.active_returned_chart_id
        if target_id and target_id in self.returned_charts:
            return self.returned_charts[target_id]
        return self.returned_chart

    def set_returned_chart(self, state: ReturnedChartState) -> ReturnedChartState:
        self.returned_charts[state.artifact_id] = state
        self.active_returned_chart_id = state.artifact_id
        self.returned_chart = state
        return state

    @model_validator(mode="after")
    def sync_returned_artifacts(self) -> AgentContext:
        if self.returned_dataset is not None:
            self.returned_datasets[self.returned_dataset.artifact_id] = self.returned_dataset
            self.active_returned_dataset_id = self.returned_dataset.artifact_id
        elif self.active_returned_dataset_id and self.active_returned_dataset_id in self.returned_datasets:
            self.returned_dataset = self.returned_datasets[self.active_returned_dataset_id]

        if self.returned_chart is not None:
            self.returned_charts[self.returned_chart.artifact_id] = self.returned_chart
            self.active_returned_chart_id = self.returned_chart.artifact_id
        elif self.active_returned_chart_id and self.active_returned_chart_id in self.returned_charts:
            self.returned_chart = self.returned_charts[self.active_returned_chart_id]
        return self

    async def to_snapshot(self, *, connectors: Any = None) -> AgentContextSnapshot:
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
        )

    def _get_ref_name(self, key: str = str(uuid4())[:8], subdir: str = "artifacts") -> str:
        return f"{self.session_id}/{subdir}/{key}"


__all__ = [
    "AgentContext",
    "AgentContextSnapshot",
    "AgentMessage",
    "AgentMessageContent",
    "ReturnedChartState",
    "ReturnedDatasetState",
    "SessionState",
]
