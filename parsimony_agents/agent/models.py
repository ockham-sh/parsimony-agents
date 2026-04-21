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
from parsimony_agents.notebook import DEFAULT_NOTEBOOK_PATH, Script
from parsimony_agents.variable import Variable, VariableStore


class ReturnedDatasetState(BaseModel):
    """Session-only bookkeeping for a dataset the agent has returned.

    Holds the executor variable name (so refresh / re-run tools can find
    the live frame again) and a snapshot of the curation envelope. Does
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
    snapshot of the curation envelope. The source dataset is referenced by
    its workspace-relative snapshot path (``source_dataset_path``); see
    :func:`parsimony_agents.artifacts.snapshot_path`. Path is identity.
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


def _default_notebooks() -> dict[str, Script]:
    return {DEFAULT_NOTEBOOK_PATH: Script(path=DEFAULT_NOTEBOOK_PATH)}


class AgentContextSnapshot(MessageContent):
    type: Literal["agent_context_snapshot"] = "agent_context_snapshot"
    data_context: VariableStore
    notebooks: dict[str, Script] = Field(default_factory=_default_notebooks)
    files_list: list[str]
    active_notebook_path: str = DEFAULT_NOTEBOOK_PATH
    #: Pre-rendered catalog of connectors bound into the executor this turn,
    #: as produced by :func:`parsimony_agents.agent.helpers.render_connector_catalog`.
    #: Empty string means no connectors are bound and the corresponding XML
    #: block is omitted from :meth:`to_llm`.
    connectors_catalog: str = ""

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

        chunks.extend(
            [
                {"type": "text", "text": "<data_context>\n"},
                *self.data_context.to_llm(mode=mode),
                {"type": "text", "text": "\n</data_context>\n"},
            ]
        )

        _parts = [
            "<session_files>",
            "Files available in this session (access directly by filename in code):",
        ]
        for f in self.files_list:
            _parts.append(f" - {f}")
        _parts.append("</session_files>")
        chunks.append(
            {
                "type": "text",
                "text": "\n".join(_parts),
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

        """

        if self.working_memory:
            _parts.extend([
                "<working_memory>",
                f"{self.working_memory}",
                "</working_memory>",
            ])


        if current_stage:
            _parts.extend([
                "<current_stage>",
                f"{current_stage['stage']} with plan: {current_stage['plan']}",
                "</current_stage>",
            ])

        """

        if len(self.notebooks) == 1 and DEFAULT_NOTEBOOK_PATH in self.notebooks:
            chunks.extend(
                [
                    {"type": "text", "text": "<notebook>\n"},
                    *self.notebooks[DEFAULT_NOTEBOOK_PATH].to_llm(mode=mode),
                    {"type": "text", "text": "\n</notebook>\n"},
                ]
            )
        else:
            chunks.append({"type": "text", "text": "<notebooks>\n"})
            for notebook_path, notebook in self.notebooks.items():
                chunks.append(
                    {
                        "type": "text",
                        "text": f'<notebook path="{notebook_path}" active="{str(notebook_path == self.active_notebook_path).lower()}">\n',
                    }
                )
                chunks.extend(notebook.to_llm(mode=mode))
                chunks.append({"type": "text", "text": "\n</notebook>\n"})
            chunks.append({"type": "text", "text": "</notebooks>\n"})

        chunks.append(
            {
                "type": "text",
                "text": "\n</context>\n",
            }
        )

        return chunks


AgentMessageContent = Annotated[
    Chart | Dataset | VariableStore | Script | AgentContextSnapshot | UtilityToolOutput | SystemToolOutput | SystemToolMessage | KernelOutput | Reasoning | Variable | Text | str | list[dict[str, Any]],
    Field(union_mode="smart"),
]


class AgentMessage(Message):
    content: AgentMessageContent | None = Field(default=None, description="Content of the message")


class AgentContext(MessageContent):
    session_id: str
    data_context: VariableStore = Field(default_factory=VariableStore)
    messages: list[AgentMessage] = Field(default_factory=list)
    notebooks: dict[str, Script] = Field(default_factory=_default_notebooks)
    active_notebook_path: str = Field(default=DEFAULT_NOTEBOOK_PATH)
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

    files_list: list[str] = Field(default_factory=list)

    #: Monotonic counter; executor warm-skip compares this to the sandbox-recorded version.
    state_version: int = Field(default=1, ge=1)

    def bump_state_version(self) -> None:
        """Invalidate warm-executor assumptions after context or notebooks change."""
        self.state_version += 1

    def get_or_create_notebook(
        self,
        path: str,
    ) -> Script:
        """Look up a notebook by its workspace path (creating it if missing).

        ``path`` is a relative workspace path like
        ``notebooks/inflation.py``. Path safety is enforced by the
        storage boundary on persistence; here we only require non-empty.
        """
        normalized = path.strip()
        if not normalized:
            raise ValueError("notebook path must be a non-empty string.")
        if normalized not in self.notebooks:
            self.notebooks[normalized] = Script(path=normalized)
        return self.notebooks[normalized]

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

    @property
    def notebook(self) -> Script:
        return self.get_or_create_notebook(DEFAULT_NOTEBOOK_PATH)

    @notebook.setter
    def notebook(self, value: Script) -> None:
        self.notebooks[value.path] = value
        if value.path == DEFAULT_NOTEBOOK_PATH:
            self.active_notebook_path = DEFAULT_NOTEBOOK_PATH

    async def execute_notebooks(
        self,
        *,
        code_executor: Any,
        update_outputs: bool = True,
    ) -> dict[str, KernelOutput]:
        outputs: dict[str, KernelOutput] = {}
        for notebook_path, notebook in self.notebooks.items():
            outputs[notebook_path] = await notebook.execute(
                code_executor=code_executor, update_outputs=update_outputs
            )
        return outputs

    async def to_snapshot(self, *, connectors: Any = None) -> AgentContextSnapshot:
        from parsimony_agents.agent.helpers import render_connector_catalog

        files_list = await self.files.list_files() if self.files is not None else []
        return AgentContextSnapshot(
            data_context=self.data_context.model_copy(deep=True),
            notebooks={path: notebook.model_copy(deep=True) for path, notebook in self.notebooks.items()},
            files_list=files_list,
            active_notebook_path=self.active_notebook_path,
            connectors_catalog=render_connector_catalog(connectors),
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
]
