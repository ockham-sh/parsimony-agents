"""Agent session state and message models (no FastAPI / SSE dependencies)."""

from __future__ import annotations

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
from parsimony_agents.artifacts import (
    Chart,
    Dataset,
    _stable_chart_artifact_id,
    _stable_dataset_artifact_id,
)
from parsimony_agents.execution import KernelOutput
from parsimony_agents.messages import Message, MessageContent, Reasoning, Text
from parsimony_agents.notebook import Script
from parsimony_agents.variable import Variable, VariableStore


class ReturnedDatasetState(BaseModel):
    """Persistent state for a dataset artifact that has been returned by the agent."""

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
            self.artifact_id = _stable_dataset_artifact_id(
                variable_name=self.dataset_variable_name,
                notebook_refs=self.notebook_refs,
            )
        if self.version < 1:
            self.version = 1
        return self


class ReturnedChartState(BaseModel):
    """Persistent state for a chart artifact that has been returned by the agent."""

    artifact_id: str = ""
    version: int = 1
    title: str = ""
    source_dataset_artifact_id: str = ""
    source_dataset_variable_name: str
    source_dataset_version: int = 1
    latest_source_dataset_version: int = 1
    is_stale: bool = False
    chart_variable_name: str
    chart_notebook_ref: str | None = None
    description: str = ""
    notes: list[str] = Field(default_factory=list)
    last_refreshed_at: datetime | None = None

    @model_validator(mode="after")
    def populate_identity(self) -> ReturnedChartState:
        if not self.source_dataset_artifact_id:
            self.source_dataset_artifact_id = _stable_dataset_artifact_id(
                variable_name=self.source_dataset_variable_name,
            )
        if not self.artifact_id:
            self.artifact_id = _stable_chart_artifact_id(
                source_dataset_artifact_id=self.source_dataset_artifact_id,
                chart_variable_name=self.chart_variable_name,
                chart_notebook_ref=self.chart_notebook_ref or "",
            )
        if self.version < 1:
            self.version = 1
        if self.source_dataset_version < 1:
            self.source_dataset_version = 1
        if self.latest_source_dataset_version < self.source_dataset_version:
            self.latest_source_dataset_version = self.source_dataset_version
        self.is_stale = self.source_dataset_version < self.latest_source_dataset_version
        return self


def _default_notebooks() -> dict[str, Script]:
    return {"main": Script(id="main")}


class AgentContextSnapshot(MessageContent):
    """Serializable snapshot of agent state for injection into the LLM context window."""

    type: Literal["agent_context_snapshot"] = "agent_context_snapshot"
    data_context: VariableStore
    notebooks: dict[str, Script] = Field(default_factory=_default_notebooks)
    files_list: list[str]
    active_notebook_name: str = "main"

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

        if len(self.notebooks) == 1 and "main" in self.notebooks:
            chunks.extend(
                [
                    {"type": "text", "text": "<notebook>\n"},
                    *self.notebooks["main"].to_llm(mode=mode),
                    {"type": "text", "text": "\n</notebook>\n"},
                ]
            )
        else:
            chunks.append({"type": "text", "text": "<notebooks>\n"})
            for notebook_name, notebook in self.notebooks.items():
                chunks.append(
                    {
                        "type": "text",
                        "text": f'<notebook name="{notebook_name}" active="{str(notebook_name == self.active_notebook_name).lower()}">\n',
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
    """A conversation message in an agent session (may carry structured artifacts)."""

    content: AgentMessageContent | None = Field(default=None, description="Content of the message")


class AgentContext(MessageContent):
    """Live agent session state: messages, notebooks, data context, and returned artifacts."""

    session_id: str
    data_context: VariableStore = Field(default_factory=VariableStore)
    messages: list[AgentMessage] = Field(default_factory=list)
    notebooks: dict[str, Script] = Field(default_factory=_default_notebooks)
    active_notebook_name: str = Field(default="main")
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
        notebook_name: str,
    ) -> Script:
        """Return the named notebook, creating it if it does not exist."""
        normalized_name = notebook_name.strip()
        if not normalized_name:
            raise ValueError("notebook_name must be a non-empty string.")
        if normalized_name not in self.notebooks:
            self.notebooks[normalized_name] = Script(id=normalized_name)
        return self.notebooks[normalized_name]

    def get_returned_dataset(self, artifact_id: str | None = None) -> ReturnedDatasetState | None:
        """Look up a returned dataset by ID, defaulting to the active one."""
        target_id = artifact_id or self.active_returned_dataset_id
        if target_id and target_id in self.returned_datasets:
            return self.returned_datasets[target_id]
        return self.returned_dataset

    def set_returned_dataset(self, state: ReturnedDatasetState) -> ReturnedDatasetState:
        """Register and activate a returned dataset state."""
        self.returned_datasets[state.artifact_id] = state
        self.active_returned_dataset_id = state.artifact_id
        self.returned_dataset = state
        return state

    def get_returned_chart(self, artifact_id: str | None = None) -> ReturnedChartState | None:
        """Look up a returned chart by ID, defaulting to the active one."""
        target_id = artifact_id or self.active_returned_chart_id
        if target_id and target_id in self.returned_charts:
            return self.returned_charts[target_id]
        return self.returned_chart

    def set_returned_chart(self, state: ReturnedChartState) -> ReturnedChartState:
        """Register and activate a returned chart state."""
        self.returned_charts[state.artifact_id] = state
        self.active_returned_chart_id = state.artifact_id
        self.returned_chart = state
        return state

    def mark_charts_stale_for_dataset(self, *, dataset_artifact_id: str, latest_version: int) -> None:
        """Flag all charts derived from the given dataset as stale when a new version is available."""
        for artifact_id, chart_state in list(self.returned_charts.items()):
            if chart_state.source_dataset_artifact_id != dataset_artifact_id:
                continue
            updated = chart_state.model_copy(
                update={
                    "latest_source_dataset_version": latest_version,
                    "is_stale": chart_state.source_dataset_version < latest_version,
                }
            )
            self.returned_charts[artifact_id] = updated
            if self.active_returned_chart_id == artifact_id:
                self.returned_chart = updated

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
        return self.get_or_create_notebook("main")

    @notebook.setter
    def notebook(self, value: Script) -> None:
        self.notebooks[value.id] = value
        if value.id == "main":
            self.active_notebook_name = "main"

    async def execute_notebooks(
        self,
        *,
        code_executor: Any,
        update_outputs: bool = True,
    ) -> dict[str, KernelOutput]:
        """Execute all session notebooks and return a dict of their kernel outputs."""
        outputs: dict[str, KernelOutput] = {}
        for notebook_name, notebook in self.notebooks.items():
            outputs[notebook_name] = await notebook.execute(
                code_executor=code_executor, update_outputs=update_outputs
            )
        return outputs

    async def to_snapshot(self) -> AgentContextSnapshot:
        """Build a serializable snapshot of the current context for LLM injection."""
        files_list = await self.files.list_files() if self.files is not None else []
        return AgentContextSnapshot(
            data_context=self.data_context.model_copy(deep=True),
            notebooks={name: notebook.model_copy(deep=True) for name, notebook in self.notebooks.items()},
            files_list=files_list,
            active_notebook_name=self.active_notebook_name,
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
