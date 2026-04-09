"""Artifacts: published deliverables (datasets, charts) and supporting types."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    Field,
    model_validator,
)

from ockham_agents.execution.outputs import FigureObject
from ockham_agents.messages import MessageContent

# ============================================================================
# Identity helpers
# ============================================================================


def _stable_dataset_artifact_id(
    *,
    variable_name: str | None,
    notebook_refs: list[str] | None = None,
) -> str:
    seed = "::".join(
        [
            "returned-dataset",
            (variable_name or "").strip(),
            *[(r or "").strip() for r in (notebook_refs or [])],
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _stable_chart_artifact_id(
    *,
    source_dataset_artifact_id: str,
    chart_variable_name: str,
    chart_notebook_ref: str,
) -> str:
    seed = "::".join(
        [
            "returned-chart",
            source_dataset_artifact_id.strip(),
            chart_variable_name.strip(),
            chart_notebook_ref.strip(),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


# ============================================================================
# Artifacts
# ============================================================================


class Dataset(MessageContent):
    """A curated, versioned dataset deliverable.

    Variables are working state. Artifacts are published outputs accepted by the user.
    """

    type: Literal["dataset"] = "dataset"
    artifact_id: str = ""
    version: int = 1

    # What variable this came from
    variable_name: str = ""
    variable_preview: dict[str, Any] = Field(default_factory=dict)

    # User-facing metadata
    title: str = ""
    description: str = ""
    notes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    # Notebook provenance
    notebook_refs: list[str] = Field(
        default_factory=list,
        description="Notebook names used to produce this dataset.",
    )

    @model_validator(mode="after")
    def populate_identity(self) -> Dataset:
        if not self.artifact_id:
            self.artifact_id = _stable_dataset_artifact_id(
                variable_name=self.variable_name,
                notebook_refs=self.notebook_refs,
            )
        if self.version < 1:
            self.version = 1
        return self

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f'<dataset variable_name="{self.variable_name}">\n'},
        ]
        if self.description:
            blocks.append({"type": "text", "text": f"<description>{self.description}</description>\n"})
        if self.notes:
            blocks.append({"type": "text", "text": "<notes>\n"})
            blocks.extend({"type": "text", "text": f"- {note}\n"} for note in self.notes)
            blocks.append({"type": "text", "text": "</notes>\n"})
        blocks.append({"type": "text", "text": "</dataset>\n"})
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "variable_name": self.variable_name,
            "variable_preview": self.variable_preview,
            "title": self.title,
            "description": self.description,
            "notes": self.notes,
            "tags": self.tags,
            "notebook_refs": self.notebook_refs,
        }


class Chart(MessageContent):
    type: Literal["chart"] = "chart"
    artifact_id: str = ""
    version: int = 1
    title: str = ""
    source_dataset_artifact_id: str = ""
    source_dataset_variable_name: str
    source_dataset_version: int = 1
    latest_source_dataset_version: int = 1
    is_stale: bool = False
    chart_variable_name: str
    figure: FigureObject
    chart_notebook_ref: str
    description: str = ""
    notes: list[str] = Field(default_factory=list)
    last_refreshed_at: datetime | None = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def populate_identity(self) -> Chart:
        dataset_artifact_id = self.source_dataset_artifact_id.strip()
        if not dataset_artifact_id:
            dataset_artifact_id = _stable_dataset_artifact_id(variable_name=self.source_dataset_variable_name)
            self.source_dataset_artifact_id = dataset_artifact_id
        if not self.artifact_id:
            self.artifact_id = _stable_chart_artifact_id(
                source_dataset_artifact_id=dataset_artifact_id,
                chart_variable_name=self.chart_variable_name,
                chart_notebook_ref=self.chart_notebook_ref,
            )
        if self.version < 1:
            self.version = 1
        if self.source_dataset_version < 1:
            self.source_dataset_version = 1
        if self.latest_source_dataset_version < self.source_dataset_version:
            self.latest_source_dataset_version = self.source_dataset_version
        self.is_stale = self.source_dataset_version < self.latest_source_dataset_version
        return self

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"<chart source_dataset_variable_name=\"{self.source_dataset_variable_name}\" "
                    f"chart_variable_name=\"{self.chart_variable_name}\" "
                    f"chart_notebook_ref=\"{self.chart_notebook_ref}\">\n"
                ),
            }
        ]
        if self.description:
            blocks.append({"type": "text", "text": f"<description>{self.description}</description>\n"})
        if self.notes:
            blocks.append({"type": "text", "text": "<notes>\n"})
            blocks.extend({"type": "text", "text": f"- {note}\n"} for note in self.notes)
            blocks.append({"type": "text", "text": "</notes>\n"})
        blocks.append({"type": "text", "text": "</chart>\n"})
        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "title": self.title,
            "source_dataset_artifact_id": self.source_dataset_artifact_id,
            "source_dataset_variable_name": self.source_dataset_variable_name,
            "source_dataset_version": self.source_dataset_version,
            "latest_source_dataset_version": self.latest_source_dataset_version,
            "is_stale": self.is_stale,
            "chart_variable_name": self.chart_variable_name,
            "figure": self.figure.to_frontend_dict(),
            "chart_notebook_ref": self.chart_notebook_ref,
            "description": self.description,
            "notes": self.notes,
            "last_refreshed_at": self.last_refreshed_at.isoformat() if self.last_refreshed_at else None,
        }


