"""Kernel output metadata models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MetadataItem(BaseModel):
    """Metadata item for a variable."""

    name: str
    value: Any
    exclude_from_llm_view: bool = False

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]] | None:
        if not self.exclude_from_llm_view:
            return [{"type": "text", "text": f"{self.name}: {self.value}"}]
        return []


class DatasetRefreshRecipe(BaseModel):
    """Replay contract for rematerializing a returned dataset."""

    dataset_variable_name: str
    source_datasets: list[str] = Field(default_factory=list)
    notebook_refs: list[str] = Field(default_factory=list)


RefreshStatus = Literal["idle", "running", "failed"]

PrimitiveTypes = str | int | float | bool | None
