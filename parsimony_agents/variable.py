"""Variable and VariableStore: the core data layer for code-generation agents."""

from __future__ import annotations

from typing import Any, Literal

from parsimony.result import Provenance
from pydantic import BaseModel, Field, TypeAdapter, computed_field, model_validator

from parsimony_agents.execution.metadata import MetadataItem
from parsimony_agents.execution.outputs import (
    DataFrameObject,
    ExceptionObject,
    KernelOutputType,
    PrimitiveObject,
)
from parsimony_agents.quality.data import inspect_object


class Variable(BaseModel):
    """A named execution output with lightweight provenance.

    Replaces DataObject + Metadata. The output field holds any KernelOutputType
    (DataFrameObject, PrimitiveObject, FigureObject, ExceptionObject) which
    already knows how to render itself for LLM and frontend.
    """

    type: Literal["variable"] = "variable"
    name: str
    output: KernelOutputType | None = None
    source: str = ""
    source_description: str = ""
    notebook_ref: str | None = None
    source_datasets: list[str] = Field(default_factory=list)
    hidden: bool = False
    provenance: Provenance | None = None
    additional_metadata: list[MetadataItem] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @property
    def is_tabular(self) -> bool:
        return isinstance(self.output, DataFrameObject)

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        header = f'type="variable" name="{self.name}"'
        blocks.append({"type": "text", "text": f"<variable {header}>\n"})

        if self.output:
            blocks.extend([
                {"type": "text", "text": "<data>\n"},
                *self.output.to_llm(mode=mode),
                {"type": "text", "text": "\n</data>\n"},
            ])

        # Metadata items
        if self.additional_metadata:
            blocks.append({"type": "text", "text": "<metadata>"})
            for item in self.additional_metadata:
                item_blocks = item.to_llm(mode=mode)
                if item_blocks:
                    blocks.extend(item_blocks)
            blocks.append({"type": "text", "text": "</metadata>"})

        # Data quality report for tabular data
        if self.is_tabular:
            report = self._get_data_quality_report()
            if report:
                blocks.append({"type": "text", "text": f"<data_quality_report>\n{report}\n</data_quality_report>\n"})

        blocks.append({"type": "text", "text": "</variable>\n"})
        return blocks

    def _get_data_quality_report(self) -> str | None:
        if not isinstance(self.output, DataFrameObject):
            return None
        try:
            return inspect_object(self.output.value)
        except Exception:
            return None

    def to_frontend_dict(self) -> dict[str, Any]:
        if self.output is None:
            output_dict = None
        elif isinstance(self.output, DataFrameObject):
            output_dict = self.output.to_frontend_dict()
        elif isinstance(self.output, PrimitiveObject):
            output_dict = {"type": self.output.type, "value": self.output.value}
        else:
            output_dict = self.output.to_frontend_dict()

        return {
            "type": self.type,
            "name": self.name,
            "output": output_dict,
            "source": self.source,
            "hidden": self.hidden,
        }


class VariableStore(BaseModel):
    """Thin typed dict of named variables. Replaces DataContext.

    Exists for: (1) session persistence, (2) sandbox seeding, (3) LLM context.
    The sandbox is the source of truth during execution.
    """

    type: Literal["variable_store"] = "variable_store"
    variables: dict[str, Variable] = Field(default_factory=dict, exclude=True)
    version: int = 1

    @model_validator(mode="before")
    @classmethod
    def _reconstruct(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        variable_adapter = TypeAdapter(Variable)

        # If variables dict is provided directly (e.g. from code), validate entries
        if "variables" in values and isinstance(values["variables"], dict):
            validated: dict[str, Variable] = {}
            for key, raw in values["variables"].items():
                obj = variable_adapter.validate_python(raw)
                validated[str(key)] = obj
            values["variables"] = validated
            return values

        # If variable_list is provided (from serialized JSON), rebuild dict
        variable_list = values.get("variable_list")
        if variable_list is not None and "variables" not in values:
            variables_adapter = TypeAdapter(list[Variable])
            validated_list = variables_adapter.validate_python(variable_list)
            values["variables"] = {var.name: var for var in validated_list}

        return values

    @computed_field
    @property
    def variable_list(self) -> list[Variable]:
        return list(self.variables.values())

    def to_locals(self) -> dict[str, Any]:
        """Seed the sandbox with Python values."""
        result: dict[str, Any] = {}
        for var in self.variables.values():
            if isinstance(var.output, ExceptionObject) or var.output is None:
                continue
            result[var.name] = var.output.value
        return result

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for var in self.variables.values():
            if var.hidden:
                continue
            blocks.extend(var.to_llm(mode=mode))
            blocks.append({"type": "text", "text": "\n---\n"})
        return blocks or [{"type": "text", "text": "Empty Variable Store"}]

    def extend(self, variables: list[Variable]) -> None:
        for var in variables:
            self.variables[var.name] = var

    def increment_version(self) -> None:
        self.version += 1

    def __contains__(self, name: object) -> bool:
        return name in self.variables

    def __getitem__(self, name: str) -> Variable:
        return self.variables[name]

    def __setitem__(self, name: str, var: Variable) -> None:
        self.variables[name] = var

    def __len__(self) -> int:
        return len(self.variables)
