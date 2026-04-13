"""Structured kernel outputs: dataframes, figures, primitives, exceptions."""

from __future__ import annotations

import base64
import json
import traceback
from functools import cached_property
from typing import Annotated, Any, Literal

import altair as alt
import pandas as pd
from parsimony.result import Provenance
from pydantic import BaseModel, Field, TypeAdapter, computed_field, field_serializer, field_validator

from parsimony_agents.execution.dataframe_ref import DataframeRef
from parsimony_agents.execution.pagination import TablePaginator, get_output_header
from parsimony_agents.messages import MessageContent
from parsimony_agents.theme import PARSIMONY_FIGURE_HEIGHT, PARSIMONY_FIGURE_WIDTH
from parsimony_agents.util import truncate_text
from parsimony_agents.views import get_llm_view_defaults


class BaseOutputObject(MessageContent):
    class Config:
        arbitrary_types_allowed = True


class ExceptionObject(BaseOutputObject):
    """Exception for a kernel execution."""

    type: Literal["exception"] = "exception"
    value: str

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, v: Any) -> str:
        if isinstance(v, Exception):
            tb_lines = traceback.format_exception(type(v), v, v.__traceback__)
            return f"{type(v).__name__}: {v}\nTraceback: {' '.join(tb_lines)}"
        if isinstance(v, str):
            return v
        raise ValueError(f"Value is not an exception: {type(v)}")

    def to_llm(self, mode="default") -> list[dict[str, Any]]:
        text = truncate_text(str(self.value), per_line=False)
        return [{"type": "text", "text": text}]

    def to_frontend_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DataFrameObject(BaseOutputObject):
    type: Literal["dataframe"] = "dataframe"
    ref: DataframeRef
    include_full_value_in_frontend: bool = Field(default=False)

    @cached_property
    def value(self) -> pd.DataFrame:
        return self.ref.materialize_sync()

    @computed_field
    @property
    def head(self) -> dict[str, Any]:
        column_to_drop = self.value.index.name if self.value.index.name is not None else "index"
        value = self.value.drop(columns=[column_to_drop], errors="ignore")

        if len(value) <= 10:
            return json.loads(value.to_json(orient="table"))
        return json.loads(value.head(5).to_json(orient="table"))

    @computed_field
    @property
    def tail(self) -> dict[str, Any] | None:
        column_to_drop = self.value.index.name if self.value.index.name is not None else "index"
        value = self.value.drop(columns=[column_to_drop], errors="ignore")

        if len(value) <= 10:
            return None
        return json.loads(value.tail(5).to_json(orient="table"))

    def to_llm(self, mode: Literal["default", "minimal"] = "default", overrides: dict[str, Any] | None = None):
        overrides = overrides or {}
        view_cfg = get_llm_view_defaults("dataframe")[mode].model_copy(update=overrides)

        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"DataFrame {get_output_header(self.type, mode)}\n"}
        ]

        if self.value.shape[0] == 0:
            blocks.append({"type": "text", "text": "DataFrame is empty."})
            return blocks

        paginator = TablePaginator(self.value, rows_per_page=view_cfg.page_rows, show_dtypes=view_cfg.show_dtypes)
        max_cell = getattr(view_cfg, "max_cell_length", None)
        if max_cell is None:
            max_cell = 1000
        page_blocks = "\n".join(
            paginator.iter_pages(view_cfg.display_pages, na_rep="<NULL>", max_cell_length=max_cell)
        )

        if view_cfg.show_dtypes:
            blocks.extend(
                [
                    {
                        "type": "text",
                        "text": f"DataFrame with {self.value.shape[0]} rows and {self.value.shape[1]} columns.\n",
                    },
                    {
                        "type": "text",
                        "text": "Data in CSV format [index=False, na_rep=`<NULL>`; column names are displayed as `<column_name> (<dtype>)` (access with `df['<column_name>'])`]:\n",
                    },
                ]
            )

        blocks.append({"type": "text", "text": page_blocks if page_blocks else "(No pages selected.)"})

        if max_cell and max_cell > 0 and " ..." in (page_blocks or ""):
            blocks.append({"type": "text", "text": "\nNote: Some cells are truncated.\n"})

        if view_cfg.show_sample_unique_values:
            blocks.append({"type": "text", "text": "\nSample unique values:\n"})
            for col in self.value.columns:
                try:
                    unique_vals = self.value[col].unique().tolist()
                except TypeError:
                    unique_vals = "(complex type - list/array)"
                blocks.append(
                    {"type": "text", "text": f"- {col}: {truncate_text(str(unique_vals), max_length=100)}\n"}
                )

        return blocks

    def to_frontend_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "head": self.head,
            "tail": self.tail,
            "row_count": len(self.value),
            "column_count": len(self.value.columns),
        }

        if self.include_full_value_in_frontend:
            column_to_drop = self.value.index.name if self.value.index.name is not None else "index"
            full = self.value.drop(columns=[column_to_drop], errors="ignore")
            payload["value"] = json.loads(full.to_json(orient="table"))

        return payload


def finalize_spec(spec: dict) -> dict:
    """Apply Parsimony default sizing/autosize rules to a spec dict."""
    spec["width"] = PARSIMONY_FIGURE_WIDTH
    spec.setdefault("height", PARSIMONY_FIGURE_HEIGHT)
    spec.setdefault("autosize", {"type": "fit", "contains": "padding"})
    return spec


class FigureObject(BaseOutputObject):
    type: Literal["figure"] = "figure"
    value: alt.TopLevelMixin | dict[str, Any]
    name: str | None = None
    base64_image: str | None = Field(default=None, exclude=True)

    @field_serializer("value")
    def serialize_value(self, value: alt.TopLevelMixin | dict[str, Any], _info) -> dict:
        if isinstance(value, dict):
            return finalize_spec(value)
        alt.data_transformers.disable_max_rows()
        return finalize_spec(value.to_dict())

    def to_llm(self, mode="default") -> list[dict[str, Any]]:
        tag = self.name or "figure"
        if mode == "default":
            if not self.base64_image:
                self.calc_base64_image()
            return [
                {"type": "text", "text": f"<{tag}>"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.base64_image}"},
                },
                {"type": "text", "text": f"</{tag}>"},
            ]

        value = f"<{tag}>\n(Figure minimized)\n</{tag}>\n"
        return [{"type": "text", "text": value}]

    def to_frontend_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"base64_image"})

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, v: Any) -> alt.TopLevelMixin | dict[str, Any]:
        if isinstance(v, alt.TopLevelMixin):
            return v
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict):
            return v
        raise ValueError(f"Value is not an Altair Figure or dict: {type(v)}")

    def calc_base64_image(self, force_recalc: bool = False, **kwargs):
        if self.base64_image and not force_recalc:
            return self.base64_image

        import vl_convert as vlc

        alt.data_transformers.disable_max_rows()
        spec = self.value if isinstance(self.value, dict) else self.value.to_dict()
        spec = finalize_spec(spec)

        img_bytes = vlc.vegalite_to_png(vl_spec=json.dumps(spec), scale=1.0)

        self.base64_image = base64.b64encode(img_bytes).decode("utf-8")
        return self.base64_image


class PrimitiveObject(BaseOutputObject):
    type: Literal["primitive"] = "primitive"
    value: str | int | float | bool | None

    def to_llm(self, mode="default", overrides: dict[str, Any] | None = None):
        text = str(self.value)

        overrides = overrides or {}
        view_cfg = get_llm_view_defaults("primitive")[mode].model_copy(update=overrides)

        if len(text) <= view_cfg.page_chars:
            if view_cfg.minimal:
                value = text
            else:
                value = f"{get_output_header(self.type, mode)}\n{text}".strip()
            return [{"type": "text", "text": value}]

        from parsimony_agents.execution.pagination import StringPaginator

        paginator = StringPaginator(text, chars_per_page=view_cfg.page_chars)
        page_blocks = "\n".join(paginator.iter_pages(view_cfg.display_pages))

        parts: list[str] = []
        if not view_cfg.minimal:
            parts.append(get_output_header(self.type, mode))

        if page_blocks:
            parts.append(page_blocks)

        return [{"type": "text", "text": "\n".join(parts).strip()}]

    def to_frontend_dict(self) -> dict[str, Any]:
        val = self.model_dump(mode="json")
        val["value"] = truncate_text(str(self.value), max_length=5000)
        return val


KernelOutputType = Annotated[
    DataFrameObject | FigureObject | PrimitiveObject | ExceptionObject,
    Field(discriminator="type"),
]

_kernel_output_type_adapter = TypeAdapter(KernelOutputType)


class FetchLogEntry(BaseModel):
    """One recorded data operation from sandbox code (via the product ``fetch()`` wrapper → ``_fetch_log``)."""

    source: str
    source_description: str = ""
    params: dict[str, Any]
    row_count: int
    column_names: list[str]
    columns: list[dict[str, Any]]
    provenance: Provenance = Field(default_factory=Provenance)
    head: dict[str, Any] | None = None
    tail: dict[str, Any] | None = None


class KernelOutput(MessageContent):
    """Message for a tool execution."""

    type: Literal["kernel_output"] = "kernel_output"
    outputs: list[KernelOutputType] = Field(..., description="Kernel outputs")
    metadata: dict[str, Any] | None = Field(default=None, description="Lightweight metadata (e.g. source, source_description, code)")
    fetch_log: list[FetchLogEntry] = Field(
        default_factory=list,
        description="Data fetches performed during this execution (from executor locals _fetch_log)",
    )

    class Config:
        arbitrary_types_allowed = True

    def get_figures(self) -> list[FigureObject]:
        return [output for output in self.outputs if isinstance(output, FigureObject)]

    def to_llm(self, mode: str = "default") -> list[dict[str, Any]]:
        blocks = [
            {
                "type": "text",
                "text": "Out:\n---\n",
            }
        ]
        for enum, output in enumerate(self.outputs):
            blocks.extend(output.to_llm(mode=mode))
            blocks.append(
                {
                    "type": "text",
                    "text": "\n---\n",
                }
            )
        return blocks

    def to_frontend_dict(self):
        return {
            "type": self.type,
            "outputs": [output.to_frontend_dict() for output in self.outputs],
            "metadata": self.metadata,
            "fetch_log": [e.model_dump(mode="json") for e in self.fetch_log],
        }
