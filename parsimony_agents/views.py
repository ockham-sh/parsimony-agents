from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DataFrameViewConfig(BaseModel):
    kind: Literal["dataframe"] = "dataframe"
    page_rows: int = 10
    show_dtypes: bool = True
    show_sample_unique_values: bool = True
    display_pages: list[int] = [0, 1, -2, -1]
    max_cell_length: int | None = 1000


class SearchViewConfig(BaseModel):
    kind: Literal["search"] = "search"
    page_rows: int = 10
    show_dtypes: bool = False
    show_sample_unique_values: bool = False
    display_pages: list[int] = [0, 1]
    max_cell_length: int | None = 1000


class PrimitiveViewConfig(BaseModel):
    kind: Literal["primitive"] = "primitive"
    page_chars: int = 1000
    minimal: bool = False
    display_pages: list[int] = [0, -1]


LLMViewConfigType = DataFrameViewConfig | PrimitiveViewConfig | SearchViewConfig


LLM_VIEW_DEFAULTS: dict[str, dict[str, LLMViewConfigType]] = {
    "dataframe": {
        "default": DataFrameViewConfig(),
        "minimal": DataFrameViewConfig(show_sample_unique_values=False, display_pages=[0]),
    },
    "search": {
        "default": SearchViewConfig(),
        "minimal": SearchViewConfig(display_pages=[0]),
    },
    "primitive": {
        "default": PrimitiveViewConfig(),
        "minimal": PrimitiveViewConfig(minimal=True, display_pages=[0]),
    },
}


def get_llm_view_defaults(kind: str) -> dict[str, LLMViewConfigType]:
    if kind not in LLM_VIEW_DEFAULTS:
        raise ValueError(f"Unsupported llm_view.kind: {kind}")
    return LLM_VIEW_DEFAULTS[kind]
