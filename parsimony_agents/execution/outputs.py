"""Structured kernel outputs: dataframes, figures, primitives, exceptions."""

from __future__ import annotations

import base64
import json
import traceback
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

import altair as alt
import pandas as pd
from parsimony.errors import ConnectorError
from parsimony.result import Provenance
from parsimony.transport import redact_sensitive_text
from pydantic import BaseModel, Field, TypeAdapter, computed_field, field_serializer, field_validator

from parsimony_agents.agent.xml_render import escape_attr
from parsimony_agents.execution.dataframe_ref import DataframeRef
from parsimony_agents.execution.pagination import TablePaginator, get_output_header
from parsimony_agents.identity import ArtifactRef
from parsimony_agents.messages import MessageContent
from parsimony_agents.theme import PARSIMONY_FIGURE_HEIGHT, PARSIMONY_FIGURE_WIDTH
from parsimony_agents.util import truncate_text
from parsimony_agents.views import get_llm_view_defaults

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_DATAFRAME_FULL_SHOW_THRESHOLD = 10  # Rows at or below this count: show all; above: show head + tail
_DATAFRAME_HEAD_TAIL_SIZE = 5        # Number of rows in head/tail preview slices
_DEFAULT_MAX_CELL_LENGTH = 1000      # Fallback max characters per cell in LLM output


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
        # Typed parsimony errors carry kernel-built, agent-safe messages
        # that already include the class semantics and the appropriate
        # agent-loop directive (DO NOT retry / pick a different connector
        # / etc.).  Surface ``str(exc)`` directly — no traceback frames,
        # no extra redaction needed (the message text is kernel-controlled
        # for typed subclasses; bare ``ConnectorError`` is contractually
        # author-controlled-but-redaction-clean).  Skipping the traceback
        # also keeps the agent's context budget tight on common failure
        # paths like rate limits and missing credentials.
        if isinstance(v, ConnectorError):
            return str(v)
        if isinstance(v, Exception):
            tb_text = "".join(traceback.format_exception(type(v), v, v.__traceback__))
            return redact_sensitive_text(f"{type(v).__name__}: {v}\nTraceback:\n{tb_text}")
        if isinstance(v, str):
            return redact_sensitive_text(v)
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

    @classmethod
    def from_pandas(
        cls,
        dataframe: pd.DataFrame | pd.Series,
        *,
        local_dir: str | Path,
        ref: str = "anonymous",
    ) -> DataFrameObject:
        """Build a self-contained DataFrameObject from a pandas frame.

        Convenience for tests, scripts, and any non-executor caller that
        needs the canonical executor wrapper without going through a live
        kernel. Production agents always receive ``DataFrameObject`` from
        the executor itself; this factory exists so the same payload type
        is the only one anything in the codebase ever has to manufacture.
        """

        return cls(
            ref=DataframeRef.from_pandas(dataframe, ref=ref, local_dir=local_dir)
        )

    @computed_field
    @property
    def head(self) -> dict[str, Any]:
        column_to_drop = self.value.index.name if self.value.index.name is not None else "index"
        value = self.value.drop(columns=[column_to_drop], errors="ignore")

        if len(value) <= _DATAFRAME_FULL_SHOW_THRESHOLD:
            return json.loads(value.to_json(orient="table"))
        return json.loads(value.head(_DATAFRAME_HEAD_TAIL_SIZE).to_json(orient="table"))

    @computed_field
    @property
    def tail(self) -> dict[str, Any] | None:
        column_to_drop = self.value.index.name if self.value.index.name is not None else "index"
        value = self.value.drop(columns=[column_to_drop], errors="ignore")

        if len(value) <= _DATAFRAME_FULL_SHOW_THRESHOLD:
            return None
        return json.loads(value.tail(_DATAFRAME_HEAD_TAIL_SIZE).to_json(orient="table"))

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
            max_cell = _DEFAULT_MAX_CELL_LENGTH
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
    """One recorded data operation from sandbox code.

    ``provenance`` is the upstream identity. ``data_object_ref`` pins
    the persisted snapshot — typed (``kind="data_object"``) so downstream
    code consumes a structured ref rather than a path string. ``version``
    is the 1-based snapshot index for that data_object's ``log.jsonl``;
    same ``v{N}`` semantic as datasets/charts/reports under the unified
    versioning model — so the agent can tell ``"GDPC1 v1"`` (first
    fetch) from ``"GDPC1 v3"`` (data refreshed twice in between).

    Both ``data_object_ref`` and ``version`` are ``None`` when the
    executor was configured without a persister.
    """

    model_config = {"arbitrary_types_allowed": True}

    provenance: Provenance
    row_count: int
    column_names: list[str]
    columns: list[dict[str, Any]]
    head: dict[str, Any] | None = None
    tail: dict[str, Any] | None = None
    data_object_ref: ArtifactRef | None = None
    version: int | None = None

    @property
    def source(self) -> str:
        return self.provenance.source

    @property
    def source_description(self) -> str:
        return self.provenance.source_description

    @property
    def params(self) -> dict[str, Any]:
        return self.provenance.params


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
        for output in self.outputs:
            blocks.extend(output.to_llm(mode=mode))
            blocks.append(
                {
                    "type": "text",
                    "text": "\n---\n",
                }
            )
        nb_block = self._notebook_ref_to_llm()
        if nb_block:
            blocks.append({"type": "text", "text": nb_block})
        fetch_block = self._fetch_log_to_llm()
        if fetch_block:
            blocks.append({"type": "text", "text": fetch_block})
        return blocks

    def _notebook_ref_to_llm(self) -> str:
        """Render ``metadata['notebook_ref']`` as a copy-pasteable XML block.

        Set by code tools (``return_notebook`` / ``edit_notebook`` with
        ``execute=True``) so the agent has the canonical
        ``(kind, logical_id, content_sha)`` triplet for the notebook the
        kernel just ran. The agent's ``return_dataset`` /
        ``return_chart`` / ``return_report`` calls require this triplet
        verbatim — without it the agent has to recompute the hash, which
        diverges from the framework's whitespace-stripped canonical form.
        """

        from parsimony_agents.identity import ArtifactRef

        raw = (self.metadata or {}).get("notebook_ref")
        if not isinstance(raw, dict):
            return ""
        try:
            ref = ArtifactRef.from_dict(raw)
        except (KeyError, ValueError):
            return ""
        return ref.to_self_closing_tag("notebook_ref") + "\n"

    def _fetch_log_to_llm(self) -> str:
        """Render persisted fetches as a copy-pasteable ``<fetch_log>`` block.

        Each entry surfaces its ``data_object_ref`` triplet
        ``(kind, logical_id, content_sha)`` so the agent can pass it
        directly as a ``source_ref`` in ``return_dataset`` /
        ``return_chart`` / ``return_report`` without inventing or
        recomputing hashes.
        """

        if not self.fetch_log:
            return ""
        lines: list[str] = ["<fetch_log>"]
        for entry in self.fetch_log:
            params_inline = escape_attr(
                json.dumps(entry.params or {}, sort_keys=True, default=str)
            )
            v_attr = f' version="{escape_attr(entry.version)}"' if entry.version is not None else ""
            lines.append(
                f'  <entry source="{escape_attr(entry.source)}" params="{params_inline}"{v_attr}>'
            )
            if entry.data_object_ref is not None:
                lines.append(
                    "    " + entry.data_object_ref.to_self_closing_tag("data_object_ref")
                )
            lines.append("  </entry>")
        lines.append("</fetch_log>")
        lines.append(
            "<note>Each &lt;data_object_ref&gt; above is the typed ArtifactRef "
            "for the persisted fetch. To use one as a source_ref in "
            "return_dataset / return_chart / return_report, copy "
            "{kind, logical_id, content_sha} verbatim — do not invent "
            "or recompute hashes.</note>"
        )
        return "\n".join(lines) + "\n"

    def to_frontend_dict(self):
        return {
            "type": self.type,
            "outputs": [output.to_frontend_dict() for output in self.outputs],
            "metadata": self.metadata,
            "fetch_log": [_fetch_entry_safe_dump(e) for e in self.fetch_log],
        }


def _fetch_entry_safe_dump(entry: FetchLogEntry) -> dict[str, Any]:
    """Wire-safe projection: replace ``provenance`` with its ``safe_dump`` and serialize ref."""
    raw = entry.model_dump(mode="json", exclude={"data_object_ref"})
    raw["provenance"] = entry.provenance.safe_dump()
    raw["data_object_ref"] = (
        entry.data_object_ref.to_dict() if entry.data_object_ref is not None else None
    )
    return raw
