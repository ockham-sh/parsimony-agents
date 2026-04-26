"""Typed session-state summaries for the agent context (kernel + workspace pointers).

These models are product-agnostic: the terminal (or any host) fills them; :meth:`to_llm`
produces a bounded XML block for the context snapshot.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import altair as alt
import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter

KernelValueKind = Literal["dataframe", "series", "altair_chart", "primitive", "omitted"]


def _column_sample(names: list[str], *, max_cols: int = 8) -> str:
    if not names:
        return ""
    if len(names) <= max_cols:
        return ", ".join(names)
    return ", ".join(names[:max_cols]) + f", … ({len(names)} cols)"


def _summarize_pandas_dataframe(df: pd.DataFrame) -> str:
    n, m = int(len(df)), int(len(df.columns))
    col_names = [str(c) for c in df.columns.tolist()]
    return f"{n} rows × {m} cols: {_column_sample(col_names)}"


def _summarize_pandas_series(s: pd.Series) -> str:
    return f"length {len(s)}: name={s.name!r}"


def summarize_kernel_value(
    name: str,
    value: Any,
) -> tuple[KernelValueKind, str] | None:
    """Return a (kind, detail) for *value*, or None to omit from session state."""
    if not name or not isinstance(name, str):
        return None
    if name.startswith("_"):
        return None

    try:
        from parsimony.connector import Connectors
    except ImportError:
        Connectors = ()  # type: ignore[assignment, misc]

    if Connectors and isinstance(value, Connectors):
        return None
    if callable(value) and not isinstance(value, type):
        return None

    if isinstance(value, pd.DataFrame):
        return "dataframe", _summarize_pandas_dataframe(value)
    if isinstance(value, pd.Series):
        return "series", _summarize_pandas_series(value)
    if isinstance(value, alt.TopLevelMixin):
        return "altair_chart", "altair chart (TopLevel)"
    if isinstance(value, (int, float, bool)) or value is None:
        s = "null" if value is None else repr(value)[:120]
        return "primitive", s
    if isinstance(value, str):
        t = value.replace("\n", " ")
        if len(t) > 160:
            t = t[:160] + "…"
        return "primitive", t

    return None


class KernelVariableSummary(BaseModel):
    name: str
    kind: KernelValueKind
    detail: str = ""


def kernel_summaries_from_locals_map(locals_map: dict[str, Any]) -> list[KernelVariableSummary]:
    out: list[KernelVariableSummary] = []
    for name in sorted(locals_map.keys()):
        got = summarize_kernel_value(name, locals_map[name])
        if got is None:
            continue
        kind, detail = got
        out.append(KernelVariableSummary(name=name, kind=kind, detail=detail))
    return out


class WorkspaceArtifactLine(BaseModel):
    """One file-backed row in the session summary (path + one-line blurb)."""

    path: str
    kind: str
    summary: str = ""


class SessionState(BaseModel):
    """Ephemeral: kernel variable hints plus pointers to key workspace files."""

    kernel: list[KernelVariableSummary] = Field(default_factory=list)
    workspace_artifacts: list[WorkspaceArtifactLine] = Field(default_factory=list)

    def to_llm_text(self) -> str:
        """Bounded XML for injection into the agent context."""
        lines: list[str] = ["<session_state>"]
        if self.kernel:
            lines.append("  <kernel_variables>")
            for v in self.kernel:
                kind_attr = re.sub(r"[^a-z0-9_]+", "_", v.kind)
                d = (v.detail or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lines.append(
                    f'    <var name="{v.name}" kind="{kind_attr}">{d}</var>'
                )
            lines.append("  </kernel_variables>")
        if self.workspace_artifacts:
            lines.append("  <workspace_artifacts>")
            for a in self.workspace_artifacts:
                sa = a.summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                lines.append(
                    f'    <artifact path="{a.path}" kind="{a.kind}">{sa}</artifact>'
                )
            lines.append("  </workspace_artifacts>")
        lines.append(
            "  <note>Ephemeral: kernel variables are cleared on kernel restart. "
            "User-visible deliverables require return_dataset / return_chart. "
            "Use read_artifact for file details.</note>"
        )
        lines.append("</session_state>")
        return "\n".join(lines) + "\n"


def parse_kernel_summaries_from_remote(body: Any) -> list[KernelVariableSummary]:
    """Normalise sandbox JSON (list of objects or legacy name-&gt;type map) into summaries."""
    if body is None:
        return []
    if isinstance(body, list):
        return TypeAdapter(list[KernelVariableSummary]).validate_python(body)
    if not isinstance(body, dict):
        return []
    if "name" in body and "kind" in body:
        return [KernelVariableSummary.model_validate(body)]
    out: list[KernelVariableSummary] = []
    for k, v in body.items():
        if isinstance(v, str):
            kind: KernelValueKind = (
                "dataframe"
                if v == "dataframe"
                else "series"
                if v == "series"
                else "omitted"
            )
            out.append(KernelVariableSummary(name=k, kind=kind, detail=""))
        elif isinstance(v, dict):
            out.append(
                KernelVariableSummary(
                    name=str(v.get("name", k)),
                    kind=v.get("kind", "omitted"),  # type: ignore[arg-type]
                    detail=str(v.get("detail", "")),
                )
            )
    out.sort(key=lambda x: x.name)
    return out


__all__ = [
    "KernelValueKind",
    "KernelVariableSummary",
    "SessionState",
    "WorkspaceArtifactLine",
    "kernel_summaries_from_locals_map",
    "parse_kernel_summaries_from_remote",
    "summarize_kernel_value",
]
