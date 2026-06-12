"""Per-variable kernel namespace summaries, as JSON-ready rows.

Execution-domain: summarize a kernel's locals map into bounded
:class:`KernelVariableSummary` rows the host (or the sandboxed kernel's RPC
handler) ships out of the kernel process. The agent layer renders these into
session-state XML; this module knows nothing about that rendering.
"""

from __future__ import annotations

__all__ = [
    "KernelValueKind",
    "KernelVariableSummary",
    "kernel_summaries_from_locals_map",
    "parse_kernel_summaries_from_remote",
    "summarize_kernel_value",
]

from typing import Any, Literal

import altair as alt
import pandas as pd
from pydantic import BaseModel, TypeAdapter

KernelValueKind = Literal["dataframe", "series", "altair_chart", "primitive", "omitted"]


_DF_MAX_COLS = 12
_DF_VALUE_REPR_MAX = 32


def _truncate_repr(value: Any) -> str:
    """Compact repr for a single dataframe cell — bounded length, NaN-aware."""
    try:
        if pd.isna(value):
            return "NaN"
    except (TypeError, ValueError):
        pass
    try:
        s = repr(value)
    except Exception:  # pragma: no cover — defensive only
        return "<unrepr>"
    if len(s) > _DF_VALUE_REPR_MAX:
        return s[:_DF_VALUE_REPR_MAX] + "…"
    return s


def _summarize_pandas_dataframe(df: pd.DataFrame) -> str:
    """Multi-line summary the agent can use to skip a confirmation roundtrip.

    Three lines: ``shape=(rows, cols)``, ``columns: name:dtype, …`` (capped at
    :data:`_DF_MAX_COLS`), and ``first_row: {col: value, …}`` (cells truncated).
    Empty dataframes get only the first two lines.
    """
    n, m = int(len(df)), int(len(df.columns))
    cols = [str(c) for c in df.columns.tolist()]
    visible = cols[:_DF_MAX_COLS]
    extra = "" if len(cols) <= _DF_MAX_COLS else f", … (+{len(cols) - _DF_MAX_COLS} more)"

    col_pieces: list[str] = []
    for c in visible:
        try:
            dt = str(df[c].dtype)
        except Exception:  # pragma: no cover
            dt = "?"
        col_pieces.append(f"{c}:{dt}")
    columns_line = ", ".join(col_pieces) + extra

    lines = [f"shape=({n}, {m})", f"columns: {columns_line}"]

    if n > 0:
        try:
            row = df.iloc[0]
            items: list[str] = [
                f"{c!r}: {_truncate_repr(row[c])}" for c in visible
            ]
            inner = ", ".join(items)
            if len(cols) > _DF_MAX_COLS:
                inner = inner + ", …"
            lines.append("first_row: {" + inner + "}")
        except Exception:  # pragma: no cover
            lines.append("first_row: <unavailable>")

    return "\n".join(lines)


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


def parse_kernel_summaries_from_remote(body: Any) -> list[KernelVariableSummary]:
    """Normalise sandbox JSON (list of objects or legacy name->type map) into summaries."""
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
