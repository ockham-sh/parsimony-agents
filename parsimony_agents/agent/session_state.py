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

from parsimony_agents.agent.xml_render import escape_attr, escape_text
from parsimony_agents.identity import ArtifactRef

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


class WorkspaceArtifactLine(BaseModel):
    """One file-backed row in the session summary.

    ``ref`` is the typed :class:`~parsimony_agents.identity.ArtifactRef`
    the agent must copy verbatim into ``return_dataset`` /
    ``return_chart`` / ``return_report`` lineage fields. It is populated
    for every kind the agent can ref (``notebook``, ``dataset``,
    ``chart``, ``report``) and ``None`` for everything else (e.g. raw
    ``data_objects`` whose refs only flow through ``<fetch_log>``, or
    user data files that aren't part of the lineage graph).

    ``new`` is set when the ref was minted or advanced *during the
    current turn* (Task 15). It surfaces as ``new="true"`` in the
    rendered ``<turn_artifacts>`` block so the agent can distinguish
    just-published refs from cross-turn ones at a glance.
    """

    path: str
    kind: str
    summary: str = ""
    ref: ArtifactRef | None = None
    new: bool = False


def fuse_workspace_artifacts(
    cross_turn: list[WorkspaceArtifactLine],
    minted: list[ArtifactRef],
) -> list[WorkspaceArtifactLine]:
    """Merge turn-start workspace_artifacts with this-turn's minted refs.

    Dedup by ``(kind, logical_id)``: when a minted ref shares logical_id
    with an existing line, the minted ref's ``content_sha`` wins (the
    artifact was advanced this turn), and the line is marked ``new=True``.
    Minted refs that don't match any existing line are appended as new
    lines using their canonical ``.ockham/...`` path; the line carries no
    summary because the host hasn't observed it yet.

    Order: existing lines keep their slot; brand-new minted lines are
    appended at the end. Preserves the agent's mental model of "what was
    here before, then what just got added".
    """
    if not minted:
        return list(cross_turn)

    out: list[WorkspaceArtifactLine] = []
    minted_by_key: dict[tuple[str, str], ArtifactRef] = {
        (r.kind, r.logical_id): r for r in minted
    }
    matched: set[tuple[str, str]] = set()

    for line in cross_turn:
        key = (line.kind, line.ref.logical_id) if line.ref else None
        replacement = minted_by_key.get(key) if key else None
        if replacement is None:
            out.append(line)
            continue
        matched.add(key)
        out.append(
            line.model_copy(
                update={"ref": replacement, "new": True}
            )
        )

    for key, ref in minted_by_key.items():
        if key in matched:
            continue
        out.append(
            WorkspaceArtifactLine(
                path=ref.workspace_file_path,
                kind=ref.kind,
                summary="",
                ref=ref,
                new=True,
            )
        )

    return out


class SessionState(BaseModel):
    """Ephemeral: kernel variable hints plus pointers to key workspace files."""

    kernel: list[KernelVariableSummary] = Field(default_factory=list)
    workspace_artifacts: list[WorkspaceArtifactLine] = Field(default_factory=list)

    def to_llm_text(self, *, minted_refs: list[ArtifactRef] | None = None) -> str:
        """Bounded XML for injection into the agent context.

        When ``minted_refs`` is provided (set by the agent loop each
        iteration), the rendered block fuses turn-start
        ``workspace_artifacts`` with this-turn's minted refs into a
        single ``<turn_artifacts>`` view — the agent's canonical surface
        for "what artifacts exist right now, with their latest refs".
        """
        artifacts = fuse_workspace_artifacts(
            self.workspace_artifacts, minted_refs or []
        )

        lines: list[str] = ["<session_state>"]
        if self.kernel:
            lines.append("  <kernel_variables>")
            for v in self.kernel:
                kind_attr = re.sub(r"[^a-z0-9_]+", "_", v.kind)
                lines.append(
                    f'    <var name="{escape_attr(v.name)}" kind="{kind_attr}">'
                    f"{escape_text(v.detail or '')}</var>"
                )
            lines.append("  </kernel_variables>")
        if artifacts:
            lines.append("  <turn_artifacts>")
            for a in artifacts:
                # When the path is content-addressed, surface the typed
                # ref's full ``kind/logical_id/content_sha`` triplet so the
                # agent can copy it verbatim. Otherwise fall back to just
                # the bare ``kind`` (data_objects, user data files).
                attrs = a.ref.to_xml_attrs() if a.ref else f'kind="{escape_attr(a.kind)}"'
                if a.new:
                    attrs = f'{attrs} new="true"'
                lines.append(
                    f'    <artifact path="{escape_attr(a.path)}" {attrs}>'
                    f"{escape_text(a.summary)}</artifact>"
                )
            lines.append("  </turn_artifacts>")
        lines.append(
            "  <note>Kernel variables clear on kernel restart. "
            "&lt;turn_artifacts&gt; is the single canonical view of the workspace's "
            "current artifacts: cross-turn entries plus this turn's minted refs "
            '(marked new="true"). Copy {kind, logical_id, content_sha} verbatim from '
            "the matching &lt;artifact&gt; — do not invent or recompute hashes.</note>"
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
    "fuse_workspace_artifacts",
    "kernel_summaries_from_locals_map",
    "parse_kernel_summaries_from_remote",
    "summarize_kernel_value",
]
