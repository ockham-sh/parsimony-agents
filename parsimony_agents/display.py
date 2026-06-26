"""Rich terminal display for agent events.

Renders agent streaming events as polished, color-coded terminal output
with tool progress, Markdown responses, dataset tables, and syntax-highlighted code.

Requires ``rich`` for the full experience::

    pip install parsimony-agents[display]

Falls back to plain ``print()`` + ``tabulate`` if ``rich`` is not installed.

Usage::

    from parsimony_agents import Agent
    from parsimony_agents.display import stream_to_display

    agent = Agent(model="...", connectors=...)
    result = await stream_to_display(agent, "Show me US GDP trends")
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Protocol

import pandas as pd
from parsimony.result import ColumnRole

from parsimony_agents.agent.agent import Agent, AgentResult
from parsimony_agents.artifacts import Dataset
from parsimony_agents.execution.outputs import FetchLogEntry

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text as RichText

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_MAX_WIDTH = 100

_TOOL_ICONS = {
    "code": ">_",
    "utility": "<<",
    "return": "=>",
    "system": "::",
}

_TOOL_COLORS = {
    "code": "bright_magenta",
    "utility": "bright_blue",
    "return": "green",
    "system": "dim",
}

# Columns hidden from previews when schema roles are unavailable (legacy fallback).
_FALLBACK_HIDDEN_COLUMNS = frozenset({"index"})


def _schema_hidden_columns(column_schema: list[dict[str, Any]] | None) -> set[str]:
    """Column names to omit from previews based on output schema roles."""
    if not column_schema:
        return set(_FALLBACK_HIDDEN_COLUMNS)
    hidden: set[str] = set()
    for col in column_schema:
        name = col.get("name")
        if not isinstance(name, str):
            continue
        role = col.get("role")
        if role in (ColumnRole.METADATA, ColumnRole.KEY, ColumnRole.TITLE, "metadata", "key", "title"):
            hidden.add(name)
        if col.get("exclude_from_llm_view"):
            hidden.add(name)
    return hidden


def _title_from_preview(preview_df: pd.DataFrame | None, column_schema: list[dict[str, Any]] | None) -> str:
    """Best-effort title from a TITLE-role column in the preview row."""
    if preview_df is None or preview_df.empty or not column_schema:
        return ""
    title_cols = [
        c["name"]
        for c in column_schema
        if c.get("role") in (ColumnRole.TITLE, "title") and isinstance(c.get("name"), str)
    ]
    for col in title_cols:
        if col in preview_df.columns:
            val = preview_df[col].iloc[0]
            if pd.notna(val):
                return str(val)
    return ""


def _format_columns(names: list[str], max_cols: int = 5) -> str:
    """Format column names into a compact bracket string."""
    if not names:
        return ""
    shown = names[:max_cols]
    extra = len(names) - max_cols
    cols_str = ", ".join(shown)
    if extra > 0:
        cols_str += f", +{extra} more"
    return f"[{cols_str}]"


def _format_params(params: dict[str, Any], max_items: int = 4) -> str:
    """Format fetch params as compact key=value pairs."""
    if not params:
        return ""
    items = list(params.items())[:max_items]
    parts = [f"{k}: {v}" for k, v in items]
    extra = len(params) - max_items
    if extra > 0:
        parts.append(f"+{extra} more")
    return " · ".join(parts)


def _head_to_dataframe(head: dict[str, Any] | None) -> pd.DataFrame | None:
    """Convert a FetchLogEntry.head (orient='table' JSON) to a DataFrame."""
    if not head or "data" not in head:
        return None
    try:
        data = head["data"]
        if isinstance(data, list) and data:
            return pd.DataFrame(data)
        if isinstance(data, str):
            return None
    except Exception:
        pass
    return None


def _dataset_dataframe(artifact: Any) -> pd.DataFrame | None:
    """Best-effort DataFrame for terminal preview from a returned :class:`Dataset`."""
    if not isinstance(artifact, Dataset):
        return None
    pl = artifact.payload
    if pl is None:
        return None
    try:
        df = pl.value
    except Exception:
        return None
    return df if isinstance(df, pd.DataFrame) else None


def _collect_fetch_entries(result: AgentResult) -> list[FetchLogEntry]:
    """Extract deduplicated fetch-log entries from notebook-execute tool results."""
    entries: list[FetchLogEntry] = []
    seen: set[tuple] = set()
    for event in result.events:
        if getattr(event, "type", None) != "tool_event":
            continue
        r = getattr(event, "result", None)
        if not isinstance(r, dict):
            continue
        nb = r.get("notebook")
        if nb is None:
            continue
        for entry in getattr(nb, "data_objects", []) or []:
            if not isinstance(entry, FetchLogEntry):
                try:
                    entry = FetchLogEntry.model_validate(entry)
                except Exception:
                    continue
            key = (entry.source, str(sorted((entry.params or {}).items())))
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return entries


def _pick_display_columns(
    df: pd.DataFrame,
    *,
    column_schema: list[dict[str, Any]] | None = None,
    max_cols: int = 6,
) -> list[str]:
    """Select preview columns, hiding KEY/METADATA and LLM-excluded schema roles."""
    hidden = _schema_hidden_columns(column_schema)
    cols = [c for c in df.columns if c not in hidden]
    if len(cols) > max_cols:
        cols = cols[:max_cols]
    return cols


def _render_chart_to_png(spec: dict[str, Any]) -> str | None:
    """Render a Vega-Lite spec to a temp PNG file. Returns the path, or None on failure."""
    try:
        import json
        import tempfile

        import vl_convert as vlc

        png_bytes = vlc.vegalite_to_png(vl_spec=json.dumps(spec), scale=2.0)
        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as f:
            f.write(png_bytes)
            return f.name
    except Exception:
        return None


def _chart_summary(spec: dict[str, Any]) -> tuple[str, str, pd.DataFrame | None]:
    """Summarize a Vega-Lite spec for terminal display.

    Returns ``(mark, encodings, preview_df)``:
    - ``mark``: the mark type (e.g. ``"bar"``), or ``""`` if absent.
    - ``encodings``: compact ``"x=field, y=field, color=field"`` string built
      from the spec's ``encoding`` channels (falls back to a channel's
      ``aggregate`` when it carries no ``field``).
    - ``preview_df``: a DataFrame of inline ``data.values`` rows, or ``None``
      when the spec's data is a URL or absent.
    """
    mark_raw = spec.get("mark")
    if isinstance(mark_raw, dict):
        mark = str(mark_raw.get("type", ""))
    elif isinstance(mark_raw, str):
        mark = mark_raw
    else:
        mark = ""

    enc_parts: list[str] = []
    encoding = spec.get("encoding")
    if isinstance(encoding, dict):
        for channel, defn in encoding.items():
            if not isinstance(defn, dict):
                continue
            field = defn.get("field")
            if field is None:
                field = defn.get("aggregate")
            if field is not None:
                enc_parts.append(f"{channel}={field}")
    encodings = ", ".join(enc_parts)

    preview_df: pd.DataFrame | None = None
    data = spec.get("data")
    if isinstance(data, dict):
        values = data.get("values")
        if isinstance(values, list) and values and isinstance(values[0], dict):
            try:
                preview_df = pd.DataFrame(values)
            except Exception as exc:  # noqa: BLE001 — display must never crash
                logger.debug("_chart_summary: could not build preview DataFrame: %s", exc)
                preview_df = None
    return mark, encodings, preview_df


def _build_preview_table(
    df: pd.DataFrame,
    cols: list[str],
    *,
    rows: int,
    tail: bool = False,
    extra_cols: int = 0,
) -> Any:
    """Build a Rich ``Table`` previewing ``rows`` of ``df[cols]``.

    ``tail`` shows the last ``rows`` instead of the first; ``extra_cols`` (> 0)
    appends a dim ``"+N cols"`` column to signal hidden columns. Rich-only —
    called from :class:`_RichDisplay`.
    """
    table = Table(
        show_header=True,
        header_style="bold",
        show_lines=False,
        padding=(0, 1),
        pad_edge=True,
    )
    for col in cols:
        justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
        table.add_column(str(col), justify=justify, max_width=30)
    if extra_cols > 0:
        table.add_column(f"+{extra_cols} cols", style="dim", max_width=10)
    preview = df.tail(rows) if tail else df.head(rows)
    for _, row in preview.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                cells.append("[dim]--[/]")
            elif isinstance(val, float):
                cells.append(f"{val:.2f}")
            else:
                s = str(val)
                cells.append(s[:30] + "..." if len(s) > 30 else s)
        if extra_cols > 0:
            cells.append("")
        table.add_row(*cells)
    return table


def _open_file(path: str) -> None:
    """Best-effort open a file with the OS default viewer. Silent on failure."""
    import os
    import platform
    import subprocess

    try:
        system = platform.system()
        if "microsoft" in platform.uname().release.lower():
            # WSL — convert to Windows path and open with cmd.exe
            win_path = (
                subprocess.check_output(
                    ["wslpath", "-w", path],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", win_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(
                ["open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Windows" or os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Display backend protocol
# ---------------------------------------------------------------------------


class DisplayBackend(Protocol):
    def banner(self, question: str) -> None: ...
    def spinner_update(self, label: str) -> None: ...
    def spinner_stop(self) -> None: ...
    def tool_completed(self, tool_type: str, label: str, elapsed: float) -> None: ...
    def start_response(self) -> None: ...
    def stream_text(self, chunk: str) -> None: ...
    def end_response(self, full_text: str) -> None: ...
    def show_error(self, message: str, error_type: str | None = None) -> None: ...
    def show_datasets(self, datasets: dict[str, Any], max_rows: int, context: Any | None = None) -> None: ...
    def show_code(self, code: dict[str, Any], max_lines: int) -> None: ...
    def show_charts(self, charts: dict[str, Any], *, open_charts: bool = False) -> None: ...
    def show_reports(self, reports: dict[str, Any], max_lines: int = 40) -> None: ...
    def show_fetches(self, entries: list[FetchLogEntry]) -> None: ...
    def show_status(
        self,
        ok: bool,
        elapsed: float,
        tool_count: int,
        dataset_count: int,
        chart_count: int,
        notebook_count: int,
        error_count: int,
        report_count: int = 0,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Rich backend
# ---------------------------------------------------------------------------


class _RichDisplay:
    def __init__(self, console: Any | None = None) -> None:
        self._console = console or Console(width=_MAX_WIDTH, highlight=False)
        self._status: Any = None
        self._live: Any = None
        self._response_buffer = ""

    def banner(self, question: str) -> None:
        q = question[:120] + ("..." if len(question) > 120 else "")
        self._console.print()
        line = RichText()
        line.append("  In ", style="bold bright_blue")
        line.append("│ ", style="bright_blue")
        line.append(q)
        self._console.print(line)
        self._console.print()
        # Start spinner
        self._status = self._console.status("[yellow]Thinking...[/]", spinner="dots")
        self._status.start()

    def spinner_update(self, label: str) -> None:
        if self._status:
            self._status.update(f"[yellow]{label}[/]")

    def spinner_stop(self) -> None:
        if self._status:
            self._status.stop()
            self._status = None

    def tool_completed(self, tool_type: str, label: str, elapsed: float) -> None:
        icon = _TOOL_ICONS.get(tool_type, "  ")
        color = _TOOL_COLORS.get(tool_type, "dim")
        line = RichText()
        line.append(f"  {icon} ", style="green")
        line.append(label, style=f"bold {color}")
        time_str = f"{elapsed:.1f}s"
        pad = max(1, _MAX_WIDTH - len(f"  {icon} {label}") - len(time_str) - 2)
        line.append(" " * pad, style="dim")
        line.append(time_str, style="dim")
        if self._status:
            self._status.stop()
        self._console.print(line)
        if self._status:
            self._status.start()

    def start_response(self) -> None:
        self.spinner_stop()
        self._console.print()
        self._console.print(Rule("Response", style="bold"))
        self._console.print()
        # Drive a Live region holding the response as it streams, re-rendered as
        # Markdown on each chunk. Using Markdown (not console markup) also makes
        # ``[...]`` in the text render as links/literals instead of being parsed
        # as Rich markup tags. ``vertical_overflow="visible"`` keeps long
        # responses in scrollback rather than cropping to terminal height.
        self._response_buffer = ""
        try:
            self._live = Live(
                Markdown(""),
                console=self._console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start()
        except Exception:
            self._live = None

    def stream_text(self, chunk: str) -> None:
        self._response_buffer += chunk
        if self._live is not None:
            try:
                self._live.update(Markdown(self._response_buffer))
                return
            except Exception:
                # Live failed mid-stream — drop to raw passthrough for the rest.
                with contextlib.suppress(Exception):
                    self._live.stop()
                self._live = None
        # Fallback path: raw text, markup disabled so brackets are not mangled.
        self._console.print(chunk, end="", markup=False, highlight=False)

    def end_response(self, full_text: str) -> None:
        text = full_text or self._response_buffer
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.update(Markdown(text))
            with contextlib.suppress(Exception):
                self._live.stop()
            self._live = None
        self._response_buffer = ""
        self._console.print()

    def show_datasets(self, datasets: dict[str, Any], max_rows: int = 5, context: Any | None = None) -> None:
        if not datasets:
            return
        _ = context
        self._console.print()
        self._console.print(Rule("Datasets", style="bright_blue"))
        self._console.print()
        for aid, artifact in datasets.items():
            df = _dataset_dataframe(artifact)
            if not isinstance(df, pd.DataFrame):
                continue
            title = artifact.title or aid
            self._console.print(f"  [bold bright_blue]# {title}[/]")

            desc = artifact.description or ""
            if desc:
                self._console.print(f"  [dim]{desc}[/]")

            tags = list(artifact.tags) if artifact.tags else []
            if tags:
                self._console.print(f"  [dim]{' · '.join(tags)}[/]")

            for note in getattr(artifact, "notes", []):
                self._console.print(f"  [dim]  - {note}[/]")

            self._console.print()
            # Table
            rows, cols = df.shape
            display_cols = _pick_display_columns(df)
            table = _build_preview_table(
                df,
                display_cols,
                rows=max_rows,
                tail=True,
                extra_cols=cols - len(display_cols),
            )
            self._console.print(table)
            earlier = rows - max_rows
            if earlier > 0:
                self._console.print(f"  [dim]... {earlier} earlier rows[/]")
            self._console.print()

    def show_code(self, code: dict[str, Any], max_lines: int = 30) -> None:
        if not code:
            return
        self._console.print()
        self._console.print(Rule("Code", style="bright_magenta"))
        self._console.print()
        for nb_name, nb in code.items():
            source = getattr(nb, "code", nb) if not isinstance(nb, str) else nb
            if not source or not source.strip():
                continue
            self._console.print(f"  [bold bright_magenta]{{ }} {nb_name}[/]")
            self._console.print()
            lines = source.splitlines()
            if len(lines) > max_lines:
                truncated = "\n".join(lines[: max_lines - 3])
                truncated += f"\n\n# ... {len(lines) - max_lines + 3} more lines"
            else:
                truncated = source
            syntax = Syntax(
                truncated,
                "python",
                theme="monokai",
                line_numbers=True,
                padding=1,
            )
            self._console.print(Panel(syntax, border_style="dim", expand=True))
            self._console.print()

    def show_fetches(self, entries: list[FetchLogEntry]) -> None:
        if not entries:
            return
        self._console.print()
        self._console.print(Rule("Data", style="bright_blue"))
        self._console.print()

        for entry in entries:
            prov = entry.provenance
            rows, _cols = entry.row_count, len(entry.column_names)
            preview_df = _head_to_dataframe(entry.head)

            param_id = str(next(iter(entry.params.values()), "")) if entry.params else ""
            title = _title_from_preview(preview_df, entry.columns) or prov.source_description or ""
            header = RichText()
            header.append(f"  # {param_id or prov.source}", style="bold bright_blue")
            if title and title.lower() != param_id.lower():
                header.append(f" · {title}", style="bright_blue")
            self._console.print(header)

            parts: list[str] = []
            source = prov.source or entry.source
            if source:
                parts.append(source)
            if entry.params:
                parts.append(_format_params(entry.params))
            if parts:
                self._console.print(f"  [dim]{' · '.join(parts)}[/]")

            self._console.print()

            if preview_df is not None and not preview_df.empty:
                display_cols = _pick_display_columns(preview_df, column_schema=entry.columns, max_cols=5)
                max_preview_rows = 3
                table = _build_preview_table(
                    preview_df,
                    display_cols,
                    rows=max_preview_rows,
                    extra_cols=len(preview_df.columns) - len(display_cols),
                )
                self._console.print(table)
                remaining = rows - max_preview_rows
                if remaining > 0:
                    self._console.print(f"  [dim]... {remaining:,} more rows[/]")
            self._console.print()

    def show_charts(self, charts: dict[str, Any], *, open_charts: bool = False) -> None:
        if not charts:
            return
        self._console.print()
        self._console.print(Rule("Charts", style="green"))
        self._console.print()
        for name, chart in charts.items():
            # Resolve the Vega-Lite spec from the Chart object
            figure = getattr(chart, "figure", None)
            if figure is None:
                continue
            value = getattr(figure, "value", None)
            if value is None:
                continue
            spec = value if isinstance(value, dict) else value.to_dict()

            # Header: title or variable name
            title = getattr(chart, "title", "") or name
            self._console.print(f"  [bold green]# {title}[/]")
            desc = getattr(chart, "description", "")
            if desc:
                self._console.print(f"  [dim]{desc}[/]")
            for note in getattr(chart, "notes", []):
                self._console.print(f"  [dim]  - {note}[/]")

            # Textual summary — always shown, so the chart's content survives even
            # when image rendering is unavailable.
            mark, encodings, preview_df = _chart_summary(spec)
            summary_bits = [b for b in (mark, encodings) if b]
            if summary_bits:
                self._console.print(f"  [green]{' · '.join(summary_bits)}[/]")
            self._console.print()
            if preview_df is not None and not preview_df.empty:
                display_cols = _pick_display_columns(preview_df, max_cols=5)
                if display_cols:
                    table = _build_preview_table(
                        preview_df,
                        display_cols,
                        rows=3,
                        extra_cols=len(preview_df.columns) - len(display_cols),
                    )
                    self._console.print(table)
                    remaining = len(preview_df) - 3
                    if remaining > 0:
                        self._console.print(f"  [dim]... {remaining:,} more rows[/]")

            # Image is a convenience: save the PNG and only pop the OS viewer when
            # explicitly requested via ``open_charts``.
            path = _render_chart_to_png(spec)
            if path:
                self._console.print(f"  [dim]→ saved: {path}[/]")
                if open_charts:
                    _open_file(path)
            else:
                self._console.print("  [dim]~ image render unavailable[/]")
            self._console.print()

    def show_reports(self, reports: dict[str, Any], max_lines: int = 40) -> None:
        if not reports:
            return
        self._console.print()
        self._console.print(Rule("Reports", style="yellow"))
        self._console.print()
        for rid, report in reports.items():
            title = getattr(report, "title", "") or rid
            self._console.print(f"  [bold yellow]# {title}[/]")
            subtitle = getattr(report, "subtitle", "")
            if subtitle:
                self._console.print(f"  [dim]{subtitle}[/]")
            desc = getattr(report, "description", "")
            if desc:
                self._console.print(f"  [dim]{desc}[/]")
            tags = list(getattr(report, "tags", []) or [])
            if tags:
                self._console.print(f"  [dim]{' · '.join(tags)}[/]")
            for note in getattr(report, "notes", []):
                self._console.print(f"  [dim]  - {note}[/]")

            # Meta line: requested formats + embedded-artifact count.
            formats = list(getattr(report, "formats", []) or [])
            try:
                embeds = len(getattr(report, "embedded_refs", []) or [])
            except Exception:
                embeds = 0
            meta_bits = []
            if formats:
                meta_bits.append(f"formats: {', '.join(formats)}")
            meta_bits.append(f"embeds: {embeds}")
            self._console.print(f"  [dim]{' · '.join(meta_bits)}[/]")
            self._console.print()

            # Body rendered as Markdown, truncated by source lines. Truncation may
            # clip a trailing code fence — acceptable for a terminal preview.
            body = getattr(report, "markdown", "") or ""
            lines = body.splitlines()
            hidden = max(0, len(lines) - max_lines)
            shown = "\n".join(lines[:max_lines]) if hidden else body
            if shown.strip():
                self._console.print(Markdown(shown))
            if hidden > 0:
                self._console.print(f"  [dim]... {hidden} more lines[/]")
            self._console.print()

    def show_error(self, message: str, error_type: str | None = None) -> None:
        # Stop spinner so the panel is visible
        if self._status:
            self._status.stop()
            self._status = None
        title = f"Error: {error_type}" if error_type else "Error"
        self._console.print()
        self._console.print(Panel(message, title=title, border_style="red", expand=False))
        self._console.print()

    def show_status(
        self,
        ok: bool,
        elapsed: float,
        tool_count: int,
        dataset_count: int,
        chart_count: int,
        notebook_count: int,
        error_count: int,
        report_count: int = 0,
    ) -> None:
        label = "ok" if ok else "!!"
        style = "green" if ok else "red"
        self._console.print(Rule(label, style=style))
        parts = [f"Completed in {elapsed:.1f}s"]
        if tool_count:
            parts.append(f"{tool_count} tool call{'s' if tool_count != 1 else ''}")
        if dataset_count:
            parts.append(f"{dataset_count} dataset{'s' if dataset_count != 1 else ''}")
        if chart_count:
            parts.append(f"{chart_count} chart{'s' if chart_count != 1 else ''}")
        if report_count:
            parts.append(f"{report_count} report{'s' if report_count != 1 else ''}")
        if notebook_count:
            parts.append(f"{notebook_count} notebook{'s' if notebook_count != 1 else ''}")
        if error_count:
            parts.append(f"[red]{error_count} error{'s' if error_count != 1 else ''}[/]")
        icon = f"[bold {style}]{label}[/]"
        self._console.print(f"  {icon}  {' | '.join(parts)}")
        self._console.print()


# ---------------------------------------------------------------------------
# Plain-text fallback backend
# ---------------------------------------------------------------------------


class _PlainDisplay:
    def banner(self, question: str) -> None:
        q = question[:120] + ("..." if len(question) > 120 else "")
        print()
        print(f"  In | {q}")
        print()
        print("  ...working")

    def spinner_update(self, label: str) -> None:
        pass

    def spinner_stop(self) -> None:
        pass

    def tool_completed(self, tool_type: str, label: str, elapsed: float) -> None:
        icon = _TOOL_ICONS.get(tool_type, "  ")
        print(f"  {icon} {label}  ({elapsed:.1f}s)")

    def start_response(self) -> None:
        print()
        print("--- Response " + "-" * 47)
        print()

    def stream_text(self, chunk: str) -> None:
        print(chunk, end="", flush=True)

    def end_response(self, full_text: str) -> None:
        print()

    def show_datasets(self, datasets: dict[str, Any], max_rows: int = 5, context: Any | None = None) -> None:
        if not datasets:
            return
        _ = context
        print()
        print("--- Datasets " + "-" * 47)
        print()
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
        for aid, artifact in datasets.items():
            df = _dataset_dataframe(artifact)
            if not isinstance(df, pd.DataFrame):
                continue
            rows, cols = df.shape
            display_cols = _pick_display_columns(df)
            title = artifact.title or aid
            print(f"  # {title}")
            desc = artifact.description or ""
            if desc:
                print(f"  {desc}")
            tags = list(artifact.tags) if artifact.tags else []
            if tags:
                print(f"  {' · '.join(tags)}")
            for note in getattr(artifact, "notes", []):
                print(f"    - {note}")
            print()
            preview = df[display_cols].tail(max_rows)
            if tabulate:
                print(tabulate(preview, headers="keys", tablefmt="simple", showindex=False))
            else:
                print(preview.to_string(index=False))
            earlier = rows - max_rows
            if earlier > 0:
                print(f"  ... {earlier} earlier rows")
            print()

    def show_code(self, code: dict[str, Any], max_lines: int = 30) -> None:
        if not code:
            return
        print()
        print("--- Code " + "-" * 51)
        print()
        for nb_name, nb in code.items():
            source = getattr(nb, "code", nb) if not isinstance(nb, str) else nb
            if not source or not source.strip():
                continue
            print(f"  {{ }} {nb_name}")
            print()
            lines = source.splitlines()
            show = lines[:max_lines]
            for i, line in enumerate(show, 1):
                print(f"  {i:>3} | {line}")
            if len(lines) > max_lines:
                print(f"  ... {len(lines) - max_lines} more lines")
            print()

    def show_fetches(self, entries: list[FetchLogEntry]) -> None:
        if not entries:
            return
        print()
        print("--- Data " + "-" * 51)
        print()
        try:
            from tabulate import tabulate as _tab
        except ImportError:
            _tab = None
        for entry in entries:
            prov = entry.provenance
            rows, _cols = entry.row_count, len(entry.column_names)
            preview_df = _head_to_dataframe(entry.head)

            param_id = str(next(iter(entry.params.values()), "")) if entry.params else ""
            title = _title_from_preview(preview_df, entry.columns) or prov.source_description or ""
            header = f"  # {param_id or prov.source}"
            if title and title.lower() != param_id.lower():
                header += f" · {title}"
            print(header)

            parts = []
            source = prov.source or entry.source
            if source:
                parts.append(source)
            if entry.params:
                parts.append(_format_params(entry.params))
            if parts:
                print(f"  {' · '.join(parts)}")
            print()

            if preview_df is not None and not preview_df.empty:
                display_cols = _pick_display_columns(preview_df, column_schema=entry.columns, max_cols=5)
                preview = preview_df[display_cols].head(3)
                if _tab:
                    print(_tab(preview, headers="keys", tablefmt="simple", showindex=False))
                else:
                    print(preview.to_string(index=False))
                remaining = rows - 3
                if remaining > 0:
                    print(f"  ... {remaining:,} more rows")
            print()

    def show_charts(self, charts: dict[str, Any], *, open_charts: bool = False) -> None:
        if not charts:
            return
        print()
        print("--- Charts " + "-" * 49)
        print()
        try:
            from tabulate import tabulate as _tab
        except ImportError:
            _tab = None
        for name, chart in charts.items():
            figure = getattr(chart, "figure", None)
            if figure is None:
                continue
            value = getattr(figure, "value", None)
            if value is None:
                continue
            spec = value if isinstance(value, dict) else value.to_dict()
            title = getattr(chart, "title", "") or name
            print(f"  # {title}")
            desc = getattr(chart, "description", "")
            if desc:
                print(f"  {desc}")
            for note in getattr(chart, "notes", []):
                print(f"    - {note}")

            mark, encodings, preview_df = _chart_summary(spec)
            summary_bits = [b for b in (mark, encodings) if b]
            if summary_bits:
                print(f"  {' · '.join(summary_bits)}")
            if preview_df is not None and not preview_df.empty:
                display_cols = _pick_display_columns(preview_df, max_cols=5)
                if display_cols:
                    preview = preview_df[display_cols].head(3)
                    if _tab:
                        print(_tab(preview, headers="keys", tablefmt="simple", showindex=False))
                    else:
                        print(preview.to_string(index=False))
                    remaining = len(preview_df) - 3
                    if remaining > 0:
                        print(f"  ... {remaining:,} more rows")

            path = _render_chart_to_png(spec)
            if path:
                print(f"  → saved: {path}")
                if open_charts:
                    _open_file(path)
            else:
                print("  ~ image render unavailable")
            print()
        print()

    def show_reports(self, reports: dict[str, Any], max_lines: int = 40) -> None:
        if not reports:
            return
        print()
        print("--- Reports " + "-" * 48)
        print()
        for rid, report in reports.items():
            title = getattr(report, "title", "") or rid
            print(f"  # {title}")
            subtitle = getattr(report, "subtitle", "")
            if subtitle:
                print(f"  {subtitle}")
            desc = getattr(report, "description", "")
            if desc:
                print(f"  {desc}")
            tags = list(getattr(report, "tags", []) or [])
            if tags:
                print(f"  {' · '.join(tags)}")
            for note in getattr(report, "notes", []):
                print(f"    - {note}")
            formats = list(getattr(report, "formats", []) or [])
            try:
                embeds = len(getattr(report, "embedded_refs", []) or [])
            except Exception:
                embeds = 0
            meta_bits = []
            if formats:
                meta_bits.append(f"formats: {', '.join(formats)}")
            meta_bits.append(f"embeds: {embeds}")
            print(f"  {' · '.join(meta_bits)}")
            print()
            body = getattr(report, "markdown", "") or ""
            lines = body.splitlines()
            hidden = max(0, len(lines) - max_lines)
            for line in lines[:max_lines]:
                print(f"  {line}")
            if hidden > 0:
                print(f"  ... {hidden} more lines")
            print()

    def show_error(self, message: str, error_type: str | None = None) -> None:
        header = f"--- Error ({error_type}) " if error_type else "--- Error "
        print()
        print(header + "-" * max(0, 60 - len(header)))
        print(f"  {message}")
        print()

    def show_status(
        self,
        ok: bool,
        elapsed: float,
        tool_count: int,
        dataset_count: int,
        chart_count: int,
        notebook_count: int,
        error_count: int,
        report_count: int = 0,
    ) -> None:
        label = "ok" if ok else "!!"
        parts = [f"Completed in {elapsed:.1f}s"]
        if tool_count:
            parts.append(f"{tool_count} tool call{'s' if tool_count != 1 else ''}")
        if dataset_count:
            parts.append(f"{dataset_count} dataset{'s' if dataset_count != 1 else ''}")
        if chart_count:
            parts.append(f"{chart_count} chart{'s' if chart_count != 1 else ''}")
        if report_count:
            parts.append(f"{report_count} report{'s' if report_count != 1 else ''}")
        if notebook_count:
            parts.append(f"{notebook_count} notebook{'s' if notebook_count != 1 else ''}")
        if error_count:
            parts.append(f"{error_count} error{'s' if error_count != 1 else ''}")
        print(f"--- {label} " + "-" * (55 - len(label)))
        print(f"  {label}  {' | '.join(parts)}")
        print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _make_backend(console: Any | None = None) -> DisplayBackend:
    if HAS_RICH:
        return _RichDisplay(console=console)  # type: ignore[return-value]
    return _PlainDisplay()  # type: ignore[return-value]


def _bullet_section(lines: list[str], title: str, items: list[str]) -> None:
    """Append a titled bullet block to ``lines`` if ``items`` is non-empty."""
    if not items:
        return
    lines.append("")
    lines.append(f"{title}:")
    lines.extend(f"  - {item}" for item in items)


def _format_handoff(event: Any) -> str:
    """Render a ``Handoff`` event into a human-readable error body."""
    lines = [getattr(event, "rationale", "") or "The agent could not complete the task."]
    _bullet_section(lines, "Blockers", list(getattr(event, "blockers", None) or []))
    _bullet_section(lines, "Suggested next steps", list(getattr(event, "suggested_next_steps", None) or []))
    return "\n".join(lines)


def _format_partial_summary(event: Any) -> str:
    """Render a ``PartialRunSummary`` event into a human-readable error body."""
    lines = ["The run stopped before completing."]
    plan = getattr(event, "next_step_plan", None)
    if plan:
        lines.extend(["", plan])
    _bullet_section(lines, "Missing", list(getattr(event, "missing", None) or []))
    _bullet_section(lines, "What was established", list(getattr(event, "learned_facts", None) or []))
    return "\n".join(lines)


async def stream_to_display(
    agent: Agent,
    message: str,
    *,
    ctx: Any | None = None,
    console: Any | None = None,
    show_code: bool = True,
    show_data: bool = True,
    max_table_rows: int = 5,
    max_code_lines: int = 30,
    open_charts: bool = False,
) -> AgentResult:
    """Run the agent with live terminal display, returning an :class:`AgentResult`.

    Wraps ``agent.run()`` with polished terminal output: a spinner during
    analysis, numbered tool progress lines, streamed Markdown response, dataset
    tables, syntax-highlighted code, chart previews, and rendered reports.

    Args:
        agent: The Agent instance.
        message: User message / question.
        ctx: Optional AgentContext for multi-turn continuation.
        console: Optional ``rich.console.Console`` (auto-created if omitted).
        show_code: Display generated code notebooks (default True).
        show_data: Display dataset previews (default True).
        max_table_rows: Max rows per dataset preview (default 5).
        max_code_lines: Max lines per code notebook (default 30).
        open_charts: Also open each chart's rendered PNG in the OS image viewer
            (default False — the in-terminal summary is always shown).

    Returns:
        AgentResult with ``.text``, ``.datasets``, ``.charts``, ``.reports``, ``.context``.
    """
    # Suppress noisy internal logging during display
    _quiet_loggers = [
        logging.getLogger("parsimony_agents.theme"),
        logging.getLogger("parsimony_agents.errors"),
    ]
    _saved_levels = [(lg, lg.level) for lg in _quiet_loggers]
    for lg in _quiet_loggers:
        lg.setLevel(logging.CRITICAL)

    display = _make_backend(console)
    result = AgentResult()
    tool_start_times: dict[str, float] = {}
    tool_count = 0
    error_count = 0
    response_started = False
    start = time.monotonic()

    display.banner(message)

    try:
        async for event in agent.run(message, ctx=ctx):
            result._collect(event)
            etype = getattr(event, "type", None)

            if etype == "reasoning_delta":
                display.spinner_update("Thinking...")

            elif etype == "tool_event":
                if not event.completed:
                    tool_start_times[event.tool_call_id] = time.monotonic()
                    label = event.ui_message or event.tool_name.replace("_", " ").title()
                    display.spinner_update(f"{label}...")
                else:
                    tool_count += 1
                    elapsed = time.monotonic() - tool_start_times.pop(event.tool_call_id, start)
                    label = (
                        getattr(event, "ui_message_completed", None)
                        or event.ui_message
                        or event.tool_name.replace("_", " ").title()
                    )
                    display.tool_completed(event.tool_type, label, elapsed)

            elif etype == "text_delta":
                # Skip non-delta (full replacement) events — they duplicate streamed text
                if not getattr(event, "delta", True):
                    continue
                if not response_started:
                    response_started = True
                    display.start_response()
                display.stream_text(event.content)

            elif etype == "error":
                error_count += 1
                display.spinner_stop()
                display.show_error(
                    getattr(event, "message", "Unknown error"),
                    error_type=getattr(event, "error_type", None),
                )

            elif etype == "handoff":
                # Terminal, non-interactive failure: the agent gave up. Carries
                # no ``error`` event, so surface it explicitly — otherwise the
                # run renders as a deceptive "ok".
                error_count += 1
                display.spinner_stop()
                display.show_error(_format_handoff(event), error_type="handoff")

            elif etype == "partial_run_summary":
                # Companion to handoff: the run stopped before completing (e.g.
                # budget exhausted with policy=stop). Also failure, also silent.
                error_count += 1
                display.spinner_stop()
                display.show_error(_format_partial_summary(event), error_type="incomplete")

    finally:
        # Restore logger levels
        for lg, level in _saved_levels:
            lg.setLevel(level)
        # Always tear down the live regions (spinner + the Markdown ``Live``), even
        # if ``run()`` raised mid-stream. A leaked ``Live`` keeps Rich's render
        # thread writing to the terminal and corrupts any output the caller emits
        # after catching the exception.
        display.spinner_stop()
        if response_started:
            display.end_response(result.text)

    # Post-response sections (normal completion only — skipped if ``run()`` raised)
    if show_data:
        fetch_entries = _collect_fetch_entries(result)
        display.show_fetches(fetch_entries)

    if show_code:
        display.show_code({}, max_lines=max_code_lines)

    if show_data:
        display.show_datasets(result.datasets, max_rows=max_table_rows, context=result.context)

    display.show_charts(result.charts, open_charts=open_charts)

    display.show_reports(result.reports)

    elapsed = time.monotonic() - start
    display.show_status(
        ok=result.ok,
        elapsed=elapsed,
        tool_count=tool_count,
        dataset_count=len(result.datasets),
        chart_count=len(result.charts),
        notebook_count=0,
        error_count=error_count,
        report_count=len(result.reports),
    )

    return result


def display_result(
    result: AgentResult,
    *,
    console: Any | None = None,
    show_code: bool = True,
    show_data: bool = True,
    max_table_rows: int = 5,
    max_code_lines: int = 30,
    open_charts: bool = False,
) -> None:
    """Render a completed :class:`AgentResult` to the terminal.

    Use this when you already have a result from ``agent.ask()`` and want
    to display it after the fact (no streaming).
    """
    display = _make_backend(console)

    if result.text:
        display.start_response()
        display.stream_text(result.text)
        display.end_response(result.text)

    if show_data:
        fetch_entries = _collect_fetch_entries(result)
        display.show_fetches(fetch_entries)

    if show_code:
        display.show_code({}, max_lines=max_code_lines)

    if show_data:
        display.show_datasets(result.datasets, max_rows=max_table_rows, context=result.context)

    display.show_charts(result.charts, open_charts=open_charts)

    display.show_reports(result.reports)

    display.show_status(
        ok=result.ok,
        elapsed=0.0,
        tool_count=sum(
            1 for e in result.events if getattr(e, "type", None) == "tool_event" and getattr(e, "completed", False)
        ),
        dataset_count=len(result.datasets),
        chart_count=len(result.charts),
        notebook_count=0,
        error_count=sum(1 for e in result.events if getattr(e, "type", None) == "error"),
        report_count=len(result.reports),
    )
