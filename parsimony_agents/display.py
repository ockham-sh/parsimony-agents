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

import logging
import time
from typing import Any, Protocol

import pandas as pd

from parsimony_agents.agent.agent import Agent, AgentResult
from parsimony_agents.execution.outputs import FetchLogEntry

try:
    from rich.console import Console
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

# Metadata columns to hide from dataset preview (keep the useful ones)
_HIDDEN_COLUMNS = {
    "index", "realtime_start", "realtime_end", "series_id",
    "frequency_short", "seasonal_adjustment_short", "units_short",
}


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


def _get_variable(context: Any | None, name: str) -> Any | None:
    """Look up a Variable by name from the agent context."""
    if context is None:
        return None
    var_store = getattr(context, "data_context", None)
    if var_store and hasattr(var_store, "variables"):
        return var_store.variables.get(name)
    return None


def _collect_fetch_entries(result: AgentResult) -> list[FetchLogEntry]:
    """Extract deduplicated FetchLogEntry objects from notebooks in the result."""
    entries: list[FetchLogEntry] = []
    seen: set[tuple] = set()
    ctx = result.context
    if ctx is None:
        return entries
    for nb in getattr(ctx, "notebooks", {}).values():
        for entry in getattr(nb, "data_objects", []):
            # Deduplicate by (source, params hash)
            key = (entry.source, str(sorted(entry.params.items())))
            if key not in seen:
                seen.add(key)
                entries.append(entry)
    return entries


def _pick_display_columns(df: pd.DataFrame, max_cols: int = 6) -> list[str]:
    """Select the most useful columns for display, hiding metadata."""
    cols = [c for c in df.columns if c not in _HIDDEN_COLUMNS]
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


def _open_file(path: str) -> None:
    """Best-effort open a file with the OS default viewer. Silent on failure."""
    import os
    import platform
    import subprocess

    try:
        system = platform.system()
        if "microsoft" in platform.uname().release.lower():
            # WSL — convert to Windows path and open with cmd.exe
            win_path = subprocess.check_output(
                ["wslpath", "-w", path], stderr=subprocess.DEVNULL,
            ).decode().strip()
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", win_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(
                ["open", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Windows" or os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
    def show_datasets(self, datasets: dict[str, Any], max_rows: int, context: Any | None = None) -> None: ...
    def show_code(self, code: dict[str, Any], max_lines: int) -> None: ...
    def show_charts(self, charts: dict[str, Any]) -> None: ...
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
    ) -> None: ...


# ---------------------------------------------------------------------------
# Rich backend
# ---------------------------------------------------------------------------


class _RichDisplay:
    def __init__(self, console: Any | None = None) -> None:
        self._console = console or Console(width=_MAX_WIDTH, highlight=False)
        self._status: Any = None

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

    def stream_text(self, chunk: str) -> None:
        self._console.print(chunk, end="", highlight=False)

    def end_response(self, full_text: str) -> None:
        self._console.print()

    def show_datasets(self, datasets: dict[str, Any], max_rows: int = 5, context: Any | None = None) -> None:
        if not datasets:
            return
        self._console.print()
        self._console.print(Rule("Datasets", style="bright_blue"))
        self._console.print()
        for name, artifact in datasets.items():
            # Resolve DataFrame from context
            var = _get_variable(context, name)
            df = getattr(getattr(var, "output", None), "value", None) if var else None
            if not isinstance(df, pd.DataFrame):
                continue

            # Header: artifact title or variable name
            title = getattr(artifact, "title", "") or name
            self._console.print(f"  [bold bright_blue]# {title}[/]")

            # Description
            desc = getattr(artifact, "description", "")
            if desc:
                self._console.print(f"  [dim]{desc}[/]")

            # Tags
            tags = getattr(artifact, "tags", [])
            if tags:
                self._console.print(f"  [dim]{' · '.join(tags)}[/]")

            # Notes
            for note in getattr(artifact, "notes", []):
                self._console.print(f"  [dim]  - {note}[/]")

            self._console.print()
            # Table
            rows, cols = df.shape
            display_cols = _pick_display_columns(df)
            has_extra = len(display_cols) < cols
            table = Table(
                show_header=True,
                header_style="bold",
                show_lines=False,
                padding=(0, 1),
                pad_edge=True,
            )
            for col in display_cols:
                justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
                table.add_column(str(col), justify=justify, max_width=30)
            if has_extra:
                table.add_column(f"+{cols - len(display_cols)} cols", style="dim", max_width=10)
            preview = df.tail(max_rows)
            for _, row in preview.iterrows():
                cells = []
                for col in display_cols:
                    val = row[col]
                    if pd.isna(val):
                        cells.append("[dim]--[/]")
                    elif isinstance(val, float):
                        cells.append(f"{val:.2f}")
                    else:
                        s = str(val)
                        cells.append(s[:30] + "..." if len(s) > 30 else s)
                if has_extra:
                    cells.append("")
                table.add_row(*cells)
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
                truncated = "\n".join(lines[:max_lines - 3])
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

            # Header: # UNRATE · Unemployment Rate
            param_id = str(next(iter(entry.params.values()), "")) if entry.params else ""
            title = prov.title or ""
            header = RichText()
            header.append(f"  # {param_id}", style="bold bright_blue")
            if title:
                header.append(f" · {title}", style="bright_blue")
            self._console.print(header)

            # Subtitle: source · params · visible metadata values
            # Values already shown in header (param_id, title) are skipped dynamically.
            header_values = {param_id.lower(), title.lower()} - {""}
            parts: list[str] = []
            source = prov.source or entry.source
            if source:
                parts.append(source)
            if entry.params:
                parts.append(_format_params(entry.params))
            for item in prov.properties.get("metadata", []):
                if not isinstance(item, dict) or item.get("exclude_from_llm_view", False):
                    continue
                value = str(item.get("value", ""))
                if value and value.lower() not in header_values:
                    parts.append(value)
            if parts:
                self._console.print(f"  [dim]{' · '.join(parts)}[/]")

            self._console.print()

            # Table preview from head (same style as show_datasets)
            preview_df = _head_to_dataframe(entry.head)
            if preview_df is not None and not preview_df.empty:
                display_cols = _pick_display_columns(preview_df, max_cols=5)
                has_extra = len(display_cols) < len(preview_df.columns)
                table = Table(
                    show_header=True,
                    header_style="bold",
                    show_lines=False,
                    padding=(0, 1),
                    pad_edge=True,
                )
                for col in display_cols:
                    justify = "right" if pd.api.types.is_numeric_dtype(preview_df[col]) else "left"
                    table.add_column(str(col), justify=justify, max_width=30)
                if has_extra:
                    table.add_column(f"+{len(preview_df.columns) - len(display_cols)} cols", style="dim", max_width=10)
                max_preview_rows = 3
                for _, row in preview_df.head(max_preview_rows).iterrows():
                    cells = []
                    for col in display_cols:
                        val = row[col]
                        if pd.isna(val):
                            cells.append("[dim]--[/]")
                        elif isinstance(val, float):
                            cells.append(f"{val:.2f}")
                        else:
                            s = str(val)
                            cells.append(s[:30] + "..." if len(s) > 30 else s)
                    if has_extra:
                        cells.append("")
                    table.add_row(*cells)
                self._console.print(table)
                remaining = rows - max_preview_rows
                if remaining > 0:
                    self._console.print(f"  [dim]... {remaining:,} more rows[/]")
            self._console.print()

    def show_charts(self, charts: dict[str, Any]) -> None:
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

            # Render and open
            path = _render_chart_to_png(spec)
            if path:
                self._console.print(f"  [dim]→ {path}[/]")
                _open_file(path)
            else:
                self._console.print("  [dim]~ render failed[/]")
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
        print()
        print("--- Datasets " + "-" * 47)
        print()
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
        for name, artifact in datasets.items():
            var = _get_variable(context, name)
            df = getattr(getattr(var, "output", None), "value", None) if var else None
            if not isinstance(df, pd.DataFrame):
                continue
            rows, cols = df.shape
            display_cols = _pick_display_columns(df)
            title = getattr(artifact, "title", "") or name
            print(f"  # {title}")
            desc = getattr(artifact, "description", "")
            if desc:
                print(f"  {desc}")
            tags = getattr(artifact, "tags", [])
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
            param_id = str(next(iter(entry.params.values()), "")) if entry.params else ""
            title = prov.title or ""
            header = f"  # {param_id}"
            if title:
                header += f" · {title}"
            print(header)

            # Subtitle
            header_values = {param_id.lower(), title.lower()} - {""}
            parts = []
            source = prov.source or entry.source
            if source:
                parts.append(source)
            if entry.params:
                parts.append(_format_params(entry.params))
            for item in prov.properties.get("metadata", []):
                if not isinstance(item, dict) or item.get("exclude_from_llm_view", False):
                    continue
                value = str(item.get("value", ""))
                if value and value.lower() not in header_values:
                    parts.append(value)
            if parts:
                print(f"  {' · '.join(parts)}")
            print()

            preview_df = _head_to_dataframe(entry.head)
            if preview_df is not None and not preview_df.empty:
                display_cols = _pick_display_columns(preview_df, max_cols=5)
                preview = preview_df[display_cols].head(3)
                if _tab:
                    print(_tab(preview, headers="keys", tablefmt="simple", showindex=False))
                else:
                    print(preview.to_string(index=False))
                remaining = rows - 3
                if remaining > 0:
                    print(f"  ... {remaining:,} more rows")
            print()

    def show_charts(self, charts: dict[str, Any]) -> None:
        if not charts:
            return
        print()
        print("--- Charts " + "-" * 49)
        print()
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
            path = _render_chart_to_png(spec)
            if path:
                print(f"  → {path}")
                _open_file(path)
            else:
                print("  ~ render failed")
            print()
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
    ) -> None:
        label = "ok" if ok else "!!"
        parts = [f"Completed in {elapsed:.1f}s"]
        if tool_count:
            parts.append(f"{tool_count} tool call{'s' if tool_count != 1 else ''}")
        if dataset_count:
            parts.append(f"{dataset_count} dataset{'s' if dataset_count != 1 else ''}")
        if chart_count:
            parts.append(f"{chart_count} chart{'s' if chart_count != 1 else ''}")
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
) -> AgentResult:
    """Run the agent with live terminal display, returning an :class:`AgentResult`.

    Wraps ``agent.run()`` with polished terminal output: a spinner during
    analysis, numbered tool progress lines, streamed response text, dataset
    tables, and syntax-highlighted code.

    Args:
        agent: The Agent instance.
        message: User message / question.
        ctx: Optional AgentContext for multi-turn continuation.
        console: Optional ``rich.console.Console`` (auto-created if omitted).
        show_code: Display generated code notebooks (default True).
        show_data: Display dataset previews (default True).
        max_table_rows: Max rows per dataset preview (default 5).
        max_code_lines: Max lines per code notebook (default 30).

    Returns:
        AgentResult with ``.text``, ``.datasets``, ``.code``, ``.charts``, ``.context``.
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

    finally:
        # Restore logger levels
        for lg, level in _saved_levels:
            lg.setLevel(level)

    # Ensure spinner is stopped even if no response text was emitted
    display.spinner_stop()

    # Finish response section
    if response_started:
        display.end_response(result.text)

    # Post-response sections
    if show_data:
        fetch_entries = _collect_fetch_entries(result)
        display.show_fetches(fetch_entries)

    if show_code:
        display.show_code(result.code, max_lines=max_code_lines)

    if show_data:
        display.show_datasets(result.datasets, max_rows=max_table_rows, context=result.context)

    display.show_charts(result.charts)

    elapsed = time.monotonic() - start
    display.show_status(
        ok=result.ok,
        elapsed=elapsed,
        tool_count=tool_count,
        dataset_count=len(result.datasets),
        chart_count=len(result.charts),
        notebook_count=len(result.code),
        error_count=error_count,
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
        display.show_code(result.code, max_lines=max_code_lines)

    if show_data:
        display.show_datasets(result.datasets, max_rows=max_table_rows, context=result.context)

    display.show_charts(result.charts)

    display.show_status(
        ok=result.ok,
        elapsed=0.0,
        tool_count=sum(1 for e in result.events if getattr(e, "type", None) == "tool_event" and getattr(e, "completed", False)),
        dataset_count=len(result.datasets),
        chart_count=len(result.charts),
        notebook_count=len(result.code),
        error_count=sum(1 for e in result.events if getattr(e, "type", None) == "error"),
    )
