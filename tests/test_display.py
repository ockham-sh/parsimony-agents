"""Tests for terminal display helpers (fetch log previews)."""

from __future__ import annotations

from parsimony.result import ColumnRole, Provenance

from types import SimpleNamespace

from parsimony_agents.agent.agent import AgentResult
from parsimony_agents.agent.events import Handoff, PartialRunSummary
from parsimony_agents.artifacts import Report
from parsimony_agents.display import (
    _PlainDisplay,
    _RichDisplay,
    _chart_summary,
    _format_handoff,
    _format_partial_summary,
    _pick_display_columns,
    _title_from_preview,
)
from parsimony_agents.execution.outputs import FetchLogEntry


def _fred_fetch_entry() -> FetchLogEntry:
    return FetchLogEntry(
        provenance=Provenance(
            source="fred_fetch",
            source_description="St. Louis Fed FRED",
            params={"series_id": "UNRATE"},
        ),
        row_count=3,
        column_names=[
            "series_id",
            "title",
            "units_short",
            "date",
            "value",
        ],
        columns=[
            {"name": "series_id", "role": ColumnRole.KEY},
            {"name": "title", "role": ColumnRole.TITLE},
            {"name": "units_short", "role": ColumnRole.METADATA},
            {"name": "date", "role": ColumnRole.DATA},
            {"name": "value", "role": ColumnRole.DATA},
        ],
        head={
            "schema": {"fields": []},
            "data": [
                {
                    "series_id": "UNRATE",
                    "title": "Unemployment Rate",
                    "units_short": "Percent",
                    "date": "2024-01-01",
                    "value": 3.7,
                }
            ],
        },
    )


def test_pick_display_columns_hides_key_and_metadata_roles() -> None:
    import pandas as pd

    entry = _fred_fetch_entry()
    df = pd.DataFrame(entry.head["data"])
    cols = _pick_display_columns(df, column_schema=entry.columns)
    assert cols == ["date", "value"]


def test_title_from_preview_uses_title_role_column() -> None:
    import pandas as pd

    entry = _fred_fetch_entry()
    df = pd.DataFrame(entry.head["data"])
    assert _title_from_preview(df, entry.columns) == "Unemployment Rate"


def test_format_handoff_includes_rationale_blockers_and_steps() -> None:
    body = _format_handoff(
        Handoff(
            rationale="The AI provider rejected the request (AuthenticationError): no api key",
            blockers=["ANTHROPIC_API_KEY is not set"],
            suggested_next_steps=["Set ANTHROPIC_API_KEY and re-run"],
        )
    )
    assert "AuthenticationError" in body
    assert "ANTHROPIC_API_KEY is not set" in body
    assert "Set ANTHROPIC_API_KEY and re-run" in body


def test_format_handoff_falls_back_when_rationale_empty() -> None:
    body = _format_handoff(Handoff(rationale=""))
    assert "could not complete" in body


def test_format_partial_summary_lists_missing() -> None:
    body = _format_partial_summary(
        PartialRunSummary(missing=["unemployment series"], next_step_plan="Fetch UNRATE")
    )
    assert "Fetch UNRATE" in body
    assert "unemployment series" in body


# ---------------------------------------------------------------------------
# Fixtures for the text / chart / report render paths
# ---------------------------------------------------------------------------


def _recording_console():
    """A Rich Console that captures output as a terminal (so Live renders)."""
    from rich.console import Console

    return Console(record=True, width=100, force_terminal=True, color_system=None)


def _make_report() -> Report:
    return Report(
        logical_id="rpt1",
        title="Quarterly Review",
        subtitle="FY24",
        description="Exec summary",
        tags=["finance"],
        markdown="## Summary\n\nRevenue grew **21%** in EMEA.\n\n- Q1 to Q2: +20%\n",
        formats=["html", "pdf"],
    )


def _bar_spec() -> dict:
    return {
        "mark": "bar",
        "encoding": {
            "x": {"field": "quarter", "type": "nominal"},
            "y": {"field": "revenue", "type": "quantitative"},
            "color": {"field": "region", "type": "nominal"},
        },
        "data": {
            "values": [
                {"quarter": "Q1", "revenue": 120.0, "region": "EMEA"},
                {"quarter": "Q2", "revenue": 145.0, "region": "EMEA"},
                {"quarter": "Q3", "revenue": 130.0, "region": "EMEA"},
                {"quarter": "Q4", "revenue": 160.0, "region": "EMEA"},
            ]
        },
    }


def _fake_chart(spec: dict):
    """A duck-typed stand-in for a Chart: show_charts only reads via getattr."""
    return SimpleNamespace(
        figure=SimpleNamespace(value=spec),
        title="Quarterly Revenue",
        description="",
        notes=[],
    )


# ---------------------------------------------------------------------------
# Report collection
# ---------------------------------------------------------------------------


def test_collect_captures_report() -> None:
    result = AgentResult()
    report = _make_report()
    result._collect(SimpleNamespace(type="tool_event", completed=True, result=report))
    assert result.reports == {"rpt1": report}


def test_collect_ignores_report_without_logical_id() -> None:
    result = AgentResult()
    report = Report(logical_id="", title="x", markdown="body")
    result._collect(SimpleNamespace(type="tool_event", completed=True, result=report))
    assert result.reports == {}


# ---------------------------------------------------------------------------
# Chart spec summary
# ---------------------------------------------------------------------------


def test_chart_summary_string_mark_and_inline_data() -> None:
    mark, encodings, df = _chart_summary(_bar_spec())
    assert mark == "bar"
    assert "x=quarter" in encodings
    assert "y=revenue" in encodings
    assert "color=region" in encodings
    assert df is not None and len(df) == 4


def test_chart_summary_dict_mark_and_url_data_has_no_preview() -> None:
    spec = {
        "mark": {"type": "line"},
        "encoding": {"x": {"field": "t"}, "y": {"aggregate": "mean"}},
        "data": {"url": "http://example.com/data.json"},
    }
    mark, encodings, df = _chart_summary(spec)
    assert mark == "line"
    assert "x=t" in encodings
    assert "y=mean" in encodings  # falls back to aggregate when no field
    assert df is None


# ---------------------------------------------------------------------------
# show_charts (Rich): textual fallback + opt-in open
# ---------------------------------------------------------------------------


def test_show_charts_shows_summary_when_image_unavailable(monkeypatch) -> None:
    import parsimony_agents.display as disp

    monkeypatch.setattr(disp, "_render_chart_to_png", lambda spec: None)
    opened: list[str] = []
    monkeypatch.setattr(disp, "_open_file", lambda p: opened.append(p))

    console = _recording_console()
    disp._RichDisplay(console=console).show_charts({"c": _fake_chart(_bar_spec())})
    out = console.export_text()

    assert "Quarterly Revenue" in out
    assert "bar" in out
    assert "x=quarter" in out
    assert "image render unavailable" in out
    assert opened == []  # never opens when render failed


def test_show_charts_saves_path_and_open_is_opt_in(monkeypatch) -> None:
    import parsimony_agents.display as disp

    monkeypatch.setattr(disp, "_render_chart_to_png", lambda spec: "/tmp/fake_chart.png")
    opened: list[str] = []
    monkeypatch.setattr(disp, "_open_file", lambda p: opened.append(p))

    # Default: save path printed, viewer NOT opened.
    console = _recording_console()
    disp._RichDisplay(console=console).show_charts({"c": _fake_chart(_bar_spec())})
    assert "saved:" in console.export_text()
    assert opened == []

    # Opt-in: viewer opened.
    disp._RichDisplay(console=_recording_console()).show_charts(
        {"c": _fake_chart(_bar_spec())}, open_charts=True
    )
    assert opened == ["/tmp/fake_chart.png"]


# ---------------------------------------------------------------------------
# show_reports
# ---------------------------------------------------------------------------


def test_show_reports_rich_renders_title_formats_and_body() -> None:
    console = _recording_console()
    _RichDisplay(console=console).show_reports({"rpt1": _make_report()})
    out = console.export_text()
    assert "Quarterly Review" in out
    assert "html" in out  # a requested format
    assert "Summary" in out  # body heading
    assert "Revenue grew" in out


def test_show_reports_rich_noop_on_empty() -> None:
    console = _recording_console()
    _RichDisplay(console=console).show_reports({})
    assert console.export_text().strip() == ""


def test_show_reports_rich_truncates_long_body() -> None:
    body = "\n".join(f"line {i}" for i in range(60))
    report = Report(logical_id="r", title="Long", markdown=body, formats=["html"])
    console = _recording_console()
    _RichDisplay(console=console).show_reports({"r": report}, max_lines=10)
    assert "more lines" in console.export_text()


def test_show_reports_plain_renders_body() -> None:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _PlainDisplay().show_reports({"rpt1": _make_report()})
    out = buf.getvalue()
    assert "Quarterly Review" in out
    assert "Revenue grew" in out
    assert "formats: html, pdf" in out


# ---------------------------------------------------------------------------
# Text rendering: Markdown + bracket safety
# ---------------------------------------------------------------------------


def test_stream_text_preserves_brackets_and_renders_markdown() -> None:
    console = _recording_console()
    d = _RichDisplay(console=console)
    text = "See [docs](http://x) and note [1].\n\n## Heading text\n"
    d.start_response()
    d.stream_text(text)
    d.end_response(text)
    out = console.export_text()
    # No crash, and bracketed content + markdown heading survive (not eaten as markup).
    assert "docs" in out
    assert "Heading text" in out
    assert "1" in out


def test_show_charts_skips_table_when_all_columns_hidden(monkeypatch) -> None:
    import parsimony_agents.display as disp

    monkeypatch.setattr(disp, "_render_chart_to_png", lambda spec: None)
    # Only column is "index", which _pick_display_columns hides → no preview table.
    spec = {"mark": "point", "encoding": {"x": {"field": "index"}}, "data": {"values": [{"index": 1}]}}
    console = _recording_console()
    disp._RichDisplay(console=console).show_charts({"c": _fake_chart(spec)})
    out = console.export_text()
    assert "point" in out  # summary still shown, no crash


def test_stream_to_display_tears_down_live_when_run_raises(monkeypatch) -> None:
    """A mid-stream exception must still stop the Live region (end_response)."""
    import asyncio

    import pytest

    import parsimony_agents.display as disp

    class _SpyDisplay:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def banner(self, q):
            self.calls.append("banner")

        def spinner_update(self, label):
            pass

        def spinner_stop(self):
            self.calls.append("spinner_stop")

        def tool_completed(self, *a, **k):
            pass

        def start_response(self):
            self.calls.append("start_response")

        def stream_text(self, chunk):
            self.calls.append("stream_text")

        def end_response(self, full_text):
            self.calls.append("end_response")

        def show_error(self, *a, **k):
            pass

        def show_datasets(self, *a, **k):
            self.calls.append("show_datasets")

        def show_code(self, *a, **k):
            pass

        def show_charts(self, *a, **k):
            pass

        def show_reports(self, *a, **k):
            pass

        def show_fetches(self, *a, **k):
            pass

        def show_status(self, *a, **k):
            self.calls.append("show_status")

    class _BoomAgent:
        def run(self, message, ctx=None):
            async def gen():
                yield SimpleNamespace(type="text_delta", delta=True, content="partial")
                raise RuntimeError("boom")

            return gen()

    spy = _SpyDisplay()
    monkeypatch.setattr(disp, "_make_backend", lambda console=None: spy)

    with pytest.raises(RuntimeError):
        asyncio.run(disp.stream_to_display(_BoomAgent(), "q"))

    # Live was torn down even though run() raised, and post-response sections
    # (which would render on top of a half-open Live) were skipped.
    assert "end_response" in spy.calls
    assert "show_status" not in spy.calls
    assert "show_datasets" not in spy.calls
