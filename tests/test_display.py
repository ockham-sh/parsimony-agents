"""Tests for terminal display helpers (fetch log previews)."""

from __future__ import annotations

from parsimony.result import ColumnRole, Provenance

from parsimony_agents.agent.events import Handoff, PartialRunSummary
from parsimony_agents.display import (
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
