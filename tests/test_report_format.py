"""Tests for the report snapshot ``formats:`` line parser."""

from __future__ import annotations

import pytest

from parsimony_agents.report_format import (
    DEFAULT_FORMATS,
    compose_snapshot,
    parse_snapshot,
)


def test_parse_explicit_formats_line() -> None:
    formats, body = parse_snapshot("formats: html,pdf\n\n# Title\n\nBody.\n")
    assert formats == ["html", "pdf"]
    assert body == "# Title\n\nBody.\n"


def test_parse_single_format() -> None:
    formats, body = parse_snapshot("formats: pptx\n\n# T\n")
    assert formats == ["pptx"]
    assert body == "# T\n"


def test_parse_missing_formats_defaults_to_html() -> None:
    formats, body = parse_snapshot("# Title\n\nBody.\n")
    assert formats == list(DEFAULT_FORMATS)
    assert body == "# Title\n\nBody.\n"


def test_compose_round_trip() -> None:
    text = compose_snapshot(["html", "pdf"], "# Title\n")
    formats, body = parse_snapshot(text)
    assert formats == ["html", "pdf"]
    assert body == "# Title\n"


def test_parse_dedupes_and_strips() -> None:
    formats, _body = parse_snapshot("formats: html, pdf , html ,pptx\n\nx")
    assert formats == ["html", "pdf", "pptx"]


def test_parse_empty_formats_falls_back_to_default() -> None:
    formats, _body = parse_snapshot("formats:\n\nx")
    assert formats == list(DEFAULT_FORMATS)


@pytest.mark.parametrize(
    "formats",
    [["html"], ["html", "pdf"], ["pptx", "dashboard"], ["revealjs"]],
)
def test_round_trip_arbitrary_format_lists(formats: list[str]) -> None:
    body = "# Hi\n\nText.\n"
    text = compose_snapshot(formats, body)
    out_formats, out_body = parse_snapshot(text)
    assert out_formats == formats
    assert out_body == body
