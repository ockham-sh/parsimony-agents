"""Tests for the report snapshot YAML frontmatter parser/composer.

The snapshot bytes are a valid Quarto ``.qmd`` document with a
``parsimony:`` block carrying ``title``, optional ``subtitle``,
``formats``, and ``pins``. These tests pin the round-trip contract:
the output of ``compose_snapshot`` round-trips through
``parse_snapshot`` and recomposes byte-identically.
"""

from __future__ import annotations

import pytest

from parsimony_agents.identity import ArtifactRef
from parsimony_agents.report_format import (
    DEFAULT_FORMATS,
    ParsedSnapshot,
    compose_snapshot,
    parse_snapshot,
)


def _sha(byte: str) -> str:
    """Generate a fake 64-char content_sha for test fixtures."""
    return byte * 64


_TREND_REF = ArtifactRef(kind="chart", logical_id="trend-lid", content_sha=_sha("a"))
_SALES_REF = ArtifactRef(kind="dataset", logical_id="sales-lid", content_sha=_sha("b"))


def test_parse_explicit_fields_and_pins() -> None:
    text = (
        "---\n"
        "parsimony:\n"
        "  title: My Report\n"
        "  subtitle: A useful subtitle\n"
        "  formats:\n"
        "  - html\n"
        "  - pdf\n"
        "  pins:\n"
        f"    trend:\n"
        f"      kind: chart\n"
        f"      logical_id: trend-lid\n"
        f"      content_sha: {_sha('a')}\n"
        "---\n\n"
        "Body content.\n"
    )
    snap = parse_snapshot(text)
    assert isinstance(snap, ParsedSnapshot)
    assert snap.title == "My Report"
    assert snap.subtitle == "A useful subtitle"
    assert snap.formats == ["html", "pdf"]
    assert snap.pins == {"trend": _TREND_REF}
    assert snap.body == "Body content.\n"


def test_compose_round_trip_with_pins_and_subtitle() -> None:
    text = compose_snapshot(
        ["html", "pdf"],
        {"trend": _TREND_REF, "sales": _SALES_REF},
        "Body.\n",
        title="My Report",
        subtitle="A note about the quarter",
    )
    snap = parse_snapshot(text)
    assert snap.title == "My Report"
    assert snap.subtitle == "A note about the quarter"
    assert snap.formats == ["html", "pdf"]
    assert snap.pins == {"trend": _TREND_REF, "sales": _SALES_REF}
    assert snap.body == "Body.\n"


def test_compose_round_trip_empty_pins_and_no_subtitle() -> None:
    """Reports without subtitle or embedded artifacts still serialize cleanly."""
    text = compose_snapshot(["html"], {}, "Body.\n", title="Plain")
    snap = parse_snapshot(text)
    assert snap.title == "Plain"
    assert snap.subtitle == ""
    assert snap.formats == ["html"]
    assert snap.pins == {}
    assert snap.body == "Body.\n"


def test_compose_omits_subtitle_key_when_empty() -> None:
    """Byte-stability across the subtitle introduction: a report with no
    subtitle must not carry a ``subtitle: ''`` line in YAML."""
    text = compose_snapshot(["html"], {}, "Body.\n", title="Plain")
    assert "subtitle" not in text


def test_compose_emits_byte_identical_output_for_same_inputs() -> None:
    """``content_sha`` stability depends on deterministic compose. Pin map keys
    are sorted; format list preserves caller order."""
    pins = {"sales": _SALES_REF, "trend": _TREND_REF}
    a = compose_snapshot(["html", "pdf"], pins, "T\n", title="Same")
    b = compose_snapshot(["html", "pdf"], pins, "T\n", title="Same")
    assert a == b


def test_compose_rejects_empty_title() -> None:
    """Title is mandatory — empty or whitespace-only fails fast."""
    with pytest.raises(ValueError, match="title"):
        compose_snapshot(["html"], {}, "Body.\n", title="")
    with pytest.raises(ValueError, match="title"):
        compose_snapshot(["html"], {}, "Body.\n", title="   ")


def test_parse_missing_frontmatter_raises() -> None:
    """No frontmatter means the bytes weren't produced by ``compose_snapshot``."""
    with pytest.raises(ValueError, match="frontmatter"):
        parse_snapshot("Body.\n")


def test_parse_unterminated_frontmatter_raises() -> None:
    with pytest.raises(ValueError, match="closing"):
        parse_snapshot("---\nparsimony:\n  title: T\n  formats: [html]\nBody\n")


def test_parse_missing_parsimony_block_raises() -> None:
    with pytest.raises(ValueError, match="parsimony"):
        parse_snapshot("---\ntitle: Hi\n---\n\nBody.\n")


def test_parse_missing_title_raises() -> None:
    """Title is mandatory on parse just as on compose."""
    text = "---\nparsimony:\n  formats: [html]\n  pins: {}\n---\n\nBody\n"
    with pytest.raises(ValueError, match="title"):
        parse_snapshot(text)


def test_parse_empty_formats_falls_back_to_default() -> None:
    text = compose_snapshot([], {}, "T\n", title="T")
    snap = parse_snapshot(text)
    assert snap.formats == list(DEFAULT_FORMATS)


@pytest.mark.parametrize(
    "formats",
    [["html"], ["html", "pdf"], ["pptx", "dashboard"], ["revealjs"]],
)
def test_round_trip_arbitrary_format_lists(formats: list[str]) -> None:
    body = "Body content.\n"
    text = compose_snapshot(
        formats, {"sales": _SALES_REF}, body, title="Report"
    )
    snap = parse_snapshot(text)
    assert snap.formats == formats
    assert snap.pins == {"sales": _SALES_REF}
    assert snap.body == body
