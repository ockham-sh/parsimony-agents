"""Tests for :mod:`parsimony_agents._naming`."""

from __future__ import annotations

import pytest

from parsimony_agents._naming import short_sha, slug_from_title


def test_slug_from_title_ascii() -> None:
    assert slug_from_title("  US CPI Headline  ") == "us_cpi_headline"


def test_slug_from_title_punctuation() -> None:
    assert slug_from_title("CPI, YoY% (all items)") == "cpi_yoy_all_items"


def test_slug_from_title_unicode_folded() -> None:
    assert slug_from_title("café / naïve") == "cafe_naive"


def test_slug_from_title_max_len() -> None:
    long_ = "a" * 100
    s = slug_from_title(long_, max_len=10)
    assert len(s) == 10
    assert s == "aaaaaaaaaa"


def test_slug_from_title_empty() -> None:
    assert slug_from_title("") == "untitled"
    assert slug_from_title("   ") == "untitled"
    assert slug_from_title("!!!") == "untitled"


def test_short_sha_prefix() -> None:
    h = "a" * 64
    assert short_sha(h) == "a" * 12
    assert short_sha(h, n=8) == "a" * 8


def test_short_sha_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least 12"):
        short_sha("ab")
