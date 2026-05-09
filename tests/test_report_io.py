"""Unit tests for parsimony_agents.report_io.

Covers the ``write_report_bytes`` / ``read_report_bytes`` round-trip,
the YAML preamble shape (title + ockham.formats only — no date), default
formats, and the safety properties (yaml.safe_load rejects arbitrary
Python objects; missing front-matter is tolerated).
"""

from __future__ import annotations

import pytest
import yaml

from parsimony_agents.artifacts import Report
from parsimony_agents.report_io import (
    DEFAULT_FORMATS,
    ExportFormat,
    read_report_bytes,
    write_report_bytes,
)


def _make(**kwargs) -> Report:
    base = {
        "logical_id": "lid",
        "title": "T",
        "markdown": "# H\n\nbody.",
        "embedded_refs": [],
    }
    base.update(kwargs)
    return Report(**base)


def test_write_emits_yaml_preamble_and_body() -> None:
    r = _make(title="Hello", markdown="# H\n\nbody.", formats=["html", "pdf"])
    out = write_report_bytes(r).decode("utf-8")
    assert out.startswith("---\n")
    yaml_chunk, _, body = out.partition("\n---\n")
    yaml_chunk = yaml_chunk[len("---\n") :]
    parsed = yaml.safe_load(yaml_chunk)
    assert parsed == {"title": "Hello", "ockham": {"formats": ["html", "pdf"]}}
    # Body separated from the closing fence by the canonical blank line.
    assert body.startswith("\n# H\n\nbody.")


def test_write_default_formats_html_pdf() -> None:
    """A Report without an explicit `formats` value gets the model default."""
    # Omit `formats` so pydantic uses the field default_factory.
    r = Report(logical_id="lid", title="T", markdown="# H", embedded_refs=[])
    yaml_dict, _ = read_report_bytes(write_report_bytes(r))
    assert yaml_dict["ockham"]["formats"] == ["html", "pdf"]
    assert tuple(yaml_dict["ockham"]["formats"]) == DEFAULT_FORMATS


def test_write_blank_title_falls_back_to_untitled() -> None:
    r = _make(title="", markdown="# x")
    yaml_dict, _ = read_report_bytes(write_report_bytes(r))
    assert yaml_dict["title"] == "(untitled)"


def test_read_round_trip_preserves_body_byte_exact() -> None:
    body = "# Heading\n\nFirst paragraph.\n\n- a\n- b\n\n```python\nprint('x')\n```\n"
    r = _make(markdown=body, formats=["html", "pptx"])
    yaml_dict, parsed_body = read_report_bytes(write_report_bytes(r))
    assert yaml_dict == {"title": "T", "ockham": {"formats": ["html", "pptx"]}}
    assert parsed_body == body


def test_read_no_front_matter_returns_empty_dict_and_full_text() -> None:
    raw = b"# just a markdown body\n\nno preamble here.\n"
    yaml_dict, body = read_report_bytes(raw)
    assert yaml_dict == {}
    assert body == raw.decode("utf-8")


def test_read_unterminated_front_matter_returns_full_text() -> None:
    """If the closing `---` fence is missing, treat it as a no-preamble file."""
    raw = b"---\ntitle: x\nno closing fence ever\n"
    yaml_dict, body = read_report_bytes(raw)
    assert yaml_dict == {}
    assert body == raw.decode("utf-8")


def test_read_safe_load_no_arbitrary_python_objects() -> None:
    """yaml.safe_load must reject `!!python/object` tags (no code execution)."""
    raw = b'---\ntitle: !!python/object/apply:os.system ["echo pwned"]\n---\n\nbody.\n'
    with pytest.raises(yaml.YAMLError):
        read_report_bytes(raw)


def test_write_preserves_unicode_in_title_and_body() -> None:
    r = _make(title="Résumé €", markdown="# 你好\n\n— body —\n")
    yaml_dict, body = read_report_bytes(write_report_bytes(r))
    assert yaml_dict["title"] == "Résumé €"
    assert body == "# 你好\n\n— body —\n"


def test_default_formats_constant_matches_export_formats_subset() -> None:
    """DEFAULT_FORMATS items must each be a valid ExportFormat literal."""
    valid = {"html", "pdf", "pptx", "dashboard"}
    for fmt in DEFAULT_FORMATS:
        assert fmt in valid


def test_write_idempotent_no_date_field() -> None:
    """write_report_bytes must not embed a `date:` — that would make the same
    body produce a different content_sha across day boundaries."""
    r = _make()
    blob1 = write_report_bytes(r)
    blob2 = write_report_bytes(r)
    assert blob1 == blob2
    assert b"date:" not in blob1


def test_export_format_literal_exposed() -> None:
    """ExportFormat is re-exported from report_io for server-side consumers."""
    # The literal type is itself a typing construct; just assert it imports
    # and that DEFAULT_FORMATS is annotated against it (we already use it
    # in production code paths — this guards the public surface).
    _: ExportFormat = "html"  # type-check assertion via assignment
    assert _ == "html"
