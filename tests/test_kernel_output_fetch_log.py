"""KernelOutput._fetch_log_to_llm honors the ``mode`` argument.

In ``default`` mode the fetch_log renders one ``<entry/>`` per fetch
with source, params, row count, and version. In ``minimal`` mode the
per-entry detail is dropped and only a single-line summary remains, so
prior-turn tool results stay short and byte-stable inside the cached
prefix.
"""

from __future__ import annotations

from parsimony.result import Provenance

from parsimony_agents.execution.outputs import FetchLogEntry, KernelOutput


def _entry(source: str = "fred", rows: int = 100) -> FetchLogEntry:
    return FetchLogEntry(
        provenance=Provenance(source=source, source_description=source, params={"series": "GDP"}),
        row_count=rows,
        column_names=["date", "value"],
        columns=[{"name": "date"}, {"name": "value"}],
    )


def test_fetch_log_default_mode_renders_full_entries():
    output = KernelOutput(outputs=[], fetch_log=[_entry(), _entry(source="bls", rows=200)])
    rendered = output._fetch_log_to_llm(mode="default")
    assert "<fetch_log>" in rendered
    assert "</fetch_log>" in rendered
    assert 'source="fred"' in rendered
    assert 'source="bls"' in rendered
    assert 'rows="100"' in rendered
    assert 'rows="200"' in rendered


def test_fetch_log_minimal_mode_collapses_to_summary():
    output = KernelOutput(outputs=[], fetch_log=[_entry(), _entry(), _entry()])
    rendered = output._fetch_log_to_llm(mode="minimal")
    assert rendered == '<fetch_log entries="3"/>\n'
    assert "source=" not in rendered
    assert "rows=" not in rendered


def test_fetch_log_minimal_is_byte_stable_for_same_entries():
    """Two identical fetch_logs produce byte-equal minimal renderings."""
    a = KernelOutput(outputs=[], fetch_log=[_entry(), _entry(source="bls")])
    b = KernelOutput(outputs=[], fetch_log=[_entry(), _entry(source="bls")])
    assert a._fetch_log_to_llm(mode="minimal") == b._fetch_log_to_llm(mode="minimal")


def test_fetch_log_empty_returns_empty_in_both_modes():
    output = KernelOutput(outputs=[], fetch_log=[])
    assert output._fetch_log_to_llm(mode="default") == ""
    assert output._fetch_log_to_llm(mode="minimal") == ""


def test_kernel_output_to_llm_passes_mode_to_fetch_log():
    """End-to-end: to_llm(mode='minimal') propagates to the fetch_log block."""
    output = KernelOutput(outputs=[], fetch_log=[_entry()])
    blocks = output.to_llm(mode="minimal")
    fetch_text = next((b["text"] for b in blocks if "fetch_log" in b.get("text", "")), None)
    assert fetch_text == '<fetch_log entries="1"/>\n'

    blocks_default = output.to_llm(mode="default")
    fetch_text_default = next((b["text"] for b in blocks_default if "fetch_log" in b.get("text", "")), None)
    assert "<entry " in fetch_text_default
