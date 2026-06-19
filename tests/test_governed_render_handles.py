"""Governed single render + honest partial-view cue.

Regression cover for the result-rendering rebuild: one governed render for
tabular outputs (honest header, exclude_from_llm_view enforced on the paginated
path) and an honest partial-view cue that points a coding agent back at the
value — slice it or search it with the core catalog — rather than at a
content-addressed retrieval tool (those were removed).
"""

from __future__ import annotations

import tempfile

import pandas as pd
from parsimony.result import Column, ColumnRole

from parsimony_agents.execution.dataframe_ref import DataframeRef, set_default_local_root
from parsimony_agents.execution.outputs import DataFrameObject, PrimitiveObject


def _text(blocks) -> str:
    return "\n".join(b["text"] for b in blocks)


def _big_codelist_object() -> DataFrameObject:
    d = tempfile.mkdtemp()
    set_default_local_root(d)
    df = pd.DataFrame(
        {
            "code": [f"c{i}" for i in range(932)],
            "label": [f"L{i}" for i in range(932)],
            "url": ["x"] * 932,
        }
    )
    cols = [
        Column(name="code", role=ColumnRole.KEY, namespace="ESTAT:COICOP18"),
        Column(name="label", role=ColumnRole.DATA),
        Column(name="url", role=ColumnRole.METADATA, exclude_from_llm_view=True),
    ]
    return DataFrameObject(ref=DataframeRef.from_pandas(df, ref="anonymous", local_dir=d), columns=cols)


def test_tabular_render_is_honest_and_governed() -> None:
    text = _text(_big_codelist_object().to_llm())
    # Honest header: the real counts, including how many columns are LLM-hidden.
    assert "932 rows × 3 columns (1 hidden from LLM view)" in text
    # Governance applied on the paginated path: the hidden column never appears.
    assert "url" not in text
    # Schema block carries role + namespace.
    assert "- code: object (KEY ns:ESTAT:COICOP18)" in text
    assert "- label: object (DATA)" in text


def test_tabular_partial_view_cue_points_at_the_value_not_a_tool() -> None:
    text = _text(_big_codelist_object().to_llm())
    assert "This is a partial view" in text
    # Codemode: slice or search the variable; the core catalog is the search path.
    assert "df.iloc[start:stop]" in text
    assert "auto_catalog(df).search(" in text
    # The removed retrieval apparatus is gone — no handle, no tools advertised.
    assert "output_search" not in text
    assert "output_read" not in text


def test_render_pagination_reaches_buried_rows() -> None:
    # The aggregates a decomposition needs sit deep in a large codelist; a
    # specific page must be reachable via the render's display_pages (the
    # codelist-blindness fix), independent of any retrieval tool.
    obj = _big_codelist_object()
    deep = _text(obj.to_llm(overrides={"display_pages": [89]}))
    assert "c890" in deep and "c899" in deep


def test_small_tabular_shows_all_rows_no_cue() -> None:
    d = tempfile.mkdtemp()
    set_default_local_root(d)
    df = pd.DataFrame({"a": [1, 2, 3]})
    obj = DataFrameObject(ref=DataframeRef.from_pandas(df, ref="anonymous", local_dir=d))
    text = _text(obj.to_llm())
    assert "3 rows × 1 columns" in text
    # Everything fits one page -> no partial-view cue.
    assert "This is a partial view" not in text


def test_single_page_frame_not_rendered_twice() -> None:
    # Default display_pages [0, 1, -2, -1] collapse to page 0 on a single-page
    # frame; the row content must appear once, not duplicated.
    d = tempfile.mkdtemp()
    set_default_local_root(d)
    df = pd.DataFrame({"v": ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]})
    obj = DataFrameObject(ref=DataframeRef.from_pandas(df, ref="anonymous", local_dir=d))
    text = _text(obj.to_llm())
    # The single page is emitted exactly once (no [0, 1, -2, -1] re-render).
    assert text.count("Page 1/1") == 1
    assert "This is a partial view" not in text


def test_large_primitive_partial_view_cue_points_at_the_value() -> None:
    obj = PrimitiveObject(value="word " * 4000)
    text = _text(obj.to_llm())
    assert "chars" in text
    assert "This is a partial view" in text
    assert "text[start:stop]" in text
    assert "grep it with Python" in text
    assert "output_read" not in text
