"""Governed single render + content-addressed handle retrieval.

Regression cover for the result-rendering rebuild: one governed render for
tabular outputs (honest header, exclude_from_llm_view enforced on the paginated
path, retrieval cue) and a server-side handle map that survives a dry_run cell.
"""

from __future__ import annotations

import tempfile

import pandas as pd
from parsimony.result import Column, ColumnRole

from parsimony_agents.agent.agent import _OUTPUT_HANDLE_LIMIT, Agent
from parsimony_agents.execution.dataframe_ref import DataframeRef, set_default_local_root
from parsimony_agents.execution.outputs import DataFrameObject, KernelOutput, PrimitiveObject


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


def test_tabular_render_emits_handle_retrieval_cue() -> None:
    obj = _big_codelist_object()
    text = _text(obj.to_llm())
    assert obj.handle in text
    assert f"output_search(variable_name='{obj.handle}'" in text
    assert f"output_read(variable_name='{obj.handle}'" in text


def test_output_read_pages_reach_buried_rows() -> None:
    # The aggregates a decomposition needs sit deep in a large codelist; a
    # specific page must be reachable (this is the codelist-blindness fix).
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
    # Everything fits one page -> no retrieval cue.
    assert "output_search(variable_name=" not in text


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
    assert "output_search(variable_name=" not in text


def test_large_primitive_has_handle_and_cue() -> None:
    obj = PrimitiveObject(value="word " * 4000)
    text = _text(obj.to_llm())
    assert "chars" in text
    assert obj.handle in text
    assert "output_read(variable_name=" in text


def test_register_outputs_indexes_handles_and_is_bounded() -> None:
    agent = Agent(model="test-model")
    obj = _big_codelist_object()
    agent._register_outputs(KernelOutput(outputs=[obj]))
    # The handle the agent advertised in the cue resolves back to the object.
    assert agent._output_handles.get(obj.handle) is obj

    # Bound holds: flooding past the cap evicts oldest, keeps the newest.
    d = tempfile.mkdtemp()
    set_default_local_root(d)
    last = None
    for i in range(_OUTPUT_HANDLE_LIMIT + 50):
        last = PrimitiveObject(value=f"payload-{i}-" + "z" * 2000)
        agent._register_outputs(KernelOutput(outputs=[last]))
    assert len(agent._output_handles) <= _OUTPUT_HANDLE_LIMIT
    assert agent._output_handles.get(last.handle) is last
