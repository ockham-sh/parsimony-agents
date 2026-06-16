"""Regression tests for :class:`DataframeRef` handling of nested columns.

Connector results sometimes carry structured metadata in a column — the
canonical case is an SDMX ``dsd`` column whose cells are lists of dimension
descriptors (``[{"dimension_id": ..., "name": ...}, ...]``). Such list/dict
cells are not hashable by ``hash_pandas_object`` and are not reliably
Arrow-serializable, which used to abort ``from_pandas`` at the content-hash
step. The display path caught the raise and fell back to a plain-text dump of
the whole frame, so the agent lost the table view of every search result.

These tests pin the tolerant behaviour: nested columns are stringified for the
content hash and the parquet write, the rest of the frame survives as a normal
table, the caller's live frame is never mutated, and the round-trip is stable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parsimony_agents.execution.dataframe_ref import DataframeRef


def _search_like_frame() -> pd.DataFrame:
    """A frame shaped like an ``sdmx_datasets_search`` result: scalar columns
    plus a nested ``dsd`` column of dimension descriptors."""
    return pd.DataFrame(
        {
            "flow_id": ["ECB/IRS", "ECB/YC"],
            "dataset_id": ["IRS", "YC"],
            "title": ["Interest Rate Statistics", "Yield Curve"],
            "dsd": [
                [
                    {"dimension_id": "FREQ", "name": "Frequency"},
                    {"dimension_id": "REF_AREA", "name": "Reference area"},
                ],
                [{"dimension_id": "FREQ", "name": "Frequency"}],
            ],
        }
    )


def test_from_pandas_persists_frame_with_nested_column(tmp_path: Path) -> None:
    """A frame with a nested ``dsd`` column persists and re-materialises as a table."""
    df = _search_like_frame()

    ref = DataframeRef.from_pandas(df, ref="search", local_dir=tmp_path)

    assert ref.content_hash  # hashing succeeded instead of raising
    restored = ref.materialize_sync()
    # Scalar columns survive structurally; the nested column is now a string.
    assert list(restored["flow_id"]) == ["ECB/IRS", "ECB/YC"]
    assert list(restored["dataset_id"]) == ["IRS", "YC"]
    assert restored["dsd"].map(lambda v: isinstance(v, str)).all()
    # The stringified cell is still a faithful, parseable rendering.
    import json

    first = json.loads(restored["dsd"].iloc[0])
    assert {"dimension_id": "FREQ", "name": "Frequency"} in first


def test_from_pandas_does_not_mutate_caller_frame(tmp_path: Path) -> None:
    """Stringifying happens on a copy — the caller's live frame is untouched."""
    df = _search_like_frame()
    original_first_cell = df["dsd"].iloc[0]

    DataframeRef.from_pandas(df, ref="search", local_dir=tmp_path)

    # The live frame the agent keeps manipulating still holds the real list.
    assert df["dsd"].iloc[0] is original_first_cell
    assert isinstance(df["dsd"].iloc[0], list)
    assert df["dsd"].iloc[0][0]["dimension_id"] == "FREQ"


def test_from_pandas_hash_is_stable_for_nested_column(tmp_path: Path) -> None:
    """Two equal nested frames hash identically (deterministic stringification)."""
    a = DataframeRef.from_pandas(_search_like_frame(), ref="a", local_dir=tmp_path)
    b = DataframeRef.from_pandas(_search_like_frame(), ref="b", local_dir=tmp_path)
    assert a.content_hash == b.content_hash


def test_from_pandas_plain_frame_unaffected(tmp_path: Path) -> None:
    """Frames with only hashable columns take the fast path and round-trip exactly."""
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "value": [1.0, 2.0]})

    ref = DataframeRef.from_pandas(df, ref="plain", local_dir=tmp_path)
    restored = ref.materialize_sync()

    pd.testing.assert_frame_equal(restored.reset_index(drop=True), df.reset_index(drop=True))
