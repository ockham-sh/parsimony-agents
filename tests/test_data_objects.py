"""Tests for :mod:`parsimony_agents.execution.data_objects`.

Validates the content-addressed persister:

* Path layout under ``.ockham/data_objects/<title_slug>_<short_sha>.parquet``.
* Same content + same provenance (modulo ``fetched_at``) → same path
  (dedup).
* Different params or different bytes → different path.
* The persisted file is a valid parquet that round-trips through
  ``Result.from_arrow``.
* Errors degrade gracefully (return ``None``, no raise).
"""

from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from parsimony.result import Provenance, Result

from parsimony_agents.execution.data_objects import (
    DATA_OBJECTS_NAMESPACE,
    make_data_object_persister,
)


def _make_result(
    df: pd.DataFrame,
    *,
    source: str = "src",
    params: dict | None = None,
    fetched_at: datetime | None = None,
) -> Result:
    return Result(
        data=df,
        provenance=Provenance(
            source=source,
            source_description="test fixture",
            params=params or {},
            fetched_at=fetched_at,
        ),
    )


async def test_persister_writes_under_data_objects_namespace(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    rel = await persist(_make_result(pd.DataFrame({"a": [1, 2, 3]})))
    assert rel is not None
    assert rel.startswith(f"{DATA_OBJECTS_NAMESPACE}/")
    assert rel.endswith(".parquet")
    base = Path(rel).name
    assert re.match(r"^src_[0-9a-f]{12}\.parquet$", base)
    assert (tmp_path / rel).exists()


async def test_persister_dedups_identical_fetches(tmp_path: Path) -> None:
    """Same data + same provenance (sans fetched_at) → same path, one file."""
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    a = await persist(_make_result(df, fetched_at=datetime(2024, 1, 1)))
    b = await persist(_make_result(df, fetched_at=datetime(2025, 6, 1)))
    assert a == b
    listing = list((tmp_path / DATA_OBJECTS_NAMESPACE).glob("*.parquet"))
    assert len(listing) == 1


async def test_persister_distinguishes_params(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1]})
    a = await persist(_make_result(df, params={"x": 1}))
    b = await persist(_make_result(df, params={"x": 2}))
    assert a is not None and b is not None
    assert a != b


async def test_persister_distinguishes_data(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    a = await persist(_make_result(pd.DataFrame({"a": [1, 2]})))
    b = await persist(_make_result(pd.DataFrame({"a": [1, 2, 3]})))
    assert a is not None and b is not None
    assert a != b


async def test_persisted_file_is_readable_parquet(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.5]})
    rel = await persist(_make_result(df, source="us_cpi", params={"q": "cpi"}))
    assert rel is not None
    blob = (tmp_path / rel).read_bytes()
    table = pq.read_table(BytesIO(blob))
    round_tripped = Result.from_arrow(table)
    pd.testing.assert_frame_equal(round_tripped.df.reset_index(drop=True), df)
    assert round_tripped.provenance.source == "us_cpi"
    assert round_tripped.provenance.params == {"q": "cpi"}


async def test_persister_swallows_codec_failures(tmp_path: Path) -> None:
    """A misbehaving codec must not kill the agent turn — graceful ``None``."""
    persist = make_data_object_persister(tmp_path)

    class _Bad:
        provenance = Provenance(source="x", source_description="x")

        def to_arrow(self):  # noqa: D401
            raise RuntimeError("boom")

    assert await persist(_Bad()) is None


async def test_persisted_file_carries_only_result_provenance(tmp_path: Path) -> None:
    """A data_object is the leaf of the lineage graph — its identity is
    the fetch. The persister writes ``parsimony.result`` metadata only;
    ``parsimony_agents`` curation is intentionally absent (a data_object
    is never curated and has no upstream sources of its own).

    The dataset codec round-trips the absent curation as ``Dataset()``
    (empty), and the viewer surfaces ``Result.provenance`` as the
    artifact's identity instead of as a self-referential "Source".
    """
    from parsimony_agents.dataset_io import deserialize_dataset

    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.5]})
    rel = await persist(
        _make_result(df, source="fred_fetch", params={"series_id": "UNRATE"})
    )
    assert rel is not None
    blob = (tmp_path / rel).read_bytes()
    result, dataset = deserialize_dataset(blob)
    # Provenance round-trips on the Result side.
    assert result.provenance.source == "fred_fetch"
    assert result.provenance.params == {"series_id": "UNRATE"}
    # Dataset side is empty — the data_object is a leaf, not a curation.
    assert dataset.sources == []
    assert dataset.title == ""
    assert dataset.description == ""
