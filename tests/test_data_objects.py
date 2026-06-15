"""Tests for :mod:`parsimony_agents.execution.data_objects`.

Validates the immutable object-pool persister:

* Files land under ``.ockham/objects/<sha[:2]>/<sha[2:]>.parquet``.
* Same upstream (provenance modulo ``fetched_at``) → same logical_id.
* Same content → same content_sha → idempotent file write.
* Different params → different logical_id.
* Different data → same logical_id, different content_sha (separate pool files).
* The persisted file is a valid parquet that round-trips through
  ``TabularResult.from_arrow``.
* Errors degrade gracefully (return ``None``, no raise).
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from parsimony.result import Provenance, TabularResult

from parsimony_agents.execution.data_objects import make_data_object_persister
from parsimony_agents.identity import OBJECTS_NAMESPACE, ArtifactRef, object_pool_path


def _make_result(
    df: pd.DataFrame,
    *,
    source: str = "src",
    params: dict | None = None,
    fetched_at: datetime | None = None,
) -> TabularResult:
    return TabularResult(
        data=df,
        provenance=Provenance(
            source=source,
            source_description="test fixture",
            params=params or {},
            fetched_at=fetched_at,
        ),
    )


def test_persister_returns_typed_artifact_ref_without_version(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    out = persist(_make_result(pd.DataFrame({"a": [1, 2, 3]})))
    assert out is not None
    ref, version = out
    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "data_object"
    assert ref.workspace_file_path == object_pool_path(ref.content_sha)
    assert ref.workspace_file_path.startswith(f"{OBJECTS_NAMESPACE}/")
    assert (tmp_path / ref.workspace_file_path).exists()
    assert version is None


def test_same_provenance_same_data_dedupes_file(tmp_path: Path) -> None:
    """Same logical_id + same content_sha → one pool file."""
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    a = persist(_make_result(df, fetched_at=datetime(2024, 1, 1)))
    b = persist(_make_result(df, fetched_at=datetime(2025, 6, 1)))
    assert a is not None and b is not None
    ref_a, _ = a
    ref_b, _ = b
    assert ref_a == ref_b
    parquets = list((tmp_path / OBJECTS_NAMESPACE).rglob("*.parquet"))
    assert len(parquets) == 1


def test_persister_distinguishes_params(tmp_path: Path) -> None:
    """Different params → different logical_id."""
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1]})
    a = persist(_make_result(df, params={"x": 1}))
    b = persist(_make_result(df, params={"x": 2}))
    assert a is not None and b is not None
    ref_a, _ = a
    ref_b, _ = b
    assert ref_a.logical_id != ref_b.logical_id


def test_same_logical_id_different_content_writes_two_pool_files(tmp_path: Path) -> None:
    """Same provenance + different bytes → two pool entries, same logical_id."""
    persist = make_data_object_persister(tmp_path)
    a = persist(_make_result(pd.DataFrame({"a": [1, 2]})))
    b = persist(_make_result(pd.DataFrame({"a": [1, 2, 3]})))
    assert a is not None and b is not None
    ref_a, _ = a
    ref_b, _ = b
    assert ref_a.logical_id == ref_b.logical_id
    assert ref_a.content_sha != ref_b.content_sha
    assert (tmp_path / ref_a.workspace_file_path).exists()
    assert (tmp_path / ref_b.workspace_file_path).exists()


def test_persisted_file_is_readable_parquet(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.5]})
    out = persist(_make_result(df, source="us_cpi", params={"q": "cpi"}))
    assert out is not None
    ref, _ = out
    blob = (tmp_path / ref.workspace_file_path).read_bytes()
    table = pq.read_table(BytesIO(blob))
    round_tripped = TabularResult.from_arrow(table)
    pd.testing.assert_frame_equal(round_tripped.df.reset_index(drop=True), df)
    assert round_tripped.provenance.source == "us_cpi"
    assert round_tripped.provenance.params == {"q": "cpi"}


def test_persister_swallows_codec_failures(tmp_path: Path) -> None:
    """A misbehaving codec must not kill the agent turn — graceful ``None``."""
    persist = make_data_object_persister(tmp_path)

    class _Bad:
        provenance = Provenance(source="x", source_description="x")

        def to_arrow(self):  # noqa: D401
            raise RuntimeError("boom")

    assert persist(_Bad()) is None


def test_persisted_file_carries_only_result_provenance(tmp_path: Path) -> None:
    """A data_object is the leaf of the lineage graph — its identity is the fetch."""
    from parsimony_agents.dataset_io import deserialize_dataset

    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.5]})
    out = persist(_make_result(df, source="fred_fetch", params={"series_id": "UNRATE"}))
    assert out is not None
    ref, _ = out
    blob = (tmp_path / ref.workspace_file_path).read_bytes()
    result, dataset = deserialize_dataset(blob)
    assert result.provenance.source == "fred_fetch"
    assert result.provenance.params == {"series_id": "UNRATE"}
    assert dataset.source_refs == []
    assert dataset.title == ""
