"""Tests for :mod:`parsimony_agents.dataset_io`.

Validates the contract: open Parquet on disk, two embedded metadata
namespaces (``parsimony.result`` for provenance, ``parsimony_agents`` for
curation), and round-trip via the typed ``Dataset.save`` /
``deserialize_dataset`` API.

Note: the durable on-disk shape *is* the :class:`Dataset` Pydantic model;
there is no separate ``Curation`` type. ``deserialize_dataset`` returns a
``(Result, Dataset)`` pair so callers get both the live frame + provenance
and the curation envelope without translation.

Payload contract: every codec call in production receives a
:class:`DataFrameObject` (the executor's wrapper). Tests therefore build
the same wrapper via :meth:`DataFrameObject.from_pandas` rather than
passing raw DataFrames — that keeps the test fixtures faithful to what
the streaming dispatcher actually hands to the codec.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from parsimony.result import Provenance, TabularResult

from parsimony_agents import Dataset, deserialize_dataset
from parsimony_agents.dataset_io import CURATION_META_KEY, write_dataset_bytes
from parsimony_agents.execution.outputs import DataFrameObject
from parsimony_agents.identity import ArtifactRef


def _nb_ref(sha: str = "nb-csha") -> ArtifactRef:
    return ArtifactRef(kind="notebook", logical_id=sha, content_sha=sha)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "value": [1.0, 2.0, 3.0],
        }
    )


def _payload(df: pd.DataFrame, tmp_path: Path) -> DataFrameObject:
    return DataFrameObject.from_pandas(df, local_dir=tmp_path / "_dfo")


def test_write_dataset_bytes_roundtrips(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    dataset = Dataset(
        title="Demo",
        description="Curation round-trip",
        tags=["demo", "test"],
        notebook_refs=[_nb_ref()],
        logical_id="lid-abc",
        content_sha="csha-init",
    )
    blob = write_dataset_bytes(dataset, _payload(sample_df, tmp_path))

    assert isinstance(blob, bytes)
    assert blob.startswith(b"PAR1")  # parquet magic

    result, recovered = deserialize_dataset(blob)
    pd.testing.assert_frame_equal(result.df, sample_df)
    assert recovered.title == "Demo"
    assert recovered.description == "Curation round-trip"
    assert recovered.tags == ["demo", "test"]
    assert recovered.notebook_refs == [_nb_ref()]
    assert recovered.logical_id == "lid-abc"




def test_deserialize_handles_vanilla_parquet(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """A pandas-written Parquet without metadata must still deserialize cleanly."""

    target = tmp_path / "vanilla.parquet"
    sample_df.to_parquet(target)

    result, dataset = deserialize_dataset(target.read_bytes())

    pd.testing.assert_frame_equal(result.df, sample_df)
    assert dataset.title == ""
    assert dataset.description == ""
    assert dataset.tags == []
    assert dataset.source_refs == []
    assert dataset.notebook_refs == []


def test_deserialize_returns_empty_dataset_for_vanilla_parquet(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    r = TabularResult(
        data=sample_df,
        provenance=Provenance(source="fred", source_description="fred", params={"series_id": "GDPC1"}),
    )
    target = tmp_path / "connector.parquet"
    r.to_parquet(target)

    result, dataset = deserialize_dataset(target.read_bytes())
    assert dataset.source_refs == []
    assert result.provenance.source == "fred"
    assert result.provenance.params == {"series_id": "GDPC1"}


def test_write_dataset_bytes_persists_variable_name(
    sample_df: pd.DataFrame, tmp_path: Path
) -> None:
    """R2: ``variable_name`` survives the parquet round-trip via embedded curation."""
    dataset = Dataset(
        title="Demo",
        notebook_refs=[_nb_ref()],
        logical_id="lid-vn",
        variable_name="gdp_df",
    )
    blob = write_dataset_bytes(dataset, _payload(sample_df, tmp_path))
    _, recovered = deserialize_dataset(blob)
    assert recovered.variable_name == "gdp_df"


def test_dataset_save_via_typed_api(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    """``Dataset.save`` is the typed entry point: build the model, attach a payload, save."""

    dataset = Dataset(title="Typed", tags=["typed"]).with_payload(_payload(sample_df, tmp_path))

    target = tmp_path / "typed.parquet"
    dataset.save(target)

    _, recovered = deserialize_dataset(target.read_bytes())
    assert recovered.title == "Typed"
    assert recovered.tags == ["typed"]


def test_dataset_save_rejects_unattached_payload(tmp_path: Path) -> None:
    dataset = Dataset(title="No payload")
    with pytest.raises(ValueError, match="no payload attached"):
        dataset.save(tmp_path / "x.parquet")


def test_dataset_save_rejects_non_parquet_path(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    dataset = Dataset(title="Bad ext").with_payload(_payload(sample_df, tmp_path))
    with pytest.raises(ValueError, match="must end in .parquet"):
        dataset.save(tmp_path / "demo.csv")


def test_dataset_with_payload_rejects_raw_dataframe(sample_df: pd.DataFrame) -> None:
    """The payload contract is single-typed: only DataFrameObject is accepted."""

    dataset = Dataset(title="Bad payload")
    with pytest.raises(TypeError, match="DataFrameObject"):
        dataset.with_payload(sample_df)  # type: ignore[arg-type]


def test_write_dataset_bytes_rejects_raw_dataframe(sample_df: pd.DataFrame) -> None:
    dataset = Dataset(title="Bad")
    with pytest.raises(TypeError, match="DataFrameObject"):
        write_dataset_bytes(dataset, sample_df)  # type: ignore[arg-type]


def test_metadata_key_is_present_on_disk(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    dataset = Dataset(title="Inspect").with_payload(_payload(sample_df, tmp_path))
    target = tmp_path / "demo.parquet"
    dataset.save(target)

    table = pq.read_table(target)
    metadata = table.schema.metadata or {}
    assert CURATION_META_KEY in metadata
    assert b"parsimony.result" in metadata


