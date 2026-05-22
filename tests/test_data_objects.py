"""Tests for :mod:`parsimony_agents.execution.data_objects`.

Validates the content-addressed persister against the new dual-identity
layout (``CONTENT_ADDRESSED_ARTIFACTS_PLAN.md`` §2.3):

* Files land under ``.ockham/data_objects/<logical_id>/<content_sha>.parquet``.
* ``log.jsonl`` records each refresh; same content → no second log entry.
* Same upstream (provenance modulo ``fetched_at``) → same logical_id.
* Same content → same content_sha → idempotent file write.
* Different params → different logical_id (different folder).
* Different data → same logical_id, different content_sha.
* The persisted file is a valid parquet that round-trips through
  ``TabularResult.from_arrow``.
* Errors degrade gracefully (return ``None``, no raise).
"""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from parsimony.result import Provenance, TabularResult

from parsimony_agents.execution.data_objects import (
    DATA_OBJECTS_NAMESPACE,
    make_data_object_persister,
)
from parsimony_agents.identity import ArtifactRef


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


async def test_persister_returns_typed_artifact_ref_and_version(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    out = await persist(_make_result(pd.DataFrame({"a": [1, 2, 3]})))
    assert out is not None
    ref, version = out
    assert isinstance(ref, ArtifactRef)
    assert ref.kind == "data_object"
    assert ref.workspace_file_path.startswith(f"{DATA_OBJECTS_NAMESPACE}/")
    assert ref.workspace_file_path.endswith(".parquet")
    assert (tmp_path / ref.workspace_file_path).exists()
    # First fetch is always v1.
    assert version == 1


async def test_same_provenance_same_data_dedupes_file_and_log(tmp_path: Path) -> None:
    """Same logical_id + same content_sha → one file, one log entry, same version."""
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    a = await persist(_make_result(df, fetched_at=datetime(2024, 1, 1)))
    b = await persist(_make_result(df, fetched_at=datetime(2025, 6, 1)))
    assert a is not None and b is not None
    ref_a, v_a = a
    ref_b, v_b = b
    assert ref_a == ref_b
    assert v_a == v_b == 1  # identical content republish → same version
    parquets = list((tmp_path / DATA_OBJECTS_NAMESPACE).rglob("*.parquet"))
    assert len(parquets) == 1
    log = (tmp_path / DATA_OBJECTS_NAMESPACE / ref_a.logical_id / "log.jsonl").read_text()
    log_lines = [json.loads(line) for line in log.splitlines() if line.strip()]
    assert len(log_lines) == 1
    assert log_lines[0]["content_sha"] == ref_a.content_sha


async def test_persister_distinguishes_params(tmp_path: Path) -> None:
    """Different params → different logical_id (different folder)."""
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1]})
    a = await persist(_make_result(df, params={"x": 1}))
    b = await persist(_make_result(df, params={"x": 2}))
    assert a is not None and b is not None
    ref_a, _ = a
    ref_b, _ = b
    assert ref_a.logical_id != ref_b.logical_id


async def test_same_logical_id_different_content_appends_log(tmp_path: Path) -> None:
    """Same provenance + different bytes → same logical_id, different content_sha,
    two snapshots side-by-side, two log entries, monotonic versions."""
    persist = make_data_object_persister(tmp_path)
    a = await persist(_make_result(pd.DataFrame({"a": [1, 2]})))
    b = await persist(_make_result(pd.DataFrame({"a": [1, 2, 3]})))
    assert a is not None and b is not None
    ref_a, v_a = a
    ref_b, v_b = b
    assert ref_a.logical_id == ref_b.logical_id
    assert ref_a.content_sha != ref_b.content_sha
    assert v_a == 1 and v_b == 2
    parquets = list((tmp_path / DATA_OBJECTS_NAMESPACE / ref_a.logical_id).glob("*.parquet"))
    assert len(parquets) == 2
    log = (tmp_path / DATA_OBJECTS_NAMESPACE / ref_a.logical_id / "log.jsonl").read_text()
    log_lines = [json.loads(line) for line in log.splitlines() if line.strip()]
    assert len(log_lines) == 2
    assert {l["content_sha"] for l in log_lines} == {ref_a.content_sha, ref_b.content_sha}


async def test_persisted_file_is_readable_parquet(tmp_path: Path) -> None:
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.5]})
    out = await persist(_make_result(df, source="us_cpi", params={"q": "cpi"}))
    assert out is not None
    ref, _v = out
    blob = (tmp_path / ref.workspace_file_path).read_bytes()
    table = pq.read_table(BytesIO(blob))
    round_tripped = TabularResult.from_arrow(table)
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
    """
    from parsimony_agents.dataset_io import deserialize_dataset

    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"date": ["2024-01-01"], "value": [1.5]})
    out = await persist(
        _make_result(df, source="fred_fetch", params={"series_id": "UNRATE"})
    )
    assert out is not None
    ref, _v = out
    blob = (tmp_path / ref.workspace_file_path).read_bytes()
    result, dataset = deserialize_dataset(blob)
    # Provenance round-trips on the Result side (modulo fetched_at, which the
    # canonicalizer strips for byte-level dedup).
    assert result.provenance.source == "fred_fetch"
    assert result.provenance.params == {"series_id": "UNRATE"}
    # Dataset side is empty — the data_object is a leaf, not a curation.
    assert dataset.source_refs == []
    assert dataset.title == ""
    assert dataset.description == ""


async def test_persister_writes_curation_sidecar(tmp_path: Path) -> None:
    """Each persisted data_object has a ``curation.json`` mirroring its provenance.

    Read consumers (ref_enrichment, detail-tab payload) consult the
    sidecar instead of unpacking parquet arrow metadata, so the sidecar
    must exist alongside the parquet and carry the same fields.
    """
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1, 2]})
    out = await persist(
        _make_result(
            df,
            source="fred_fetch",
            params={"series_id": "GDPC1"},
            fetched_at=datetime(2026, 1, 1),
        )
    )
    assert out is not None
    ref, _ = out
    sidecar = tmp_path / DATA_OBJECTS_NAMESPACE / ref.logical_id / "curation.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["kind"] == "data_object"
    assert payload["logical_id"] == ref.logical_id
    assert payload["source"] == "fred_fetch"
    assert payload["params"] == {"series_id": "GDPC1"}
    assert payload["created_at"]
    assert payload["updated_at"]


async def test_persister_curation_preserves_created_at_on_rewrite(tmp_path: Path) -> None:
    """Re-persisting the same data_object refreshes ``updated_at`` but not ``created_at``."""
    persist = make_data_object_persister(tmp_path)
    df = pd.DataFrame({"a": [1]})
    a = await persist(_make_result(df, source="x", params={"k": "v"}))
    assert a is not None
    ref, _ = a
    sidecar = tmp_path / DATA_OBJECTS_NAMESPACE / ref.logical_id / "curation.json"
    first = json.loads(sidecar.read_text(encoding="utf-8"))
    # Re-persist (idempotent on bytes; sidecar gets rewritten).
    b = await persist(_make_result(df, source="x", params={"k": "v"}))
    assert b is not None
    second = json.loads(sidecar.read_text(encoding="utf-8"))
    assert second["created_at"] == first["created_at"]
