"""Tests for the ``load_dataset`` kernel primitive.

Contract from brief §4:

- Slug → resolved DataFrame.
- Miss → :class:`LoadDatasetError` (subclass of KeyError) with guidance.
- Ambiguous → error.
- Non-string arg → TypeError.
- Load events surface on the active :class:`RunScope` and only the
  active scope (scratch reads do not produce lineage edges).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from parsimony_agents.artifacts import Dataset
from parsimony_agents.dataset_io import write_dataset_bytes
from parsimony_agents.execution.load import (
    LoadDatasetError,
    build_load_dataset,
    resolve_dataset_slug,
)
from parsimony_agents.execution.outputs import DataFrameObject
from parsimony_agents.execution.run_scope import OriginLedger
from parsimony_agents.identity import content_sha


def _seed_dataset(
    root: Path, logical_id: str, live_name: str, df: pd.DataFrame
) -> str:
    """Persist a dataset snapshot + curation + log on disk; return content_sha."""
    payload = DataFrameObject.from_pandas(df, local_dir=root)
    dataset = Dataset(
        logical_id=logical_id,
        title=live_name,
        description="",
        variable_name="result",
        live_name=live_name,
    )
    blob = write_dataset_bytes(dataset, payload)
    csha = content_sha(blob)
    base = root / ".ockham" / "datasets" / logical_id
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{csha}.parquet").write_bytes(blob)
    (base / "log.jsonl").write_text(
        json.dumps({"ts": "t1", "content_sha": csha, "inputs": {}}) + "\n",
        encoding="utf-8",
    )
    (base / "curation.json").write_text(
        json.dumps(
            {
                "kind": "dataset",
                "logical_id": logical_id,
                "title": live_name,
                "live_name": live_name,
                "tags": [],
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    return csha


def test_resolve_dataset_slug_happy_path(tmp_path: Path) -> None:
    _seed_dataset(tmp_path, "lid1", "us_gdp", pd.DataFrame({"v": [1]}))
    ref = resolve_dataset_slug(tmp_path, "us_gdp")
    assert ref.kind == "dataset"
    assert ref.logical_id == "lid1"


def test_resolve_dataset_slug_missing(tmp_path: Path) -> None:
    with pytest.raises(LoadDatasetError) as excinfo:
        resolve_dataset_slug(tmp_path, "ghost")
    assert "ghost" in str(excinfo.value)
    assert "return_dataset" in str(excinfo.value)


def test_resolve_dataset_slug_ambiguous(tmp_path: Path) -> None:
    _seed_dataset(tmp_path, "lid1", "twin", pd.DataFrame({"v": [1]}))
    _seed_dataset(tmp_path, "lid2", "twin", pd.DataFrame({"v": [2]}))
    with pytest.raises(LoadDatasetError, match="ambiguous"):
        resolve_dataset_slug(tmp_path, "twin")


def test_load_dataset_returns_dataframe(tmp_path: Path) -> None:
    _seed_dataset(tmp_path, "lid1", "us_gdp", pd.DataFrame({"v": [1, 2, 3]}))
    ledger = OriginLedger()
    load = build_load_dataset(lambda: tmp_path, ledger)
    df = load("us_gdp")
    assert isinstance(df, pd.DataFrame)
    assert list(df["v"]) == [1, 2, 3]


def test_load_dataset_rejects_non_string(tmp_path: Path) -> None:
    ledger = OriginLedger()
    load = build_load_dataset(lambda: tmp_path, ledger)
    with pytest.raises(TypeError, match="live_name string"):
        load({"kind": "dataset", "logical_id": "x", "content_sha": "y"})


def test_load_dataset_rejects_extra_kwargs(tmp_path: Path) -> None:
    ledger = OriginLedger()
    load = build_load_dataset(lambda: tmp_path, ledger)
    with pytest.raises(TypeError, match="single positional"):
        load("foo", version=1)


def test_load_dataset_records_on_active_scope(tmp_path: Path) -> None:
    _seed_dataset(tmp_path, "lid1", "us_gdp", pd.DataFrame({"v": [1]}))
    ledger = OriginLedger()
    load = build_load_dataset(lambda: tmp_path, ledger)

    with ledger.scope("notebooks/chart.py") as scope:
        _ = load("us_gdp")
        assert len(scope.load_refs) == 1
        assert scope.load_refs[0].kind == "dataset"
        assert scope.load_refs[0].logical_id == "lid1"


def test_load_dataset_scratch_does_not_record(tmp_path: Path) -> None:
    """Outside a producing scope, loads still work but produce no lineage."""
    _seed_dataset(tmp_path, "lid1", "us_gdp", pd.DataFrame({"v": [1]}))
    ledger = OriginLedger()
    load = build_load_dataset(lambda: tmp_path, ledger)
    _ = load("us_gdp")
    assert ledger.current is None
    # No origin has been stamped — scratch reads are observational only.
    assert ledger.get("anything") is None
