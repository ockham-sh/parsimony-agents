"""Tests for :func:`parsimony_agents.artifacts.snapshot_path` and path parsing."""

from __future__ import annotations

import pytest

from parsimony_agents.artifacts import artifact_id_from_dataset_snapshot_path, snapshot_path


def test_snapshot_path_for_dataset() -> None:
    assert snapshot_path(
        artifact_id="abc",
        version=3,
        kind="dataset",
        title="US CPI",
    ) == ".ockham/cards/abc/v3/us_cpi.parquet"


def test_snapshot_path_for_chart() -> None:
    assert snapshot_path(
        artifact_id="def",
        version=1,
        kind="chart",
        title="Headline CPI",
    ) == ".ockham/cards/def/v1/headline_cpi.vl.json"


def test_snapshot_path_rejects_empty_artifact_id() -> None:
    with pytest.raises(ValueError, match="non-empty artifact_id"):
        snapshot_path(artifact_id="", version=1, kind="dataset", title="x")


def test_snapshot_path_rejects_zero_version() -> None:
    with pytest.raises(ValueError, match="version >="):
        snapshot_path(artifact_id="x", version=0, kind="dataset", title="x")


def test_snapshot_path_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported artifact kind"):
        snapshot_path(artifact_id="x", version=1, kind="bogus", title="x")


def test_artifact_id_from_dataset_snapshot_path_new_layout() -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    p = f".ockham/cards/{pid}/v1/my_dataset.parquet"
    assert artifact_id_from_dataset_snapshot_path(p) == pid


def test_artifact_id_from_dataset_snapshot_path_empty() -> None:
    assert artifact_id_from_dataset_snapshot_path("") is None
