"""Tests for :mod:`parsimony_agents.lineage_diff`.

A diff compares the dependency closures of two snapshots (content_shas) of the
SAME artifact (kind + logical_id) and reports which lineage nodes moved. The
realistic case: a refresh re-fetches a data_object (new content_sha), which
propagates a new content_sha up to the dataset that references it.

Persist helpers mirror test_closure.py / test_refresh.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from parsimony_agents.artifacts import Dataset
from parsimony_agents.dataset_io import write_dataset_bytes
from parsimony_agents.execution.outputs import DataFrameObject
from parsimony_agents.identity import (
    ArtifactRef,
    content_sha,
    dataset_logical_id,
    notebook_content_sha,
)
from parsimony_agents.lineage_diff import diff_artifacts


class _ReadOnlyExecutor:
    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)
        self._cwd_path = cwd

    async def read_workspace_file(self, path: str) -> bytes:
        target = (self._cwd_path / path).resolve()
        target.relative_to(self._cwd_path.resolve())
        if not target.exists():
            raise FileNotFoundError(path)
        return target.read_bytes()


def _nb_ref(name: str, code: str) -> ArtifactRef:
    return ArtifactRef(kind="notebook", logical_id=name, content_sha=notebook_content_sha(code))


def _do_ref(provenance_id: str, content: str) -> ArtifactRef:
    return ArtifactRef(
        kind="data_object",
        logical_id=f"do-{provenance_id}",
        content_sha=content_sha(content.encode("utf-8")),
    )


def _write_bytes(executor: _ReadOnlyExecutor, path: str, data: bytes) -> None:
    p = Path(executor.cwd) / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _write_log(executor: _ReadOnlyExecutor, kind: str, logical_id: str, csha: str) -> None:
    log_path = Path(executor.cwd) / f".ockham/{kind}s/{logical_id}/log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({"content_sha": csha, "inputs": {}}, sort_keys=True) + "\n")


def _persist_dataset(
    executor: _ReadOnlyExecutor,
    *,
    notebook_refs: list[ArtifactRef],
    source_refs: list[ArtifactRef],
    variable_name: str,
    df: pd.DataFrame,
) -> ArtifactRef:
    lid = dataset_logical_id(notebook_refs=notebook_refs, variable_name=variable_name, source_refs=source_refs)
    payload = DataFrameObject.from_pandas(df, local_dir=Path(executor.cwd) / "_dfo")
    dataset = Dataset(
        logical_id=lid,
        title="demo",
        notebook_refs=notebook_refs,
        source_refs=source_refs,
        variable_name=variable_name,
    )
    blob = write_dataset_bytes(dataset, payload)
    csha = content_sha(blob)
    ref = ArtifactRef(kind="dataset", logical_id=lid, content_sha=csha)
    _write_bytes(executor, ref.workspace_file_path, blob)
    _write_log(executor, "dataset", lid, csha)
    return ref


@pytest.mark.asyncio
async def test_diff_reports_changed_upstream_data_object(tmp_path: Path) -> None:
    ex = _ReadOnlyExecutor(tmp_path)
    nb = _nb_ref("build", "df = fetch()")
    df = pd.DataFrame({"a": [1, 2, 3]})

    # Same data_object logical_id, different content_sha across the two versions —
    # the dataset's logical_id is stable (derives from source LOGICAL ids) but its
    # bytes (and so content_sha) move because it embeds the source content_sha.
    do_v1 = _do_ref("fred-gdp", "old-bytes")
    do_v2 = _do_ref("fred-gdp", "new-bytes")
    ds_v1 = _persist_dataset(ex, notebook_refs=[nb], source_refs=[do_v1], variable_name="gdp", df=df)
    ds_v2 = _persist_dataset(ex, notebook_refs=[nb], source_refs=[do_v2], variable_name="gdp", df=df)

    assert ds_v1.logical_id == ds_v2.logical_id  # same artifact
    assert ds_v1.content_sha != ds_v2.content_sha  # different snapshot

    diff = await diff_artifacts(ds_v1, ds_v2, executor=ex)

    assert diff.content_changed is True
    assert diff.added == ()
    assert diff.removed == ()
    assert len(diff.changed) == 1
    (change,) = diff.changed
    assert change.kind == "data_object"
    assert change.logical_id == "do-fred-gdp"
    assert change.before == do_v1.content_sha
    assert change.after == do_v2.content_sha
    assert not diff.is_empty
    assert "do-fred-gdp" in diff.summary()


@pytest.mark.asyncio
async def test_diff_identical_snapshot_is_empty(tmp_path: Path) -> None:
    ex = _ReadOnlyExecutor(tmp_path)
    nb = _nb_ref("build", "df = fetch()")
    ds = _persist_dataset(
        ex, notebook_refs=[nb], source_refs=[_do_ref("x", "b")], variable_name="v", df=pd.DataFrame({"a": [1]})
    )
    diff = await diff_artifacts(ds, ds, executor=ex)
    assert diff.is_empty
    assert diff.content_changed is False
    assert diff.summary().endswith("unchanged")


@pytest.mark.asyncio
async def test_diff_rejects_unrelated_artifacts(tmp_path: Path) -> None:
    ex = _ReadOnlyExecutor(tmp_path)
    a = ArtifactRef(kind="dataset", logical_id="L1", content_sha="a" * 64)
    b = ArtifactRef(kind="dataset", logical_id="L2", content_sha="b" * 64)
    with pytest.raises(ValueError, match="SAME artifact"):
        await diff_artifacts(a, b, executor=ex)
