"""Tests for the framework artifact-persistence registry."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.execution.artifact_store import (
    ReportValidationError,
    SnapshotIntegrityError,
    log_inputs_for,
    persist_artifact,
    persist_notebook,
    render_artifact_bytes,
)
from parsimony_agents.execution.outputs import DataFrameObject
from parsimony_agents.identity import ArtifactRef, content_sha, notebook_content_sha


class FakeExecutor:
    """In-memory executor exposing only the persist seam."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def read_workspace_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        self.files[path] = data


def _dataset(tmp_path: Path, *, lid: str = "d1", live_name: str = "unrate") -> Dataset:
    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "value": [3.6, 3.5]})
    payload = DataFrameObject.from_pandas(df, local_dir=tmp_path / "_dfo")
    return Dataset(logical_id=lid, title="US Unemployment", live_name=live_name).with_payload(payload)


# ---------------------------------------------------------------------------
# render_artifact_bytes / log_inputs_for
# ---------------------------------------------------------------------------


def test_render_dataset_bytes_roundtrips(tmp_path: Path) -> None:
    from parsimony_agents.dataset_io import deserialize_dataset

    blob = render_artifact_bytes(_dataset(tmp_path), "dataset")
    result, ds = deserialize_dataset(blob)
    assert ds.live_name == "unrate"
    assert len(result.df) == 2


def test_render_dataset_without_payload_raises() -> None:
    with pytest.raises(ValueError, match="payload"):
        render_artifact_bytes(Dataset(logical_id="d1", title="x"), "dataset")


def _chart(tmp_path: Path, *, lid: str = "c1", live_name: str = "mychart") -> Chart:
    import altair as alt

    from parsimony_agents.execution.factory import OutputFactory

    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    fig = OutputFactory(local_dir=str(tmp_path)).from_value(alt.Chart(df).mark_line().encode(x="x:Q", y="y:Q"), ref="c")
    return Chart(
        logical_id=lid,
        title="My Chart",
        description="d",
        live_name=live_name,
        notebook_ref=ArtifactRef(kind="notebook", logical_id="nb", content_sha="csha_nb"),
    ).with_payload(fig)


def test_render_chart_bytes_roundtrips(tmp_path: Path) -> None:
    from parsimony_agents.chart_io import deserialize_chart

    blob = render_artifact_bytes(_chart(tmp_path), "chart")
    chart, _spec = deserialize_chart(blob)
    assert chart.live_name == "mychart"


def test_render_chart_without_payload_raises() -> None:
    chart = Chart(
        logical_id="c1",
        title="x",
        notebook_ref=ArtifactRef(kind="notebook", logical_id="nb", content_sha="s"),
    )
    with pytest.raises(ValueError, match="payload"):
        render_artifact_bytes(chart, "chart")


def test_render_chart_without_notebook_ref_raises(tmp_path: Path) -> None:
    import altair as alt

    from parsimony_agents.execution.factory import OutputFactory

    fig = OutputFactory(local_dir=str(tmp_path)).from_value(
        alt.Chart(pd.DataFrame({"x": [1]})).mark_point().encode(x="x:Q"), ref="c"
    )
    chart = Chart(logical_id="c1", title="x").with_payload(fig)  # no notebook_ref
    with pytest.raises(ValueError, match="notebook_ref"):
        render_artifact_bytes(chart, "chart")


@pytest.mark.asyncio
async def test_persist_chart_writes_triplet(tmp_path: Path) -> None:
    ex = FakeExecutor()
    chart = _chart(tmp_path)
    ref = await persist_artifact(
        ex,
        kind="chart",
        artifact=chart,
        blob=render_artifact_bytes(chart, "chart"),
        log_inputs=log_inputs_for(chart, "chart"),
    )
    assert ref.workspace_file_path in ex.files
    cur = json.loads(ex.files[".ockham/charts/c1/curation.json"])
    assert cur["live_name"] == "mychart" and cur["kind"] == "chart"
    assert ".ockham/charts/c1/log.jsonl" in ex.files


@pytest.mark.asyncio
async def test_persist_report_writes_triplet() -> None:
    ex = FakeExecutor()
    report = Report(
        logical_id="r1",
        title="My Report",
        description="d",
        live_name="myreport",
        markdown="# My Report\n\nSome prose.\n",
    )
    blob = render_artifact_bytes(report, "report")
    ref = await persist_artifact(
        ex, kind="report", artifact=report, blob=blob, log_inputs=log_inputs_for(report, "report")
    )
    assert ref.workspace_file_path in ex.files
    cur = json.loads(ex.files[".ockham/reports/r1/curation.json"])
    assert cur["live_name"] == "myreport" and cur["kind"] == "report"
    log = json.loads(ex.files[".ockham/reports/r1/log.jsonl"].decode().strip())
    assert "formats" in log["inputs"]  # report log carries the formats list


def test_render_report_without_markdown_raises() -> None:
    with pytest.raises(ValueError, match="empty markdown"):
        render_artifact_bytes(Report(logical_id="r1", title="x", markdown="  "), "report")


def test_log_inputs_for_dataset_lists_lineage_shas() -> None:
    ds = Dataset(
        logical_id="d1",
        title="x",
        notebook_refs=[ArtifactRef(kind="notebook", logical_id="nb", content_sha="csha_nb")],
        source_refs=[ArtifactRef(kind="data_object", logical_id="do", content_sha="csha_do")],
    )
    assert log_inputs_for(ds, "dataset") == {"notebooks": ["csha_nb"], "sources": ["csha_do"]}


# ---------------------------------------------------------------------------
# persist_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_artifact_writes_triplet_and_stamps_sha(tmp_path: Path) -> None:
    ex = FakeExecutor()
    ds = _dataset(tmp_path)
    blob = render_artifact_bytes(ds, "dataset")

    ref = await persist_artifact(ex, kind="dataset", artifact=ds, blob=blob, log_inputs=log_inputs_for(ds, "dataset"))

    assert ds.content_sha == ref.content_sha and ref.content_sha
    assert ref.workspace_file_path in ex.files
    cur = json.loads(ex.files[".ockham/datasets/d1/curation.json"])
    assert cur["live_name"] == "unrate"
    assert cur["kind"] == "dataset"
    assert cur["created_at"] and cur["updated_at"]
    log_lines = ex.files[".ockham/datasets/d1/log.jsonl"].decode().strip().splitlines()
    assert len(log_lines) == 1
    assert json.loads(log_lines[0])["content_sha"] == ref.content_sha


@pytest.mark.asyncio
async def test_persist_artifact_records_variable_name_on_curation(tmp_path: Path) -> None:
    ex = FakeExecutor()
    df = pd.DataFrame({"x": [1]})
    payload = DataFrameObject.from_pandas(df, local_dir=tmp_path / "_dfo")
    ds = Dataset(logical_id="d1", title="x", live_name="d1", variable_name="cpi").with_payload(payload)
    await persist_artifact(
        ex,
        kind="dataset",
        artifact=ds,
        blob=render_artifact_bytes(ds, "dataset"),
        log_inputs=log_inputs_for(ds, "dataset"),
    )
    cur = json.loads(ex.files[".ockham/datasets/d1/curation.json"])
    assert cur["variable_name"] == "cpi"


@pytest.mark.asyncio
async def test_persist_artifact_is_idempotent_on_same_bytes(tmp_path: Path) -> None:
    ex = FakeExecutor()
    ds = _dataset(tmp_path)
    blob = render_artifact_bytes(ds, "dataset")
    # Different log_inputs on the second call: dedup is on content_sha, so the
    # second publish must NOT append a new entry (and the first entry's inputs win).
    await persist_artifact(ex, kind="dataset", artifact=ds, blob=blob, log_inputs={"notebooks": ["a"], "sources": []})
    await persist_artifact(ex, kind="dataset", artifact=ds, blob=blob, log_inputs={"notebooks": ["b"], "sources": []})
    log_lines = ex.files[".ockham/datasets/d1/log.jsonl"].decode().strip().splitlines()
    assert len(log_lines) == 1
    assert json.loads(log_lines[0])["inputs"]["notebooks"] == ["a"]


@pytest.mark.asyncio
async def test_persist_artifact_preserves_created_at(tmp_path: Path) -> None:
    ex = FakeExecutor()
    ex.files[".ockham/datasets/d1/curation.json"] = json.dumps(
        {"kind": "dataset", "logical_id": "d1", "created_at": "2020-01-01T00:00:00Z"}
    ).encode()
    ds = _dataset(tmp_path)
    await persist_artifact(ex, kind="dataset", artifact=ds, blob=render_artifact_bytes(ds, "dataset"), log_inputs={})
    cur = json.loads(ex.files[".ockham/datasets/d1/curation.json"])
    assert cur["created_at"] == "2020-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_persist_artifact_requires_logical_id(tmp_path: Path) -> None:
    ex = FakeExecutor()
    ds = Dataset(title="x").with_payload(_dataset(tmp_path).payload)
    with pytest.raises(ValueError, match="logical_id"):
        await persist_artifact(ex, kind="dataset", artifact=ds, blob=b"x", log_inputs={})


# ---------------------------------------------------------------------------
# persist_notebook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_notebook_writes_triplet(tmp_path: Path) -> None:
    ex = FakeExecutor()
    code = "import pandas as pd\nresult_df = pd.DataFrame({'x': [1]})\n"
    ref = ArtifactRef(kind="notebook", logical_id="unrate", content_sha=notebook_content_sha(code))

    out = await persist_notebook(ex, ref=ref, code=code, notebook_path="notebooks/unrate.py")

    assert out is ref
    assert f".ockham/notebooks/unrate/{ref.content_sha}.py" in ex.files
    cur = json.loads(ex.files[".ockham/notebooks/unrate/curation.json"])
    assert cur["kind"] == "notebook"
    assert cur["live_name"] == "unrate"
    log_lines = ex.files[".ockham/notebooks/unrate/log.jsonl"].decode().strip().splitlines()
    assert json.loads(log_lines[0])["inputs"] == {"path": "notebooks/unrate.py"}


@pytest.mark.asyncio
async def test_persist_notebook_snapshot_is_readable_by_read_latest(tmp_path: Path) -> None:
    from parsimony_agents.notebook_io import deserialize_notebook, read_latest_notebook

    ex = FakeExecutor()
    code = "result_df = 1\n"
    ref = ArtifactRef(kind="notebook", logical_id="nb", content_sha=notebook_content_sha(code))
    await persist_notebook(ex, ref=ref, code=code, notebook_path="notebooks/nb.py")

    raw, csha = await read_latest_notebook(ex, logical_id="nb")
    assert csha == ref.content_sha
    script = deserialize_notebook(raw, path=f".ockham/notebooks/nb/{csha}.py")
    assert "result_df = 1" in script.code


@pytest.mark.asyncio
async def test_persist_notebook_rejects_non_notebook_ref() -> None:
    ex = FakeExecutor()
    ref = ArtifactRef(kind="dataset", logical_id="d1", content_sha="x")
    with pytest.raises(ValueError, match="notebook"):
        await persist_notebook(ex, ref=ref, code="x", notebook_path="notebooks/x.py")


# ---------------------------------------------------------------------------
# verify-after-write (snapshot integrity floor)
# ---------------------------------------------------------------------------


class _CorruptingExecutor(FakeExecutor):
    """Reads a snapshot back as corrupt bytes to exercise verify-after-write.

    The curation/log paths read back faithfully; only the immutable
    ``<content_sha>.<ext>`` snapshot read is corrupted on its *first* read
    (the verify-after-write round-trip), so the integrity check trips.
    """

    def __init__(self, corrupt_suffixes: tuple[str, ...]) -> None:
        super().__init__()
        self._corrupt_suffixes = corrupt_suffixes

    async def read_workspace_file(self, path: str) -> bytes:
        data = await super().read_workspace_file(path)
        if path.endswith(self._corrupt_suffixes):
            return data + b"\x00corrupted"
        return data


@pytest.mark.asyncio
async def test_persist_artifact_verify_after_write_raises_on_corruption(tmp_path: Path) -> None:
    ex = _CorruptingExecutor(corrupt_suffixes=(".parquet",))
    ds = _dataset(tmp_path)
    with pytest.raises(SnapshotIntegrityError, match="corrupt"):
        await persist_artifact(
            ex,
            kind="dataset",
            artifact=ds,
            blob=render_artifact_bytes(ds, "dataset"),
            log_inputs=log_inputs_for(ds, "dataset"),
        )
    # The bad write must not be treated as durable: no curation/log committed.
    assert ".ockham/datasets/d1/curation.json" not in ex.files


@pytest.mark.asyncio
async def test_persist_notebook_verify_after_write_raises_on_corruption() -> None:
    ex = _CorruptingExecutor(corrupt_suffixes=(".py",))
    code = "result_df = 1\n"
    ref = ArtifactRef(kind="notebook", logical_id="nb", content_sha=notebook_content_sha(code))
    with pytest.raises(SnapshotIntegrityError, match="corrupt"):
        await persist_notebook(ex, ref=ref, code=code, notebook_path="notebooks/nb.py")
    assert ".ockham/notebooks/nb/curation.json" not in ex.files


@pytest.mark.asyncio
async def test_persist_artifact_skip_on_exists_accepts_existing_snapshot(tmp_path: Path) -> None:
    """Documents the skip-on-exists assumption (verify-after-write retry gap).

    ``_write_snapshot_verified`` leaves an existing content-addressed path untouched
    and unverified — it trusts that "same path implies same bytes" because the
    executor writes atomically (tmp + replace), so a failed write never leaves a
    partial file at the final path. This pins that behaviour: a pre-existing
    snapshot is NOT overwritten or re-verified, so a (hypothetical, non-atomic
    backend) corrupt-but-present snapshot would be accepted on retry. If a future
    backend cannot guarantee atomic writes, this test must change alongside a
    re-verify in ``_write_snapshot_verified``.
    """
    ex = FakeExecutor()
    ds = _dataset(tmp_path)
    blob = render_artifact_bytes(ds, "dataset")
    snap_path = f".ockham/datasets/d1/{content_sha(blob)}.parquet"
    ex.files[snap_path] = b"pre-existing bytes (would be corrupt on a non-atomic backend)"

    await persist_artifact(ex, kind="dataset", artifact=ds, blob=blob, log_inputs=log_inputs_for(ds, "dataset"))

    # Left untouched (skip-on-exists), not re-verified.
    assert ex.files[snap_path] == b"pre-existing bytes (would be corrupt on a non-atomic backend)"


# ---------------------------------------------------------------------------
# report_validator chokepoint (trust boundary at write time)
# ---------------------------------------------------------------------------


def _report(*, pins: dict[str, ArtifactRef] | None = None) -> Report:
    return Report(
        logical_id="r1",
        title="My Report",
        description="d",
        live_name="myreport",
        markdown="# My Report\n\nSome prose.\n",
        live_name_pins=pins or {},
    )


@pytest.mark.asyncio
async def test_persist_report_validator_rejects_writes_nothing() -> None:
    """A rejecting report_validator stops the write before any byte lands."""
    ex = FakeExecutor()
    report = _report()
    blob = render_artifact_bytes(report, "report")

    def _reject(body: str, *, pin_map_keys: frozenset[str] | None = None) -> None:
        raise ValueError("nope")

    with pytest.raises(ReportValidationError, match="nope"):
        await persist_artifact(
            ex,
            kind="report",
            artifact=report,
            blob=blob,
            log_inputs=log_inputs_for(report, "report"),
            report_validator=_reject,
        )
    # Nothing — not the snapshot, not curation, not the log.
    assert ex.files == {}


@pytest.mark.asyncio
async def test_persist_report_validator_receives_pin_map_keys() -> None:
    """persist_artifact passes the snapshot's pin-map live_names to the validator."""
    ex = FakeExecutor()
    pins = {"unrate": ArtifactRef(kind="dataset", logical_id="d1", content_sha="abc")}
    report = _report(pins=pins)
    seen: dict[str, frozenset[str] | None] = {}

    def _capture(body: str, *, pin_map_keys: frozenset[str] | None = None) -> None:
        seen["keys"] = pin_map_keys

    await persist_artifact(
        ex,
        kind="report",
        artifact=report,
        blob=render_artifact_bytes(report, "report"),
        log_inputs=log_inputs_for(report, "report"),
        report_validator=_capture,
    )
    assert seen["keys"] == frozenset({"unrate"})
    assert ".ockham/reports/r1/curation.json" in ex.files  # clean body persisted


@pytest.mark.asyncio
async def test_persist_report_normalizes_host_validator_error() -> None:
    """Any host validator exception is normalized to ReportValidationError."""
    ex = FakeExecutor()
    report = _report()

    def _boom(body: str, *, pin_map_keys: frozenset[str] | None = None) -> None:
        raise RuntimeError("host blew up")

    with pytest.raises(ReportValidationError, match="host blew up"):
        await persist_artifact(
            ex,
            kind="report",
            artifact=report,
            blob=render_artifact_bytes(report, "report"),
            log_inputs=log_inputs_for(report, "report"),
            report_validator=_boom,
        )


@pytest.mark.asyncio
async def test_persist_dataset_ignores_report_validator(tmp_path: Path) -> None:
    """report_validator only gates reports; a dataset persist never invokes it."""
    ex = FakeExecutor()
    ds = _dataset(tmp_path)

    def _explode(body: str, *, pin_map_keys: frozenset[str] | None = None) -> None:
        raise AssertionError("validator must not run for datasets")

    await persist_artifact(
        ex,
        kind="dataset",
        artifact=ds,
        blob=render_artifact_bytes(ds, "dataset"),
        log_inputs=log_inputs_for(ds, "dataset"),
        report_validator=_explode,
    )
    assert ".ockham/datasets/d1/curation.json" in ex.files
