"""Tests for the refresh orchestrator (Phase R3).

Refresh re-derives an artifact bottom-up: data_objects via re-running
notebooks, datasets/charts via re-extracting kernel variables,
reports via rewriting markdown. The orchestrator persists each layer
through ``executor.write_workspace_file`` so the cascade can build new
parent refs against fresh child content_shas.

These tests exercise the orchestrator with a stub executor so they
don't depend on a real kernel or connectors. End-to-end coverage with
the actual ``CodeExecutor`` lives in the agent integration tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from parsimony_agents.artifacts import Chart, Dataset, Report
from parsimony_agents.chart_io import write_chart_bytes
from parsimony_agents.dataset_io import write_dataset_bytes
from parsimony_agents.execution.outputs import (
    DataFrameObject,
    FetchLogEntry,
    FigureObject,
    KernelOutput,
)
from parsimony_agents.identity import (
    ArtifactRef,
    chart_logical_id,
    content_sha,
    dataset_logical_id,
    report_logical_id,
)
from parsimony_agents.refresh import refresh_artifact

# ---------------------------------------------------------------------------
# Stub executor
# ---------------------------------------------------------------------------


class _StubExecutor:
    """Minimal executor stub for refresh.

    - Real local FS for read/write under ``cwd``.
    - Pluggable ``execute`` / ``get`` so tests control kernel state.
    - ``next_executes`` is a list of ``KernelOutput``s consumed in order;
      ``next_variables`` is a dict the next ``get`` call reads from.
    """

    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)
        self._cwd_path = cwd
        self.next_executes: list[KernelOutput] = []
        self.next_variables: dict[str, Any] = {}
        self.execute_calls: list[str] = []
        self.get_calls: list[str] = []

    def _resolve(self, path: str) -> Path:
        candidate = (self._cwd_path / path).resolve()
        candidate.relative_to(self._cwd_path.resolve())
        return candidate

    async def read_workspace_file(self, path: str) -> bytes:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(path)
        return p.read_bytes()

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            tmp.replace(p)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    async def execute(self, code: str) -> KernelOutput:
        self.execute_calls.append(code)
        if self.next_executes:
            return self.next_executes.pop(0)
        return KernelOutput(outputs=[], fetch_log=[])

    async def get(self, key: str) -> Any:
        self.get_calls.append(key)
        return self.next_variables.get(key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df(values: list[int]) -> DataFrameObject:
    return DataFrameObject.from_pandas(pd.DataFrame({"x": values}), local_dir=Path("/tmp"))


def _vega_spec(value: int) -> FigureObject:
    return FigureObject(
        value={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "line",
            "encoding": {"x": {"field": "x", "type": "quantitative"}},
            "data": {"values": [{"x": value}]},
        }
    )


def _write_notebook(executor: _StubExecutor, live_name: str, code: str) -> None:
    """Persist a notebook the way the production pipeline does.

    Writes the canonical content-addressed snapshot at
    ``.ockham/notebooks/<lid>/<csha>.py``, a curation sidecar, and a
    ``log.jsonl`` entry. Refresh reads from the snapshot via
    ``log.jsonl`` — it never touches the transient working copy at
    ``notebooks/<live_name>.py`` (deleted in production after persist).
    """
    from parsimony_agents.identity import notebook_content_sha

    csha = notebook_content_sha(code)
    snap = Path(executor.cwd) / f".ockham/notebooks/{live_name}/{csha}.py"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_bytes(code.encode("utf-8"))
    _write_log_entry(executor, "notebook", live_name, csha, {"path": f"notebooks/{live_name}.py"})
    _write_curation(executor, "notebook", live_name, title=live_name)


def _persist_dataset(
    executor: _StubExecutor,
    *,
    notebook_refs: list[ArtifactRef],
    source_refs: list[ArtifactRef],
    variable_name: str,
    title: str,
    df: pd.DataFrame,
) -> tuple[ArtifactRef, Dataset]:
    """Build + write a dataset snapshot at v1 (mirrors persist_return_artifact)."""
    lid = dataset_logical_id(
        notebook_refs=notebook_refs, variable_name=variable_name, source_refs=source_refs
    )
    payload = DataFrameObject.from_pandas(df, local_dir=Path(executor.cwd) / "_dfo")
    dataset = Dataset(
        logical_id=lid,
        title=title,
        notebook_refs=notebook_refs,
        source_refs=source_refs,
        variable_name=variable_name,
    )
    blob = write_dataset_bytes(dataset, payload)
    csha = content_sha(blob)
    dataset.content_sha = csha
    ref = ArtifactRef(kind="dataset", logical_id=lid, content_sha=csha)
    p = Path(executor.cwd) / ref.workspace_file_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    _write_log_entry(executor, "dataset", lid, csha, {
        "notebooks": [r.content_sha for r in notebook_refs],
        "sources": [r.content_sha for r in source_refs],
    })
    _write_curation(
        executor, "dataset", lid,
        title=title, variable_name=variable_name,
    )
    return ref, dataset.with_payload(payload)


def _persist_chart(
    executor: _StubExecutor,
    *,
    notebook_ref: ArtifactRef,
    source_dataset_refs: list[ArtifactRef],
    source_refs: list[ArtifactRef],
    variable_name: str,
    title: str,
    spec: FigureObject,
) -> tuple[ArtifactRef, Chart]:
    lid = chart_logical_id(
        notebook_ref=notebook_ref,
        chart_variable_name=variable_name,
        source_dataset_refs=source_dataset_refs,
        source_refs=source_refs,
    )
    chart = Chart(
        logical_id=lid,
        title=title,
        notebook_ref=notebook_ref,
        source_dataset_refs=source_dataset_refs,
        source_refs=source_refs,
        variable_name=variable_name,
    )
    blob = write_chart_bytes(chart, spec)
    csha = content_sha(blob)
    chart.content_sha = csha
    ref = ArtifactRef(kind="chart", logical_id=lid, content_sha=csha)
    p = Path(executor.cwd) / ref.workspace_file_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    _write_log_entry(executor, "chart", lid, csha, {
        "notebook": notebook_ref.content_sha,
        "source_datasets": [r.content_sha for r in source_dataset_refs],
        "sources": [r.content_sha for r in source_refs],
    })
    _write_curation(
        executor, "chart", lid,
        title=title, variable_name=variable_name,
    )
    return ref, chart.with_payload(spec)


def _persist_report(
    executor: _StubExecutor,
    *,
    pin_map: dict[str, ArtifactRef],
    title: str,
    markdown: str,
) -> tuple[ArtifactRef, Report]:
    """Persist a report snapshot using the new YAML-frontmatter shape.

    Caller supplies ``pin_map`` (live_name → ArtifactRef); body must
    reference each live_name via ``file://./charts/<n>.vl.json`` or
    ``file://./data/<n>.parquet``. The pin map is the sole source of
    truth for embeds — ``Report.embedded_refs`` derives from it.
    """
    from parsimony_agents.report_format import compose_snapshot

    embedded_refs = list(pin_map.values())
    lid = report_logical_id(embedded_refs=embedded_refs, title=title)
    snapshot_text = compose_snapshot(["html"], pin_map, markdown, title=title)
    blob = snapshot_text.encode("utf-8")
    csha = content_sha(blob)
    ref = ArtifactRef(kind="report", logical_id=lid, content_sha=csha)
    p = Path(executor.cwd) / ref.workspace_file_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    _write_log_entry(executor, "report", lid, csha, {
        "embedded": [r.content_sha for r in embedded_refs],
    })
    _write_curation(executor, "report", lid, title=title)
    report = Report(
        logical_id=lid, title=title, markdown=markdown,
        live_name_pins=pin_map,
        formats=["html"], content_sha=csha,
    )
    return ref, report


def _write_log_entry(
    executor: _StubExecutor,
    kind: str,
    logical_id: str,
    csha: str,
    inputs: dict[str, Any],
) -> None:
    log_path = Path(executor.cwd) / f".ockham/{kind}s/{logical_id}/log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": "2026-05-08T00:00:00Z", "content_sha": csha, "inputs": inputs}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def _write_curation(
    executor: _StubExecutor,
    kind: str,
    logical_id: str,
    *,
    title: str,
    variable_name: str | None = None,
) -> None:
    p = Path(executor.cwd) / f".ockham/{kind}s/{logical_id}/curation.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "kind": kind,
        "logical_id": logical_id,
        "title": title,
        "description": "",
        "tags": [],
        "notes": [],
        "live_name": title.lower().replace(" ", "_"),
        "created_at": "2026-05-08T00:00:00Z",
        "updated_at": "2026-05-08T00:00:00Z",
    }
    if variable_name is not None:
        payload["variable_name"] = variable_name
    p.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def _nb_ref(name: str, code: str) -> ArtifactRef:
    from parsimony_agents.identity import notebook_content_sha

    return ArtifactRef(
        kind="notebook", logical_id=name, content_sha=notebook_content_sha(code)
    )


def _do_ref(provenance_id: str, content: str) -> ArtifactRef:
    return ArtifactRef(
        kind="data_object",
        logical_id=f"do-{provenance_id}",
        content_sha=content_sha(content.encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# Tests: dataset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_dataset_idempotent_when_nothing_changes(tmp_path: Path) -> None:
    """Two consecutive refreshes with no upstream change → same content_sha both times."""
    executor = _StubExecutor(tmp_path)
    nb_code = "df = ...\n"
    _write_notebook(executor, "macro", nb_code)
    nb_ref = _nb_ref("macro", nb_code)
    do = _do_ref("gdp", "v1")

    df = pd.DataFrame({"x": [1, 2, 3]})
    ref_v1, _dataset = _persist_dataset(
        executor,
        notebook_refs=[nb_ref],
        source_refs=[do],
        variable_name="df",
        title="GDP",
        df=df,
    )

    # Stub: re-running notebook returns the same fetch_log; kernel get
    # returns the same DataFrameObject.
    df_payload = DataFrameObject.from_pandas(df, local_dir=tmp_path / "_dfo1")
    executor.next_variables = {"df": df_payload}
    executor.next_executes = [
        KernelOutput(outputs=[], fetch_log=[
            FetchLogEntry(
                provenance={"source": "fred", "source_description": "FRED", "params": {"id": "GDP"}},
                row_count=3, column_names=["x"], columns=[],
                data_object_ref=do,
                version=1,
            ),
        ])
    ]

    out_a = await refresh_artifact(ref_v1, executor=executor)

    df_payload2 = DataFrameObject.from_pandas(df, local_dir=tmp_path / "_dfo2")
    executor.next_variables = {"df": df_payload2}
    executor.next_executes = [
        KernelOutput(outputs=[], fetch_log=[
            FetchLogEntry(
                provenance={"source": "fred", "source_description": "FRED", "params": {"id": "GDP"}},
                row_count=3, column_names=["x"], columns=[],
                data_object_ref=do,
                version=1,
            ),
        ])
    ]
    out_b = await refresh_artifact(ref_v1, executor=executor)

    assert out_a == out_b
    assert out_a.logical_id == ref_v1.logical_id
    # Original content_sha should match — same bytes round-trip.
    assert out_a.content_sha == ref_v1.content_sha


@pytest.mark.asyncio
async def test_refresh_dataset_advances_when_kernel_data_changes(tmp_path: Path) -> None:
    """Different DataFrame in the kernel → new content_sha; same logical_id; v2 in log."""
    executor = _StubExecutor(tmp_path)
    nb_code = "df = ...\n"
    _write_notebook(executor, "macro", nb_code)
    nb_ref = _nb_ref("macro", nb_code)

    ref_v1, _ = _persist_dataset(
        executor,
        notebook_refs=[nb_ref],
        source_refs=[],
        variable_name="df",
        title="Macro",
        df=pd.DataFrame({"x": [1, 2, 3]}),
    )

    fresh_payload = DataFrameObject.from_pandas(
        pd.DataFrame({"x": [10, 20, 30]}), local_dir=tmp_path / "_dfo"
    )
    executor.next_variables = {"df": fresh_payload}
    executor.next_executes = [KernelOutput(outputs=[], fetch_log=[])]

    out = await refresh_artifact(ref_v1, executor=executor)
    assert out.logical_id == ref_v1.logical_id
    assert out.content_sha != ref_v1.content_sha

    log = (tmp_path / f".ockham/datasets/{ref_v1.logical_id}/log.jsonl").read_text()
    assert log.count("\n") == 2  # v1 + v2


@pytest.mark.asyncio
async def test_refresh_dataset_missing_variable_name_raises(tmp_path: Path) -> None:
    """Datasets persisted before R2 (no variable_name) cannot refresh."""
    executor = _StubExecutor(tmp_path)
    nb_code = "df = ...\n"
    nb_ref = _nb_ref("macro", nb_code)
    _write_notebook(executor, "macro", nb_code)

    # Persist a dataset *without* variable_name (simulate pre-R2 artifact).
    df = pd.DataFrame({"x": [1]})
    payload = DataFrameObject.from_pandas(df, local_dir=tmp_path / "_dfo")
    lid = "legacy-dataset"
    dataset = Dataset(
        logical_id=lid,
        title="Legacy",
        notebook_refs=[nb_ref],
        source_refs=[],
        variable_name="",  # missing!
    )
    blob = write_dataset_bytes(dataset, payload)
    csha = content_sha(blob)
    ref = ArtifactRef(kind="dataset", logical_id=lid, content_sha=csha)
    p = tmp_path / ref.workspace_file_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)

    with pytest.raises(ValueError, match="predates R2"):
        await refresh_artifact(ref, executor=executor)


@pytest.mark.asyncio
async def test_refresh_dataset_missing_snapshot_raises(tmp_path: Path) -> None:
    executor = _StubExecutor(tmp_path)
    ref = ArtifactRef(kind="dataset", logical_id="x", content_sha="missing")
    with pytest.raises(ValueError, match="snapshot bytes missing"):
        await refresh_artifact(ref, executor=executor)


@pytest.mark.asyncio
async def test_refresh_dataset_kernel_variable_missing_raises(tmp_path: Path) -> None:
    """If the re-run notebook doesn't produce the expected variable, refresh errors clearly."""
    executor = _StubExecutor(tmp_path)
    nb_code = "df = 1\n"
    _write_notebook(executor, "macro", nb_code)
    nb_ref = _nb_ref("macro", nb_code)
    ref, _ = _persist_dataset(
        executor,
        notebook_refs=[nb_ref], source_refs=[],
        variable_name="df", title="x",
        df=pd.DataFrame({"x": [1]}),
    )
    executor.next_variables = {}  # df not in kernel after re-run
    executor.next_executes = [KernelOutput(outputs=[], fetch_log=[])]
    with pytest.raises(ValueError, match="not produced"):
        await refresh_artifact(ref, executor=executor)


# ---------------------------------------------------------------------------
# Tests: chart cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_chart_cascades_into_source_dataset(tmp_path: Path) -> None:
    """Refreshing a chart triggers refresh of its source dataset(s)."""
    executor = _StubExecutor(tmp_path)
    nb_code_ds = "ds = ...\n"
    nb_code_chart = "fig = ...\n"
    _write_notebook(executor, "ds_nb", nb_code_ds)
    _write_notebook(executor, "chart_nb", nb_code_chart)
    nb_ds_ref = _nb_ref("ds_nb", nb_code_ds)
    nb_chart_ref = _nb_ref("chart_nb", nb_code_chart)

    ds_ref, _ = _persist_dataset(
        executor,
        notebook_refs=[nb_ds_ref], source_refs=[],
        variable_name="ds", title="Source",
        df=pd.DataFrame({"x": [1]}),
    )
    chart_ref, _ = _persist_chart(
        executor,
        notebook_ref=nb_chart_ref,
        source_dataset_refs=[ds_ref], source_refs=[],
        variable_name="fig", title="Trend", spec=_vega_spec(1),
    )

    # When refreshing the chart, the orchestrator first refreshes the
    # source dataset (re-runs ds_nb, gets new ``ds``), then re-runs
    # chart_nb and gets ``fig`` with the new spec.
    fresh_ds = DataFrameObject.from_pandas(
        pd.DataFrame({"x": [42]}), local_dir=tmp_path / "_dfo"
    )
    fresh_fig = _vega_spec(42)
    executor.next_variables = {"ds": fresh_ds, "fig": fresh_fig}
    executor.next_executes = [
        KernelOutput(outputs=[], fetch_log=[]),  # ds re-run
        KernelOutput(outputs=[], fetch_log=[]),  # chart re-run
    ]

    out = await refresh_artifact(chart_ref, executor=executor)

    # Chart logical_id is stable; content_sha advances because the
    # source_dataset_refs now point at a new dataset content_sha.
    assert out.logical_id == chart_ref.logical_id
    assert out.content_sha != chart_ref.content_sha

    # Both notebooks were re-run (one for the source dataset, one for
    # the chart). The deserialize_notebook step may trim trailing
    # whitespace, so match by substring rather than exact bytes.
    assert len(executor.execute_calls) == 2
    assert any("ds = ..." in c for c in executor.execute_calls)
    assert any("fig = ..." in c for c in executor.execute_calls)

    # Source dataset advanced under the same logical_id.
    ds_log = (tmp_path / f".ockham/datasets/{ds_ref.logical_id}/log.jsonl").read_text()
    assert ds_log.count("\n") == 2


# ---------------------------------------------------------------------------
# Tests: report cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_report_rewrites_markdown_with_new_content_shas(
    tmp_path: Path,
) -> None:
    """Refresh on a report → embedded chart refreshes; markdown rewritten."""
    executor = _StubExecutor(tmp_path)
    nb_code_ds = "ds = ...\n"
    nb_code_chart = "fig = ...\n"
    _write_notebook(executor, "ds_nb", nb_code_ds)
    _write_notebook(executor, "chart_nb", nb_code_chart)
    nb_ds_ref = _nb_ref("ds_nb", nb_code_ds)
    nb_chart_ref = _nb_ref("chart_nb", nb_code_chart)

    ds_ref, _ = _persist_dataset(
        executor,
        notebook_refs=[nb_ds_ref], source_refs=[],
        variable_name="ds", title="Source",
        df=pd.DataFrame({"x": [1]}),
    )
    chart_ref, _ = _persist_chart(
        executor,
        notebook_ref=nb_chart_ref,
        source_dataset_refs=[ds_ref], source_refs=[],
        variable_name="fig", title="Trend", spec=_vega_spec(1),
    )

    # Body addresses the embed by live_name; pin map ties live_name to
    # the chart's frozen ArtifactRef. After refresh the body is
    # byte-stable (same live_name), only the pin map's ref drifts.
    markdown = (
        "# Q1 review\n\n"
        "![chart](file://./charts/trend.vl.json)\n"
    )
    report_ref, _ = _persist_report(
        executor,
        pin_map={"trend": chart_ref},
        title="Q1 review", markdown=markdown,
    )

    fresh_ds = DataFrameObject.from_pandas(
        pd.DataFrame({"x": [99]}), local_dir=tmp_path / "_dfo"
    )
    fresh_fig = _vega_spec(99)
    executor.next_variables = {"ds": fresh_ds, "fig": fresh_fig}
    executor.next_executes = [
        KernelOutput(outputs=[], fetch_log=[]),  # ds re-run
        KernelOutput(outputs=[], fetch_log=[]),  # chart re-run
    ]

    out = await refresh_artifact(report_ref, executor=executor)
    assert out.logical_id == report_ref.logical_id
    assert out.content_sha != report_ref.content_sha

    from parsimony_agents.report_format import parse_snapshot
    new_text = (tmp_path / out.workspace_file_path).read_text()
    new_snap = parse_snapshot(new_text)
    # Body is byte-stable — embed URI still names the live_name.
    assert "file://./charts/trend.vl.json" in new_snap.body
    # Pin map now points at the refreshed chart snapshot (same lid, new csha).
    assert new_snap.pins["trend"].logical_id == chart_ref.logical_id
    assert new_snap.pins["trend"].content_sha != chart_ref.content_sha


# ---------------------------------------------------------------------------
# Tests: kind validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_unsupported_kind_raises(tmp_path: Path) -> None:
    """Refreshing a notebook ref or data_object ref is explicitly rejected."""
    executor = _StubExecutor(tmp_path)
    nb_ref = ArtifactRef(kind="notebook", logical_id="x", content_sha="y")
    do_ref = ArtifactRef(kind="data_object", logical_id="x", content_sha="y")
    with pytest.raises(ValueError, match="unsupported kind"):
        await refresh_artifact(nb_ref, executor=executor)
    with pytest.raises(ValueError, match="unsupported kind"):
        await refresh_artifact(do_ref, executor=executor)


@pytest.mark.asyncio
async def test_refresh_reads_notebook_from_snapshot_not_working_copy(
    tmp_path: Path,
) -> None:
    """Regression: refresh must read notebook bytes from .ockham/, not notebooks/<live_name>.py.

    The working copy is deleted by the streaming layer after persist, so
    a refresh in a subsequent turn cannot rely on it. This test ensures
    refresh works even when ``notebooks/<live_name>.py`` is absent —
    the snapshot tree is the canonical source of truth.
    """
    executor = _StubExecutor(tmp_path)
    nb_code = "df = ...\n"
    _write_notebook(executor, "macro", nb_code)
    nb_ref = _nb_ref("macro", nb_code)

    # Sanity: working copy does NOT exist (production state post-persist).
    assert not (tmp_path / "notebooks" / "macro.py").exists()
    # But the snapshot does.
    assert any((tmp_path / ".ockham/notebooks/macro").glob("*.py"))

    ref, _ = _persist_dataset(
        executor,
        notebook_refs=[nb_ref], source_refs=[],
        variable_name="df", title="Macro",
        df=pd.DataFrame({"x": [1]}),
    )

    fresh = DataFrameObject.from_pandas(
        pd.DataFrame({"x": [99]}), local_dir=tmp_path / "_dfo"
    )
    executor.next_variables = {"df": fresh}
    executor.next_executes = [KernelOutput(outputs=[], fetch_log=[])]

    out = await refresh_artifact(ref, executor=executor)
    assert out.logical_id == ref.logical_id
    assert out.content_sha != ref.content_sha


@pytest.mark.asyncio
async def test_refresh_errors_when_notebook_log_missing(tmp_path: Path) -> None:
    """Refresh on a dataset whose notebook was never persisted → clear error."""
    executor = _StubExecutor(tmp_path)
    # Build a dataset whose notebook_refs point at a logical_id that has
    # no .ockham/notebooks/<lid>/log.jsonl on disk.
    nb_ref = ArtifactRef(kind="notebook", logical_id="ghost", content_sha="abc")
    ref, _ = _persist_dataset(
        executor,
        notebook_refs=[nb_ref], source_refs=[],
        variable_name="df", title="orphan",
        df=pd.DataFrame({"x": [1]}),
    )
    with pytest.raises(ValueError, match="no persisted snapshot"):
        await refresh_artifact(ref, executor=executor)


@pytest.mark.asyncio
async def test_refresh_dataset_data_object_source_uses_fresh_fetch(tmp_path: Path) -> None:
    """Notebook re-run produces a new data_object ref → dataset's source_refs advance."""
    executor = _StubExecutor(tmp_path)
    nb_code = "df = ...\n"
    _write_notebook(executor, "macro", nb_code)
    nb_ref = _nb_ref("macro", nb_code)

    do_v1 = _do_ref("gdp", "v1")
    ref, _ = _persist_dataset(
        executor,
        notebook_refs=[nb_ref], source_refs=[do_v1],
        variable_name="df", title="GDP",
        df=pd.DataFrame({"x": [1]}),
    )

    do_v2 = ArtifactRef(
        kind="data_object",
        logical_id=do_v1.logical_id,  # same logical_id
        content_sha="new-csha-from-fresh-fetch",
    )
    fresh_df = DataFrameObject.from_pandas(
        pd.DataFrame({"x": [42]}), local_dir=tmp_path / "_dfo"
    )
    executor.next_variables = {"df": fresh_df}
    executor.next_executes = [KernelOutput(outputs=[], fetch_log=[
        FetchLogEntry(
            provenance={"source": "fred", "source_description": "FRED", "params": {"id": "GDP"}},
            row_count=1, column_names=["x"], columns=[],
            data_object_ref=do_v2,
            version=2,
        ),
    ])]

    out = await refresh_artifact(ref, executor=executor)
    # logical_id unchanged.
    assert out.logical_id == ref.logical_id
    # log entry pins the new data_object content_sha.
    log_text = (tmp_path / f".ockham/datasets/{ref.logical_id}/log.jsonl").read_text()
    assert "new-csha-from-fresh-fetch" in log_text
