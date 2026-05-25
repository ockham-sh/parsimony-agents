"""Tests for :mod:`parsimony_agents.closure`.

Two primitives under test:

- :func:`child_refs` — single-step typed-edge enumeration. Per kind:
  reports follow ``pin_map`` values; charts follow notebook_ref +
  source_dataset_refs + source_refs; datasets follow notebook_refs +
  source_refs; data_object and notebook are leaves.

- :func:`enumerate_closure` — transitive post-order DFS. Returns
  dependencies before dependents; deduplicates by ``(kind, logical_id,
  content_sha)``; cycle-safe.

The persist helpers mirror those in ``test_refresh.py`` (same artifact
shapes, same on-disk layout); kept inline here so closure tests don't
take a fixture dependency on refresh tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from parsimony_agents.artifacts import Chart, Dataset
from parsimony_agents.chart_io import write_chart_bytes
from parsimony_agents.closure import child_refs, enumerate_closure
from parsimony_agents.dataset_io import write_dataset_bytes
from parsimony_agents.execution.outputs import DataFrameObject, FigureObject
from parsimony_agents.identity import (
    ArtifactRef,
    chart_logical_id,
    content_sha,
    dataset_logical_id,
    notebook_content_sha,
    report_logical_id,
)

# ---------------------------------------------------------------------------
# Read-only stub executor — closure never writes.
# ---------------------------------------------------------------------------


class _ReadOnlyExecutor:
    """Minimal executor surface for closure: just ``read_workspace_file``."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = str(cwd)
        self._cwd_path = cwd

    async def read_workspace_file(self, path: str) -> bytes:
        target = (self._cwd_path / path).resolve()
        target.relative_to(self._cwd_path.resolve())
        if not target.exists():
            raise FileNotFoundError(path)
        return target.read_bytes()


# ---------------------------------------------------------------------------
# Persist helpers (mirrors test_refresh.py)
# ---------------------------------------------------------------------------


def _vega_spec(value: int) -> FigureObject:
    return FigureObject(
        value={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "mark": "line",
            "encoding": {"x": {"field": "x", "type": "quantitative"}},
            "data": {"values": [{"x": value}]},
        }
    )


def _nb_ref(name: str, code: str) -> ArtifactRef:
    return ArtifactRef(
        kind="notebook", logical_id=name, content_sha=notebook_content_sha(code)
    )


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


def _write_notebook(executor: _ReadOnlyExecutor, live_name: str, code: str) -> None:
    csha = notebook_content_sha(code)
    _write_bytes(executor, f".ockham/notebooks/{live_name}/{csha}.py", code.encode("utf-8"))
    _write_log(executor, "notebook", live_name, csha, {"path": f"notebooks/{live_name}.py"})


def _persist_dataset(
    executor: _ReadOnlyExecutor,
    *,
    notebook_refs: list[ArtifactRef],
    source_refs: list[ArtifactRef],
    variable_name: str,
    title: str,
    df: pd.DataFrame,
) -> ArtifactRef:
    lid = dataset_logical_id(
        notebook_refs=notebook_refs,
        variable_name=variable_name,
        source_refs=source_refs,
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
    ref = ArtifactRef(kind="dataset", logical_id=lid, content_sha=csha)
    _write_bytes(executor, ref.workspace_file_path, blob)
    _write_log(executor, "dataset", lid, csha, {})
    return ref


def _persist_chart(
    executor: _ReadOnlyExecutor,
    *,
    notebook_ref: ArtifactRef,
    source_dataset_refs: list[ArtifactRef],
    source_refs: list[ArtifactRef],
    variable_name: str,
    title: str,
    spec: FigureObject,
) -> ArtifactRef:
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
    ref = ArtifactRef(kind="chart", logical_id=lid, content_sha=csha)
    _write_bytes(executor, ref.workspace_file_path, blob)
    _write_log(executor, "chart", lid, csha, {})
    return ref


def _persist_report(
    executor: _ReadOnlyExecutor,
    *,
    pin_map: dict[str, ArtifactRef],
    title: str,
    markdown: str,
) -> ArtifactRef:
    from parsimony_agents.report_format import compose_snapshot

    embedded_refs = list(pin_map.values())
    lid = report_logical_id(embedded_refs=embedded_refs, title=title)
    snapshot_text = compose_snapshot(["html"], pin_map, markdown, title=title)
    blob = snapshot_text.encode("utf-8")
    csha = content_sha(blob)
    ref = ArtifactRef(kind="report", logical_id=lid, content_sha=csha)
    _write_bytes(executor, ref.workspace_file_path, blob)
    _write_log(executor, "report", lid, csha, {})
    return ref


def _write_log(
    executor: _ReadOnlyExecutor,
    kind: str,
    logical_id: str,
    csha: str,
    inputs: dict[str, Any],
) -> None:
    log_path = Path(executor.cwd) / f".ockham/{kind}s/{logical_id}/log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": "2026-05-23T00:00:00Z", "content_sha": csha, "inputs": inputs}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Helpers for asserting topological order
# ---------------------------------------------------------------------------


def _index_of(refs: list[ArtifactRef], target: ArtifactRef) -> int:
    """Index of ``target`` in ``refs`` by ``(kind, logical_id, content_sha)``."""
    for i, r in enumerate(refs):
        if (
            r.kind == target.kind
            and r.logical_id == target.logical_id
            and r.content_sha == target.content_sha
        ):
            return i
    raise AssertionError(f"ref not in closure: {target}")


def _assert_before(closure: list[ArtifactRef], dep: ArtifactRef, dependent: ArtifactRef) -> None:
    """Assert ``dep`` precedes ``dependent`` in a topological closure."""
    assert _index_of(closure, dep) < _index_of(closure, dependent), (
        f"{dep.kind}/{dep.logical_id} must come before "
        f"{dependent.kind}/{dependent.logical_id} in topological order"
    )


# ---------------------------------------------------------------------------
# child_refs — per-kind edge enumeration
# ---------------------------------------------------------------------------


class TestChildRefs:
    @pytest.mark.asyncio
    async def test_notebook_is_a_leaf(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        nb = _nb_ref("nb_a", "x = 1")
        _write_notebook(executor, "nb_a", "x = 1")
        # Notebook bytes don't carry a persisted fetch_log — kernel-run state
        # is not part of the snapshot. Closure walker honours this and emits
        # the notebook as a leaf.
        assert await child_refs(nb, executor=executor) == []

    @pytest.mark.asyncio
    async def test_data_object_is_a_leaf_without_reading(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        do = _do_ref("connector_x", "raw")
        # Note: no on-disk parquet written. data_object leaves don't need
        # the snapshot to be present — they have no source refs to discover.
        assert await child_refs(do, executor=executor) == []

    @pytest.mark.asyncio
    async def test_dataset_emits_notebooks_then_source_refs(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        nb_code = "import pandas as pd\nds = pd.DataFrame({'x': [1, 2]})"
        nb = _nb_ref("etl", nb_code)
        _write_notebook(executor, "etl", nb_code)
        do = _do_ref("src_a", "rows")
        ds_ref = _persist_dataset(
            executor,
            notebook_refs=[nb],
            source_refs=[do],
            variable_name="ds",
            title="My Dataset",
            df=pd.DataFrame({"x": [1, 2]}),
        )

        refs = await child_refs(ds_ref, executor=executor)
        assert [r.kind for r in refs] == ["notebook", "data_object"]
        assert refs[0].logical_id == nb.logical_id
        assert refs[1].logical_id == do.logical_id

    @pytest.mark.asyncio
    async def test_chart_emits_notebook_then_source_datasets_then_source_refs(
        self, tmp_path: Path
    ) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        nb_code = "fig = {'mark': 'line'}"
        nb = _nb_ref("chart_nb", nb_code)
        _write_notebook(executor, "chart_nb", nb_code)
        ds_nb_code = "import pandas as pd\nds = pd.DataFrame({'x': [1]})"
        ds_nb = _nb_ref("etl", ds_nb_code)
        _write_notebook(executor, "etl", ds_nb_code)
        ds = _persist_dataset(
            executor,
            notebook_refs=[ds_nb],
            source_refs=[],
            variable_name="ds",
            title="Source Dataset",
            df=pd.DataFrame({"x": [1]}),
        )
        do = _do_ref("uncommon_path", "raw")
        chart_ref = _persist_chart(
            executor,
            notebook_ref=nb,
            source_dataset_refs=[ds],
            source_refs=[do],
            variable_name="fig",
            title="Bar",
            spec=_vega_spec(1),
        )

        refs = await child_refs(chart_ref, executor=executor)
        # Declared-field order: notebook_ref first, then source_dataset_refs,
        # then source_refs.
        assert [r.kind for r in refs] == ["notebook", "dataset", "data_object"]
        assert refs[0].logical_id == nb.logical_id
        assert refs[1].logical_id == ds.logical_id
        assert refs[2].logical_id == do.logical_id

    @pytest.mark.asyncio
    async def test_report_emits_pin_map_values(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        ds_nb_code = "import pandas as pd\nds = pd.DataFrame({'x': [1]})"
        ds_nb = _nb_ref("etl", ds_nb_code)
        _write_notebook(executor, "etl", ds_nb_code)
        ds = _persist_dataset(
            executor,
            notebook_refs=[ds_nb],
            source_refs=[],
            variable_name="ds",
            title="ds",
            df=pd.DataFrame({"x": [1]}),
        )
        chart_nb_code = "fig = {'mark': 'bar'}"
        chart_nb = _nb_ref("chart_nb", chart_nb_code)
        _write_notebook(executor, "chart_nb", chart_nb_code)
        ch = _persist_chart(
            executor,
            notebook_ref=chart_nb,
            source_dataset_refs=[ds],
            source_refs=[],
            variable_name="fig",
            title="ch",
            spec=_vega_spec(1),
        )
        report = _persist_report(
            executor,
            pin_map={"my_chart": ch, "my_data": ds},
            title="R",
            markdown=(
                "# R\n\n"
                "![chart](file://./charts/my_chart.vl.json)\n\n"
                "![data](file://./data/my_data.parquet)\n"
            ),
        )

        refs = await child_refs(report, executor=executor)
        # Both pin_map values appear; order matches pin_map insertion.
        assert {(r.kind, r.logical_id) for r in refs} == {
            ("chart", ch.logical_id),
            ("dataset", ds.logical_id),
        }

    @pytest.mark.asyncio
    async def test_missing_snapshot_raises_named_error(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        ghost = ArtifactRef(kind="dataset", logical_id="nope", content_sha="a" * 64)
        with pytest.raises(ValueError, match="closure: snapshot bytes missing"):
            await child_refs(ghost, executor=executor)

    @pytest.mark.asyncio
    async def test_unsupported_kind_raises(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        # ArtifactRef rejects unknown kinds at construction, so synthesise a
        # bare object with the right shape to exercise the closure guard.
        class _FakeRef:
            kind = "mystery"
            logical_id = "x"
            content_sha = "y"
            workspace_file_path = ".ockham/mystery/x/y"

        with pytest.raises(ValueError, match="closure: unsupported kind"):
            await child_refs(_FakeRef(), executor=executor)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# enumerate_closure — transitive post-order DFS
# ---------------------------------------------------------------------------


class TestEnumerateClosure:
    @pytest.mark.asyncio
    async def test_single_leaf_returns_self(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        do = _do_ref("only_one", "rows")
        closure = await enumerate_closure(do, executor=executor)
        assert closure == [do]

    @pytest.mark.asyncio
    async def test_dataset_with_data_object_source(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        nb_code = "ds = 1"
        nb = _nb_ref("etl", nb_code)
        _write_notebook(executor, "etl", nb_code)
        do = _do_ref("connector_y", "rows")
        ds = _persist_dataset(
            executor,
            notebook_refs=[nb],
            source_refs=[do],
            variable_name="ds",
            title="ds",
            df=pd.DataFrame({"x": [1]}),
        )

        closure = await enumerate_closure(ds, executor=executor)
        # Leaves before the dataset; root last.
        assert closure[-1] == ds
        _assert_before(closure, nb, ds)
        _assert_before(closure, do, ds)
        assert len(closure) == 3

    @pytest.mark.asyncio
    async def test_full_chain_report_chart_dataset_notebook_data_object(
        self, tmp_path: Path
    ) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        # Source notebook + data_object
        ds_nb_code = "import pandas as pd\nds = pd.DataFrame({'x': [1]})"
        ds_nb = _nb_ref("etl", ds_nb_code)
        _write_notebook(executor, "etl", ds_nb_code)
        do = _do_ref("source_z", "rows")
        # Dataset
        ds = _persist_dataset(
            executor,
            notebook_refs=[ds_nb],
            source_refs=[do],
            variable_name="ds",
            title="ds",
            df=pd.DataFrame({"x": [1]}),
        )
        # Chart on top
        ch_nb_code = "fig = {'mark': 'bar'}"
        ch_nb = _nb_ref("chart_nb", ch_nb_code)
        _write_notebook(executor, "chart_nb", ch_nb_code)
        ch = _persist_chart(
            executor,
            notebook_ref=ch_nb,
            source_dataset_refs=[ds],
            source_refs=[],
            variable_name="fig",
            title="ch",
            spec=_vega_spec(1),
        )
        # Report wrapping the chart and citing the dataset
        report = _persist_report(
            executor,
            pin_map={"my_chart": ch, "my_data": ds},
            title="R",
            markdown=(
                "# R\n\n"
                "![chart](file://./charts/my_chart.vl.json)\n\n"
                "![data](file://./data/my_data.parquet)\n"
            ),
        )

        closure = await enumerate_closure(report, executor=executor)
        # Every level is present exactly once.
        assert {(r.kind, r.logical_id) for r in closure} == {
            ("notebook", "etl"),
            ("data_object", do.logical_id),
            ("dataset", ds.logical_id),
            ("notebook", "chart_nb"),
            ("chart", ch.logical_id),
            ("report", report.logical_id),
        }
        assert closure[-1] == report
        # Topological invariants.
        _assert_before(closure, ds_nb, ds)
        _assert_before(closure, do, ds)
        _assert_before(closure, ds, ch)
        _assert_before(closure, ch_nb, ch)
        _assert_before(closure, ch, report)
        _assert_before(closure, ds, report)

    @pytest.mark.asyncio
    async def test_diamond_dag_deduplicates_shared_dependency(
        self, tmp_path: Path
    ) -> None:
        """Two charts share a dataset → dataset appears once in the closure."""
        executor = _ReadOnlyExecutor(tmp_path)
        ds_nb_code = "ds = 1"
        ds_nb = _nb_ref("etl", ds_nb_code)
        _write_notebook(executor, "etl", ds_nb_code)
        ds = _persist_dataset(
            executor,
            notebook_refs=[ds_nb],
            source_refs=[],
            variable_name="ds",
            title="shared",
            df=pd.DataFrame({"x": [1]}),
        )
        ch1_nb_code = "fig = {'mark': 'line'}"
        ch1_nb = _nb_ref("ch1_nb", ch1_nb_code)
        _write_notebook(executor, "ch1_nb", ch1_nb_code)
        ch1 = _persist_chart(
            executor,
            notebook_ref=ch1_nb,
            source_dataset_refs=[ds],
            source_refs=[],
            variable_name="fig",
            title="ch1",
            spec=_vega_spec(1),
        )
        ch2_nb_code = "fig = {'mark': 'bar'}"
        ch2_nb = _nb_ref("ch2_nb", ch2_nb_code)
        _write_notebook(executor, "ch2_nb", ch2_nb_code)
        ch2 = _persist_chart(
            executor,
            notebook_ref=ch2_nb,
            source_dataset_refs=[ds],
            source_refs=[],
            variable_name="fig",
            title="ch2",
            spec=_vega_spec(2),
        )
        report = _persist_report(
            executor,
            pin_map={"line": ch1, "bar": ch2},
            title="R",
            markdown=(
                "# R\n\n"
                "![](file://./charts/line.vl.json)\n\n"
                "![](file://./charts/bar.vl.json)\n"
            ),
        )

        closure = await enumerate_closure(report, executor=executor)
        # Shared dataset appears exactly once.
        ds_count = sum(1 for r in closure if r.kind == "dataset" and r.logical_id == ds.logical_id)
        assert ds_count == 1
        # Shared dataset's source notebook appears exactly once.
        nb_count = sum(
            1 for r in closure if r.kind == "notebook" and r.logical_id == "etl"
        )
        assert nb_count == 1
        # Both charts present.
        chart_lids = {r.logical_id for r in closure if r.kind == "chart"}
        assert chart_lids == {ch1.logical_id, ch2.logical_id}

    @pytest.mark.asyncio
    async def test_visited_set_terminates_a_synthetic_cycle(self, tmp_path: Path) -> None:
        """Defensive: a back-edge in the DAG must not infinite-loop.

        Healthy typed-ref graphs are acyclic (no path lets a dataset's
        source_refs point at its own dependents), but the walker is
        defensive — it terminates on the visited-set check before
        re-emitting any ref it has already seen.
        """
        # Simulate a cycle by stubbing child_refs at runtime: chase the same
        # ref twice via a custom executor whose snapshot bytes encode the
        # same edge back. Easiest is to monkeypatch closure._dataset_children
        # through a wrapper that returns a self-referential edge.
        executor = _ReadOnlyExecutor(tmp_path)
        nb_code = "ds = 1"
        nb = _nb_ref("etl", nb_code)
        _write_notebook(executor, "etl", nb_code)
        ds = _persist_dataset(
            executor,
            notebook_refs=[nb],
            source_refs=[],
            variable_name="ds",
            title="ds",
            df=pd.DataFrame({"x": [1]}),
        )

        # Patch child_refs to inject a self-cycle on the dataset.
        import parsimony_agents.closure as closure_mod

        original = closure_mod.child_refs

        async def cyclic_child_refs(ref: ArtifactRef, *, executor: Any) -> list[ArtifactRef]:
            if ref.kind == "dataset":
                # Self-loop: dataset claims itself as a child. Walker must
                # terminate on the visited-set check.
                return [ref, *await original(ref, executor=executor)]
            return await original(ref, executor=executor)

        closure_mod.child_refs = cyclic_child_refs
        try:
            closure = await enumerate_closure(ds, executor=executor)
        finally:
            closure_mod.child_refs = original

        # Exactly one dataset entry — the self-edge was deduped.
        assert sum(1 for r in closure if r.kind == "dataset") == 1
        assert closure[-1] == ds

    @pytest.mark.asyncio
    async def test_root_always_last_in_closure(self, tmp_path: Path) -> None:
        executor = _ReadOnlyExecutor(tmp_path)
        nb_code = "ds = 1"
        nb = _nb_ref("etl", nb_code)
        _write_notebook(executor, "etl", nb_code)
        ds = _persist_dataset(
            executor,
            notebook_refs=[nb],
            source_refs=[],
            variable_name="ds",
            title="ds",
            df=pd.DataFrame({"x": [1]}),
        )

        closure = await enumerate_closure(ds, executor=executor)
        assert closure[-1] == ds

    @pytest.mark.asyncio
    async def test_uncommon_chart_from_data_objects_reaches_them(
        self, tmp_path: Path
    ) -> None:
        """The 'uncommon' case (chart.source_refs holds data_objects directly)
        must include those data_objects in the closure — that's the only path
        by which they're reachable, since notebooks don't statically expose
        their fetch_log."""
        executor = _ReadOnlyExecutor(tmp_path)
        nb_code = "fig = {'mark': 'bar'}"
        nb = _nb_ref("chart_nb", nb_code)
        _write_notebook(executor, "chart_nb", nb_code)
        do_a = _do_ref("connector_a", "rows-a")
        do_b = _do_ref("connector_b", "rows-b")
        ch = _persist_chart(
            executor,
            notebook_ref=nb,
            source_dataset_refs=[],
            source_refs=[do_a, do_b],
            variable_name="fig",
            title="ch",
            spec=_vega_spec(1),
        )

        closure = await enumerate_closure(ch, executor=executor)
        data_object_lids = {r.logical_id for r in closure if r.kind == "data_object"}
        assert data_object_lids == {do_a.logical_id, do_b.logical_id}
        _assert_before(closure, do_a, ch)
        _assert_before(closure, do_b, ch)
