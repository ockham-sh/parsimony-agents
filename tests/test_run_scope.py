"""Tests for producer-scoped attribution (brief §6).

The producing-run scope is what makes the agent never have to type a
ref. A variable assigned by a notebook run carries that notebook's
identity as its origin. A scratch / verification cell between the
producing run and the publish must NOT overwrite that origin.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from parsimony_agents.execution.executor import CodeExecutor
from parsimony_agents.execution.factory import OutputFactory
from parsimony_agents.execution.run_scope import OriginLedger, RunScope
from parsimony_agents.identity import ArtifactRef


def _executor(tmp_path: Path) -> CodeExecutor:
    factory = OutputFactory(local_dir=str(tmp_path))
    ex = CodeExecutor(cwd=str(tmp_path), output_factory=factory)
    return ex


def test_origin_ledger_scope_lifecycle(tmp_path: Path) -> None:
    ledger = OriginLedger()
    assert ledger.current is None
    with ledger.scope("notebooks/foo.py") as scope:
        assert ledger.current is scope
        assert scope.notebook_path == "notebooks/foo.py"
    assert ledger.current is None


def test_nested_scope_raises() -> None:
    ledger = OriginLedger()
    with ledger.scope("a") as _, pytest.raises(RuntimeError, match="already open"), ledger.scope("b"):
        pass


def test_executor_stamps_origin_for_producing_run(tmp_path: Path) -> None:
    """A `producer_notebook_path` run stamps every assigned name."""

    async def _go() -> None:
        ex = _executor(tmp_path)
        await ex.execute(
            "df = 1\n",
            producer_notebook_path="notebooks/producer.py",
        )
        origin = ex.origin_ledger.get("df")
        assert origin is not None
        assert origin.notebook_path == "notebooks/producer.py"

    asyncio.run(_go())


def test_scratch_run_does_not_stamp_origin(tmp_path: Path) -> None:
    """A run without ``producer_notebook_path`` produces no origin entry."""

    async def _go() -> None:
        ex = _executor(tmp_path)
        await ex.execute("df = 1\n")
        assert ex.origin_ledger.get("df") is None

    asyncio.run(_go())


def test_verification_cell_does_not_overwrite_origin(tmp_path: Path) -> None:
    """The brief §6 case: a scratch verification cell must not steal lineage."""

    async def _go() -> None:
        ex = _executor(tmp_path)
        # 1. Producing run assigns `df`.
        await ex.execute(
            "df = 42\n",
            producer_notebook_path="notebooks/producer.py",
        )
        producer_origin = ex.origin_ledger.get("df")
        assert producer_origin is not None
        assert producer_origin.notebook_path == "notebooks/producer.py"

        # 2. Scratch verification cell reads `df` (and may even reassign
        #    to something derived — but it is not a producer run).
        await ex.execute("assert df == 42\n")

        # 3. Origin is still the producer.
        post = ex.origin_ledger.get("df")
        assert post is not None
        assert post.notebook_path == "notebooks/producer.py"

    asyncio.run(_go())


def test_rebind_in_producer_replaces_origin(tmp_path: Path) -> None:
    """``df = df.dropna()`` in producer B replaces producer A's origin.

    set-diff alone would miss this (the name pre-existed); the AST
    half of the union earns its keep here.
    """

    async def _go() -> None:
        ex = _executor(tmp_path)
        await ex.execute(
            "df = pd.DataFrame({'x': [1, 2, None]})\n",
            producer_notebook_path="notebooks/a.py",
        )
        first = ex.origin_ledger.get("df")
        assert first is not None and first.notebook_path == "notebooks/a.py"

        await ex.execute(
            "df = df.dropna()\n",
            producer_notebook_path="notebooks/b.py",
        )
        second = ex.origin_ledger.get("df")
        assert second is not None and second.notebook_path == "notebooks/b.py"

    asyncio.run(_go())


def test_exception_mid_run_skips_stamping(tmp_path: Path) -> None:
    """If a producer run raises, the AST-claimed names must NOT be stamped.

    Locks in the "union is sound" invariant: stamping happens only
    after successful completion, so AST's optimism over names that
    never got bound is bounded by control flow.
    """

    async def _go() -> None:
        ex = _executor(tmp_path)
        ko = await ex.execute(
            "a = 1\nraise RuntimeError('boom')\nb = 2\n",
            producer_notebook_path="notebooks/explodes.py",
        )
        # KernelOutput captures the exception rather than raising — the
        # stamp branch must have been skipped in the except path.
        from parsimony_agents.execution.outputs import ExceptionObject

        assert any(isinstance(o, ExceptionObject) for o in ko.outputs)
        assert ex.origin_ledger.get("a") is None
        assert ex.origin_ledger.get("b") is None

    asyncio.run(_go())


def test_clear_namespace_clears_origins(tmp_path: Path) -> None:
    async def _go() -> None:
        ex = _executor(tmp_path)
        await ex.execute(
            "df = 1\n",
            producer_notebook_path="notebooks/foo.py",
        )
        assert ex.origin_ledger.get("df") is not None
        await ex.clear_namespace()
        assert ex.origin_ledger.get("df") is None

    asyncio.run(_go())


def test_run_scope_dedupes_repeated_refs() -> None:
    """A producing run that loads/fetches the same artifact twice records it once.

    Realistic: ``load_dataset('us_gdp')`` called from two cells in one
    notebook, or a connector fetch repeated inside a loop. Without
    dedup the VariableOrigin would carry duplicate refs that then
    round-trip across the HTTP boundary as phantom lineage edges.
    """
    scope = RunScope(notebook_path="notebooks/n.py")
    ref = ArtifactRef(kind="dataset", logical_id="lid", content_sha="csha")
    scope.record_load(ref)
    scope.record_load(ref)
    assert len(scope.load_refs) == 1

    fref = ArtifactRef(kind="data_object", logical_id="fmp_quote_AAPL", content_sha="csha2")
    scope.record_fetch(fref)
    scope.record_fetch(fref)
    assert len(scope.fetch_refs) == 1


def test_origin_captures_load_refs(tmp_path: Path) -> None:
    """When a producing run calls load_dataset, the ref lands on the origin."""
    import json

    from parsimony_agents.artifacts import Dataset
    from parsimony_agents.dataset_io import write_dataset_bytes
    from parsimony_agents.execution.outputs import DataFrameObject
    from parsimony_agents.identity import content_sha

    df = pd.DataFrame({"v": [1, 2]})
    payload = DataFrameObject.from_pandas(df, local_dir=tmp_path)
    ds = Dataset(
        logical_id="lid_up", title="us_gdp", live_name="us_gdp",
        variable_name="r",
    )
    blob = write_dataset_bytes(ds, payload)
    csha = content_sha(blob)
    base = tmp_path / ".ockham" / "datasets" / "lid_up"
    base.mkdir(parents=True, exist_ok=True)
    (base / f"{csha}.parquet").write_bytes(blob)
    (base / "log.jsonl").write_text(
        json.dumps({"ts": "t1", "content_sha": csha, "inputs": {}}) + "\n"
    )
    (base / "curation.json").write_text(
        json.dumps(
            {"kind": "dataset", "logical_id": "lid_up", "live_name": "us_gdp",
             "title": "us_gdp", "tags": [], "notes": []}
        )
    )

    async def _go() -> None:
        ex = _executor(tmp_path)
        await ex.execute(
            "got = load_dataset('us_gdp')\n",
            producer_notebook_path="notebooks/consumer.py",
        )
        origin = ex.origin_ledger.get("got")
        assert origin is not None
        assert len(origin.load_refs) == 1
        assert origin.load_refs[0].kind == "dataset"
        assert origin.load_refs[0].logical_id == "lid_up"

        # Repeated load inside the same producing run is deduped — the
        # RunScope.record_load workspace_file_path invariant, end-to-end.
        await ex.execute(
            "a = load_dataset('us_gdp')\nb = load_dataset('us_gdp')\n",
            producer_notebook_path="notebooks/consumer2.py",
        )
        for var in ("a", "b"):
            origin = ex.origin_ledger.get(var)
            assert origin is not None
            assert len(origin.load_refs) == 1

    asyncio.run(_go())
