"""Cross-terminal collision detection for the agent-side resolvers.

Covers ``Agent._resolve_artifact_slug``, ``Agent._resolve_slug_to_latest_ref``
and ``parsimony_agents.execution.load.resolve_dataset_slug``. The
host-side notebook resolver is tested separately under the terminal
package.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.execution.load import (
    LoadDatasetError,
    resolve_dataset_slug,
)
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.identity import LiveNameCollisionError


class _Executor:
    """In-memory FS stub keyed by relative workspace paths."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.cwd: str | None = None

    async def get(self, key: str) -> object:  # noqa: ARG002
        return None

    async def clear_namespace(self) -> None:
        return None

    async def set_cwd(self, path: str, session_id: str | None = None) -> None:  # noqa: ARG002
        self.cwd = path

    async def set_connectors(self, _c) -> None:  # noqa: ANN001
        return None

    async def execute(  # noqa: ARG002
        self, code: str, dry_run: bool = False, timeout_seconds: float | None = None
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    async def read_workspace_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        return [(p, len(d)) for p, d in self.files.items() if p.startswith(prefix)]


def _agent_with_executor(executor: _Executor) -> Agent:
    agent = Agent(model_config={"model": "test-model"}, instructions="t", session_id="s")
    agent.code_executor = executor  # type: ignore[assignment]
    return agent


def _seed_report_curation(ex: _Executor, *, lid: str, live_name: str) -> None:
    ex.files[f".ockham/reports/{lid}/curation.json"] = json.dumps(
        {"kind": "report", "logical_id": lid, "live_name": live_name}
    ).encode("utf-8")


def _seed_dataset_curation(ex: _Executor, *, lid: str, live_name: str) -> None:
    ex.files[f".ockham/datasets/{lid}/curation.json"] = json.dumps(
        {"kind": "dataset", "logical_id": lid, "live_name": live_name}
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_resolve_artifact_slug_empty_seen_raises() -> None:
    ex = _Executor()
    _seed_report_curation(ex, lid="rep1", live_name="us_report")
    agent = _agent_with_executor(ex)

    with pytest.raises(LiveNameCollisionError) as info:
        await agent._resolve_artifact_slug("us_report", kind="report", seen_live_names=set())
    assert info.value.existing_logical_id == "rep1"
    assert info.value.kind == "report"
    assert "read_artifact" in str(info.value)


@pytest.mark.asyncio
async def test_resolve_artifact_slug_seen_includes_pair_returns_lid() -> None:
    ex = _Executor()
    _seed_report_curation(ex, lid="rep1", live_name="us_report")
    agent = _agent_with_executor(ex)

    lid = await agent._resolve_artifact_slug(
        "us_report",
        kind="report",
        seen_live_names={("report", "us_report")},
    )
    assert lid == "rep1"


@pytest.mark.asyncio
async def test_resolve_artifact_slug_unrelated_seen_still_raises() -> None:
    ex = _Executor()
    _seed_report_curation(ex, lid="rep1", live_name="us_report")
    agent = _agent_with_executor(ex)

    with pytest.raises(LiveNameCollisionError):
        await agent._resolve_artifact_slug(
            "us_report",
            kind="report",
            seen_live_names={("report", "other_report"), ("dataset", "us_report")},
        )


@pytest.mark.asyncio
async def test_resolve_artifact_slug_none_seen_skips_gate() -> None:
    """seen_live_names=None preserves legacy / programmatic behaviour."""
    ex = _Executor()
    _seed_report_curation(ex, lid="rep1", live_name="us_report")
    agent = _agent_with_executor(ex)

    lid = await agent._resolve_artifact_slug("us_report", kind="report", seen_live_names=None)
    assert lid == "rep1"


@pytest.mark.asyncio
async def test_resolve_artifact_slug_no_match_raises_value_error() -> None:
    """Miss case is unchanged: ValueError, not LiveNameCollisionError."""
    ex = _Executor()
    agent = _agent_with_executor(ex)

    with pytest.raises(ValueError, match="No report has live_name"):
        await agent._resolve_artifact_slug("missing", kind="report", seen_live_names=set())


@pytest.mark.asyncio
async def test_resolve_slug_to_latest_ref_surfaces_collision() -> None:
    ex = _Executor()
    _seed_dataset_curation(ex, lid="ds1", live_name="us_gdp")
    ex.files[".ockham/datasets/ds1/log.jsonl"] = (json.dumps({"content_sha": "csha_a"}) + "\n").encode("utf-8")
    agent = _agent_with_executor(ex)

    with pytest.raises(LiveNameCollisionError) as info:
        await agent._resolve_slug_to_latest_ref("us_gdp", seen_live_names=set())
    assert info.value.kind == "dataset"
    assert info.value.existing_logical_id == "ds1"


@pytest.mark.asyncio
async def test_resolve_slug_to_latest_ref_seen_returns_ref() -> None:
    ex = _Executor()
    _seed_dataset_curation(ex, lid="ds1", live_name="us_gdp")
    ex.files[".ockham/datasets/ds1/log.jsonl"] = (json.dumps({"content_sha": "csha_a"}) + "\n").encode("utf-8")
    agent = _agent_with_executor(ex)

    ref = await agent._resolve_slug_to_latest_ref("us_gdp", seen_live_names={("dataset", "us_gdp")})
    assert ref.kind == "dataset"
    assert ref.logical_id == "ds1"
    assert ref.content_sha == "csha_a"


def _seed_filesystem_dataset(tmp_path: Path, *, lid: str, live_name: str, csha: str) -> None:
    """Materialise a dataset curation + log on disk under tmp_path."""
    root = tmp_path / ".ockham" / "datasets" / lid
    root.mkdir(parents=True)
    (root / "curation.json").write_text(json.dumps({"kind": "dataset", "logical_id": lid, "live_name": live_name}))
    (root / "log.jsonl").write_text(json.dumps({"content_sha": csha}) + "\n")


def test_resolve_dataset_slug_empty_seen_raises(tmp_path: Path) -> None:
    _seed_filesystem_dataset(tmp_path, lid="ds1", live_name="us_gdp", csha="csha_a")

    with pytest.raises(LiveNameCollisionError) as info:
        resolve_dataset_slug(tmp_path, "us_gdp", seen_live_names=set())
    assert info.value.kind == "dataset"
    assert info.value.existing_logical_id == "ds1"


def test_resolve_dataset_slug_seen_returns_ref(tmp_path: Path) -> None:
    _seed_filesystem_dataset(tmp_path, lid="ds1", live_name="us_gdp", csha="csha_a")

    ref = resolve_dataset_slug(tmp_path, "us_gdp", seen_live_names={("dataset", "us_gdp")})
    assert ref.logical_id == "ds1"
    assert ref.content_sha == "csha_a"


def test_resolve_dataset_slug_none_seen_skips_gate(tmp_path: Path) -> None:
    _seed_filesystem_dataset(tmp_path, lid="ds1", live_name="us_gdp", csha="csha_a")

    ref = resolve_dataset_slug(tmp_path, "us_gdp", seen_live_names=None)
    assert ref.logical_id == "ds1"


def test_load_dataset_wraps_collision_as_load_dataset_error(tmp_path: Path) -> None:
    """``load_dataset`` should surface the collision recovery message but as
    ``LoadDatasetError`` so existing ``except KeyError`` paths still trap it."""
    from parsimony_agents.execution.load import build_load_dataset
    from parsimony_agents.execution.run_scope import OriginLedger

    _seed_filesystem_dataset(tmp_path, lid="ds1", live_name="us_gdp", csha="csha_a")
    ledger = OriginLedger()
    load = build_load_dataset(
        workspace_root_provider=lambda: tmp_path,
        ledger=ledger,
        seen_live_names_provider=lambda: set(),
    )

    with pytest.raises(LoadDatasetError) as info:
        load("us_gdp")
    assert "read_artifact" in str(info.value)
