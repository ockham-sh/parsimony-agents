"""Tests for the standalone local-store artifact discovery surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsimony_agents.agent.local_store import (
    build_local_session_state,
    collect_local_artifact_lines,
    list_local_artifacts,
    read_local_artifact,
)
from parsimony_agents.agent.outputs import ArtifactNotFound

_EXT = {"notebook": ".py", "dataset": ".parquet", "chart": ".vl.json", "report": ".qmd"}


def _write_artifact(
    root: Path,
    *,
    kind: str,
    lid: str,
    live_name: str | None,
    title: str = "",
    description: str = "",
    tags: list[str] | None = None,
    snapshot_bytes: bytes | None = b"snapshot",
) -> Path:
    """Create ``.ockham/<kind>s/<lid>/{curation.json, log.jsonl, <csha><ext>}``."""
    logical_dir = root / ".ockham" / f"{kind}s" / lid
    logical_dir.mkdir(parents=True, exist_ok=True)
    body: dict = {"kind": kind, "logical_id": lid}
    if live_name is not None:
        body["live_name"] = live_name
    if title:
        body["title"] = title
    if description:
        body["description"] = description
    if tags:
        body["tags"] = tags
    (logical_dir / "curation.json").write_text(json.dumps(body))
    csha = f"sha_{lid}"
    (logical_dir / "log.jsonl").write_text(json.dumps({"content_sha": csha}) + "\n")
    if snapshot_bytes is not None:
        (logical_dir / f"{csha}{_EXT[kind]}").write_bytes(snapshot_bytes)
    return logical_dir


# ---------------------------------------------------------------------------
# list_local_artifacts
# ---------------------------------------------------------------------------


def test_list_empty_tree(tmp_path: Path) -> None:
    assert list_local_artifacts(tmp_path, None, None, 20) == []


def test_list_returns_rows_with_expected_keys(tmp_path: Path) -> None:
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name="unrate", title="US Unemployment Rate")
    rows = list_local_artifacts(tmp_path, None, None, 20)
    assert len(rows) == 1
    row = rows[0]
    assert row["live_name"] == "unrate"
    assert row["kind"] == "dataset"
    assert row["summary"] == "US Unemployment Rate"
    assert row["logical_id"] == "d1"


def test_list_filters_by_kind(tmp_path: Path) -> None:
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name="unrate")
    _write_artifact(tmp_path, kind="notebook", lid="n1", live_name="unrate_nb")
    assert {r["kind"] for r in list_local_artifacts(tmp_path, None, "dataset", 20)} == {"dataset"}


def test_list_query_matches_title_case_insensitive(tmp_path: Path) -> None:
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name="unrate", title="Unemployment")
    _write_artifact(tmp_path, kind="dataset", lid="d2", live_name="gdp", title="Gross Domestic Product")
    rows = list_local_artifacts(tmp_path, "UNEMPLOY", None, 20)
    assert [r["live_name"] for r in rows] == ["unrate"]


def test_list_skips_curation_without_live_name(tmp_path: Path) -> None:
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name=None, title="orphan")
    assert list_local_artifacts(tmp_path, None, None, 20) == []


def test_list_skips_nonnotebook_without_snapshot(tmp_path: Path) -> None:
    # A dataset curated but never written has no snapshot → not readable → hidden.
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name="pending", snapshot_bytes=None)
    assert list_local_artifacts(tmp_path, None, None, 20) == []


def test_list_respects_limit(tmp_path: Path) -> None:
    for i in range(5):
        _write_artifact(tmp_path, kind="dataset", lid=f"d{i}", live_name=f"ds{i}")
    assert len(list_local_artifacts(tmp_path, None, None, 2)) == 2


# ---------------------------------------------------------------------------
# collect_local_artifact_lines / build_local_session_state
# ---------------------------------------------------------------------------


def test_collect_lines_carry_live_name_and_ref(tmp_path: Path) -> None:
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name="unrate", title="Unemployment")
    lines = collect_local_artifact_lines(tmp_path)
    assert len(lines) == 1
    line = lines[0]
    assert line.live_name == "unrate"
    assert line.kind == "dataset"
    assert line.ref is not None
    assert line.ref.logical_id == "d1"
    assert line.path == ".ockham/datasets/d1/sha_d1.parquet"


def test_collect_skips_snapshot_with_unrecognised_extension(tmp_path: Path) -> None:
    # A stray non-canonical file (e.g. a partial write) selected as "latest"
    # yields no ArtifactRef → the row is skipped rather than emitted ref-less.
    logical_dir = tmp_path / ".ockham" / "datasets" / "d1"
    logical_dir.mkdir(parents=True)
    (logical_dir / "curation.json").write_text(json.dumps({"kind": "dataset", "logical_id": "d1", "live_name": "x"}))
    (logical_dir / "sha_d1.parquet.part").write_bytes(b"partial")
    assert collect_local_artifact_lines(tmp_path) == []


def test_build_session_state_lists_artifacts(tmp_path: Path) -> None:
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name="unrate")
    state = build_local_session_state(executor=None, local_dir=tmp_path)
    assert [a.live_name for a in state.workspace_artifacts] == ["unrate"]
    assert state.kernel == []


# ---------------------------------------------------------------------------
# read_local_artifact
# ---------------------------------------------------------------------------


def test_read_notebook_returns_source(tmp_path: Path) -> None:
    _write_artifact(
        tmp_path, kind="notebook", lid="n1", live_name="unrate_nb",
        snapshot_bytes=b"df = load_dataset('unrate')\n",
    )
    result = read_local_artifact(tmp_path, "unrate_nb", "notebook", {})
    assert "load_dataset('unrate')" in result.text
    assert 'live_name="unrate_nb"' in result.text


def test_read_dataset_returns_preview(tmp_path: Path) -> None:
    import pandas as pd

    from parsimony_agents import Dataset
    from parsimony_agents.dataset_io import write_dataset_bytes
    from parsimony_agents.execution.outputs import DataFrameObject

    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "value": [3.6, 3.5]})
    dataset = Dataset(title="US Unemployment Rate", live_name="unrate")
    payload = DataFrameObject.from_pandas(df, local_dir=tmp_path / "_dfo")
    blob = write_dataset_bytes(dataset, payload)
    _write_artifact(tmp_path, kind="dataset", lid="d1", live_name="unrate", snapshot_bytes=blob)

    result = read_local_artifact(tmp_path, "unrate", "dataset", {})
    assert 'live_name="unrate"' in result.text
    assert "2 rows x 2 columns" in result.text
    assert "value" in result.text


def test_read_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFound):
        read_local_artifact(tmp_path, "nope", "dataset", {})


def test_read_unknown_kind_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactNotFound):
        read_local_artifact(tmp_path, "x", "banana", {})


# ---------------------------------------------------------------------------
# local_discovery composition with the cross-turn seen-set filter
# ---------------------------------------------------------------------------


def _ctx_with_artifact(*, local_discovery: bool):
    from parsimony_agents.agent.models import AgentContext
    from parsimony_agents.agent.session_state import SessionState, WorkspaceArtifactLine

    return AgentContext(
        session_id="s",
        messages=[],  # no prior-turn messages → empty extracted seen-set
        session_state=SessionState(
            workspace_artifacts=[
                WorkspaceArtifactLine(
                    path=".ockham/datasets/d1/x.parquet", kind="dataset",
                    live_name="unrate", summary="US Unemployment",
                )
            ]
        ),
        local_discovery=local_discovery,
    )


def _snapshot_text(snap) -> str:
    return "".join(c["text"] for c in snap.to_llm() if c.get("type") == "text")


@pytest.mark.asyncio
async def test_local_discovery_preseeds_seen_set_so_disk_artifact_survives_filter() -> None:
    # With local_discovery, a disk-discovered artifact is admitted past the
    # cross-turn filter even though messages carry no seen-set — this is what
    # stops the reuse loop when ctx is not threaded between turns.
    snap = await _ctx_with_artifact(local_discovery=True).to_snapshot()
    assert ("dataset", "unrate") in {tuple(p) for p in snap.seen_live_names_pairs}
    assert "unrate" in _snapshot_text(snap)


@pytest.mark.asyncio
async def test_without_local_discovery_empty_seen_set_filters_artifact_out() -> None:
    # Contrast: in host (multi-terminal) mode the same empty seen-set drops the
    # cross-turn row — the sibling-terminal hiding the fix must not disturb.
    snap = await _ctx_with_artifact(local_discovery=False).to_snapshot()
    assert ("dataset", "unrate") not in {tuple(p) for p in snap.seen_live_names_pairs}
    assert "unrate" not in _snapshot_text(snap)
