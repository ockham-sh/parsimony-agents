"""Turn-artifacts ledger: fused view of cross-turn + this-turn refs (Task 15).

The agent's ref discipline used to require scanning across two surfaces —
``<workspace_artifacts>`` (frozen at turn start) and tool-message self-ref
tags. The ledger collapses these into a single ``<turn_artifacts>`` block,
regenerated each iteration. These tests verify:

- ``fuse_workspace_artifacts`` correctly merges cross-turn + minted refs,
  with minted refs winning on logical_id collision.
- The rendered XML uses ``<turn_artifacts>`` (renamed from
  ``<workspace_artifacts>``) and marks fresh refs with ``new="true"``.
- Brand-new minted refs (no matching cross-turn entry) appear at the end
  with their canonical ``.ockham/...`` path.
"""

from __future__ import annotations

from parsimony_agents.agent.session_state import (
    SessionState,
    WorkspaceArtifactLine,
    fuse_workspace_artifacts,
)
from parsimony_agents.identity import ArtifactRef


# ---------------------------------------------------------------------------
# fuse_workspace_artifacts
# ---------------------------------------------------------------------------


def test_fuse_with_no_minted_returns_cross_turn_unchanged() -> None:
    cross_turn = [
        WorkspaceArtifactLine(
            path="data/x.parquet",
            kind="dataset",
            summary="x",
            ref=ArtifactRef(kind="dataset", logical_id="lid-x", content_sha="cs1"),
        ),
    ]
    out = fuse_workspace_artifacts(cross_turn, [])
    assert len(out) == 1
    assert out[0].new is False
    assert out[0].ref.content_sha == "cs1"


def test_fuse_replaces_cross_turn_when_logical_id_matches() -> None:
    """A minted ref with the same logical_id wins on content_sha and marks new=True."""
    cross_turn = [
        WorkspaceArtifactLine(
            path="data/x.parquet",
            kind="dataset",
            summary="orig",
            ref=ArtifactRef(kind="dataset", logical_id="lid-x", content_sha="cs1"),
        ),
    ]
    minted = [ArtifactRef(kind="dataset", logical_id="lid-x", content_sha="cs2")]
    out = fuse_workspace_artifacts(cross_turn, minted)
    assert len(out) == 1
    assert out[0].new is True
    assert out[0].ref.content_sha == "cs2"
    # Path and summary preserved from the cross-turn line.
    assert out[0].path == "data/x.parquet"
    assert out[0].summary == "orig"


def test_fuse_appends_brand_new_minted_with_canonical_path() -> None:
    """A minted ref with no cross-turn match becomes a new line at the end."""
    cross_turn: list[WorkspaceArtifactLine] = []
    minted = [ArtifactRef(kind="chart", logical_id="lid-c", content_sha="cs-c")]
    out = fuse_workspace_artifacts(cross_turn, minted)
    assert len(out) == 1
    assert out[0].new is True
    assert out[0].path == ".ockham/charts/lid-c/cs-c.vl.json"
    assert out[0].kind == "chart"
    assert out[0].summary == ""


def test_fuse_preserves_cross_turn_order_with_appended_minted() -> None:
    cross_turn = [
        WorkspaceArtifactLine(
            path="notebooks/a.py",
            kind="notebook",
            ref=ArtifactRef(kind="notebook", logical_id="nb1", content_sha="ns1"),
        ),
        WorkspaceArtifactLine(
            path="data/b.parquet",
            kind="dataset",
            ref=ArtifactRef(kind="dataset", logical_id="ds1", content_sha="cs1"),
        ),
    ]
    minted = [
        ArtifactRef(kind="dataset", logical_id="ds1", content_sha="cs2"),  # match
        ArtifactRef(kind="chart", logical_id="ch1", content_sha="cscs"),  # new
    ]
    out = fuse_workspace_artifacts(cross_turn, minted)
    assert [a.kind for a in out] == ["notebook", "dataset", "chart"]
    assert out[0].new is False  # notebook unchanged
    assert out[1].new is True   # dataset advanced
    assert out[2].new is True   # chart brand-new


# ---------------------------------------------------------------------------
# SessionState.to_llm_text
# ---------------------------------------------------------------------------


def test_to_llm_text_renders_turn_artifacts_block_with_new_flag() -> None:
    state = SessionState(
        kernel=[],
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="data/x.parquet",
                kind="dataset",
                summary="cross-turn dataset",
                ref=ArtifactRef(kind="dataset", logical_id="lid-x", content_sha="cs1"),
            ),
        ],
    )
    minted = [ArtifactRef(kind="chart", logical_id="lid-c", content_sha="cs-c")]
    text = state.to_llm_text(minted_refs=minted)

    # Renamed from <workspace_artifacts>.
    assert "<turn_artifacts>" in text
    assert "<workspace_artifacts>" not in text
    # Cross-turn artifact: no new attribute.
    assert 'logical_id="lid-x" content_sha="cs1"' in text
    # Brand-new minted artifact: marked new="true".
    assert 'logical_id="lid-c" content_sha="cs-c" new="true"' in text
    # Brand-new line uses canonical .ockham/ path.
    assert ".ockham/charts/lid-c/cs-c.vl.json" in text


def test_to_llm_text_no_minted_renders_clean_cross_turn_view() -> None:
    state = SessionState(
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="data/x.parquet",
                kind="dataset",
                ref=ArtifactRef(kind="dataset", logical_id="lid-x", content_sha="cs1"),
            ),
        ],
    )
    text = state.to_llm_text(minted_refs=None)
    assert "<turn_artifacts>" in text
    # No new="true" attribute on any rendered artifact line — extract just
    # the artifact block, since the human-readable <note> mentions new="true"
    # as documentation.
    artifacts_block = text.split("<turn_artifacts>")[1].split("</turn_artifacts>")[0]
    assert 'new="true"' not in artifacts_block


def test_to_llm_text_replaces_advanced_artifact_in_place() -> None:
    """Same logical_id, new content_sha → in-place replacement marked new=true."""
    state = SessionState(
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="reports/r.report.qmd",
                kind="report",
                summary="orig",
                ref=ArtifactRef(kind="report", logical_id="lid-r", content_sha="cs1"),
            ),
        ],
    )
    minted = [ArtifactRef(kind="report", logical_id="lid-r", content_sha="cs2")]
    text = state.to_llm_text(minted_refs=minted)
    # Old content_sha is gone (replaced).
    assert 'content_sha="cs1"' not in text
    # New content_sha present, with new="true" attribute.
    assert 'content_sha="cs2" new="true"' in text


# ---------------------------------------------------------------------------
# Sanity: TurnState carries minted_refs
# ---------------------------------------------------------------------------


def test_turn_state_starts_with_empty_minted_refs() -> None:
    from parsimony_agents.agent.helpers import TurnState

    state = TurnState()
    assert state.minted_refs == []
    state.minted_refs.append(
        ArtifactRef(kind="dataset", logical_id="lid", content_sha="cs")
    )
    assert len(state.minted_refs) == 1
