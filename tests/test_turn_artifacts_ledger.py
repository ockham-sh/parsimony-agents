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
# SessionState.render_block
# ---------------------------------------------------------------------------


def test_render_block_renders_turn_artifacts_block_with_new_flag() -> None:
    state = SessionState(
        kernel=[],
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="data/x.parquet",
                kind="dataset",
                summary="cross-turn dataset",
                live_name="my_dataset",
                ref=ArtifactRef(kind="dataset", logical_id="lid-x", content_sha="cs1"),
            ),
        ],
    )
    minted = [ArtifactRef(kind="chart", logical_id="lid-c", content_sha="cs-c")]
    # Carry the agent-typed slug for the minted ref. Required for the
    # rendered tag to carry ``live_name="..."`` — without it, the next
    # iteration's seen-set extractor cannot recognise the artifact and
    # the next ``return_*`` raises against this terminal's own write.
    text = state.render_block(
        minted_refs=minted,
        minted_live_names={"chart:lid-c": "my_chart"},
    )

    # Renamed from <workspace_artifacts>.
    assert "<turn_artifacts>" in text
    assert "<workspace_artifacts>" not in text
    # Cross-turn artifact: surfaces by live_name, no new attribute.
    assert 'kind="dataset" live_name="my_dataset"' in text
    assert 'new="true"' not in text.split("my_dataset")[0]
    # Brand-new minted chart: live_name from the carrier, new="true".
    assert 'kind="chart" live_name="my_chart" new="true"' in text
    # Hash triplets must not appear in the agent-facing render.
    assert "logical_id" not in text
    assert "content_sha" not in text


def test_render_block_minted_without_live_name_carrier_falls_back_to_no_attr() -> None:
    """Legacy / standalone callers that don't populate the carrier still
    render — they just omit ``live_name=`` (the seen-set extractor will
    skip the row, which is fine in single-call scenarios)."""
    state = SessionState(kernel=[], workspace_artifacts=[])
    minted = [ArtifactRef(kind="chart", logical_id="lid-c", content_sha="cs-c")]
    text = state.render_block(minted_refs=minted)  # no minted_live_names
    assert 'kind="chart" new="true"' in text
    assert 'live_name="' not in text.split("<turn_artifacts>")[1].split("</turn_artifacts>")[0]


def test_render_block_minted_live_name_propagates_to_seen_set_extractor() -> None:
    """End-to-end: when the rendering carries live_name, the seen-set
    extractor reading the rendered text picks the artifact up.

    This is the regression guard for the iter-N+1 collision bug — the
    agent loop scans ``ctx.messages`` (which contains the rendered
    snapshot) to reconstruct the seen-set before each tool call.
    """
    from parsimony_agents.agent.seen_refs import extract_seen_live_names

    state = SessionState(kernel=[], workspace_artifacts=[])
    minted = [ArtifactRef(kind="notebook", logical_id="us_gdp", content_sha="cs")]
    text = state.render_block(
        minted_refs=minted,
        minted_live_names={"notebook:us_gdp": "us_gdp"},
    )

    seen = extract_seen_live_names([{"content": text}])
    assert ("notebook", "us_gdp") in seen


def test_render_block_no_minted_renders_clean_cross_turn_view() -> None:
    state = SessionState(
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="data/x.parquet",
                kind="dataset",
                ref=ArtifactRef(kind="dataset", logical_id="lid-x", content_sha="cs1"),
            ),
        ],
    )
    text = state.render_block(minted_refs=None)
    assert "<turn_artifacts>" in text
    # No new="true" attribute on any rendered artifact line — extract just
    # the artifact block, since the human-readable <note> mentions new="true"
    # as documentation.
    artifacts_block = text.split("<turn_artifacts>")[1].split("</turn_artifacts>")[0]
    assert 'new="true"' not in artifacts_block


def test_render_block_replaces_advanced_artifact_in_place() -> None:
    """Same logical_id, new content_sha → in-place replacement marked new=true.

    With the new live_name surface, the *visible* effect is just that the
    existing slug row picks up ``new="true"`` — the agent's mental model
    "this artifact exists, and it was advanced this turn" is preserved
    without exposing any hash.
    """
    state = SessionState(
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="reports/r.qmd",
                kind="report",
                summary="orig",
                live_name="weekly_report",
                ref=ArtifactRef(kind="report", logical_id="lid-r", content_sha="cs1"),
            ),
        ],
    )
    minted = [ArtifactRef(kind="report", logical_id="lid-r", content_sha="cs2")]
    text = state.render_block(minted_refs=minted)
    assert 'kind="report" live_name="weekly_report" new="true"' in text
    # Only one row for that artifact — replacement, not addition.
    assert text.count('live_name="weekly_report"') == 1
    assert "content_sha" not in text


# ---------------------------------------------------------------------------
# Sanity: RunState carries the run-lifetime minted_refs ledger
# ---------------------------------------------------------------------------


def test_run_state_starts_with_empty_minted_refs() -> None:
    from parsimony_agents.agent.state import RunState

    state = RunState(run_id="r", session_id="s")
    assert state.minted_refs == []
    assert state.minted_live_names == {}
    state.minted_refs.append(
        ArtifactRef(kind="dataset", logical_id="lid", content_sha="cs")
    )
    state.minted_live_names["dataset:lid"] = "my_slug"
    assert len(state.minted_refs) == 1
    assert state.minted_live_names == {"dataset:lid": "my_slug"}
