"""Tests for ``parsimony_agents.agent.seen_refs.extract_seen_live_names``."""

from __future__ import annotations

from typing import Any

from parsimony_agents.agent.seen_refs import extract_seen_live_names
from parsimony_agents.messages import Text


def _msg(text: str) -> dict[str, Any]:
    """Build a minimal message-shaped dict for the scanner to walk."""
    return {"role": "user", "content": {"type": "text", "content": text}}


def test_empty_list_returns_empty_set() -> None:
    assert extract_seen_live_names([]) == set()


def test_single_self_closing_artifact_ref_is_extracted() -> None:
    msgs = [_msg('<artifact_ref kind="notebook" live_name="us_gdp"/>')]
    assert extract_seen_live_names(msgs) == {("notebook", "us_gdp")}


def test_artifact_tag_with_body_is_extracted() -> None:
    msgs = [_msg('<artifact kind="dataset" live_name="us_gdp" new="true">GDP series (1947-2024)</artifact>')]
    assert extract_seen_live_names(msgs) == {("dataset", "us_gdp")}


def test_multiple_refs_across_messages_are_deduplicated() -> None:
    msgs = [
        _msg('<artifact kind="dataset" live_name="us_gdp"/>'),
        _msg('<artifact_ref kind="dataset" live_name="us_gdp"/>'),
        _msg('<artifact kind="chart" live_name="us_gdp_plot"/>'),
    ]
    result = extract_seen_live_names(msgs)
    assert result == {("dataset", "us_gdp"), ("chart", "us_gdp_plot")}


def test_attribute_order_does_not_matter() -> None:
    msgs = [_msg('<artifact_ref live_name="alpha" kind="report"/>')]
    assert extract_seen_live_names(msgs) == {("report", "alpha")}


def test_missing_live_name_is_skipped() -> None:
    msgs = [_msg('<artifact kind="dataset" logical_id="lid_x"/>')]
    assert extract_seen_live_names(msgs) == set()


def test_missing_kind_is_skipped() -> None:
    msgs = [_msg('<artifact live_name="orphan"/>')]
    assert extract_seen_live_names(msgs) == set()


def test_unknown_kind_is_skipped() -> None:
    msgs = [_msg('<artifact kind="banana" live_name="rotten"/>')]
    assert extract_seen_live_names(msgs) == set()


def test_empty_live_name_is_skipped() -> None:
    msgs = [_msg('<artifact kind="dataset" live_name=""/>')]
    assert extract_seen_live_names(msgs) == set()


def test_plain_string_in_message_list() -> None:
    msgs = ['<artifact_ref kind="notebook" live_name="plain_str"/>']
    assert extract_seen_live_names(msgs) == {("notebook", "plain_str")}


def test_nested_dict_and_list_are_walked() -> None:
    msgs = [
        {
            "outer": [
                {"inner": '<artifact kind="dataset" live_name="nested"/>'},
                {"deeper": [{"text": '<artifact kind="chart" live_name="deeper_chart"/>'}]},
            ]
        }
    ]
    assert extract_seen_live_names(msgs) == {
        ("dataset", "nested"),
        ("chart", "deeper_chart"),
    }


def test_pydantic_text_content_is_scanned() -> None:
    msgs = [Text(content='<artifact kind="dataset" live_name="from_pydantic"/>')]
    assert extract_seen_live_names(msgs) == {("dataset", "from_pydantic")}


def test_malformed_text_does_not_raise() -> None:
    msgs = [_msg("not a real tag < > <foo bar baz < ><<<")]
    assert extract_seen_live_names(msgs) == set()


def test_all_five_snapshot_kinds_recognised() -> None:
    msgs = [
        _msg(
            "\n".join(
                [
                    '<artifact kind="notebook" live_name="nb"/>',
                    '<artifact kind="data_object" live_name="do"/>',
                    '<artifact kind="dataset" live_name="ds"/>',
                    '<artifact kind="chart" live_name="ch"/>',
                    '<artifact kind="report" live_name="rp"/>',
                ]
            )
        )
    ]
    assert extract_seen_live_names(msgs) == {
        ("notebook", "nb"),
        ("data_object", "do"),
        ("dataset", "ds"),
        ("chart", "ch"),
        ("report", "rp"),
    }


# ---------------------------------------------------------------------------
# minted_live_names structured carrier
# ---------------------------------------------------------------------------


def test_minted_live_names_dict_shape_is_recognised() -> None:
    """The structured carrier on ``AgentContextSnapshot`` is picked up.

    Without this, the calling terminal's iter-N mint isn't visible to
    iter-N+1's seen-set scan (the XML form lives in the rendered prompt
    text only, which is produced by ``to_llm`` and not part of the
    scanned message graph).
    """
    msgs = [{"minted_live_names": {"notebook:us_gdp": "us_gdp"}}]
    assert extract_seen_live_names(msgs) == {("notebook", "us_gdp")}


def test_minted_live_names_recognises_all_five_kinds() -> None:
    msgs = [
        {
            "minted_live_names": {
                "notebook:nb_lid": "nb",
                "dataset:ds_lid": "ds",
                "chart:ch_lid": "ch",
                "report:rp_lid": "rp",
                "data_object:do_lid": "do",
            }
        }
    ]
    assert extract_seen_live_names(msgs) == {
        ("notebook", "nb"),
        ("dataset", "ds"),
        ("chart", "ch"),
        ("report", "rp"),
        ("data_object", "do"),
    }


def test_unknown_kind_prefix_in_colon_keys_is_skipped() -> None:
    """Colon-keyed dicts whose prefix isn't a SNAPSHOT_KIND are ignored.

    Defensive: ordinary maps that happen to use ``foo:bar`` keys
    (e.g. URN-style identifiers) must not bleed into the seen-set.
    """
    msgs = [{"some_map": {"urn:thing": "value", "banana:lid": "nope"}}]
    assert extract_seen_live_names(msgs) == set()


def test_workspace_artifacts_list_does_not_leak_via_dict_shape() -> None:
    """Sibling-terminal artifacts in ``workspace_artifacts`` must NOT
    leak into the seen-set via the new dict recognition.

    Each WorkspaceArtifactLine dump has ordinary keys ``"kind"`` and
    ``"live_name"`` — neither contains ``":"``. The new check looks
    only at colon-encoded composite keys, so this is safe.
    """
    msgs = [
        {
            "workspace_artifacts": [
                {
                    "path": "data/x.parquet",
                    "kind": "dataset",
                    "live_name": "sibling_terminal_artifact",
                    "ref": {"kind": "dataset", "logical_id": "x", "content_sha": "y"},
                    "new": False,
                }
            ]
        }
    ]
    assert extract_seen_live_names(msgs) == set()


def test_agent_context_snapshot_round_trip_picks_up_minted_pair() -> None:
    """End-to-end: an ``AgentContextSnapshot`` carrying the dict in
    ``ctx.messages`` makes the calling terminal's own mint visible to
    the cross-terminal gate."""
    from parsimony_agents.agent.models import AgentContextSnapshot
    from parsimony_agents.agent.session_state import SessionState
    from parsimony_agents.identity import ArtifactRef

    snap = AgentContextSnapshot(
        session_state=SessionState(kernel=[], workspace_artifacts=[]),
        minted_refs=[ArtifactRef(kind="notebook", logical_id="us_gdp", content_sha="cs")],
        minted_live_names={"notebook:us_gdp": "us_gdp"},
    )
    assert extract_seen_live_names([snap]) == {("notebook", "us_gdp")}
