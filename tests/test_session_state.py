"""Session state models and kernel summary parsing."""

from __future__ import annotations

import pandas as pd
import pytest

from parsimony_agents.agent.models import AgentContextSnapshot
from parsimony_agents.agent.session_state import (
    SessionState,
    kernel_summaries_from_locals_map,
    parse_kernel_summaries_from_remote,
)


def test_kernel_summaries_omit_underscore_and_arbitrary() -> None:
    df = pd.DataFrame({"a": [1]})
    m = {"_hidden": 1, "x": object(), "ok": df}
    rows = kernel_summaries_from_locals_map(m)
    assert [r.name for r in rows] == ["ok"]
    assert rows[0].kind == "dataframe"


def test_parse_remote_list_roundtrip() -> None:
    body = [
        {"name": "s", "kind": "series", "detail": "len 3"},
    ]
    got = parse_kernel_summaries_from_remote(body)
    assert len(got) == 1
    assert got[0].name == "s"
    assert got[0].kind == "series"


def test_parse_legacy_str_map() -> None:
    got = parse_kernel_summaries_from_remote({"u": "dataframe", "v": "nope"})
    by_name = {r.name: r for r in got}
    assert by_name["u"].kind == "dataframe"
    assert by_name["v"].kind == "omitted"


def test_session_state_xml_escapes() -> None:
    s = SessionState(
        kernel=[],
        workspace_artifacts=[],
    )
    text = s.to_llm_text()
    assert "<session_state>" in text
    # The note teaches the agent about <turn_artifacts> as the canonical
    # surface — escaped because angle brackets are XML-special inside text.
    assert "&lt;turn_artifacts&gt;" in text


@pytest.mark.asyncio
async def test_read_artifact_tool_invokes_injected_fn() -> None:
    import tempfile

    from parsimony_agents.agent.agent import Agent
    from parsimony_agents.agent.models import AgentContext
    from parsimony_agents.agent.outputs import ArtifactLlmResult
    from parsimony_agents.execution.executor import CodeExecutor
    from parsimony_agents.execution.factory import OutputFactory as FrameworkOutputFactory
    from parsimony_agents.messages import Text

    async def _fn(live_name: str, kind: str, options: dict) -> ArtifactLlmResult:
        m = (options.get("view") or options.get("mode") or "summary") or "summary"
        if isinstance(m, str):
            m = m.strip().lower()
        return ArtifactLlmResult(text=f"{kind}:{live_name}|{m}")

    root = tempfile.mkdtemp()
    of = FrameworkOutputFactory(local_dir=root)
    ex = CodeExecutor(cwd=root, output_factory=of)
    agent = Agent(
        model="m",
        code_executor=ex,  # type: ignore[arg-type]
        output_factory=of,
        read_artifact_fn=_fn,
    )
    out = await agent.read_artifact(
        live_name="n", kind="notebook", mode="summary",
        context=AgentContext(session_id="s"),
    )
    assert out.data.content is not None
    assert isinstance(out.data.content, Text)
    assert "notebook:n" in out.data.content.content
    out2 = await agent.read_artifact(
        live_name="n", kind="notebook", mode="full",
        context=AgentContext(session_id="s"),
    )
    assert out2.data.content is not None
    assert isinstance(out2.data.content, Text)
    assert "full" in out2.data.content.content


def test_agent_context_snapshot_includes_session_state() -> None:
    from parsimony_agents.agent.session_state import KernelVariableSummary, WorkspaceArtifactLine

    # seen_live_names_pairs must include ("notebook", "my_notebook") for the
    # cross-turn row to survive the per-terminal filter — without it, the
    # artifact is treated as a sibling-terminal artifact and hidden.
    snap = AgentContextSnapshot(
        session_state=SessionState(
            kernel=[KernelVariableSummary(name="df", kind="dataframe", detail="1×1")],
            workspace_artifacts=[
                WorkspaceArtifactLine(
                    path="n.py", kind="notebook", summary="x", live_name="my_notebook"
                )
            ],
        ),
        seen_live_names_pairs=[("notebook", "my_notebook")],
    )
    llm = snap.to_llm()
    flat = "".join(c["text"] for c in llm if c.get("type") == "text")
    assert "<session_state>" in flat
    assert 'name="df"' in flat
    # The agent-facing block surfaces the workspace slug, not the on-disk path.
    assert 'live_name="my_notebook"' in flat


def test_agent_context_snapshot_filters_sibling_terminal_artifacts() -> None:
    """Cross-turn rows missing from seen_live_names are dropped (sibling-owned)."""
    from parsimony_agents.agent.session_state import WorkspaceArtifactLine

    snap = AgentContextSnapshot(
        session_state=SessionState(
            kernel=[],
            workspace_artifacts=[
                WorkspaceArtifactLine(
                    path="n.py",
                    kind="notebook",
                    summary="from sibling",
                    live_name="sibling_nb",
                ),
                WorkspaceArtifactLine(
                    path="ours.py",
                    kind="notebook",
                    summary="ours",
                    live_name="ours_nb",
                ),
            ],
        ),
        seen_live_names_pairs=[("notebook", "ours_nb")],
    )
    llm = snap.to_llm()
    flat = "".join(c["text"] for c in llm if c.get("type") == "text")
    assert 'live_name="ours_nb"' in flat
    assert 'live_name="sibling_nb"' not in flat


def test_workspace_artifact_xml_emits_live_name_attribute() -> None:
    """The unified <turn_artifacts> view surfaces only kind + live_name.

    Refs (logical_id / content_sha) are framework-internal — the agent
    composes by live_name, never by hash.
    """
    from parsimony_agents.agent.session_state import WorkspaceArtifactLine

    s = SessionState(
        kernel=[],
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="notebooks/demo.py",
                kind="notebook",
                summary="x",
                live_name="demo",
            ),
            WorkspaceArtifactLine(
                path=".ockham/datasets/lid1/csha1.parquet",
                kind="dataset",
                summary="y",
                live_name="us_gdp",
            ),
            WorkspaceArtifactLine(
                path="data/extra.parquet",
                kind="data_object",
                summary="z",
            ),
        ],
    )
    text = s.to_llm_text()
    assert 'kind="notebook" live_name="demo"' in text
    assert 'kind="dataset" live_name="us_gdp"' in text
    # data_object rows have no live_name (no human-facing slug).
    assert 'kind="data_object"' in text
    # Hash triplets must not appear anywhere in the agent-facing render.
    assert "logical_id" not in text
    assert "content_sha" not in text


def test_kernel_output_to_llm_omits_ref_blocks() -> None:
    """KernelOutput must NOT surface notebook_ref or data_object_ref triplets.

    With the new model the agent never types refs; lineage is derived by
    the framework from the producer-scoped run. Surfacing refs would just
    tempt the LLM to re-paste them.
    """
    from parsimony.result import Provenance

    from parsimony_agents.execution.outputs import FetchLogEntry, KernelOutput
    from parsimony_agents.identity import ArtifactRef

    ref = ArtifactRef(kind="data_object", logical_id="lid_a", content_sha="csha_a")
    entry = FetchLogEntry(
        provenance=Provenance(
            source="fred_fetch",
            source_description="FRED",
            params={"series_id": "GDPC1"},
        ),
        row_count=10,
        column_names=["date", "value"],
        columns=[],
        data_object_ref=ref,
    )
    ko = KernelOutput(
        outputs=[],
        fetch_log=[entry],
        metadata={
            "notebook_ref": {
                "kind": "notebook",
                "logical_id": "lid",
                "content_sha": "csha",
            }
        },
    )
    flat = "".join(b.get("text", "") for b in ko.to_llm())
    # The fetch log still surfaces (informational), but the ref triplet is gone.
    assert "<fetch_log>" in flat
    assert 'source="fred_fetch"' in flat
    assert 'logical_id="lid_a"' not in flat
    assert 'content_sha="csha_a"' not in flat
    # The notebook_ref metadata block must NOT render either.
    assert "<notebook_ref" not in flat
    # No fetch_log → no block.
    assert KernelOutput(outputs=[]).to_llm() == [{"type": "text", "text": "Out:\n---\n"}]
