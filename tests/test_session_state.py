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
    from parsimony_agents.agent.agent import Agent
    from parsimony_agents.agent.models import AgentContext
    from parsimony_agents.execution.executor import CodeExecutor
    from parsimony_agents.execution.factory import OutputFactory as FrameworkOutputFactory
    from parsimony_agents.messages import Text
    import tempfile

    from parsimony_agents.agent.outputs import ArtifactLlmResult

    async def _fn(path: str, options: dict) -> ArtifactLlmResult:
        m = (options.get("view") or options.get("mode") or "summary") or "summary"
        if isinstance(m, str):
            m = m.strip().lower()
        return ArtifactLlmResult(text=f"{path}|{m}")

    root = tempfile.mkdtemp()
    of = FrameworkOutputFactory(local_dir=root)
    ex = CodeExecutor(cwd=root, output_factory=of)
    agent = Agent(
        model="m",
        code_executor=ex,  # type: ignore[arg-type]
        output_factory=of,
        read_artifact_fn=_fn,
    )
    out = await agent.read_artifact(path="n.py", mode="summary", context=AgentContext(session_id="s"))
    assert out.data.content is not None
    assert isinstance(out.data.content, Text)
    assert "n.py" in out.data.content.content
    out2 = await agent.read_artifact(path="n.py", mode="full", context=AgentContext(session_id="s"))
    assert out2.data.content is not None
    assert isinstance(out2.data.content, Text)
    assert "full" in out2.data.content.content


def test_agent_context_snapshot_includes_session_state() -> None:
    from parsimony_agents.agent.session_state import KernelVariableSummary, WorkspaceArtifactLine

    snap = AgentContextSnapshot(
        session_state=SessionState(
            kernel=[KernelVariableSummary(name="df", kind="dataframe", detail="1×1")],
            workspace_artifacts=[WorkspaceArtifactLine(path="n.py", kind="notebook", summary="x")],
        )
    )
    llm = snap.to_llm()
    flat = "".join(c["text"] for c in llm if c.get("type") == "text")
    assert "<session_state>" in flat
    assert 'name="df"' in flat
    assert "n.py" in flat


def test_workspace_artifact_xml_emits_ref_attributes_when_set() -> None:
    """Logical/content sha attributes must round-trip into the XML the LLM sees.

    Without this, the agent has no canonical ref to copy verbatim into
    return_dataset / return_chart / return_report and tries to recompute it
    locally — which silently diverges from the framework's identity.
    """
    from parsimony_agents.agent.session_state import WorkspaceArtifactLine
    from parsimony_agents.identity import ArtifactRef

    s = SessionState(
        kernel=[],
        workspace_artifacts=[
            WorkspaceArtifactLine(
                path="notebooks/demo.py",
                kind="notebook",
                summary="x",
                ref=ArtifactRef(kind="notebook", logical_id="abc123", content_sha="abc123"),
            ),
            WorkspaceArtifactLine(
                path=".ockham/datasets/lid1/csha1.parquet",
                kind="dataset",
                summary="y",
                ref=ArtifactRef(kind="dataset", logical_id="lid1", content_sha="csha1"),
            ),
            WorkspaceArtifactLine(
                path="data/extra.parquet",
                kind="data_object",
                summary="z",  # ref omitted — data_object refs come via <fetch_log>
            ),
        ],
    )
    text = s.to_llm_text()
    assert 'logical_id="abc123" content_sha="abc123"' in text
    assert 'logical_id="lid1" content_sha="csha1"' in text
    # data_object entry has no canonical ref triplet (those come via fetch_log).
    assert "logical_id" not in text.split('path="data/extra.parquet"')[1].split("</artifact>")[0]


def test_kernel_output_to_llm_emits_notebook_ref_block_when_metadata_set() -> None:
    """Code-tool kernel results must surface the notebook's canonical ref.

    Without this, the agent only sees the file path and prose output —
    it has no way to learn the notebook's ``(logical_id, content_sha)``
    pair to copy verbatim into ``return_dataset.notebook_refs``.
    """
    from parsimony_agents.execution.outputs import KernelOutput
    from parsimony_agents.identity import notebook_content_sha

    code = "x = 1\n"
    csha = notebook_content_sha(code)
    lid = "some-uuid"
    ko = KernelOutput(
        outputs=[],
        metadata={
            "notebook_ref": {
                "kind": "notebook",
                "logical_id": lid,
                "content_sha": csha,
            }
        },
    )
    flat = "".join(b.get("text", "") for b in ko.to_llm())
    assert "<notebook_ref " in flat
    assert f'logical_id="{lid}"' in flat
    assert f'content_sha="{csha}"' in flat
    # Empty metadata → no block.
    assert "<notebook_ref" not in "".join(
        b.get("text", "") for b in KernelOutput(outputs=[]).to_llm()
    )


def test_kernel_output_to_llm_emits_fetch_log_block() -> None:
    """``data_object_ref`` triplets must surface in the LLM output of a code run.

    Without this block the agent would have to invent ``source_refs`` for
    return_dataset / return_chart / return_report from prose, which is what
    led to hallucinated hashes in earlier traces.
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
    ko = KernelOutput(outputs=[], fetch_log=[entry])
    blocks = ko.to_llm()
    flat = "".join(b.get("text", "") for b in blocks)
    assert "<fetch_log>" in flat
    assert 'kind="data_object"' in flat
    assert 'logical_id="lid_a"' in flat
    assert 'content_sha="csha_a"' in flat
    # No fetch_log → no block.
    assert KernelOutput(outputs=[]).to_llm() == [{"type": "text", "text": "Out:\n---\n"}]
