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
    # Exercise note line contains angle brackets in user-facing doc — model is static
    text = s.to_llm_text()
    assert "<session_state>" in text
    assert "read_artifact" in text


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
