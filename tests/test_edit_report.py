"""Tests for the ``edit_report`` agent tool.

Surgical-edit producer for reports: reads the latest snapshot via
``log.jsonl``, applies a substring replacement, re-extracts embedded
refs from the new markdown, and returns a ``Report`` whose
``logical_id`` matches the original (this is a revision, not a new
report).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsimony_agents.agent.agent import Agent
from parsimony_agents.agent.models import AgentContext, AgentMessage
from parsimony_agents.artifacts import Report
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.identity import ArtifactRef, content_sha
from parsimony_agents.messages import Text


def _ctx_with_seen(session_id: str, kind: str, live_name: str) -> AgentContext:
    """Build an AgentContext whose seen-set already carries ``(kind, live_name)``.

    Simulates a prior turn that minted or read the artifact, so the
    cross-terminal collision gate treats this terminal as the owner.
    """
    return AgentContext(
        session_id=session_id,
        messages=[
            AgentMessage(
                role="assistant",
                content=Text(
                    content=f'<artifact_ref kind="{kind}" live_name="{live_name}"/>'
                ),
            )
        ],
    )


class _ReportExecutor:
    """In-memory FS executor stub keyed by absolute workspace paths."""

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

    async def eval(  # noqa: ARG002
        self, expr: str, dry_run: bool = False, timeout_seconds: float | None = None
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    async def read_workspace_file(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def write_workspace_file(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def delete_workspace_file(self, path: str) -> None:
        self.files.pop(path, None)

    async def list_workspace_files(self, prefix: str = "") -> list[tuple[str, int]]:
        return [(p, len(d)) for p, d in self.files.items() if p.startswith(prefix)]

    async def execute_workspace(  # noqa: ARG002
        self, code: str, dry_run: bool = False, timeout_seconds: float | None = None
    ) -> KernelOutput:
        return KernelOutput(outputs=[])

    def get_locals(self) -> dict[str, object]:
        return {}


def _seed_report(
    executor: _ReportExecutor,
    *,
    logical_id: str,
    markdown: str,
    title: str = "My Report",
    description: str = "",
    tags: list[str] | None = None,
    notes: list[str] | None = None,
) -> ArtifactRef:
    """Write a published report's snapshot + log + curation into the stub FS."""
    blob = markdown.encode("utf-8")
    csha = content_sha(blob)
    snap_path = f".ockham/reports/{logical_id}/{csha}.report.md"
    log_path = f".ockham/reports/{logical_id}/log.jsonl"
    cur_path = f".ockham/reports/{logical_id}/curation.json"
    executor.files[snap_path] = blob
    executor.files[log_path] = (
        json.dumps({"ts": "t1", "content_sha": csha, "inputs": {}}) + "\n"
    ).encode("utf-8")
    executor.files[cur_path] = json.dumps(
        {
            "kind": "report",
            "logical_id": logical_id,
            "title": title,
            "description": description,
            "tags": list(tags or []),
            "notes": list(notes or []),
            "live_name": "my-report",
        }
    ).encode("utf-8")
    return ArtifactRef(kind="report", logical_id=logical_id, content_sha=csha)


def _make_agent(executor: _ReportExecutor) -> Agent:
    agent = Agent(
        model_config={"model": "test-model"},
        instructions="test",
        session_id="s",
    )
    agent.code_executor = executor  # type: ignore[assignment]
    # Bypass ref-resolve validation: that helper queries the workspace
    # API; for unit tests we trust the stub FS.
    async def _accept(_refs):
        return None

    agent._validate_refs_resolve = _accept  # type: ignore[method-assign]
    return agent


@pytest.mark.asyncio
async def test_edit_report_preserves_logical_id() -> None:
    """A surgical edit produces a new revision under the same logical_id."""
    ex = _ReportExecutor()
    _seed_report(
        ex,
        logical_id="rep1",
        markdown="# Title\n\nThe answer is 42.\n",
        title="My Report",
    )
    agent = _make_agent(ex)
    ctx = _ctx_with_seen("s", "report", "my-report")

    tr = await agent.edit_report(
        context=ctx,
        live_name="my-report",
        old_str="The answer is 42.",
        new_str="The answer is 43.",
    )

    assert tr.success, getattr(tr, "exception_message", "")
    report = tr.data
    assert isinstance(report, Report)
    assert report.logical_id == "rep1"
    assert "43" in report.markdown
    assert "42" not in report.markdown


@pytest.mark.asyncio
async def test_edit_report_rejects_unknown_slug() -> None:
    """A live_name that doesn't resolve to a report errors out clearly."""
    ex = _ReportExecutor()
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        live_name="not-a-thing",
        old_str="a",
        new_str="b",
    )
    assert not tr.success
    assert "live_name" in tr.exception_message or "report" in tr.exception_message


@pytest.mark.asyncio
async def test_edit_report_rejects_empty_old_str() -> None:
    """Full-body rewrites should go through ``return_report``, not ``edit_report``."""
    ex = _ReportExecutor()
    _seed_report(ex, logical_id="rep2", markdown="# Hi\n")
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        live_name="my-report",
        old_str="",
        new_str="anything",
    )
    assert not tr.success
    assert "old_str must be a non-empty" in tr.exception_message


@pytest.mark.asyncio
async def test_edit_report_errors_when_log_missing() -> None:
    """Editing a slug that resolves to a report with no log → clear error."""
    ex = _ReportExecutor()
    # Seed a curation but no snapshot/log files.
    ex.files[".ockham/reports/ghost/curation.json"] = json.dumps(
        {"kind": "report", "logical_id": "ghost", "title": "T",
         "tags": [], "notes": [], "live_name": "ghost-slug"}
    ).encode("utf-8")
    agent = _make_agent(ex)
    ctx = _ctx_with_seen("s", "report", "ghost-slug")

    tr = await agent.edit_report(
        context=ctx,
        live_name="ghost-slug",
        old_str="anything",
        new_str="other",
    )
    assert not tr.success
    assert "log.jsonl" in tr.exception_message or "no published artifact" in tr.exception_message.lower()


@pytest.mark.asyncio
async def test_edit_report_rejects_non_unique_old_str() -> None:
    """``old_str`` must occur exactly once."""
    ex = _ReportExecutor()
    _seed_report(
        ex,
        logical_id="rep3",
        markdown="# Hi\n\nfoo bar\n\nfoo baz\n",
    )
    agent = _make_agent(ex)
    ctx = _ctx_with_seen("s", "report", "my-report")

    tr = await agent.edit_report(
        context=ctx,
        live_name="my-report",
        old_str="foo",
        new_str="qux",
    )
    assert not tr.success
    assert "occurs multiple times" in tr.exception_message


@pytest.mark.asyncio
async def test_edit_report_resolves_to_latest_snapshot() -> None:
    """The edit always targets the latest revision in log.jsonl."""
    ex = _ReportExecutor()
    md1 = "# Hi\n\noriginal text\n"
    csha1 = content_sha(md1.encode("utf-8"))
    snap_path1 = f".ockham/reports/rep4/{csha1}.report.md"
    ex.files[snap_path1] = md1.encode("utf-8")
    md2 = "# Hi\n\nupdated text\n"
    csha2 = content_sha(md2.encode("utf-8"))
    snap_path2 = f".ockham/reports/rep4/{csha2}.report.md"
    ex.files[snap_path2] = md2.encode("utf-8")
    ex.files[".ockham/reports/rep4/log.jsonl"] = (
        json.dumps({"ts": "t1", "content_sha": csha1, "inputs": {}}) + "\n"
        + json.dumps({"ts": "t2", "content_sha": csha2, "inputs": {}}) + "\n"
    ).encode("utf-8")
    ex.files[".ockham/reports/rep4/curation.json"] = json.dumps(
        {"kind": "report", "logical_id": "rep4", "title": "T", "tags": [], "notes": [], "live_name": "report-4"}
    ).encode("utf-8")

    agent = _make_agent(ex)
    ctx = _ctx_with_seen("s", "report", "report-4")

    tr = await agent.edit_report(
        context=ctx,
        live_name="report-4",
        old_str="updated text",
        new_str="patched text",
    )
    assert tr.success, getattr(tr, "exception_message", "")
    assert "patched text" in tr.data.markdown
    assert "original text" not in tr.data.markdown
