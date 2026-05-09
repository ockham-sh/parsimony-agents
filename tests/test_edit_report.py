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
from parsimony_agents.agent.models import AgentContext
from parsimony_agents.artifacts import Report
from parsimony_agents.execution.outputs import KernelOutput
from parsimony_agents.identity import ArtifactRef, content_sha
from parsimony_agents.report_io import read_report_bytes, write_report_bytes


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
    formats: list[str] | None = None,
) -> ArtifactRef:
    """Write a published report's snapshot + log + curation into the stub FS."""
    seed = Report(
        logical_id=logical_id,
        title=title,
        description=description,
        tags=list(tags or []),
        notes=list(notes or []),
        markdown=markdown,
        embedded_refs=[],
        live_name="my-report",
        **({"formats": formats} if formats else {}),
    )
    blob = write_report_bytes(seed)
    csha = content_sha(blob)
    snap_path = f".ockham/reports/{logical_id}/{csha}.qmd"
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
    ref = _seed_report(
        ex,
        logical_id="rep1",
        markdown="# Title\n\nThe answer is 42.\n",
        title="My Report",
    )
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "report", "logical_id": ref.logical_id, "content_sha": ref.content_sha},
        old_str="The answer is 42.",
        new_str="The answer is 43.",
    )

    assert tr.success
    report = tr.data
    assert isinstance(report, Report)
    assert report.logical_id == "rep1"
    assert "43" in report.markdown
    assert "42" not in report.markdown


@pytest.mark.asyncio
async def test_edit_report_rejects_non_report_ref() -> None:
    """``ref`` must be ``kind='report'``."""
    ex = _ReportExecutor()
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "dataset", "logical_id": "d1", "content_sha": "x" * 64},
        old_str="a",
        new_str="b",
    )
    assert not tr.success
    assert "kind='report'" in tr.exception_message


@pytest.mark.asyncio
async def test_edit_report_rejects_empty_old_str() -> None:
    """Full-body rewrites should go through ``return_report``, not ``edit_report``."""
    ex = _ReportExecutor()
    ref = _seed_report(ex, logical_id="rep2", markdown="# Hi\n")
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "report", "logical_id": ref.logical_id, "content_sha": ref.content_sha},
        old_str="",
        new_str="anything",
    )
    assert not tr.success
    assert "old_str must be a non-empty" in tr.exception_message


@pytest.mark.asyncio
async def test_edit_report_errors_when_log_missing() -> None:
    """Editing a report that was never persisted → clear error."""
    ex = _ReportExecutor()
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "report", "logical_id": "ghost", "content_sha": "z" * 64},
        old_str="anything",
        new_str="other",
    )
    assert not tr.success
    assert "no log.jsonl" in tr.exception_message


@pytest.mark.asyncio
async def test_edit_report_rejects_non_unique_old_str() -> None:
    """``old_str`` must occur exactly once."""
    ex = _ReportExecutor()
    ref = _seed_report(
        ex,
        logical_id="rep3",
        markdown="# Hi\n\nfoo bar\n\nfoo baz\n",
    )
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "report", "logical_id": ref.logical_id, "content_sha": ref.content_sha},
        old_str="foo",
        new_str="qux",
    )
    assert not tr.success
    assert "occurs multiple times" in tr.exception_message


@pytest.mark.asyncio
async def test_edit_report_resolves_to_latest_snapshot() -> None:
    """When ``ref.content_sha`` is stale, the edit applies to the latest revision."""
    ex = _ReportExecutor()
    # Initial revision.
    blob1 = write_report_bytes(
        Report(logical_id="rep4", title="T", markdown="# Hi\n\noriginal text\n", embedded_refs=[])
    )
    csha1 = content_sha(blob1)
    ex.files[f".ockham/reports/rep4/{csha1}.qmd"] = blob1
    # Newer revision (refresh, etc.). Both entries in log.jsonl; latest wins.
    blob2 = write_report_bytes(
        Report(logical_id="rep4", title="T", markdown="# Hi\n\nupdated text\n", embedded_refs=[])
    )
    csha2 = content_sha(blob2)
    ex.files[f".ockham/reports/rep4/{csha2}.qmd"] = blob2
    ex.files[".ockham/reports/rep4/log.jsonl"] = (
        json.dumps({"ts": "t1", "content_sha": csha1, "inputs": {}}) + "\n"
        + json.dumps({"ts": "t2", "content_sha": csha2, "inputs": {}}) + "\n"
    ).encode("utf-8")
    ex.files[".ockham/reports/rep4/curation.json"] = json.dumps(
        {"kind": "report", "logical_id": "rep4", "title": "T", "tags": [], "notes": [], "live_name": "t"}
    ).encode("utf-8")

    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    # Caller passes a *stale* csha — the older revision.
    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "report", "logical_id": "rep4", "content_sha": csha1},
        old_str="updated text",
        new_str="patched text",
    )
    assert tr.success
    assert "patched text" in tr.data.markdown
    # Original csha1 said "original text"; the stale ref didn't pin the
    # base — edit_report worked against csha2 (latest).
    assert "original text" not in tr.data.markdown


@pytest.mark.asyncio
async def test_edit_report_preserves_formats_from_prior_yaml() -> None:
    """edit_report carries the prior snapshot's formats forward — body-only edit."""
    ex = _ReportExecutor()
    ref = _seed_report(
        ex,
        logical_id="rep_fmts",
        markdown="# Hi\n\noriginal text\n",
        formats=["html", "pptx", "dashboard"],
    )
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "report", "logical_id": ref.logical_id, "content_sha": ref.content_sha},
        old_str="original text",
        new_str="updated text",
    )
    assert tr.success
    assert tr.data.formats == ["html", "pptx", "dashboard"]


@pytest.mark.asyncio
async def test_edit_report_writes_qmd_compatible_body() -> None:
    """The Report returned by edit_report carries body-only markdown (no YAML preamble)."""
    ex = _ReportExecutor()
    ref = _seed_report(ex, logical_id="rep_body", markdown="# Hi\n\nfind me\n")
    agent = _make_agent(ex)
    ctx = AgentContext(session_id="s")

    tr = await agent.edit_report(
        context=ctx,
        ref={"kind": "report", "logical_id": ref.logical_id, "content_sha": ref.content_sha},
        old_str="find me",
        new_str="found",
    )
    assert tr.success
    # Body must NOT include the YAML preamble.
    assert "ockham:" not in tr.data.markdown
    assert tr.data.markdown.startswith("# Hi")
    # Re-emitting via write_report_bytes should round-trip.
    blob = write_report_bytes(tr.data)
    yaml_dict, body = read_report_bytes(blob)
    assert "found" in body
    assert yaml_dict["title"] == "My Report"  # carried from curation
